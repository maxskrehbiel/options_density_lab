"""Static no-arbitrage checks and sanity checks on a recovered distribution.

Covers strike monotonicity, butterfly (convexity) and calendar arbitrage, plus
mass, positivity, CDF shape and the martingale (mean equals forward) condition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._types import FloatArray
from .density import RiskNeutralDensity
from .smile import Smile

PRICE_TOLERANCE = 1e-10
BUTTERFLY_GRID_POINTS = 400
CALENDAR_GRID_POINTS = 201


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check.

    Attributes:
        name: Short name of the check.
        passed: Whether the check passed.
        n_violations: Number of violating points.
        worst: Size of the worst violation (0 when passed).
        message: One-line plain-English summary.
        details: Per-point table, where useful.
    """

    name: str
    passed: bool
    n_violations: int
    worst: float
    message: str
    details: pd.DataFrame | None = field(default=None, repr=False)


def check_call_prices(
    strikes: npt.ArrayLike, calls: npt.ArrayLike, discount: float, tol: float = PRICE_TOLERANCE
) -> CheckResult:
    """Check call prices for monotonicity and convexity across strikes.

    A higher strike can never be worth more, and no call can lose value
    faster than the discounted strike rises: ``-D <= dC/dK <= 0``. A butterfly
    (long ``K1`` and ``K3``, short ``K2``) never pays less than zero, so call
    prices must be convex: ``C(K2) <= lam C(K1) + (1 - lam) C(K3)`` with
    ``lam = (K3 - K2)/(K3 - K1)``. Works for any strike spacing.

    Args:
        strikes: Strike prices.
        calls: Call prices at those strikes.
        discount: Discount factor.
        tol: Violations smaller than this are ignored.

    Returns:
        The check result; ``details`` lists each violation.
    """
    k = np.asarray(strikes, dtype=np.float64)
    c = np.asarray(calls, dtype=np.float64)
    order = np.argsort(k)
    k, c = k[order], c[order]
    slope = np.diff(c) / np.diff(k)
    rows = [
        {
            "type": "monotonicity",
            "strike": float(k[i + 1]),
            "amount": float(max(slope[i], -discount - slope[i])),
        }
        for i in np.flatnonzero((slope > tol) | (slope < -discount - tol))
    ]
    lam = (k[2:] - k[1:-1]) / (k[2:] - k[:-2])
    excess = c[1:-1] - (lam * c[:-2] + (1.0 - lam) * c[2:])
    rows += [
        {"type": "butterfly", "strike": float(k[i + 1]), "amount": float(excess[i])}
        for i in np.flatnonzero(excess > tol)
    ]
    details = pd.DataFrame(rows, columns=["type", "strike", "amount"])
    if details.empty:
        return CheckResult(
            "call-price arbitrage",
            True,
            0,
            0.0,
            "calls are decreasing and convex in strike",
            details,
        )
    worst_row = details.iloc[int(np.argmax(details["amount"].to_numpy()))]
    message = (
        f"{len(details)} violation(s); worst {worst_row['amount']:.4g} "
        f"({worst_row['type']}) at strike {worst_row['strike']:g}"
    )
    return CheckResult(
        "call-price arbitrage", False, len(details), float(worst_row["amount"]), message, details
    )


def check_quote_butterflies(otm: pd.DataFrame, forward: float, discount: float) -> CheckResult:
    """Look for butterflies that could be traded for a riskless profit at the quotes.

    Mid-prices of noisy quotes often look slightly non-convex; that is noise,
    not arbitrage. A tradable violation needs the body sold at the bid and the
    wings bought at the ask to still come out ahead. Puts are first converted
    to calls with put-call parity, which leaves the spread unchanged.

    Args:
        otm: Out-of-the-money quotes sorted by strike, with ``strike``,
            ``option_type``, ``bid`` and ``ask`` columns.
        forward: Forward price.
        discount: Discount factor.

    Returns:
        Passes when no butterfly is tradable; the message also counts
        mid-price non-convexities.
    """
    k = otm["strike"].to_numpy(dtype=float)
    to_call = np.where((otm["option_type"] == "P").to_numpy(), discount * (forward - k), 0.0)
    bid = otm["bid"].to_numpy(dtype=float) + to_call
    ask = otm["ask"].to_numpy(dtype=float) + to_call
    mid = 0.5 * (bid + ask)
    lam = (k[2:] - k[1:-1]) / (k[2:] - k[:-2])
    tradable = bid[1:-1] - (lam * ask[:-2] + (1.0 - lam) * ask[2:])
    at_mid = mid[1:-1] - (lam * mid[:-2] + (1.0 - lam) * mid[2:])
    n_trade = int((tradable > PRICE_TOLERANCE).sum())
    n_mid = int((at_mid > PRICE_TOLERANCE).sum())
    details = pd.DataFrame({"strike": k[1:-1], "tradable_profit": tradable, "mid_excess": at_mid})
    message = f"{n_mid} non-convex at mid-prices (noise); {n_trade} tradable at bid/ask"
    worst = float(max(tradable.max(initial=0.0), 0.0))
    return CheckResult("quoted butterflies", n_trade == 0, n_trade, worst, message, details)


def check_smile_butterfly(smile: Smile, k: FloatArray | None = None) -> CheckResult:
    """Check ``g(k) >= 0`` (non-negative implied density) on a grid.

    Args:
        smile: Fitted smile.
        k: Log-moneyness grid; defaults to ``BUTTERFLY_GRID_POINTS`` across the quoted range.

    Returns:
        The check result; ``details`` holds ``g`` on the grid.
    """
    grid = np.linspace(smile.k_lo, smile.k_hi, BUTTERFLY_GRID_POINTS) if k is None else k
    g = smile.g(grid)
    n_bad = int((g < 0).sum())
    message = (
        f"implied density non-negative (min g = {g.min():.3f})"
        if n_bad == 0
        else f"negative implied density at {n_bad} grid point(s) (min g = {g.min():.3g})"
    )
    return CheckResult(
        "smile butterfly",
        n_bad == 0,
        n_bad,
        float(max(-g.min(), 0.0)),
        message,
        pd.DataFrame({"k": grid, "g": g}),
    )


def check_calendar(smiles: Sequence[Smile], tol: float = PRICE_TOLERANCE) -> CheckResult:
    """Check that total variance never falls from one expiry to the next.

    Each adjacent pair is compared across the union of their quoted ranges.

    Args:
        smiles: Smiles for several expiries, in any order.
        tol: Decreases smaller than this are ignored.

    Returns:
        The check result; ``details`` lists each violating grid point.
    """
    ordered = sorted(smiles, key=lambda s: s.maturity)
    rows = []
    for short, long in pairwise(ordered):
        grid = np.linspace(
            min(short.k_lo, long.k_lo), max(short.k_hi, long.k_hi), CALENDAR_GRID_POINTS
        )
        gap = short.total_variance(grid) - long.total_variance(grid)
        rows += [
            {
                "t_short": short.maturity,
                "t_long": long.maturity,
                "k": float(kv),
                "amount": float(gv),
            }
            for kv, gv in zip(grid[gap > tol], gap[gap > tol], strict=True)
        ]
    details = pd.DataFrame(rows, columns=["t_short", "t_long", "k", "amount"])
    if details.empty:
        return CheckResult(
            "calendar", True, 0, 0.0, "total variance rises with expiry everywhere checked", details
        )
    worst = float(details["amount"].max())
    message = (
        f"total variance falls with expiry at {len(details)} grid point(s) (worst {worst:.2e})"
    )
    return CheckResult("calendar", False, len(details), worst, message, details)


def check_density(
    dens: RiskNeutralDensity,
    mass_tol: float = 2e-3,
    mean_tol: float = 1e-3,
    pdf_rel_tol: float = 1e-6,
) -> list[CheckResult]:
    """Sanity checks on a recovered distribution.

    Args:
        dens: The recovered distribution.
        mass_tol: Allowed deviation of total probability from 1.
        mean_tol: Allowed relative deviation of the mean price from the forward.
        pdf_rel_tol: Negative density values smaller than this fraction of the
            peak are finite-difference round-off in the far tails, not arbitrage.

    Returns:
        Four results: integrates to one, non-negative, CDF monotone within
        ``[0, 1]``, and mean equals forward (the risk-neutral mean of the price
        at expiry must be the forward).
    """
    mass = dens.mass()
    min_pdf = float(dens.pdf.min())
    pdf_floor = -pdf_rel_tol * float(dens.pdf.max())
    drops = np.diff(dens.cdf)
    cdf_ok = bool(
        drops.min() >= -PRICE_TOLERANCE and dens.cdf.min() >= -1e-9 and dens.cdf.max() <= 1 + 1e-9
    )
    rel = dens.moments()["mean"] / dens.forward - 1.0
    return [
        CheckResult(
            "integrates to one",
            abs(mass - 1) < mass_tol,
            int(abs(mass - 1) >= mass_tol),
            abs(mass - 1),
            f"total probability {mass:.5f}",
        ),
        CheckResult(
            "non-negative density",
            min_pdf >= pdf_floor,
            int((dens.pdf < pdf_floor).sum()),
            max(-min_pdf, 0.0),
            f"minimum density {min_pdf:.3g}",
        ),
        CheckResult(
            "CDF monotone in [0, 1]",
            cdf_ok,
            int((drops < -PRICE_TOLERANCE).sum()),
            float(max(-drops.min(), 0.0)),
            "CDF non-decreasing and within [0, 1]"
            if cdf_ok
            else "CDF decreases somewhere or leaves [0, 1]",
        ),
        CheckResult(
            "mean equals forward",
            abs(rel) < mean_tol,
            int(abs(rel) >= mean_tol),
            abs(rel),
            f"E[S_T]/F - 1 = {rel:+.2e}",
        ),
    ]
