"""Error metrics comparing a recovered distribution with the known synthetic truth.

Only this module reads the hidden truth; the recovery pipeline never does.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ._types import FloatArray
from .density import DensityConfig, distribution_moments, rnd_from_smile
from .pipeline import ExpiryResult, SurfaceResult
from .smile import SVISmile
from .synthetic import MarketSetup

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
METRIC_DESCRIPTIONS = {
    "tv": "total-variation distance: the largest disagreement in probability about any event",
    "tv_tails": "part of tv from beyond the quoted strikes (extrapolation, not data)",
    "ks": "largest gap between the recovered and true CDFs",
    "q_err_max_pct": "worst error across the 1%-99% quantiles, in % of the forward",
    "iv_rmse_volpts": "root-mean-square smile error at the quoted strikes, in vol points",
}


@dataclass
class TruthCache:
    """True density and CDF per expiry on a fine master grid, computed once.

    The truth depends only on the market setup, not on the quote noise, so
    one cache serves every chain generated from the same setup.

    Attributes:
        setup: The market whose truth is cached.
        n: Master grid size per expiry (evenly spaced in log-price, where the
            density is smooth, so linear interpolation error stays near 1e-5).
        left_sds: Master grid lower reach, in standard deviations of the log-price.
        right_sds: Master grid upper reach.
    """

    setup: MarketSetup
    n: int = 2001
    left_sds: float = 16.0
    right_sds: float = 10.0
    _store: dict[float, tuple[FloatArray, FloatArray, FloatArray]] = field(
        default_factory=dict, repr=False
    )

    def _master(self, maturity: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        if maturity not in self._store:
            params = self.setup.heston
            fwd = self.setup.forward(maturity)
            sd = float(np.sqrt(max(params.v0, params.theta) * maturity))
            grid = fwd * np.exp(np.linspace(-self.left_sds * sd, self.right_sds * sd, self.n))
            self._store[maturity] = (
                grid,
                self.setup.true_pdf(grid, maturity),
                self.setup.true_cdf(grid, maturity),
            )
        return self._store[maturity]

    def on(self, strikes: FloatArray, maturity: float) -> tuple[FloatArray, FloatArray]:
        """True density and CDF interpolated onto a grid.

        Args:
            strikes: Price grid.
            maturity: Time to expiry in years.

        Returns:
            ``(pdf, cdf)``.
        """
        grid, pdf, cdf = self._master(maturity)
        return (
            np.asarray(np.interp(strikes, grid, pdf, left=0.0, right=0.0), dtype=np.float64),
            np.asarray(np.interp(strikes, grid, cdf, left=0.0, right=1.0), dtype=np.float64),
        )

    def quantiles(self, maturity: float, probs: tuple[float, ...] = QUANTILES) -> FloatArray:
        """True quantiles of the price at expiry.

        Args:
            maturity: Time to expiry in years.
            probs: Probabilities.

        Returns:
            Price levels.
        """
        grid, _, cdf = self._master(maturity)
        return np.asarray(np.interp(probs, np.maximum.accumulate(cdf), grid), dtype=np.float64)


def compare(result: ExpiryResult, truth: TruthCache) -> dict[str, float | str | int]:
    """Error metrics for one expiry (see ``METRIC_DESCRIPTIONS``).

    Args:
        result: One expiry's pipeline output.
        truth: Cached truth for the same market setup.

    Returns:
        Metrics plus recovered and true quantiles, moments and tail masses.
    """
    d = result.density
    t = d.maturity
    fwd_true = truth.setup.forward(t)
    pdf_t, cdf_t = truth.on(d.strikes, t)
    gap = np.abs(d.pdf - pdf_t)
    outside = (d.strikes < d.strike_lo) | (d.strikes > d.strike_hi)
    q_hat = d.quantile(np.array(QUANTILES))
    q_true = truth.quantiles(t)
    m_hat = d.moments()
    m_true = distribution_moments(d.strikes, pdf_t, fwd_true, t)
    otm = result.otm[result.used]
    iv_true = truth.setup.true_iv(otm["strike"].to_numpy(dtype=float), t)
    iv_fit = result.smile.iv(otm["k"].to_numpy(dtype=float))
    row: dict[str, float | str | int] = {
        "days": result.days,
        "method": d.method,
        "tails": d.tails,
        "tv": 0.5 * float(np.trapezoid(gap, d.strikes)),
        "tv_tails": 0.5 * float(np.trapezoid(gap * outside, d.strikes)),
        "ks": float(np.max(np.abs(d.cdf - cdf_t))),
        "q_err_max_pct": float(np.max(np.abs(q_hat - q_true)) / fwd_true * 100.0),
        "iv_rmse_volpts": float(np.sqrt(np.mean((iv_fit - iv_true) ** 2)) * 100.0),
        "fwd_err_bp": (d.forward / fwd_true - 1.0) * 1e4,
        "disc_err_bp": (d.discount / truth.setup.discount(t) - 1.0) * 1e4,
        "tail_left_hat": d.tail_mass_left,
        "tail_left_true": float(np.interp(d.strike_lo, d.strikes, cdf_t)),
        "tail_right_hat": d.tail_mass_right,
        "tail_right_true": 1.0 - float(np.interp(d.strike_hi, d.strikes, cdf_t)),
    }
    for name in ("sd_log", "skew_log", "excess_kurt_log"):
        row[f"{name}_hat"] = m_hat[name]
        row[f"{name}_true"] = m_true[name]
    for p, a, b in zip(QUANTILES, q_hat, q_true, strict=True):
        row[f"q{round(p * 100):02d}_hat"] = float(a)
        row[f"q{round(p * 100):02d}_true"] = float(b)
    return row


def lognormal_baseline(result: ExpiryResult) -> ExpiryResult:
    """The same expiry with the smile flattened to its at-the-money volatility.

    A flat smile implies a plain bell curve in log-price (the Black-Scholes
    assumption). Scoring it against the truth shows how much of the shape the
    smile-based recovery actually captures.

    Args:
        result: One expiry's pipeline output.

    Returns:
        A copy whose smile and density come from the flat volatility.
    """
    sm = result.smile
    w_atm = float(sm.total_variance(np.zeros(1))[0])
    flat = SVISmile(
        a=w_atm,
        b=0.0,
        rho=0.0,
        m=0.0,
        s=1.0,
        maturity=sm.maturity,
        forward=sm.forward,
        k_lo=sm.k_lo,
        k_hi=sm.k_hi,
        method="lognormal",
    )
    dens = rnd_from_smile(flat, result.clean.discount, DensityConfig(tails="smile"))
    return replace(result, smile=flat, density=dens)


def validate_surface(surface: SurfaceResult, truth: TruthCache) -> pd.DataFrame:
    """Error metrics for every expiry of a surface.

    Args:
        surface: Pipeline output.
        truth: Cached truth for the setup the surface's chain came from.

    Returns:
        One row per expiry.
    """
    return pd.DataFrame([compare(e, truth) for e in surface.expiries])
