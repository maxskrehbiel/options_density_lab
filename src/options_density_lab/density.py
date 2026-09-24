"""Breeden-Litzenberger (1978) recovery of the risk-neutral density and CDF from a smile.

``q(K) = (1/D) d2C/dK2`` and ``P(S_T <= K) = 1 + (1/D) dC/dK``, evaluated by
finite differences of call prices on an evenly spaced strike grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._types import FloatArray
from .black76 import call_price_from_total_variance
from .smile import Smile

logger = logging.getLogger(__name__)

TailModel = Literal["smile", "exponential"]
MIN_TAIL_DECAY = 1.05  # a right-tail decay rate <= 1 would give an infinite mean price


@dataclass(frozen=True)
class DensityConfig:
    """Settings for the density grid and tails.

    Attributes:
        n_grid: Number of grid points.
        left_sds: Grid reaches this many at-the-money standard deviations below the forward.
        right_sds: Grid reaches this many at-the-money standard deviations above the forward.
        edge_sds: Minimum margin beyond the quoted range, in the same units.
        tails: ``"smile"``, ``"exponential"``, or ``None`` for the method's default
            (``"smile"`` for SVI, ``"exponential"`` for the spline).
    """

    n_grid: int = 4001
    left_sds: float = 12.0
    right_sds: float = 8.0
    edge_sds: float = 4.0
    tails: TailModel | None = None


@dataclass(frozen=True)
class RiskNeutralDensity:
    """A recovered distribution of the price at expiry, on an evenly spaced grid.

    Attributes:
        strikes: Price grid.
        pdf: Density ``q(K)``.
        cdf: ``P(S_T <= K)``.
        forward: Forward price.
        discount: Discount factor.
        maturity: Time to expiry in years.
        strike_lo: Lowest quoted strike used.
        strike_hi: Highest quoted strike used.
        tail_mass_left: Market-implied ``P(S_T < strike_lo)``.
        tail_mass_right: Market-implied ``P(S_T > strike_hi)``.
        tails: Tail model used beyond the quoted range.
        method: Smile method the density came from.
    """

    strikes: FloatArray
    pdf: FloatArray
    cdf: FloatArray
    forward: float
    discount: float
    maturity: float
    strike_lo: float
    strike_hi: float
    tail_mass_left: float
    tail_mass_right: float
    tails: str
    method: str

    def mass(self) -> float:
        """Total probability on the grid; should be very close to 1.

        Returns:
            The integral of the density.
        """
        return float(np.trapezoid(self.pdf, self.strikes))

    def _cdf_monotone(self) -> FloatArray:
        return np.clip(np.maximum.accumulate(self.cdf), 0.0, 1.0)

    def quantile(self, p: npt.ArrayLike) -> FloatArray:
        """Price level below which the price ends with probability ``p``.

        Args:
            p: Probabilities in ``[0, 1]``.

        Returns:
            Price levels.
        """
        probs = np.asarray(p, dtype=np.float64)
        return np.asarray(np.interp(probs, self._cdf_monotone(), self.strikes), dtype=np.float64)

    def prob_below(self, level: npt.ArrayLike) -> FloatArray:
        """Probability that the price ends at or below ``level``.

        Args:
            level: Price levels.

        Returns:
            Probabilities.
        """
        levels = np.asarray(level, dtype=np.float64)
        return np.asarray(np.interp(levels, self.strikes, self._cdf_monotone()), dtype=np.float64)

    def prob_above(self, level: npt.ArrayLike) -> FloatArray:
        """Probability that the price ends above ``level``.

        Args:
            level: Price levels.

        Returns:
            Probabilities.
        """
        return 1.0 - self.prob_below(level)

    def moments(self) -> dict[str, float]:
        """Mean price and the spread and shape of the log-return ``ln(S_T/F)``.

        Returns:
            ``mean`` (should equal the forward), ``sd_log``, ``vol_annualised``
            (``sd_log / sqrt(T)``), ``skew_log`` and ``excess_kurt_log``.
        """
        return distribution_moments(self.strikes, self.pdf, self.forward, self.maturity)


def distribution_moments(
    strikes: FloatArray, pdf: FloatArray, forward: float, maturity: float
) -> dict[str, float]:
    """Moments of a density given on a grid (normalised to integrate to 1).

    Args:
        strikes: Price grid.
        pdf: Density values.
        forward: Forward price, the reference for log-returns.
        maturity: Time to expiry in years.

    Returns:
        See :meth:`RiskNeutralDensity.moments`.
    """
    q = pdf / np.trapezoid(pdf, strikes)
    x = np.log(strikes / forward)
    mean_x = float(np.trapezoid(x * q, strikes))
    var = float(np.trapezoid((x - mean_x) ** 2 * q, strikes))
    sd = float(np.sqrt(var))
    return {
        "mean": float(np.trapezoid(strikes * q, strikes)),
        "sd_log": sd,
        "vol_annualised": sd / float(np.sqrt(maturity)),
        "skew_log": float(np.trapezoid((x - mean_x) ** 3 * q, strikes)) / sd**3,
        "excess_kurt_log": float(np.trapezoid((x - mean_x) ** 4 * q, strikes)) / var**2 - 3.0,
    }


def price_grid(
    forward: float, atm_total_sd: float, k_lo: float, k_hi: float, config: DensityConfig
) -> FloatArray:
    """Evenly spaced price grid wide enough to hold essentially all the probability.

    Args:
        forward: Forward price.
        atm_total_sd: At-the-money ``sqrt(w)``, the typical size of a log move.
        k_lo: Lowest quoted log-moneyness.
        k_hi: Highest quoted log-moneyness.
        config: Grid settings.

    Returns:
        The grid.
    """
    x_lo = min(-config.left_sds * atm_total_sd, k_lo - config.edge_sds * atm_total_sd)
    x_hi = max(config.right_sds * atm_total_sd, k_hi + config.edge_sds * atm_total_sd)
    return np.linspace(forward * np.exp(x_lo), forward * np.exp(x_hi), config.n_grid)


def breeden_litzenberger(
    strikes: FloatArray, calls: FloatArray, discount: float
) -> tuple[FloatArray, FloatArray]:
    """Density and CDF from call prices by central finite differences.

    Args:
        strikes: Evenly spaced strikes including one extra point at each end.
        calls: Call prices at those strikes.
        discount: Discount factor.

    Returns:
        ``(pdf, cdf)`` at the interior points ``strikes[1:-1]``.
    """
    h = float(strikes[1] - strikes[0])
    pdf = (calls[2:] - 2.0 * calls[1:-1] + calls[:-2]) / (h * h) / discount
    cdf = 1.0 + (calls[2:] - calls[:-2]) / (2.0 * h) / discount
    return pdf, cdf


def exponential_tails(
    strikes: FloatArray,
    pdf: FloatArray,
    cdf: FloatArray,
    forward: float,
    strike_lo: float,
    strike_hi: float,
) -> tuple[FloatArray, FloatArray]:
    """Replace the density beyond the quoted range with exponential tails in log-price.

    Beyond the right edge ``x_R = ln(K_hi/F)`` the log-price density becomes
    ``f(x_R) exp(-lam (x - x_R))``, whose total mass is ``f(x_R)/lam``. Setting
    that equal to the market-implied tail mass ``p_R`` gives ``lam = f(x_R)/p_R``,
    so both the density height and the CDF are continuous at the edge. The
    left tail mirrors this.

    Args:
        strikes: Price grid.
        pdf: Density on the grid.
        cdf: CDF on the grid.
        forward: Forward price.
        strike_lo: Lowest quoted strike.
        strike_hi: Highest quoted strike.

    Returns:
        ``(pdf, cdf)`` with both tails replaced.
    """
    pdf, cdf = pdf.copy(), cdf.copy()
    x = np.log(strikes / forward)
    for side, edge in (("left", strike_lo), ("right", strike_hi)):
        x_edge = float(np.log(edge / forward))
        height = float(np.interp(edge, strikes, pdf)) * edge  # log-price density at the edge
        cdf_edge = float(np.interp(edge, strikes, cdf))
        tail = cdf_edge if side == "left" else 1.0 - cdf_edge
        mask = strikes < edge if side == "left" else strikes > edge
        if tail <= 1e-14 or height <= 0:
            pdf[mask] = 0.0
            cdf[mask] = 0.0 if side == "left" else 1.0
            continue
        lam = height / tail
        if side == "right" and lam < MIN_TAIL_DECAY:
            logger.warning(
                "right tail decay %.3f raised to %.2f to keep the mean finite", lam, MIN_TAIL_DECAY
            )
            lam = MIN_TAIL_DECAY
        dist = (x_edge - x[mask]) if side == "left" else (x[mask] - x_edge)
        decay = np.exp(-lam * dist)
        pdf[mask] = tail * lam * decay / strikes[mask]
        cdf[mask] = tail * decay if side == "left" else 1.0 - tail * decay
    return pdf, cdf


def rnd_from_smile(
    smile: Smile,
    discount: float,
    config: DensityConfig | None = None,
    strikes: FloatArray | None = None,
) -> RiskNeutralDensity:
    """Recover the risk-neutral density and CDF implied by a fitted smile.

    Args:
        smile: Fitted smile for one expiry.
        discount: Discount factor for that expiry.
        config: Grid and tail settings; defaults to ``DensityConfig()``.
        strikes: Optional evenly spaced grid to use instead of the default.

    Returns:
        The recovered distribution.

    Raises:
        ValueError: If the tail model is unknown, or ``"smile"`` tails are
            requested for a spline smile (which has no principled extrapolation).
    """
    config = config or DensityConfig()
    tails = config.tails or ("smile" if smile.method == "svi" else "exponential")
    if tails not in ("smile", "exponential"):
        raise ValueError(f"unknown tail model {tails!r}; use 'smile' or 'exponential'")
    if tails == "smile" and smile.method == "spline":
        raise ValueError(
            "the spline smile has no principled extrapolation; use tails='exponential'"
        )
    fwd = smile.forward
    if strikes is None:
        atm_sd = float(np.sqrt(smile.total_variance(np.zeros(1))[0]))
        strikes = price_grid(fwd, atm_sd, smile.k_lo, smile.k_hi, config)
    h = float(strikes[1] - strikes[0])
    ext = np.concatenate([[strikes[0] - h], strikes, [strikes[-1] + h]])
    ext = np.maximum(ext, 1e-8 * fwd)
    calls = call_price_from_total_variance(
        fwd, ext, smile.total_variance(np.log(ext / fwd)), discount
    )
    pdf, cdf = breeden_litzenberger(ext, calls, discount)

    strike_lo, strike_hi = fwd * float(np.exp(smile.k_lo)), fwd * float(np.exp(smile.k_hi))
    tail_left = float(np.interp(strike_lo, strikes, cdf))
    tail_right = 1.0 - float(np.interp(strike_hi, strikes, cdf))
    if tails == "exponential":
        pdf, cdf = exponential_tails(strikes, pdf, cdf, fwd, strike_lo, strike_hi)
    return RiskNeutralDensity(
        strikes=strikes,
        pdf=pdf,
        cdf=cdf,
        forward=fwd,
        discount=discount,
        maturity=smile.maturity,
        strike_lo=strike_lo,
        strike_hi=strike_hi,
        tail_mass_left=tail_left,
        tail_mass_right=tail_right,
        tails=tails,
        method=smile.method,
    )
