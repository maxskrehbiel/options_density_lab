"""Arbitrage-aware implied-volatility smiles in total variance: SVI and a penalised spline.

Smiles are functions ``w(k)`` of log-moneyness ``k = ln(K/F)``, where
``w = iv^2 T`` is total implied variance.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from ._types import BoolArray, FloatArray

logger = logging.getLogger(__name__)

SmileMethod = Literal["svi", "spline"]
MIN_K_SPAN = 0.05  # smallest log-moneyness range used to scale starts and bounds


def g_function(k: FloatArray, w: FloatArray, w1: FloatArray, w2: FloatArray) -> FloatArray:
    """Butterfly function of Gatheral and Jacquier (2014); the density is proportional to it.

    ``g = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2``. The density of the
    log-price is ``g(k) / sqrt(2 pi w) * exp(-d2^2/2)`` with
    ``d2 = -k/sqrt(w) - sqrt(w)/2``, so a smile is free of butterfly arbitrage
    exactly when ``g >= 0`` everywhere.

    Args:
        k: Log-moneyness.
        w: Total variance at ``k``.
        w1: First derivative of ``w`` in ``k``.
        w2: Second derivative of ``w`` in ``k``.

    Returns:
        ``g(k)``.
    """
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1**2 * (1.0 / w + 0.25) + 0.5 * w2


class Smile(ABC):
    """A fitted smile: total variance and its derivatives in log-moneyness."""

    method: str
    maturity: float
    forward: float
    k_lo: float
    k_hi: float

    @abstractmethod
    def total_variance(self, k: npt.ArrayLike, derivative: int = 0) -> FloatArray:
        """Total variance ``w(k)`` or its first or second derivative.

        Args:
            k: Log-moneyness values.
            derivative: 0, 1 or 2.

        Returns:
            Values at each ``k``.
        """

    def iv(self, k: npt.ArrayLike) -> FloatArray:
        """Implied volatility ``sqrt(w(k)/T)``.

        Args:
            k: Log-moneyness values.

        Returns:
            Annualised implied volatilities.
        """
        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / self.maturity)

    def g(self, k: npt.ArrayLike) -> FloatArray:
        """Butterfly function ``g(k)`` for this smile (see :func:`g_function`).

        Args:
            k: Log-moneyness values.

        Returns:
            ``g`` at each ``k``.
        """
        kk = np.asarray(k, dtype=np.float64)
        return g_function(
            kk, self.total_variance(kk), self.total_variance(kk, 1), self.total_variance(kk, 2)
        )

    def logprice_pdf(self, k: npt.ArrayLike) -> FloatArray:
        """Closed-form density of ``ln(S_T/F)`` implied by the smile.

        Args:
            k: Log-moneyness values.

        Returns:
            Density values.
        """
        kk = np.asarray(k, dtype=np.float64)
        w = self.total_variance(kk)
        sqrt_w = np.sqrt(w)
        d2 = -kk / sqrt_w - sqrt_w / 2.0
        density = self.g(kk) / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d2 * d2)
        return np.asarray(density, dtype=np.float64)


def svi_total_variance(
    k: npt.ArrayLike, a: float, b: float, rho: float, m: float, s: float, derivative: int = 0
) -> FloatArray:
    """Raw SVI: ``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))`` or a derivative.

    Args:
        k: Log-moneyness values.
        a: Overall variance level.
        b: Steepness of the wings.
        rho: Tilt between -1 and 1; negative means downside skew.
        m: Horizontal shift.
        s: Rounding of the vertex (``sigma`` in the literature).
        derivative: 0, 1 or 2.

    Returns:
        Values at each ``k``.

    Raises:
        ValueError: If ``derivative`` is not 0, 1 or 2.
    """
    z = np.asarray(k, dtype=np.float64) - m
    r = np.sqrt(z * z + s * s)
    if derivative == 0:
        out = a + b * (rho * z + r)
    elif derivative == 1:
        out = b * (rho + z / r)
    elif derivative == 2:
        out = b * s * s / r**3
    else:
        raise ValueError(f"derivative must be 0, 1 or 2, got {derivative}")
    return np.asarray(out, dtype=np.float64)


def _svi_with_gradient(
    k: FloatArray, a: float, b: float, rho: float, m: float, s: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """SVI ``w``, ``w'``, ``w''`` and their gradients in ``(a, b, rho, m, s)`` (one column each)."""
    z = k - m
    r = np.sqrt(z * z + s * s)
    w = a + b * (rho * z + r)
    w1 = b * (rho + z / r)
    w2 = b * s * s / r**3
    zeros, ones = np.zeros_like(k), np.ones_like(k)
    dw = np.column_stack([ones, rho * z + r, b * z, -b * (rho + z / r), b * s / r])
    dw1 = np.column_stack([zeros, rho + z / r, b * ones, -b * s * s / r**3, -b * z * s / r**3])
    dw2 = np.column_stack(
        [
            zeros,
            s * s / r**3,
            zeros,
            3.0 * b * s * s * z / r**5,
            b * s * (2.0 * r * r - 3.0 * s * s) / r**5,
        ]
    )
    return w, w1, w2, dw, dw1, dw2


@dataclass(frozen=True)
class SviConfig:
    """Settings for the SVI fit.

    Attributes:
        penalty: Weight on no-arbitrage penalty residuals, relative to data
            residuals measured in half-spreads.
        rho_starts: Starting tilts for the multi-start search.
        s_starts: Starting vertex widths, as fractions of the quoted k-range.
        wing_spans: How far beyond the quoted range (in range widths) the
            butterfly penalty is enforced.
        grid_points: Points in the penalty grid.
        max_nfev: Function-evaluation cap per start.
        tolerance: Convergence tolerance on cost, step and gradient. Tight on
            purpose: a fully converged optimum is identical across platforms,
            which keeps the committed demo output byte-stable.
        calendar_points: Grid size for the calendar penalty.
        min_s_spacings: Lower bound on the vertex rounding ``s``, in multiples
            of the median gap between quoted log-strikes. Curvature on a finer
            scale than the quotes is not identified by them, and a near-zero
            ``s`` turns the vertex into a kink that puts a spurious spike in
            the density.
    """

    penalty: float = 1e3
    rho_starts: tuple[float, ...] = (-0.7, 0.0)
    s_starts: tuple[float, ...] = (0.25,)
    wing_spans: float = 2.0
    grid_points: int = 201
    max_nfev: int = 3000
    tolerance: float = 1e-12
    calendar_points: int = 101
    min_s_spacings: float = 1.0


@dataclass(frozen=True)
class SVISmile(Smile):
    """A raw SVI smile with its parameters (see :func:`svi_total_variance`).

    Attributes:
        a: Overall variance level.
        b: Wing steepness.
        rho: Tilt.
        m: Horizontal shift.
        s: Vertex rounding.
        maturity: Time to expiry in years.
        forward: Forward price the log-moneyness is measured from.
        k_lo: Lowest quoted log-moneyness used in the fit.
        k_hi: Highest quoted log-moneyness used in the fit.
        cost: Final least-squares cost (half the sum of squared residuals).
        method: Always ``"svi"``.
    """

    a: float
    b: float
    rho: float
    m: float
    s: float
    maturity: float
    forward: float
    k_lo: float
    k_hi: float
    cost: float = float("nan")
    method: str = "svi"

    @property
    def params(self) -> tuple[float, float, float, float, float]:
        """``(a, b, rho, m, s)``."""
        return self.a, self.b, self.rho, self.m, self.s

    def total_variance(self, k: npt.ArrayLike, derivative: int = 0) -> FloatArray:
        """SVI total variance or derivative (see :meth:`Smile.total_variance`).

        Args:
            k: Log-moneyness values.
            derivative: 0, 1 or 2.

        Returns:
            Values at each ``k``.
        """
        return svi_total_variance(k, *self.params, derivative=derivative)


@dataclass(frozen=True)
class _SviObjective:
    """Least-squares residuals and their Jacobian for one SVI fit.

    Rows are: data misfits in half-spreads, butterfly penalties on a wide grid,
    the minimum-variance and Lee-bound penalties, and calendar penalties
    against the previous expiry (empty when there is none).
    """

    k: FloatArray
    w_obs: FloatArray
    w_err: FloatArray
    grid: FloatArray
    k_cal: FloatArray
    w_prev: FloatArray
    penalty: float
    scale: float

    @classmethod
    def build(
        cls,
        k: FloatArray,
        w_obs: FloatArray,
        w_err: FloatArray,
        prev: Smile | None,
        config: SviConfig,
    ) -> _SviObjective:
        k_lo, k_hi = float(k.min()), float(k.max())
        reach = config.wing_spans * max(k_hi - k_lo, MIN_K_SPAN)
        k_cal = np.empty(0)
        if prev is not None:
            k_cal = np.linspace(min(k_lo, prev.k_lo), max(k_hi, prev.k_hi), config.calendar_points)
        return cls(
            k=k,
            w_obs=w_obs,
            w_err=w_err,
            grid=np.linspace(k_lo - reach, k_hi + reach, config.grid_points),
            k_cal=k_cal,
            w_prev=prev.total_variance(k_cal) if prev is not None else np.empty(0),
            penalty=config.penalty,
            scale=float(np.median(w_err)),
        )

    @property
    def fly_scale(self) -> float:
        return self.penalty / float(np.sqrt(self.grid.size))

    @property
    def cal_scale(self) -> float:
        return self.penalty / self.scale / float(np.sqrt(max(self.k_cal.size, 1)))

    def _grid_terms(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        a, b, rho, m, s = (float(v) for v in x)
        w = np.maximum(svi_total_variance(self.grid, a, b, rho, m, s), 1e-12)
        w1 = svi_total_variance(self.grid, a, b, rho, m, s, 1)
        g = g_function(self.grid, w, w1, svi_total_variance(self.grid, a, b, rho, m, s, 2))
        return w, w1, g

    def residuals(self, x: FloatArray) -> FloatArray:
        a, b, rho, m, s = (float(v) for v in x)
        g = self._grid_terms(x)[2]
        min_var = a + b * s * np.sqrt(1.0 - rho * rho)
        lee = b * (1.0 + abs(rho)) - 2.0
        w_cal = svi_total_variance(self.k_cal, a, b, rho, m, s)
        return np.concatenate(
            [
                (svi_total_variance(self.k, a, b, rho, m, s) - self.w_obs) / self.w_err,
                self.fly_scale * np.maximum(-g, 0.0),
                [self.penalty * max(0.0, -min_var) / self.scale, self.penalty * max(0.0, lee)],
                self.cal_scale * np.maximum(self.w_prev - w_cal, 0.0),
            ]
        )

    def _butterfly_rows(self, x: FloatArray) -> tuple[npt.NDArray[np.int64], FloatArray]:
        w, w1, g = self._grid_terms(x)
        active = np.flatnonzero(g < 0)
        if not active.size:
            return active, np.zeros((0, 5))
        kk, ww, ww1 = self.grid[active], w[active], w1[active]
        _, _, _, dw, dw1, dw2 = _svi_with_gradient(kk, *(float(v) for v in x))
        # Chain rule through g = A^2 - (w1^2/4)(1/w + 1/4) + w2/2 with A = 1 - k w1/(2w).
        big_a = 1.0 - kk * ww1 / (2.0 * ww)
        dg_dw = big_a * kk * ww1 / ww**2 + 0.25 * ww1**2 / ww**2
        dg_dw1 = -big_a * kk / ww - 0.5 * ww1 * (1.0 / ww + 0.25)
        rows = -self.fly_scale * (dg_dw[:, None] * dw + dg_dw1[:, None] * dw1 + 0.5 * dw2)
        return active, rows

    def jacobian(self, x: FloatArray) -> FloatArray:
        a, b, rho, m, s = (float(v) for v in x)
        n_data, n_grid = self.k.size, self.grid.size
        jac = np.zeros((n_data + n_grid + 2 + self.k_cal.size, 5))
        jac[:n_data] = _svi_with_gradient(self.k, a, b, rho, m, s)[3] / self.w_err[:, None]
        # Penalty rows are non-zero only where a constraint binds, which is rare.
        active, rows = self._butterfly_rows(x)
        jac[n_data + active] = rows
        root = np.sqrt(1.0 - rho * rho)
        row = n_data + n_grid
        if a + b * s * root < 0:
            d_min_var = np.array([1.0, s * root, -b * s * rho / root, 0.0, b * root])
            jac[row] = -self.penalty / self.scale * d_min_var
        if b * (1.0 + abs(rho)) > 2.0:
            jac[row + 1] = self.penalty * np.array(
                [0.0, 1.0 + abs(rho), b * np.sign(rho), 0.0, 0.0]
            )
        cal = np.flatnonzero(self.w_prev > svi_total_variance(self.k_cal, a, b, rho, m, s))
        if cal.size:
            grad = _svi_with_gradient(self.k_cal[cal], a, b, rho, m, s)[3]
            jac[row + 2 + cal] = -self.cal_scale * grad
        return jac


def _svi_starts(
    k: FloatArray,
    w_obs: FloatArray,
    w_err: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    config: SviConfig,
) -> list[FloatArray]:
    """Starting points; SVI is linear in ``(a, b)``, so those are solved exactly for each."""
    k_lo, k_hi = float(k.min()), float(k.max())
    span = max(k_hi - k_lo, MIN_K_SPAN)
    sqrt_wts = 1.0 / w_err
    lower, upper = bounds
    starts = []
    for rho0 in config.rho_starts:
        for m0 in (k_lo + 0.5 * span, 0.0, k_hi - 0.25 * span):
            for s_frac in config.s_starts:
                s0 = s_frac * span
                basis = rho0 * (k - m0) + np.sqrt((k - m0) ** 2 + s0**2)
                design = np.column_stack([np.ones_like(k), basis]) * sqrt_wts[:, None]
                a0, b0 = np.linalg.lstsq(design, w_obs * sqrt_wts, rcond=None)[0]
                b0 = float(np.clip(b0, 1e-4, 1.9 / (1.0 + abs(rho0))))
                x0 = np.clip([a0, b0, rho0, m0, s0], lower + 1e-9, upper - 1e-9)
                starts.append(np.asarray(x0, dtype=np.float64))
    return starts


def _svi_bounds(
    k: FloatArray, w_obs: FloatArray, config: SviConfig
) -> tuple[FloatArray, FloatArray]:
    """Parameter bounds ``(a, b, rho, m, s)``; ``s`` is floored at the strike spacing."""
    k_lo, k_hi = float(k.min()), float(k.max())
    span = max(k_hi - k_lo, MIN_K_SPAN)
    w_max = float(w_obs.max())
    s_min = max(config.min_s_spacings * float(np.median(np.diff(np.sort(k)))), 1e-4)
    lower = np.array([-2.0 * w_max, 1e-6, -0.999, k_lo - span, s_min])
    upper = np.array([2.0 * w_max, 2.0, 0.999, k_hi + span, 2.0 * span + 0.5])
    return lower, upper


def fit_svi(
    k: FloatArray,
    w_obs: FloatArray,
    w_err: FloatArray,
    maturity: float,
    forward: float,
    prev: Smile | None = None,
    config: SviConfig | None = None,
) -> SVISmile:
    """Weighted least-squares SVI fit with no-arbitrage penalties.

    Data residuals are ``(w_model - w_obs) / w_err``: a residual of 1 means
    "off by one half-spread". Penalty residuals are added for a negative
    implied density on a wide grid (butterfly), negative minimum variance,
    wings steeper than Lee's (2004) bound ``b (1 + |rho|) <= 2``, and, if
    ``prev`` is given, total variance below the previous expiry's (calendar).

    Args:
        k: Log-moneyness of the quotes.
        w_obs: Observed total variances.
        w_err: Uncertainty of each total variance.
        maturity: Time to expiry in years.
        forward: Forward price.
        prev: The previous (shorter) expiry's smile, for the calendar penalty.
        config: Fit settings; defaults to ``SviConfig()``.

    Returns:
        The best fit over all starting points.
    """
    config = config or SviConfig()
    objective = _SviObjective.build(k, w_obs, w_err, prev, config)
    lower, upper = _svi_bounds(k, w_obs, config)
    best: tuple[float, FloatArray] | None = None
    for x0 in _svi_starts(k, w_obs, w_err, (lower, upper), config):
        sol = least_squares(
            objective.residuals,
            x0,
            jac=objective.jacobian,
            bounds=(lower, upper),
            x_scale="jac",
            ftol=config.tolerance,
            xtol=config.tolerance,
            gtol=config.tolerance,
            max_nfev=config.max_nfev,
        )
        if best is None or float(sol.cost) < best[0]:
            best = (float(sol.cost), np.asarray(sol.x, dtype=np.float64))
    if best is None:
        raise ValueError("SviConfig needs at least one starting point")
    a, b, rho, m, s = (float(v) for v in best[1])
    k_lo, k_hi = float(k.min()), float(k.max())
    return SVISmile(a, b, rho, m, s, maturity, forward, k_lo, k_hi, cost=best[0])


@dataclass(frozen=True)
class SplineConfig:
    """Settings for the penalised-spline fit.

    Attributes:
        degree: Polynomial degree of the B-spline pieces.
        penalty_order: Order of the coefficient differences penalised; 3 pulls
            a heavily smoothed curve toward a parabola.
        lam_min: Smallest smoothing weight tried by cross-validation.
        lam_max: Largest smoothing weight tried by cross-validation.
        lam_steps: Number of log-spaced smoothing weights tried.
        max_doublings: Cap on smoothing increases made by the arbitrage guard.
        min_segments: Fewest spline segments.
        max_segments: Most spline segments.
        guard_points: Grid size for the arbitrage guard's density check.
    """

    degree: int = 3
    penalty_order: int = 3
    lam_min: float = 1e-6
    lam_max: float = 1e6
    lam_steps: int = 49
    max_doublings: int = 60
    min_segments: int = 4
    max_segments: int = 20
    guard_points: int = 400


@dataclass(frozen=True)
class SplineSmile(Smile):
    """Penalised B-spline in total variance, trustworthy only on ``[k_lo, k_hi]``.

    Outside that range it continues as a straight line with the edge value
    and slope, only so finite differences at the edges work. That is not a
    tail model; pair this smile with ``tails="exponential"``.

    Attributes:
        spline: The fitted ``scipy.interpolate.BSpline``.
        maturity: Time to expiry in years.
        forward: Forward price.
        k_lo: Lowest quoted log-moneyness.
        k_hi: Highest quoted log-moneyness.
        lam: Final smoothing weight.
        lam_gcv: Smoothing weight chosen by cross-validation, before the guard.
        method: Always ``"spline"``.
    """

    spline: Any = field(repr=False)
    maturity: float
    forward: float
    k_lo: float
    k_hi: float
    lam: float = float("nan")
    lam_gcv: float = float("nan")
    method: str = "spline"

    def total_variance(self, k: npt.ArrayLike, derivative: int = 0) -> FloatArray:
        """Spline total variance or derivative (see :meth:`Smile.total_variance`).

        Args:
            k: Log-moneyness values.
            derivative: 0, 1 or 2.

        Returns:
            Values at each ``k``.
        """
        kk = np.asarray(k, dtype=np.float64)
        inside = np.clip(kk, self.k_lo, self.k_hi)
        value = np.asarray(self.spline(inside, nu=derivative), dtype=np.float64)
        if derivative == 0:
            slope = np.asarray(self.spline(inside, nu=1), dtype=np.float64)
            return value + slope * (kk - inside)
        if derivative == 2:
            return np.where(kk == inside, value, 0.0)
        return value


@dataclass(frozen=True)
class _PenalisedSystem:
    """Normal equations of a weighted P-spline fit, ready to solve for any ``lam``."""

    knots: FloatArray
    basis: FloatArray
    roughness: FloatArray
    weights: FloatArray
    w_obs: FloatArray

    @classmethod
    def build(
        cls, k: FloatArray, w_obs: FloatArray, w_err: FloatArray, config: SplineConfig
    ) -> _PenalisedSystem:
        k_lo, k_hi = float(k.min()), float(k.max())
        n_seg = int(np.clip(k.size // 2, config.min_segments, config.max_segments))
        step = (k_hi - k_lo) / n_seg
        knots = np.asarray(
            k_lo + step * np.arange(-config.degree, n_seg + config.degree + 1), dtype=np.float64
        )
        basis = BSpline.design_matrix(k, knots, config.degree, extrapolate=True).toarray()
        diff = np.diff(np.eye(basis.shape[1]), n=config.penalty_order, axis=0)
        weights = 1.0 / w_err**2
        return cls(
            knots,
            np.asarray(basis, dtype=np.float64),
            diff.T @ diff,
            weights / weights.mean(),
            w_obs,
        )

    def solve(self, lam: float) -> tuple[FloatArray, float, float]:
        """Coefficients, effective number of parameters and weighted residual sum of squares."""
        btwb = self.basis.T @ (self.basis * self.weights[:, None])
        system = btwb + lam * self.roughness
        coef = np.asarray(
            np.linalg.solve(system, self.basis.T @ (self.weights * self.w_obs)), dtype=np.float64
        )
        edf = float(np.trace(np.linalg.solve(system, btwb)))
        rss = float(np.sum(self.weights * (self.w_obs - self.basis @ coef) ** 2))
        return coef, edf, rss

    def gcv_lambda(self, config: SplineConfig) -> float:
        """The smoothing weight with the lowest generalised cross-validation score."""
        n = self.w_obs.size
        lams = np.logspace(np.log10(config.lam_min), np.log10(config.lam_max), config.lam_steps)
        scores = []
        for lam in lams:
            _, edf, rss = self.solve(float(lam))
            scores.append(n * rss / max(n - edf, 1e-3) ** 2)
        return float(lams[int(np.argmin(scores))])


def fit_spline(
    k: FloatArray,
    w_obs: FloatArray,
    w_err: FloatArray,
    maturity: float,
    forward: float,
    config: SplineConfig | None = None,
) -> SplineSmile:
    """P-spline fit (Eilers and Marx 1996) with an arbitrage guard.

    Minimises ``sum(((w_i - s(k_i)) / err_i)^2) + lam * |Delta^p c|^2`` where ``c``
    are the B-spline coefficients and ``Delta^p`` takes ``p``-th differences.
    ``lam`` starts at the generalised cross-validation choice (Craven and
    Wahba 1979) and is doubled until the implied density is non-negative
    across the quoted range and total variance stays positive.

    Args:
        k: Log-moneyness of the quotes.
        w_obs: Observed total variances.
        w_err: Uncertainty of each total variance.
        maturity: Time to expiry in years.
        forward: Forward price.
        config: Fit settings; defaults to ``SplineConfig()``.

    Returns:
        The fitted smile.
    """
    config = config or SplineConfig()
    k_lo, k_hi = float(k.min()), float(k.max())
    system = _PenalisedSystem.build(k, w_obs, w_err, config)
    lam_gcv = system.gcv_lambda(config)
    check = np.linspace(k_lo, k_hi, config.guard_points)
    lam = lam_gcv
    for _ in range(config.max_doublings):
        coef, _, _ = system.solve(lam)
        smile = SplineSmile(
            spline=BSpline(system.knots, coef, config.degree),
            maturity=maturity,
            forward=forward,
            k_lo=k_lo,
            k_hi=k_hi,
            lam=lam,
            lam_gcv=lam_gcv,
        )
        if smile.total_variance(check).min() > 0 and smile.g(check).min() >= 0:
            break
        lam *= 2.0
    else:
        logger.warning("spline arbitrage guard hit its cap; density may be negative somewhere")
    if lam > lam_gcv:
        logger.info(
            "spline smoothing raised from %.3g to %.3g to remove negative density", lam_gcv, lam
        )
    return smile


@dataclass(frozen=True)
class SmileConfig:
    """Settings for :func:`fit_smile`.

    Attributes:
        method: ``"svi"`` or ``"spline"``.
        drop_outliers: Refit once after dropping quotes far outside their own band.
        outlier_z: Drop threshold in multiples of the median-based misfit scale.
        outlier_min_spreads: A dropped quote must also be at least this many
            half-spreads away from the curve.
        max_outlier_share: Most quotes that may be dropped, as a share.
        svi: SVI settings.
        spline: Spline settings.
    """

    method: SmileMethod = "svi"
    drop_outliers: bool = True
    outlier_z: float = 4.0
    outlier_min_spreads: float = 3.0
    max_outlier_share: float = 0.1
    svi: SviConfig = field(default_factory=SviConfig)
    spline: SplineConfig = field(default_factory=SplineConfig)


def fit_smile(
    otm: pd.DataFrame,
    maturity: float,
    forward: float,
    prev: Smile | None = None,
    config: SmileConfig | None = None,
) -> tuple[Smile, BoolArray]:
    """Fit one expiry's smile from cleaned out-of-the-money quotes.

    With ``drop_outliers`` the fit runs twice: quotes that sit far outside
    their own bid-ask band after the first pass are removed before the second.

    Args:
        otm: Output of :func:`options_density_lab.clean.otm_implied_vols`.
        maturity: Time to expiry in years.
        forward: Forward price.
        prev: Previous expiry's smile (SVI calendar penalty only).
        config: Settings; defaults to ``SmileConfig()``.

    Returns:
        ``(smile, used)`` where ``used`` marks the quotes in the final fit.

    Raises:
        ValueError: If ``config.method`` is unknown.
    """
    config = config or SmileConfig()
    if config.method not in ("svi", "spline"):
        raise ValueError(f"unknown smile method {config.method!r}; use 'svi' or 'spline'")
    k = otm["k"].to_numpy(dtype=float)
    w = otm["w"].to_numpy(dtype=float)
    w_err = otm["w_err"].to_numpy(dtype=float)

    def fit(mask: BoolArray) -> Smile:
        if config.method == "svi":
            return fit_svi(k[mask], w[mask], w_err[mask], maturity, forward, prev, config.svi)
        return fit_spline(k[mask], w[mask], w_err[mask], maturity, forward, config.spline)

    used = np.ones(k.size, dtype=bool)
    smile = fit(used)
    if config.drop_outliers:
        z = (smile.total_variance(k) - w) / w_err
        scale = 1.4826 * float(np.median(np.abs(z)))  # median absolute deviation -> std
        bad = np.abs(z) > max(config.outlier_z * scale, config.outlier_min_spreads)
        n_drop = min(int(bad.sum()), int(config.max_outlier_share * k.size))
        if n_drop > 0:
            used[np.argsort(-np.abs(z))[:n_drop]] = False
            logger.info("T=%.3f: dropped %d outlying quote(s) and refit", maturity, n_drop)
            smile = fit(used)
    return smile, used
