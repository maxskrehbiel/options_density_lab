"""The Heston (1993) stochastic-volatility model, used as the known ground truth.

Its characteristic function is closed-form, so prices, the density and the
CDF of the future price all follow from one Fourier integral each.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._types import FloatArray

# Fourier grid spacings. The density integrand dies off fast; the price and CDF
# integrands carry slowly decaying 1/(u^2 + 1/4) and 1/u factors, so they need
# a finer grid to keep the trapezoid rule's aliasing error below 1e-7.
DU_DENSITY = 0.2
DU_PRICE = 0.1
DU_CDF = 0.1
CUTOFF_TOLERANCE = 1e-14
CHUNK = 256


@dataclass(frozen=True)
class HestonParams:
    """Parameters of the Heston model under the pricing measure.

    ``dS = (r - q) S dt + sqrt(v) S dW1`` and
    ``dv = kappa (theta - v) dt + sigma sqrt(v) dW2``, with ``corr(dW1, dW2) = rho``.
    The defaults are generic illustrative values, not calibrated to any market.

    Attributes:
        v0: Current variance (0.04 means 20% volatility).
        kappa: Speed at which variance reverts to ``theta``.
        theta: Long-run variance level.
        sigma: Volatility of variance ("vol of vol").
        rho: Correlation between price and variance shocks; negative values
            produce the equity-style downside skew.
    """

    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    sigma: float = 0.6
    rho: float = -0.7


def char_func(
    u: npt.ArrayLike, maturity: float, params: HestonParams
) -> npt.NDArray[np.complex128]:
    """Characteristic function ``E[exp(i u X)]`` of the log-price ``X = ln(S_T / F)``.

    Uses the arrangement of Albrecher et al. (2007), "the little Heston trap",
    which keeps the complex logarithm on its principal branch.

    Args:
        u: Real or complex Fourier argument(s).
        maturity: Time to expiry in years.
        params: Model parameters.

    Returns:
        Complex values of the characteristic function.
    """
    z = np.asarray(u, dtype=np.complex128)
    kappa, theta, sigma, rho, v0 = params.kappa, params.theta, params.sigma, params.rho, params.v0
    xi = kappa - rho * sigma * 1j * z
    d = np.sqrt(xi * xi + sigma * sigma * (z * z + 1j * z))
    g = (xi - d) / (xi + d)
    decay = np.exp(-d * maturity)
    log_ratio = np.log((1.0 - g * decay) / (1.0 - g))
    c_term = (kappa * theta / sigma**2) * ((xi - d) * maturity - 2.0 * log_ratio)
    d_term = ((xi - d) / sigma**2) * (1.0 - decay) / (1.0 - g * decay)
    return np.asarray(np.exp(c_term + d_term * v0), dtype=np.complex128)


def _u_grid(maturity: float, params: HestonParams, du: float) -> tuple[FloatArray, FloatArray]:
    """Trapezoid nodes and weights on ``[0, U]``, with ``U`` where the integrand has died.

    All three integrands below are even in ``u``, so the trapezoid rule on the
    half-line is as accurate as on the whole line: spectrally accurate.
    """
    cutoff = 10.0
    while (
        cutoff < 1e5 and abs(complex(char_func(cutoff - 0.5j, maturity, params))) > CUTOFF_TOLERANCE
    ):
        cutoff *= 1.25
    n = int(np.ceil(cutoff / du)) + 1
    u = du * np.arange(n, dtype=np.float64)
    weights = np.full(n, du)
    weights[0] = 0.5 * du
    return u, weights


def call_prices(
    forward: float,
    strikes: npt.ArrayLike,
    maturity: float,
    params: HestonParams,
    discount: float = 1.0,
) -> FloatArray:
    """European call prices by the Lewis (2001) single-integral formula.

    ``C = D (F - sqrt(F K)/pi * int_0^inf Re[exp(i u ln(F/K)) phi(u - i/2)] / (u^2 + 1/4) du)``

    Args:
        forward: Forward price for the expiry.
        strikes: Strike prices.
        maturity: Time to expiry in years.
        params: Model parameters.
        discount: Discount factor to expiry.

    Returns:
        Call prices, one per strike.
    """
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    u, weights = _u_grid(maturity, params, DU_PRICE)
    phi = char_func(u - 0.5j, maturity, params) / (u * u + 0.25)
    log_fk = np.log(forward / k)
    out = np.empty_like(k)
    for s in range(0, k.size, CHUNK):
        integrand = np.exp(1j * np.outer(log_fk[s : s + CHUNK], u)) * phi
        out[s : s + CHUNK] = forward - np.sqrt(forward * k[s : s + CHUNK]) / np.pi * (
            integrand.real @ weights
        )
    return discount * out


def logprice_pdf(x: npt.ArrayLike, maturity: float, params: HestonParams) -> FloatArray:
    """Density of ``X = ln(S_T / F)`` by Fourier inversion.

    ``f(x) = (1/pi) int_0^inf Re[exp(-i u x) phi(u)] du``

    Args:
        x: Log-moneyness values.
        maturity: Time to expiry in years.
        params: Model parameters.

    Returns:
        Density values, clipped at zero to remove round-off of order 1e-17.
    """
    xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
    u, weights = _u_grid(maturity, params, DU_DENSITY)
    phi = char_func(u, maturity, params)
    out = np.empty_like(xs)
    for s in range(0, xs.size, CHUNK):
        integrand = np.exp(-1j * np.outer(xs[s : s + CHUNK], u)) * phi
        out[s : s + CHUNK] = (integrand.real @ weights) / np.pi
    return np.maximum(out, 0.0)


def logprice_cdf(x: npt.ArrayLike, maturity: float, params: HestonParams) -> FloatArray:
    """CDF of ``X = ln(S_T / F)`` by the Gil-Pelaez (1951) inversion formula.

    ``P(X <= x) = 1/2 - (1/pi) int_0^inf Im[exp(-i u x) phi(u)] / u du``

    Args:
        x: Log-moneyness values.
        maturity: Time to expiry in years.
        params: Model parameters.

    Returns:
        Probabilities in ``[0, 1]``.
    """
    xs = np.atleast_1d(np.asarray(x, dtype=np.float64))
    u, weights = _u_grid(maturity, params, DU_CDF)
    phi = char_func(u, maturity, params)
    # The integrand is finite at u = 0, where its limit is E[X] - x.
    h = 1e-6
    mean_x = float(np.real(np.log(complex(char_func(h, maturity, params))) / 1j) / h)
    out = np.empty_like(xs)
    for s in range(0, xs.size, CHUNK):
        chunk = xs[s : s + CHUNK]
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = (np.exp(-1j * np.outer(chunk, u)) * phi).imag / u
        integrand[:, 0] = mean_x - chunk
        out[s : s + CHUNK] = 0.5 - (integrand @ weights) / np.pi
    return np.clip(out, 0.0, 1.0)


def price_pdf(
    strikes: npt.ArrayLike, forward: float, maturity: float, params: HestonParams
) -> FloatArray:
    """Density of the future price ``S_T`` (change of variables from the log-price).

    Args:
        strikes: Price levels.
        forward: Forward price for the expiry.
        maturity: Time to expiry in years.
        params: Model parameters.

    Returns:
        Density values ``q(K)``.
    """
    k = np.asarray(strikes, dtype=np.float64)
    return logprice_pdf(np.log(k / forward), maturity, params) / k


def price_cdf(
    strikes: npt.ArrayLike, forward: float, maturity: float, params: HestonParams
) -> FloatArray:
    """Probability that the future price ends at or below each level.

    Args:
        strikes: Price levels.
        forward: Forward price for the expiry.
        maturity: Time to expiry in years.
        params: Model parameters.

    Returns:
        ``P(S_T <= K)`` for each level.
    """
    k = np.asarray(strikes, dtype=np.float64)
    return logprice_cdf(np.log(k / forward), maturity, params)
