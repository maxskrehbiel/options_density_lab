"""Black-76 option prices and implied volatility, quoted on the forward price.

Every price in the package is written in terms of the forward ``F`` and the
discount factor ``D``, so rates and dividends never appear separately.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from ._types import FloatArray

IV_LOWER = 1e-4
IV_UPPER = 5.0
IV_BISECTION_STEPS = 80


def _floats(x: npt.ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _norm_cdf(x: FloatArray) -> FloatArray:
    return np.asarray(ndtr(x), dtype=np.float64)


def call_price_from_total_variance(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    total_variance: npt.ArrayLike,
    discount: npt.ArrayLike = 1.0,
) -> FloatArray:
    """Black-76 call price written in terms of total implied variance.

    ``C = D * (F N(d1) - K N(d2))`` with ``d1 = (ln(F/K) + w/2) / sqrt(w)`` and
    ``d2 = d1 - sqrt(w)``, where ``w = sigma^2 T`` is the total variance.

    Args:
        forward: Forward price of the underlying for this expiry.
        strike: Strike price(s).
        total_variance: Total implied variance ``w`` at each strike.
        discount: Discount factor to expiry.

    Returns:
        Call prices, broadcast over the inputs. Zero variance gives the
        discounted intrinsic value.
    """
    f, k, w, d = _floats(forward), _floats(strike), _floats(total_variance), _floats(discount)
    sqrt_w = np.sqrt(np.maximum(w, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(f / k) + 0.5 * w) / sqrt_w
        d2 = d1 - sqrt_w
        price = d * (f * _norm_cdf(d1) - k * _norm_cdf(d2))
    return np.where(sqrt_w > 0, price, d * np.maximum(f - k, 0.0))


def black76_price(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    maturity: npt.ArrayLike,
    vol: npt.ArrayLike,
    discount: npt.ArrayLike = 1.0,
    option_type: npt.ArrayLike = "C",
) -> FloatArray:
    """Price European calls or puts under Black-76.

    Args:
        forward: Forward price for the expiry.
        strike: Strike price(s).
        maturity: Time to expiry in years.
        vol: Implied volatility, annualised (0.20 means 20%).
        discount: Discount factor to expiry.
        option_type: ``"C"`` for calls or ``"P"`` for puts, scalar or array.

    Returns:
        Option prices, broadcast over the inputs.
    """
    total_variance = _floats(vol) ** 2 * _floats(maturity)
    call = call_price_from_total_variance(forward, strike, total_variance, discount)
    put = call - _floats(discount) * (_floats(forward) - _floats(strike))  # put-call parity
    return np.where(np.asarray(option_type) == "C", call, put)


def black76_vega(
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    maturity: npt.ArrayLike,
    vol: npt.ArrayLike,
    discount: npt.ArrayLike = 1.0,
) -> FloatArray:
    """Sensitivity of the Black-76 price to volatility (identical for calls and puts).

    Args:
        forward: Forward price for the expiry.
        strike: Strike price(s).
        maturity: Time to expiry in years.
        vol: Implied volatility, annualised.
        discount: Discount factor to expiry.

    Returns:
        Price change per unit change in volatility.
    """
    f, k, t, s, d = (_floats(a) for a in (forward, strike, maturity, vol, discount))
    sd = s * np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * sd * sd) / sd
    density = np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
    return np.asarray(d * f * density * np.sqrt(t), dtype=np.float64)


def implied_vol(
    price: npt.ArrayLike,
    forward: npt.ArrayLike,
    strike: npt.ArrayLike,
    maturity: npt.ArrayLike,
    discount: npt.ArrayLike = 1.0,
    option_type: npt.ArrayLike = "C",
) -> FloatArray:
    """Invert Black-76 for volatility by vectorised bisection.

    The price rises strictly with volatility, so halving the bracket
    ``[IV_LOWER, IV_UPPER]`` a fixed number of times cannot fail. Puts are
    turned into calls with put-call parity first, so one search serves both.

    Args:
        price: Observed option prices.
        forward: Forward price for the expiry.
        strike: Strike price(s).
        maturity: Time to expiry in years.
        discount: Discount factor to expiry.
        option_type: ``"C"`` or ``"P"``, scalar or array.

    Returns:
        Implied volatilities. Prices outside the no-arbitrage bounds (below
        discounted intrinsic value or at least the discounted forward) have no
        implied volatility and return ``NaN``.
    """
    p, f, k, t, d = np.broadcast_arrays(
        *(_floats(a) for a in (price, forward, strike, maturity, discount))
    )
    is_call = np.broadcast_to(np.asarray(option_type), p.shape) == "C"
    call_price = np.where(is_call, p, p + d * (f - k))
    valid = (
        (call_price > d * np.maximum(f - k, 0.0)) & (call_price < d * f) & np.isfinite(call_price)
    )

    lo = np.full(p.shape, IV_LOWER)
    hi = np.full(p.shape, IV_UPPER)
    for _ in range(IV_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        too_high = call_price_from_total_variance(f, k, mid * mid * t, d) > call_price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    iv = 0.5 * (lo + hi)
    # A root pinned against the bracket edge means there is no solution inside it.
    interior = (iv > IV_LOWER * 1.01) & (iv < IV_UPPER * 0.99)
    return np.where(valid & interior, iv, np.nan)
