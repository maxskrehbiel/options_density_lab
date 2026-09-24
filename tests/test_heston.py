"""The Heston truth model: characteristic function, prices, density and CDF."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from options_density_lab import heston
from options_density_lab.black76 import black76_price, implied_vol
from options_density_lab.heston import HestonParams

PARAMS = HestonParams()


def test_characteristic_function_is_one_at_zero_and_minus_i() -> None:
    # phi(0) = E[1] = 1 and phi(-i) = E[S_T / F] = 1 (the forward is the mean).
    values = heston.char_func(np.array([0.0, -1j]), 0.5, PARAMS)
    np.testing.assert_allclose(values, [1.0, 1.0], atol=1e-12)


def test_prices_reduce_to_black_scholes_without_vol_of_vol() -> None:
    flat = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=0.0)
    strikes = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
    prices = heston.call_prices(100.0, strikes, 0.5, flat, 0.98)
    np.testing.assert_allclose(prices, black76_price(100.0, strikes, 0.5, 0.2, 0.98), atol=1e-6)


def test_negative_rho_produces_downside_skew() -> None:
    strikes = np.array([80.0, 100.0, 120.0])
    prices = heston.call_prices(100.0, strikes, 0.25, PARAMS)
    iv = implied_vol(prices, 100.0, strikes, 0.25)
    assert iv[0] > iv[1] > iv[2]


def test_density_integrates_to_one_with_mean_at_forward() -> None:
    x = np.linspace(-2.5, 1.5, 4001)
    pdf = heston.logprice_pdf(x, 0.25, PARAMS)
    np.testing.assert_allclose(np.trapezoid(pdf, x), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.trapezoid(np.exp(x) * pdf, x), 1.0, atol=1e-9)


def test_cdf_matches_integrated_density() -> None:
    x = np.linspace(-2.5, 1.5, 4001)
    pdf = heston.logprice_pdf(x, 0.25, PARAMS)
    cdf = heston.logprice_cdf(x, 0.25, PARAMS)
    # Two independent inversions agree to the trapezoid rule's accuracy on this grid.
    np.testing.assert_allclose(cumulative_trapezoid(pdf, x, initial=0.0), cdf, atol=2e-5)


def test_price_space_density_is_change_of_variables() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    expected = heston.logprice_pdf(np.log(strikes / 101.0), 0.5, PARAMS) / strikes
    np.testing.assert_allclose(heston.price_pdf(strikes, 101.0, 0.5, PARAMS), expected)
    assert np.all(np.diff(heston.price_cdf(strikes, 101.0, 0.5, PARAMS)) > 0)
