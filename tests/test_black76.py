"""Black-76 pricing and implied-volatility inversion."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pytest

from options_density_lab.black76 import (
    black76_price,
    black76_vega,
    call_price_from_total_variance,
    implied_vol,
)

FORWARD, MATURITY, DISCOUNT = 100.0, 0.5, 0.98
STRIKES = np.array([60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 150.0])


def test_put_call_parity_holds() -> None:
    calls = black76_price(FORWARD, STRIKES, MATURITY, 0.25, DISCOUNT, "C")
    puts = black76_price(FORWARD, STRIKES, MATURITY, 0.25, DISCOUNT, "P")
    np.testing.assert_allclose(calls - puts, DISCOUNT * (FORWARD - STRIKES), atol=1e-12)


def test_at_the_money_price_matches_closed_form() -> None:
    vol = 0.2
    sd = vol * np.sqrt(MATURITY)
    # At K = F the formula collapses to C = D F (2 N(sd/2) - 1) = D F erf(sd / (2 sqrt 2)).
    expected = DISCOUNT * FORWARD * erf(sd / 2 / sqrt(2))
    assert float(black76_price(FORWARD, FORWARD, MATURITY, vol, DISCOUNT)) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("option_type", ["C", "P"])
def test_implied_vol_round_trip(option_type: str) -> None:
    vols = np.array([0.35, 0.15, 0.25, 0.4, 0.7, 1.0, 0.3])
    prices = black76_price(FORWARD, STRIKES, MATURITY, vols, DISCOUNT, option_type)
    recovered = implied_vol(prices, FORWARD, STRIKES, MATURITY, DISCOUNT, option_type)
    np.testing.assert_allclose(recovered, vols, rtol=1e-9)


def test_price_with_no_time_value_has_no_implied_vol() -> None:
    # A 60-strike call at 8% vol is worth its intrinsic value to machine precision.
    price = black76_price(FORWARD, 60.0, MATURITY, 0.08, DISCOUNT)
    assert np.isnan(implied_vol(price, FORWARD, 60.0, MATURITY, DISCOUNT))


def test_implied_vol_is_nan_outside_no_arbitrage_bounds() -> None:
    below_intrinsic = DISCOUNT * (FORWARD - 80.0) - 0.01
    above_max = DISCOUNT * FORWARD + 0.01
    iv = implied_vol([below_intrinsic, above_max], FORWARD, [80.0, 80.0], MATURITY, DISCOUNT, "C")
    assert np.isnan(iv).all()


def test_zero_variance_gives_discounted_intrinsic_value() -> None:
    prices = call_price_from_total_variance(FORWARD, STRIKES, 0.0, DISCOUNT)
    np.testing.assert_allclose(prices, DISCOUNT * np.maximum(FORWARD - STRIKES, 0.0))


def test_vega_matches_finite_difference() -> None:
    h = 1e-6
    up = black76_price(FORWARD, STRIKES, MATURITY, 0.2 + h, DISCOUNT)
    down = black76_price(FORWARD, STRIKES, MATURITY, 0.2 - h, DISCOUNT)
    np.testing.assert_allclose(
        black76_vega(FORWARD, STRIKES, MATURITY, 0.2, DISCOUNT), (up - down) / (2 * h), rtol=1e-6
    )
