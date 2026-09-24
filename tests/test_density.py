"""Breeden-Litzenberger recovery: exactness on a known case, tails, and distribution queries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from options_density_lab.density import DensityConfig, rnd_from_smile
from options_density_lab.pipeline import SurfaceResult
from options_density_lab.smile import SVISmile

FORWARD, MATURITY, DISCOUNT, VOL = 100.0, 0.5, 0.98, 0.25


def _flat(k_lo: float = -0.3, k_hi: float = 0.3) -> SVISmile:
    return SVISmile(
        a=VOL**2 * MATURITY,
        b=0.0,
        rho=0.0,
        m=0.0,
        s=1.0,
        maturity=MATURITY,
        forward=FORWARD,
        k_lo=k_lo,
        k_hi=k_hi,
    )


def _lognormal_pdf(strikes: np.ndarray) -> np.ndarray:
    sd = VOL * np.sqrt(MATURITY)
    z = (np.log(strikes / FORWARD) + 0.5 * sd**2) / sd
    return np.exp(-0.5 * z**2) / (strikes * sd * np.sqrt(2 * np.pi))


def test_flat_smile_recovers_the_lognormal_density() -> None:
    dens = rnd_from_smile(_flat(), DISCOUNT)
    truth = _lognormal_pdf(dens.strikes)
    np.testing.assert_allclose(dens.pdf, truth, rtol=1e-4, atol=1e-9)
    assert dens.mass() == pytest.approx(1.0, abs=1e-6)
    assert dens.moments()["mean"] == pytest.approx(FORWARD, rel=1e-6)
    assert dens.moments()["vol_annualised"] == pytest.approx(VOL, rel=1e-4)


def test_numerical_second_derivative_matches_closed_form(svi_surface: SurfaceResult) -> None:
    for e in svi_surface.expiries:
        d = e.density
        inside = (d.strikes > d.strike_lo) & (d.strikes < d.strike_hi)
        closed = e.smile.logprice_pdf(np.log(d.strikes[inside] / d.forward)) / d.strikes[inside]
        np.testing.assert_allclose(d.pdf[inside], closed, rtol=1e-3, atol=1e-7)


def test_exponential_tails_keep_market_implied_tail_mass(svi_surface: SurfaceResult) -> None:
    for e in svi_surface.expiries:
        dens = rnd_from_smile(e.smile, e.clean.discount, DensityConfig(tails="exponential"))
        assert dens.tails == "exponential"
        assert float(dens.prob_below(dens.strike_lo)) == pytest.approx(
            dens.tail_mass_left, abs=2e-4
        )
        assert float(dens.prob_above(dens.strike_hi)) == pytest.approx(
            dens.tail_mass_right, abs=2e-4
        )
        assert dens.mass() == pytest.approx(1.0, abs=2e-3)
        assert (dens.pdf >= 0).all()


def test_quantiles_invert_the_cdf(svi_surface: SurfaceResult) -> None:
    dens = svi_surface.expiries[1].density
    probs = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    np.testing.assert_allclose(dens.prob_below(dens.quantile(probs)), probs, atol=1e-4)
    assert np.all(np.diff(dens.quantile(probs)) > 0)


def test_invalid_tail_choices_raise(spline_surface: SurfaceResult) -> None:
    spline = spline_surface.expiries[0].smile
    with pytest.raises(ValueError, match="no principled extrapolation"):
        rnd_from_smile(spline, 0.99, DensityConfig(tails="smile"))
    with pytest.raises(ValueError, match="unknown tail model"):
        rnd_from_smile(_flat(), 0.99, replace(DensityConfig(), tails="normal"))  # type: ignore[arg-type]


def test_custom_grid_is_used() -> None:
    grid = np.linspace(40.0, 200.0, 801)
    dens = rnd_from_smile(_flat(), DISCOUNT, strikes=grid)
    np.testing.assert_array_equal(dens.strikes, grid)
