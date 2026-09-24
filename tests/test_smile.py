"""Smile fitting: SVI recovery, analytic gradients, penalties, spline guard, outlier removal."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from options_density_lab.smile import (
    SmileConfig,
    SVISmile,
    _svi_with_gradient,
    fit_smile,
    fit_spline,
    fit_svi,
    g_function,
    svi_total_variance,
)

TRUE_SVI = (0.01, 0.1, -0.5, 0.02, 0.1)
K = np.linspace(-0.4, 0.3, 30)


def _otm_frame(k: np.ndarray, w: np.ndarray, w_err: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"k": k, "w": w, "w_err": w_err})


def test_svi_recovers_known_parameters_from_exact_data() -> None:
    w = svi_total_variance(K, *TRUE_SVI)
    fit = fit_svi(K, w, np.full(K.size, 1e-5), maturity=0.5, forward=100.0)
    np.testing.assert_allclose(fit.params, TRUE_SVI, atol=1e-4)
    np.testing.assert_allclose(fit.total_variance(K), w, atol=1e-8)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_svi_parameter_gradients_match_finite_differences(index: int) -> None:
    x = np.array(TRUE_SVI)
    analytic = _svi_with_gradient(K, *x)[3 + index]
    for j, point in enumerate(K[::7]):

        def value(p: np.ndarray, pt: float = point) -> float:
            return float(_svi_with_gradient(np.array([pt]), *p)[index][0])

        numeric = approx_fprime(x, value, 1e-7)
        np.testing.assert_allclose(analytic[j * 7], numeric, rtol=1e-4, atol=1e-6)


def test_svi_derivatives_match_finite_differences() -> None:
    h = 1e-5
    for nu in (1, 2):
        numeric = (
            svi_total_variance(K + h, *TRUE_SVI, derivative=nu - 1)
            - svi_total_variance(K - h, *TRUE_SVI, derivative=nu - 1)
        ) / (2 * h)
        np.testing.assert_allclose(
            svi_total_variance(K, *TRUE_SVI, derivative=nu), numeric, atol=1e-6
        )
    with pytest.raises(ValueError, match="derivative"):
        svi_total_variance(K, *TRUE_SVI, derivative=3)


def test_flat_smile_has_g_one_and_normal_log_density() -> None:
    w = 0.04
    flat = SVISmile(a=w, b=0.0, rho=0.0, m=0.0, s=1.0, maturity=1.0, forward=100.0, k_lo=-1, k_hi=1)
    np.testing.assert_allclose(flat.g(K), 1.0)
    normal = np.exp(-((K + w / 2) ** 2) / (2 * w)) / np.sqrt(2 * np.pi * w)
    np.testing.assert_allclose(flat.logprice_pdf(K), normal, rtol=1e-12)
    np.testing.assert_allclose(flat.iv(K), 0.2)
    zeros = np.zeros_like(K)
    np.testing.assert_allclose(g_function(K, np.full_like(K, w), zeros, zeros), 1.0)


def test_calendar_penalty_keeps_fit_above_previous_expiry() -> None:
    prev = SVISmile(*TRUE_SVI, maturity=0.25, forward=100.0, k_lo=-0.4, k_hi=0.3)
    # Data deliberately below the previous smile on the right: an unconstrained fit would cross it.
    w = svi_total_variance(K, *TRUE_SVI) + 0.002 - 0.01 * np.clip(K, 0, None)
    fit = fit_svi(K, w, np.full(K.size, 2e-4), 0.5, 100.0, prev=prev)
    grid = np.linspace(-0.4, 0.3, 101)
    assert np.min(fit.total_variance(grid) - prev.total_variance(grid)) > -2e-5


def test_spline_fits_smooth_data_closely() -> None:
    w = svi_total_variance(K, *TRUE_SVI)
    fit = fit_spline(K, w, np.full(K.size, 1e-4), 0.5, 100.0)
    np.testing.assert_allclose(fit.total_variance(K), w, atol=2e-5)
    # Straight-line continuation outside the quoted range, with no curvature there.
    assert fit.total_variance(np.array([0.5]), 2)[0] == 0.0
    slope = fit.total_variance(np.array([0.3]), 1)[0]
    assert fit.total_variance(np.array([0.4]))[0] == pytest.approx(
        fit.total_variance(np.array([0.3]))[0] + 0.1 * slope
    )


def test_spline_guard_removes_negative_density() -> None:
    # A narrow dip in total variance makes the cross-validated fit imply a
    # negative density on the dip's shoulders; the guard must smooth it away.
    k = np.linspace(-0.4, 0.3, 60)
    w = svi_total_variance(k, *TRUE_SVI) - 0.002 * np.exp(-(k**2) / (2 * 0.02**2))
    fit = fit_spline(k, w, np.full(k.size, 1e-5), 0.5, 100.0)
    assert fit.lam > fit.lam_gcv
    assert fit.g(np.linspace(k.min(), k.max(), 400)).min() >= 0


def test_fit_smile_drops_a_gross_outlier() -> None:
    w = svi_total_variance(K, *TRUE_SVI)
    err = np.full(K.size, 1e-4)
    w_bad = w.copy()
    w_bad[12] += 50 * err[12]
    smile, used = fit_smile(_otm_frame(K, w_bad, err), 0.5, 100.0)
    assert not used[12] and used.sum() == K.size - 1
    np.testing.assert_allclose(smile.total_variance(K), w, atol=1e-6)


def test_fit_smile_spline_and_unknown_method() -> None:
    frame = _otm_frame(K, svi_total_variance(K, *TRUE_SVI), np.full(K.size, 1e-4))
    smile, _ = fit_smile(frame, 0.5, 100.0, config=replace(SmileConfig(), method="spline"))
    assert smile.method == "spline"
    with pytest.raises(ValueError, match="unknown smile method"):
        fit_smile(frame, 0.5, 100.0, config=replace(SmileConfig(), method="cubic"))  # type: ignore[arg-type]
