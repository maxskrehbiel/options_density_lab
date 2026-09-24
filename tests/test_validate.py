"""Validation metrics behave as distances: zero against the truth itself, large for a bad model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from options_density_lab.pipeline import SurfaceResult
from options_density_lab.validate import TruthCache, compare, lognormal_baseline


def test_truth_scored_against_itself_has_zero_error(
    svi_surface: SurfaceResult, truth: TruthCache
) -> None:
    e = svi_surface.expiries[1]
    pdf, cdf = truth.on(e.density.strikes, e.maturity)
    perfect = replace(e, density=replace(e.density, pdf=pdf, cdf=cdf))
    metrics = compare(perfect, truth)
    assert metrics["tv"] < 1e-3
    assert metrics["ks"] < 1e-3
    assert float(metrics["q_err_max_pct"]) < 0.05


def test_lognormal_baseline_is_much_worse_than_recovery(
    svi_surface: SurfaceResult, truth: TruthCache
) -> None:
    for e in svi_surface.expiries:
        fitted = compare(e, truth)
        baseline = compare(lognormal_baseline(e), truth)
        assert baseline["method"] == "lognormal"
        assert float(baseline["tv"]) > 3 * float(fitted["tv"])


def test_truth_quantiles_are_ordered_and_skewed(truth: TruthCache) -> None:
    q = truth.quantiles(0.5)
    fwd = truth.setup.forward(0.5)
    assert np.all(np.diff(q) > 0)
    # Downside skew: the mean (the forward) sits below the median, and the
    # 1% quantile is further below the median than the 99% is above it.
    assert fwd < q[3] < 1.05 * fwd
    assert q[3] - q[0] > q[-1] - q[3]
