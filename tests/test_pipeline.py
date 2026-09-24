"""End-to-end recovery on synthetic chains, graded against the known truth."""

from __future__ import annotations

import pandas as pd
import pytest

from options_density_lab.errors import QuoteDataError
from options_density_lab.pipeline import (
    PipelineConfig,
    SurfaceResult,
    prepare_quotes,
    run_pipeline,
)
from options_density_lab.synthetic import SyntheticChain
from options_density_lab.validate import TruthCache, validate_surface

# Tolerances sit well below the error of a flat-volatility (lognormal) baseline,
# which is 11-21% total variation on this market.
TV_TOLERANCE = 0.05
KS_TOLERANCE = 0.025


@pytest.mark.parametrize("surface_name", ["svi_surface", "spline_surface"])
def test_every_check_passes(surface_name: str, request: pytest.FixtureRequest) -> None:
    surface: SurfaceResult = request.getfixturevalue(surface_name)
    table = surface.check_table()
    assert surface.all_passed, table[~table["passed"]].to_string()
    assert len(surface.expiries) == 4


@pytest.mark.parametrize("surface_name", ["svi_surface", "spline_surface"])
def test_recovery_error_is_within_tolerance(
    surface_name: str, request: pytest.FixtureRequest, truth: TruthCache
) -> None:
    surface: SurfaceResult = request.getfixturevalue(surface_name)
    metrics = validate_surface(surface, truth)
    assert (metrics["tv"] < TV_TOLERANCE).all(), metrics[["days", "tv"]].to_string()
    assert (metrics["ks"] < KS_TOLERANCE).all(), metrics[["days", "ks"]].to_string()
    assert (metrics["fwd_err_bp"].abs() < 5).all()


def test_exact_quotes_with_spline_are_nearly_exact(
    exact_chain: SyntheticChain, truth: TruthCache
) -> None:
    surface = run_pipeline(exact_chain.quotes, PipelineConfig.for_method("spline"))
    metrics = validate_surface(surface, truth)
    assert (metrics["tv"] < 0.015).all(), metrics[["days", "tv"]].to_string()
    assert (metrics["iv_rmse_volpts"] < 0.05).all()


def test_missing_columns_raise() -> None:
    with pytest.raises(QuoteDataError, match="missing column"):
        run_pipeline(pd.DataFrame({"strike": [100.0]}))


def test_empty_table_raises_value_error() -> None:
    empty = pd.DataFrame(columns=["maturity", "strike", "option_type", "bid", "ask"])
    with pytest.raises(ValueError, match="quote table has no rows"):
        run_pipeline(empty)


def test_option_type_is_normalised(noisy_chain: SyntheticChain) -> None:
    quotes = noisy_chain.quotes.copy()
    quotes["option_type"] = quotes["option_type"].map({"C": " c", "P": "p "})
    prepared = prepare_quotes(quotes)
    assert set(prepared["option_type"]) == {"C", "P"}
    pd.testing.assert_series_equal(prepared["strike"], noisy_chain.quotes["strike"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("option_type", "X", "C or P"), ("bid", "n/a", "non-numeric"), ("maturity", None, "missing")],
)
def test_bad_values_are_rejected(
    noisy_chain: SyntheticChain, column: str, value: object, message: str
) -> None:
    quotes = noisy_chain.quotes.astype({column: object})
    quotes.loc[0, column] = value
    with pytest.raises(QuoteDataError, match=message):
        prepare_quotes(quotes)
