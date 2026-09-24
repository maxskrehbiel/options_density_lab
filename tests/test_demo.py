"""The demo's self-grading against the synthetic truth."""

from __future__ import annotations

import pandas as pd
import pytest

from options_density_lab.demo import DemoConfig, DemoGrade, DemoResult, grade
from options_density_lab.pipeline import SurfaceResult


def _validation(tv_noisy: float, tv_exact: float, tv_baseline: float) -> pd.DataFrame:
    rows = [
        ("noisy", "svi", tv_noisy, 0.01, 1.0),
        ("noisy", "spline", tv_noisy, 0.01, 1.0),
        ("exact", "spline", tv_exact, 0.001, 0.0),
        ("noisy", "lognormal", tv_baseline, 0.1, 1.0),
    ]
    return pd.DataFrame(rows, columns=["quotes", "method", "tv", "ks", "fwd_err_bp"]).assign(
        days=30
    )


@pytest.fixture
def surfaces(svi_surface: SurfaceResult, spline_surface: SurfaceResult) -> dict[str, SurfaceResult]:
    return {"svi": svi_surface, "spline": spline_surface}


def test_good_results_pass_every_grade(surfaces: dict[str, SurfaceResult]) -> None:
    grades = grade(DemoConfig(), surfaces, _validation(0.02, 0.005, 0.2))
    assert all(g.passed for g in grades), [g for g in grades if not g.passed]
    assert len(grades) == 7


@pytest.mark.parametrize(
    ("tv_noisy", "tv_exact", "tv_baseline", "failing"),
    [
        (0.06, 0.005, 0.5, "density error, noisy quotes"),
        (0.02, 0.02, 0.2, "spline density error, exact quotes"),
        (0.02, 0.005, 0.04, "fitted smile beats flat-vol baseline"),
    ],
)
def test_a_bad_result_fails_its_grade(
    surfaces: dict[str, SurfaceResult],
    tv_noisy: float,
    tv_exact: float,
    tv_baseline: float,
    failing: str,
) -> None:
    grades = grade(DemoConfig(), surfaces, _validation(tv_noisy, tv_exact, tv_baseline))
    assert [g.name for g in grades if not g.passed] == [failing]
    result = DemoResult(
        files=[], validation=pd.DataFrame(), seed_study=pd.DataFrame(), grades=grades
    )
    assert not result.all_passed


def test_grade_message_shows_measured_value_and_limit() -> None:
    g = DemoGrade("forward from put-call parity", True, "worst error 1.2 bp", "below 5 bp")
    assert g.message == "worst error 1.2 bp (limit: below 5 bp)"
