"""Summaries and markdown rendering."""

from __future__ import annotations

import pandas as pd

from options_density_lab.pipeline import SurfaceResult
from options_density_lab.report import (
    cleaning_table,
    distribution_summary,
    fixed,
    markdown_table,
    pct,
    recovery_report,
)


def test_distribution_summary_is_internally_consistent(svi_surface: SurfaceResult) -> None:
    for e in svi_surface.expiries:
        s = distribution_summary(e)
        assert s["q05_move_pct"] < s["q50_move_pct"] < s["q95_move_pct"]
        assert s["p_down_20"] < s["p_down_10"] < s["p_down_5"] < 1
        assert s["p_up_20"] < s["p_up_10"] < s["p_up_5"] < 1
        assert s["skew"] < 0  # the synthetic market has a downside skew
        assert 10 < s["vol_pct"] < 30


def test_cleaning_table_accounts_for_every_quote(svi_surface: SurfaceResult) -> None:
    table = cleaning_table(svi_surface)
    assert (table["otm_fitted"] > 10).all()
    assert (table["quotes"] >= table["zero_bid"] + table["crossed"] + table["otm_fitted"]).all()


def test_markdown_table_formats_cells() -> None:
    df = pd.DataFrame({"name": ["a", "b"], "ok": [True, False], "p": [0.1234, 0.5], "x": [1.5, 2]})
    text = markdown_table(df, {"p": pct(1), "x": fixed(1, "%")}, {"p": "prob"})
    lines = text.splitlines()
    assert lines[0] == "| name | ok | prob | x |"
    assert lines[1] == "|:---|:---|---:|---:|"
    assert lines[2] == "| a | pass | 12.3% | 1.5% |"
    assert lines[3] == "| b | FAIL | 50.0% | 2.0% |"


def test_recovery_report_has_every_section(svi_surface: SurfaceResult) -> None:
    text = recovery_report(svi_surface)
    for heading in ("## Cleaning", "## Distribution by expiry", "## Checks"):
        assert heading in text
    assert "FAIL" not in text
