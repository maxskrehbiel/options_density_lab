"""Command-line interface: generate, recover, demo, exit codes and argument checking."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from options_density_lab import __version__
from options_density_lab.cli import EXIT_BAD_INPUT, EXIT_OK, log_level, main


def test_generate_then_recover(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = tmp_path / "quotes.csv"
    assert main(["generate", "--out", str(csv), "--seed", "11"]) == EXIT_OK
    assert set(pd.read_csv(csv)["option_type"]) == {"C", "P"}

    out = tmp_path / "recovery"
    assert main(["recover", str(csv), "--method", "spline", "--out", str(out)]) == EXIT_OK
    assert (out / "recovery_report.md").read_bytes().endswith(b"\n")
    densities = pd.read_csv(out / "densities.csv")
    assert set(densities.columns) == {"days", "price", "pdf", "cdf"}
    assert "all checks passed: True" in capsys.readouterr().out


def test_generate_exact_chain(tmp_path: Path) -> None:
    csv = tmp_path / "exact.csv"
    assert main(["generate", "--out", str(csv), "--exact"]) == EXIT_OK
    quoted = pd.read_csv(csv).query("bid > 0")
    assert (quoted["bid"] == quoted["ask"]).all()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "cannot read quotes"),
        ("", "cannot read quotes"),
        ("maturity,strike,option_type,bid,ask\n", "no rows"),
        ("maturity,strike,option_type,bid,ask\n0.5,100,X,1,2\n", "C or P"),
    ],
)
def test_bad_input_exits_2_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None, message: str
) -> None:
    csv = tmp_path / "quotes.csv"
    if content is not None:
        csv.write_text(content, encoding="utf-8")
    assert main(["recover", str(csv), "--out", str(tmp_path / "out")]) == EXIT_BAD_INPUT
    err = capsys.readouterr().err
    assert err.startswith("error:") and message in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1


@pytest.mark.parametrize(
    "args", [["demo", "--seeds", "0"], ["generate", "--out", "x", "--seed", "-1"]]
)
def test_counts_and_seeds_are_validated(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == EXIT_BAD_INPUT
    assert "expected an integer" in capsys.readouterr().err


def test_demo_grades_itself(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out), "--seeds", "1"]) == EXIT_OK
    names = {p.name for p in out.iterdir()}
    assert {"summary.md", "synthetic_quotes.csv", "density_vs_truth.png", "seed_study.png"} <= names
    summary = (out / "summary.md").read_bytes()
    assert summary.endswith(b"\n") and b"\r\n" not in summary
    assert b"FAIL" not in summary
    assert str(tmp_path).encode() not in summary  # figures are linked by relative name only
    stdout = capsys.readouterr().out
    assert "self-check against the synthetic truth" in stdout
    assert "PASS" in stdout and "FAIL" not in stdout


def test_verbosity_maps_to_log_levels(tmp_path: Path) -> None:
    assert [log_level(n) for n in range(4)] == [
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
        logging.DEBUG,
    ]
    assert main(["-vv", "generate", "--out", str(tmp_path / "q.csv")]) == EXIT_OK


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
