"""Command-line interface: ``demo``, ``generate`` and ``recover`` subcommands."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .demo import DemoConfig, run_demo
from .density import DensityConfig
from .errors import OptionsDensityError, QuoteDataError
from .pipeline import PipelineConfig, run_pipeline
from .report import recovery_report
from .synthetic import MarketNoise, generate_chain

EXIT_OK = 0
EXIT_CHECKS_FAILED = 1
EXIT_BAD_INPUT = 2
LOG_LEVELS = (logging.WARNING, logging.INFO, logging.DEBUG)


def positive_int(text: str) -> int:
    """Argparse type for counts that must be at least 1.

    Args:
        text: The raw argument.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not an integer of at least 1.
    """
    value = non_negative_int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {text!r}")
    return value


def non_negative_int(text: str) -> int:
    """Argparse type for seeds and other integers that must be at least 0.

    Args:
        text: The raw argument.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not an integer of at least 0.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {text!r}")
    return value


def log_level(verbosity: int) -> int:
    """Logging level for a count of ``-v`` flags: WARNING, then INFO, then DEBUG.

    Args:
        verbosity: Number of ``-v`` flags given.

    Returns:
        A ``logging`` level.
    """
    return LOG_LEVELS[min(max(verbosity, 0), len(LOG_LEVELS) - 1)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="options_density_lab",
        description="Recover market-implied (risk-neutral) distributions from option prices.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="-v for progress, -vv for debug detail"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the synthetic demo, grade it against the truth")
    demo.add_argument("--out", type=Path, default=Path("demo_output"), help="default: demo_output")
    demo.add_argument("--seed", type=non_negative_int, default=7, help="headline chain seed")
    demo.add_argument("--seeds", type=positive_int, default=10, help="chains in the seed study")

    gen = sub.add_parser("generate", help="write a synthetic option chain to CSV")
    gen.add_argument("--out", type=Path, required=True, help="CSV file to write")
    gen.add_argument("--seed", type=non_negative_int, default=7, help="random seed (default: 7)")
    gen.add_argument("--exact", action="store_true", help="no spreads or noise: bid = ask = true")

    rec = sub.add_parser("recover", help="recover distributions from a quote CSV")
    rec.add_argument("quotes", type=Path, help="CSV with maturity, strike, option_type, bid, ask")
    rec.add_argument("--method", choices=("svi", "spline"), default="svi", help="default: svi")
    rec.add_argument(
        "--tails",
        choices=("smile", "exponential"),
        default=None,
        help="tail model (default: smile for svi, exponential for spline)",
    )
    rec.add_argument("--out", type=Path, default=Path("output"), help="default: output")
    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    result = run_demo(DemoConfig(out_dir=args.out, seed=args.seed, n_seeds=args.seeds))
    print(f"wrote {len(result.files)} files to {args.out}")
    print("self-check against the synthetic truth:")
    for grade in result.grades:
        print(f"  {'PASS' if grade.passed else 'FAIL'}  {grade.name}: {grade.message}")
    return EXIT_OK if result.all_passed else EXIT_CHECKS_FAILED


def _cmd_generate(args: argparse.Namespace) -> int:
    noise = MarketNoise.none() if args.exact else MarketNoise()
    chain = generate_chain(np.random.default_rng(args.seed), noise=noise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    chain.quotes.to_csv(args.out, index=False, float_format="%.6g", lineterminator="\n")
    print(f"wrote {len(chain.quotes)} synthetic quotes to {args.out}")
    return EXIT_OK


def _read_quotes(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError) as exc:  # missing file, empty file or malformed CSV
        raise QuoteDataError(f"cannot read quotes from {path}: {exc}") from exc


def _cmd_recover(args: argparse.Namespace) -> int:
    quotes = _read_quotes(args.quotes)
    config = replace(
        PipelineConfig.for_method(args.method), density=DensityConfig(tails=args.tails)
    )
    surface = run_pipeline(quotes, config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "recovery_report.md").write_text(
        recovery_report(surface), encoding="utf-8", newline="\n"
    )
    frames = [
        pd.DataFrame(
            {"days": e.days, "price": e.density.strikes, "pdf": e.density.pdf, "cdf": e.density.cdf}
        )
        for e in surface.expiries
    ]
    pd.concat(frames).to_csv(
        args.out / "densities.csv", index=False, float_format="%.8g", lineterminator="\n"
    )
    for e in surface.expiries:
        q05, q50, q95 = e.density.quantile(np.array([0.05, 0.5, 0.95]))
        print(
            f"{e.days:>4} days  forward {e.clean.forward:9.3f}  "
            f"5%/50%/95%: {q05:.2f} / {q50:.2f} / {q95:.2f}"
        )
    print(f"all checks passed: {surface.all_passed}; report written to {args.out}")
    return EXIT_OK if surface.all_passed else EXIT_CHECKS_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``options_density_lab`` and ``python -m options_density_lab``.

    Args:
        argv: Arguments, excluding the program name; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if a domain check failed, 2 on bad input. Usage errors
        exit with 2 through argparse.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=log_level(args.verbose), format="%(levelname)s %(name)s: %(message)s")
    handlers = {"demo": _cmd_demo, "generate": _cmd_generate, "recover": _cmd_recover}
    try:
        return handlers[args.command](args)
    except OptionsDensityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
