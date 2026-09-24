"""Plain-language summaries of recovered distributions, as tables and markdown."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .pipeline import ExpiryResult, SurfaceResult

SUMMARY_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
MOVE_SIZES = (0.05, 0.10, 0.20)


def distribution_summary(
    result: ExpiryResult,
    quantiles: Sequence[float] = SUMMARY_QUANTILES,
    moves: Sequence[float] = MOVE_SIZES,
) -> dict[str, float | int]:
    """Headline numbers for one expiry's recovered distribution.

    Moves are measured from the underlying's current price when known,
    otherwise from the forward.

    Args:
        result: One expiry's pipeline output.
        quantiles: Probabilities at which to report price levels.
        moves: Move sizes (0.10 = 10%) for the up/down probabilities.

    Returns:
        ``days``, the forward, the annualised volatility, skew and excess
        kurtosis of the log-return, each quantile as a % move, and the
        probability of a fall or rise of at least each move size.
    """
    d = result.density
    ref = result.spot if result.spot is not None else d.forward
    m = d.moments()
    out: dict[str, float | int] = {
        "days": result.days,
        "forward": d.forward,
        "implied_rate_pct": result.clean.implied_rate * 100.0,
        "vol_pct": m["vol_annualised"] * 100.0,
        "skew": m["skew_log"],
        "excess_kurt": m["excess_kurt_log"],
    }
    for p, level in zip(quantiles, d.quantile(np.asarray(quantiles)), strict=True):
        out[f"q{round(p * 100):02d}_move_pct"] = (float(level) / ref - 1.0) * 100.0
    for size in moves:
        pct = round(size * 100)
        out[f"p_down_{pct}"] = float(d.prob_below(ref * (1.0 - size)))
        out[f"p_up_{pct}"] = float(d.prob_above(ref * (1.0 + size)))
    return out


def summary_table(surface: SurfaceResult) -> pd.DataFrame:
    """:func:`distribution_summary` for every expiry.

    Args:
        surface: Pipeline output.

    Returns:
        One row per expiry.
    """
    return pd.DataFrame([distribution_summary(e) for e in surface.expiries])


def cleaning_table(surface: SurfaceResult) -> pd.DataFrame:
    """What the cleaning step kept and dropped, per expiry.

    Args:
        surface: Pipeline output.

    Returns:
        Quote counts by outcome, the forward and discount estimates, and the
        number of quotes the smile fit finally used.
    """
    rows = []
    for e in surface.expiries:
        counts = e.clean.drop_counts
        rows.append(
            {
                "days": e.days,
                "quotes": e.clean.n_raw,
                "zero_bid": counts.get("zero_bid", 0),
                "crossed": counts.get("crossed", 0),
                "wide": counts.get("wide", 0),
                "parity": counts.get("parity", 0),
                "otm_fitted": int(e.used.sum()),
                "outliers": int((~e.used).sum()),
                "forward": e.clean.forward,
                "discount": e.clean.discount,
            }
        )
    return pd.DataFrame(rows)


def _is_number_column(column: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column))


def _default_format(value: Any) -> str:
    if isinstance(value, bool | np.bool_):
        return "pass" if value else "FAIL"
    if isinstance(value, int | np.integer):
        return str(value)
    if isinstance(value, float | np.floating):
        return f"{value:.4g}"
    return str(value)


def markdown_table(
    df: pd.DataFrame,
    formats: dict[str, Callable[[Any], str]] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Args:
        df: Table to render.
        formats: Optional per-column formatting functions.
        headers: Optional display names for columns.

    Returns:
        Markdown text, one line per row.
    """
    formats = formats or {}
    headers = headers or {}
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(headers.get(c, c) for c in cols) + " |",
        "|" + "|".join("---:" if _is_number_column(df[c]) else ":---" for c in cols) + "|",
    ]
    for _, row in df.iterrows():
        cells = [formats.get(c, _default_format)(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(digits: int = 1) -> Callable[[Any], str]:
    """Formatter for a probability shown as a percentage.

    Args:
        digits: Decimal places.

    Returns:
        A function mapping 0.123 to ``"12.3%"``.
    """
    return lambda v: f"{float(v) * 100:.{digits}f}%"


def fixed(digits: int = 2, suffix: str = "") -> Callable[[Any], str]:
    """Formatter with a fixed number of decimals.

    Args:
        digits: Decimal places.
        suffix: Text appended to each value.

    Returns:
        A function mapping a number to text.
    """
    return lambda v: f"{float(v):.{digits}f}{suffix}"


def recovery_report(surface: SurfaceResult) -> str:
    """Markdown report for a pipeline run on any quote table (no truth needed).

    Args:
        surface: Pipeline output.

    Returns:
        Markdown text with cleaning, distribution and check tables.
    """
    summary = summary_table(surface)
    checks = surface.check_table()
    move_cols = [c for c in summary.columns if c.startswith(("p_down", "p_up"))]
    parts = [
        "# Risk-neutral distribution report",
        "",
        f"Smile method: `{surface.method}`; tails: `{surface.tails}`.",
        "",
        "## Cleaning",
        "",
        markdown_table(cleaning_table(surface), {"forward": fixed(3), "discount": fixed(5)}),
        "",
        "## Distribution by expiry",
        "",
        "Quantiles are shown as % moves from the current underlying price.",
        "",
        markdown_table(
            summary.drop(columns=move_cols),
            {c: fixed(2) for c in summary.columns if c not in ("days",)},
        ),
        "",
        "Probability of a move of at least the stated size by expiry:",
        "",
        markdown_table(summary[["days", *move_cols]], {c: pct() for c in move_cols}),
        "",
        "## Checks",
        "",
        markdown_table(checks),
        "",
    ]
    return "\n".join(parts)
