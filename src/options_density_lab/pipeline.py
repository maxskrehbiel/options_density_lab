"""End-to-end recovery for every expiry in a quote table: clean, fit, recover, check."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import pandas as pd

from . import checks
from ._types import BoolArray
from .clean import CleanConfig, CleanReport, clean_expiry
from .density import DensityConfig, RiskNeutralDensity, rnd_from_smile
from .errors import QuoteDataError
from .smile import Smile, SmileConfig, SmileMethod, fit_smile

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("maturity", "strike", "option_type", "bid", "ask")
NUMERIC_COLUMNS = ("maturity", "strike", "bid", "ask")
OPTIONAL_NUMERIC_COLUMNS = ("days", "underlying")
OPTION_TYPES = frozenset({"C", "P"})
DAYS_PER_YEAR = 365


@dataclass(frozen=True)
class PipelineConfig:
    """Settings for every stage.

    Attributes:
        clean: Cleaning settings.
        smile: Smile-fitting settings (including the method).
        density: Density grid and tail settings.
    """

    clean: CleanConfig = field(default_factory=CleanConfig)
    smile: SmileConfig = field(default_factory=SmileConfig)
    density: DensityConfig = field(default_factory=DensityConfig)

    @classmethod
    def for_method(cls, method: SmileMethod) -> PipelineConfig:
        """Default settings with the given smile method.

        Args:
            method: ``"svi"`` or ``"spline"``.

        Returns:
            The configuration.
        """
        return cls(smile=replace(SmileConfig(), method=method))


@dataclass(frozen=True)
class ExpiryResult:
    """Everything produced for one expiry.

    Attributes:
        maturity: Time to expiry in years.
        days: Calendar days to expiry.
        otm: Cleaned out-of-the-money quotes with implied vols.
        used: Which rows of ``otm`` the final smile fit used.
        clean: Audit record of the cleaning step.
        smile: The fitted smile.
        density: The recovered distribution.
        check_results: Quote, smile and density checks for this expiry.
        spot: Underlying price at quote time, if supplied.
    """

    maturity: float
    days: int
    otm: pd.DataFrame = field(repr=False)
    used: BoolArray = field(repr=False)
    clean: CleanReport = field(repr=False)
    smile: Smile = field(repr=False)
    density: RiskNeutralDensity = field(repr=False)
    check_results: list[checks.CheckResult] = field(default_factory=list, repr=False)
    spot: float | None = None


@dataclass(frozen=True)
class SurfaceResult:
    """Results for every expiry plus the cross-expiry calendar check.

    Attributes:
        method: Smile method used.
        tails: Tail model used.
        expiries: Per-expiry results, shortest first.
        calendar: Calendar-arbitrage check across the fitted smiles.
    """

    method: str
    tails: str
    expiries: list[ExpiryResult]
    calendar: checks.CheckResult

    def check_table(self) -> pd.DataFrame:
        """All checks as one table.

        Returns:
            Columns ``days``, ``check``, ``passed``, ``message``.
        """
        rows: list[dict[str, object]] = [
            {"days": e.days, "check": c.name, "passed": c.passed, "message": c.message}
            for e in self.expiries
            for c in e.check_results
        ]
        cal = self.calendar
        rows.append(
            {"days": "all", "check": cal.name, "passed": cal.passed, "message": cal.message}
        )
        return pd.DataFrame(rows)

    @property
    def all_passed(self) -> bool:
        """True when every check passed."""
        per_expiry = all(c.passed for e in self.expiries for c in e.check_results)
        return bool(self.calendar.passed and per_expiry)


def prepare_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw quote table and return a normalised copy.

    ``option_type`` is stripped and upper-cased, and the price columns are
    converted to numbers.

    Args:
        quotes: Raw quotes with ``REQUIRED_COLUMNS``.

    Returns:
        The normalised copy.

    Raises:
        QuoteDataError: If a column is missing, the table is empty, a numeric
            column holds text, or an option type is not ``C`` or ``P``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in quotes.columns]
    if missing:
        raise QuoteDataError(f"quote table is missing column(s): {', '.join(missing)}")
    if quotes.empty:
        raise QuoteDataError("quote table has no rows")
    out = quotes.copy()
    out["option_type"] = out["option_type"].astype(str).str.strip().str.upper()
    unknown = sorted(set(out["option_type"]) - OPTION_TYPES)
    if unknown:
        raise QuoteDataError(f"option_type must be C or P, found: {', '.join(unknown)}")
    present_optional = [c for c in OPTIONAL_NUMERIC_COLUMNS if c in out.columns]
    for col in (*NUMERIC_COLUMNS, *present_optional):
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.isna().any():
            raise QuoteDataError(f"column {col!r} has missing or non-numeric values")
        out[col] = converted.astype(float)
    return out


def _run_expiry(
    quotes: pd.DataFrame, maturity: float, prev: Smile | None, config: PipelineConfig
) -> ExpiryResult:
    otm, report = clean_expiry(quotes, config.clean)
    smile, used = fit_smile(otm, maturity, report.forward, prev, config.smile)
    dens = rnd_from_smile(smile, report.discount, config.density)
    results = [
        checks.check_quote_butterflies(otm[used], report.forward, report.discount),
        checks.check_smile_butterfly(smile),
        *checks.check_density(dens),
    ]
    days = (
        int(quotes["days"].iloc[0]) if "days" in quotes.columns else round(maturity * DAYS_PER_YEAR)
    )
    spot = float(quotes["underlying"].iloc[0]) if "underlying" in quotes.columns else None
    logger.info("T=%.3f (%s): %d quotes fitted", maturity, config.smile.method, int(used.sum()))
    return ExpiryResult(
        maturity=maturity,
        days=days,
        otm=otm,
        used=used,
        clean=report,
        smile=smile,
        density=dens,
        check_results=results,
        spot=spot,
    )


def run_pipeline(quotes: pd.DataFrame, config: PipelineConfig | None = None) -> SurfaceResult:
    """Recover one risk-neutral distribution per expiry from a raw quote table.

    Expiries run shortest first so each SVI fit can be held above the
    previous expiry's total variance.

    Args:
        quotes: Quotes with columns ``maturity`` (years), ``strike``,
            ``option_type`` (``"C"``/``"P"``), ``bid`` and ``ask``; optionally
            ``days`` and ``underlying``.
        config: Settings; defaults to ``PipelineConfig()``.

    Returns:
        Per-expiry results and the calendar check.

    Raises:
        QuoteDataError: If the quote table fails validation (see :func:`prepare_quotes`).
        CleaningError: If an expiry cannot support a forward and discount estimate.
    """
    config = config or PipelineConfig()
    table = prepare_quotes(quotes)
    results: list[ExpiryResult] = []
    prev: Smile | None = None
    for maturity in sorted(float(t) for t in table["maturity"].unique()):
        result = _run_expiry(table[table["maturity"] == maturity], maturity, prev, config)
        results.append(result)
        prev = result.smile
    return SurfaceResult(
        method=config.smile.method,
        tails=results[0].density.tails,
        expiries=results,
        calendar=checks.check_calendar([r.smile for r in results]),
    )
