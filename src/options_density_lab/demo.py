"""The end-to-end demo: generate chains, recover, grade against the truth, write outputs.

Outputs are figures, a markdown summary and the sample quote table.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .pipeline import PipelineConfig, SurfaceResult, run_pipeline
from .report import cleaning_table, fixed, markdown_table, pct, summary_table
from .smile import SmileMethod
from .synthetic import MarketNoise, MarketSetup, SyntheticChain, generate_chain
from .validate import (
    METRIC_DESCRIPTIONS,
    TruthCache,
    compare,
    lognormal_baseline,
    validate_surface,
)

logger = logging.getLogger(__name__)

METHODS: tuple[SmileMethod, ...] = ("svi", "spline")
FIGURES = {
    "cleaning": "quote_cleaning.png",
    "smiles": "smile_fits.png",
    "densities": "density_vs_truth.png",
    "cdfs": "cdf_vs_truth.png",
    "arbitrage": "arbitrage_checks.png",
    "seeds": "seed_study.png",
}
SUMMARY_FILE = "summary.md"
QUOTES_FILE = "synthetic_quotes.csv"


@dataclass(frozen=True)
class DemoConfig:
    """Settings for the demo run, including the limits it grades itself against.

    Attributes:
        out_dir: Where figures, the summary and the sample quotes are written.
        seed: Seed for the headline chain; the seed study derives its own seeds from it.
        n_seeds: Number of independent noisy chains in the seed study.
        setup: The fictional market.
        noise: Quote defects for the noisy chains.
        truth_grid: Size of the fine grid the true distribution is computed on.
        max_tv: Largest acceptable total-variation distance on noisy quotes.
        max_ks: Largest acceptable CDF gap on noisy quotes.
        max_exact_spline_tv: Largest acceptable spline error on exact quotes.
        max_forward_error_bp: Largest acceptable forward error, in basis points.
        min_baseline_ratio: The flat-volatility baseline must be at least this
            many times worse than each fitted method.
    """

    out_dir: Path = Path("demo_output")
    seed: int = 7
    n_seeds: int = 10
    setup: MarketSetup = field(default_factory=MarketSetup)
    noise: MarketNoise = field(default_factory=MarketNoise)
    truth_grid: int = 2001
    max_tv: float = 0.05
    max_ks: float = 0.025
    max_exact_spline_tv: float = 0.015
    max_forward_error_bp: float = 5.0
    min_baseline_ratio: float = 3.0


@dataclass(frozen=True)
class DemoGrade:
    """One self-check of the demo against the synthetic truth.

    Attributes:
        name: What is checked.
        passed: Whether it passed.
        measured: The measured value, as text.
        limit: The limit it is held to, as text.
    """

    name: str
    passed: bool
    measured: str
    limit: str

    @property
    def message(self) -> str:
        """``measured`` and ``limit`` in one line."""
        return f"{self.measured} (limit: {self.limit})"


@dataclass(frozen=True)
class DemoResult:
    """What the demo produced.

    Attributes:
        files: Every file written.
        validation: Error metrics for noisy and exact quotes, both methods, and the baseline.
        seed_study: Error metrics per seed, expiry and method.
        grades: The demo's checks of its own output against the truth.
    """

    files: list[Path]
    validation: pd.DataFrame
    seed_study: pd.DataFrame
    grades: list[DemoGrade]

    @property
    def all_passed(self) -> bool:
        """True when every self-check passed."""
        return all(g.passed for g in self.grades)


def run_methods(
    chain: SyntheticChain, methods: Sequence[SmileMethod] = METHODS
) -> dict[str, SurfaceResult]:
    """Run the pipeline once per smile method on the same chain.

    Args:
        chain: Synthetic chain.
        methods: Smile methods to run.

    Returns:
        Surfaces keyed by method.
    """
    return {m: run_pipeline(chain.quotes, PipelineConfig.for_method(m)) for m in methods}


def seed_study(config: DemoConfig, truth: TruthCache) -> pd.DataFrame:
    """Recovery error across independently generated noisy chains.

    Child seeds come from ``numpy.random.SeedSequence(config.seed)``, so the
    study is reproducible and independent of the headline chain.

    Args:
        config: Demo settings.
        truth: Cached truth for ``config.setup``.

    Returns:
        One row per seed, expiry and method.
    """
    frames = []
    for i, child in enumerate(np.random.SeedSequence(config.seed).spawn(config.n_seeds)):
        chain = generate_chain(np.random.default_rng(child), config.setup, config.noise)
        for surface in run_methods(chain).values():
            frames.append(validate_surface(surface, truth).assign(seed=i))
        logger.info("seed study: chain %d of %d done", i + 1, config.n_seeds)
    return pd.concat(frames, ignore_index=True)


def _worst_errors(validation: pd.DataFrame) -> dict[str, float]:
    """The worst value of each graded error across expiries and methods."""
    noisy = validation[(validation["quotes"] == "noisy") & validation["method"].isin(METHODS)]
    exact = validation[(validation["quotes"] == "exact") & (validation["method"] == "spline")]
    flat = validation[validation["method"] == "lognormal"]
    baseline_tv = dict(zip(flat["days"].tolist(), flat["tv"].tolist(), strict=True))
    ratios = [
        baseline_tv[day] / tv
        for day, tv in zip(noisy["days"].tolist(), noisy["tv"].tolist(), strict=True)
    ]
    return {
        "tv": float(noisy["tv"].max()),
        "ks": float(noisy["ks"].max()),
        "exact_tv": float(exact["tv"].max()),
        "forward_bp": float(noisy["fwd_err_bp"].abs().max()),
        "baseline_ratio": float(min(ratios)),
    }


def grade(
    config: DemoConfig, surfaces: dict[str, SurfaceResult], validation: pd.DataFrame
) -> list[DemoGrade]:
    """Check the demo's own output against the synthetic truth.

    Args:
        config: Demo settings, which hold the limits.
        surfaces: Pipeline outputs on the headline noisy chain, keyed by method.
        validation: Output of the validation step.

    Returns:
        One grade per check.
    """
    worst = _worst_errors(validation)
    arbitrage = [
        DemoGrade(
            f"arbitrage and sanity checks, {m}",
            surfaces[m].all_passed,
            "all passed" if surfaces[m].all_passed else "some failed",
            "every check passes",
        )
        for m in METHODS
    ]
    rows = [
        (
            "density error, noisy quotes",
            worst["tv"] < config.max_tv,
            f"worst TV {worst['tv']:.2%}",
            f"below {config.max_tv:.1%}",
        ),
        (
            "CDF error, noisy quotes",
            worst["ks"] < config.max_ks,
            f"worst KS {worst['ks']:.2%}",
            f"below {config.max_ks:.1%}",
        ),
        (
            "spline density error, exact quotes",
            worst["exact_tv"] < config.max_exact_spline_tv,
            f"worst TV {worst['exact_tv']:.2%}",
            f"below {config.max_exact_spline_tv:.1%}",
        ),
        (
            "forward from put-call parity",
            worst["forward_bp"] < config.max_forward_error_bp,
            f"worst error {worst['forward_bp']:.1f} bp",
            f"below {config.max_forward_error_bp:g} bp",
        ),
        (
            "fitted smile beats flat-vol baseline",
            worst["baseline_ratio"] >= config.min_baseline_ratio,
            f"baseline error at least {worst['baseline_ratio']:.1f}x larger",
            f"at least {config.min_baseline_ratio:g}x",
        ),
    ]
    return [*arbitrage, *(DemoGrade(*row) for row in rows)]


def _validate_all(
    surfaces: dict[str, SurfaceResult],
    exact_surfaces: dict[str, SurfaceResult],
    truth: TruthCache,
) -> pd.DataFrame:
    baseline = pd.DataFrame(
        [compare(lognormal_baseline(e), truth) for e in surfaces["svi"].expiries]
    )
    return pd.concat(
        [validate_surface(s, truth).assign(quotes="noisy") for s in surfaces.values()]
        + [validate_surface(s, truth).assign(quotes="exact") for s in exact_surfaces.values()]
        + [baseline.assign(quotes="noisy")],
        ignore_index=True,
    )


def run_demo(config: DemoConfig | None = None) -> DemoResult:
    """Run the demo, write its outputs and grade it against the truth.

    Args:
        config: Demo settings; defaults to ``DemoConfig()``.

    Returns:
        The written files, validation tables and self-check grades.
    """
    config = config or DemoConfig()
    out = config.out_dir
    out.mkdir(parents=True, exist_ok=True)
    noisy = generate_chain(np.random.default_rng(config.seed), config.setup, config.noise)
    exact = generate_chain(np.random.default_rng(config.seed), config.setup, MarketNoise.none())
    truth = TruthCache(config.setup, n=config.truth_grid)
    surfaces = run_methods(noisy)
    validation = _validate_all(surfaces, run_methods(exact), truth)
    study = seed_study(config, truth)
    grades = grade(config, surfaces, validation)

    files = [
        plots.plot_quote_cleaning(surfaces["svi"].expiries[0], out / FIGURES["cleaning"]),
        plots.plot_smiles(surfaces, out / FIGURES["smiles"], config.setup.true_iv),
        plots.plot_densities(surfaces, out / FIGURES["densities"], truth),
        plots.plot_cdfs(surfaces, out / FIGURES["cdfs"], truth),
        plots.plot_arbitrage(surfaces["svi"], out / FIGURES["arbitrage"]),
        plots.plot_seed_study(study, out / FIGURES["seeds"]),
    ]
    summary = render_summary(config, noisy, surfaces, validation, study, grades)
    summary_path = out / SUMMARY_FILE
    summary_path.write_text(summary, encoding="utf-8", newline="\n")
    quotes_path = out / QUOTES_FILE
    noisy.quotes.to_csv(quotes_path, index=False, float_format="%.6g", lineterminator="\n")
    files += [summary_path, quotes_path]
    return DemoResult(files=files, validation=validation, seed_study=study, grades=grades)


def _intro(config: DemoConfig, chain: SyntheticChain) -> list[str]:
    s, h = config.setup, config.setup.heston
    counts = chain.truth[["is_stale", "is_crossed"]].sum()
    return [
        "# Demo summary",
        "",
        "Everything below comes from **synthetic** option quotes generated by this package "
        f"(seed {config.seed}). No market data is used. The generating model is Heston with "
        f"v0={h.v0}, kappa={h.kappa}, theta={h.theta}, sigma={h.sigma}, rho={h.rho}; "
        f"spot {s.spot:g}, rate {s.rate:.1%}, dividend yield {s.div_yield:.1%}. "
        "These are illustrative values, not calibrated to any market.",
        "",
        f"The chain has {len(chain.quotes)} quotes across {len(s.expiry_days)} expiries, "
        f"of which {int(counts['is_stale'])} were made stale and "
        f"{int(counts['is_crossed'])} crossed on purpose.",
        "",
    ]


def _distribution_section(svi: SurfaceResult) -> list[str]:
    summary = summary_table(svi)
    move_cols = [c for c in summary.columns if c.startswith(("p_down", "p_up"))]
    shape_cols = ["implied_rate_pct", "vol_pct", "skew", "excess_kurt"]
    shape_cols += [c for c in summary.columns if c.endswith("_move_pct")]
    return [
        "## 1. Cleaning",
        "",
        f"![quote cleaning]({FIGURES['cleaning']})",
        "",
        markdown_table(cleaning_table(svi), {"forward": fixed(3), "discount": fixed(5)}),
        "",
        "## 2. Smiles",
        "",
        f"![smile fits]({FIGURES['smiles']})",
        "",
        "## 3. Recovered distribution (SVI, noisy quotes)",
        "",
        f"![densities]({FIGURES['densities']})",
        "",
        f"![cdfs]({FIGURES['cdfs']})",
        "",
        "Quantiles as % moves from the current price, and the distribution's volatility and shape:",
        "",
        markdown_table(summary[["days", *shape_cols]], {c: fixed(2) for c in shape_cols}),
        "",
        "Probability of a move of at least the stated size by expiry:",
        "",
        markdown_table(summary[["days", *move_cols]], {c: pct() for c in move_cols}),
        "",
    ]


def _checks_section(svi: SurfaceResult, grades: list[DemoGrade]) -> list[str]:
    self_checks = pd.DataFrame(
        {
            "check": [g.name for g in grades],
            "result": [g.passed for g in grades],
            "limit": [g.limit for g in grades],
        }
    )
    return [
        "## 4. No-arbitrage and sanity checks (SVI, noisy quotes)",
        "",
        f"![arbitrage checks]({FIGURES['arbitrage']})",
        "",
        markdown_table(svi.check_table()[["days", "check", "passed"]]),
        "",
        "## 5. Self-check against the known truth",
        "",
        markdown_table(self_checks),
        "",
    ]


def _validation_section(validation: pd.DataFrame) -> list[str]:
    cols = ["quotes", "method", "days", "tv", "tv_tails", "ks", "q_err_max_pct", "iv_rmse_volpts"]
    formats = {
        "tv": pct(2),
        "tv_tails": pct(2),
        "ks": pct(2),
        "q_err_max_pct": fixed(2, "%"),
        "iv_rmse_volpts": fixed(3),
    }
    headers = {
        "tv": "TV distance",
        "tv_tails": "TV from tails",
        "ks": "KS distance",
        "q_err_max_pct": "worst quantile error",
        "iv_rmse_volpts": "smile RMSE (vol pts)",
    }
    svi = validation[(validation["quotes"] == "noisy") & (validation["method"] == "svi")]
    truth_cols = [
        "days",
        "q05_hat",
        "q05_true",
        "q50_hat",
        "q50_true",
        "q95_hat",
        "q95_true",
        "skew_log_hat",
        "skew_log_true",
        "tail_left_hat",
        "tail_left_true",
        "tail_right_hat",
        "tail_right_true",
    ]
    truth_fmt = (
        {c: fixed(2) for c in truth_cols[1:7]}
        | {c: fixed(3) for c in truth_cols[7:9]}
        | {c: pct(2) for c in truth_cols[9:]}
    )
    return [
        "## 6. Validation against the known truth",
        "",
        "Metric definitions:",
        "",
        *[f"- `{k}`: {v}" for k, v in METRIC_DESCRIPTIONS.items()],
        "",
        "`exact` quotes have no spread or noise (bid = ask = true price); `noisy` are the "
        "realistic quotes above. The gap between the two rows for a method is the cost of "
        "market noise; the exact row is the method's own approximation error. The "
        "`lognormal` rows are a reference point, not a method: the bell curve implied by "
        "a single at-the-money volatility, which ignores the smile.",
        "",
        markdown_table(validation[cols], formats, headers),
        "",
        "Recovered versus true quantiles, skew and market-implied tail mass (SVI, noisy):",
        "",
        markdown_table(svi[truth_cols], truth_fmt),
        "",
    ]


def _seed_section(config: DemoConfig, study: pd.DataFrame) -> list[str]:
    grouped = study.groupby(["method", "days"])
    agg = pd.DataFrame(
        {
            "median TV": grouped["tv"].median(),
            "90th pct TV": grouped["tv"].quantile(0.9),
            "median KS": grouped["ks"].median(),
            "90th pct KS": grouped["ks"].quantile(0.9),
        }
    ).reset_index()
    return [
        f"## 7. Seed study ({config.n_seeds} independent noisy chains)",
        "",
        f"![seed study]({FIGURES['seeds']})",
        "",
        markdown_table(agg, {c: pct(2) for c in agg.columns if "TV" in c or "KS" in c}),
        "",
    ]


def render_summary(
    config: DemoConfig,
    chain: SyntheticChain,
    surfaces: dict[str, SurfaceResult],
    validation: pd.DataFrame,
    study: pd.DataFrame,
    grades: list[DemoGrade],
) -> str:
    """Compose the demo's markdown summary.

    Only pass/fail results and rounded quantities appear, never raw float
    residuals, so the file is byte-identical across runs and platforms.

    Args:
        config: Demo settings.
        chain: The headline noisy chain.
        surfaces: Pipeline outputs on that chain, keyed by method.
        validation: Output of the validation step.
        study: Output of :func:`seed_study`.
        grades: Output of :func:`grade`.

    Returns:
        Markdown text ending in a newline. Figures are linked by relative file name.
    """
    svi = surfaces["svi"]
    lines = [
        *_intro(config, chain),
        *_distribution_section(svi),
        *_checks_section(svi, grades),
        *_validation_section(validation),
        *_seed_section(config, study),
    ]
    return "\n".join(lines)
