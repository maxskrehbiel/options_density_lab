"""Static figures: quote cleaning, smiles, densities, CDFs, arbitrage checks and the seed study.

Uses matplotlib's object API (no pyplot global state), so it runs headless.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._types import FloatArray
from .black76 import implied_vol
from .clean import KEEP, is_out_of_the_money
from .pipeline import ExpiryResult, SurfaceResult
from .validate import TruthCache

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EXTRAPOLATED = "#f0efec"
DROPPED = "#d03b3b"
METHOD_COLORS = {"svi": "#2a78d6", "spline": "#eb6834"}
METHOD_LABELS = {"svi": "SVI fit", "spline": "Spline fit"}
EXPIRY_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
DPI = 130

LegendLoc = Literal["best", "upper left", "upper right", "upper center"]


def _style(ax: Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=8)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=8)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8)


def _figure(nrows: int, ncols: int, width: float, height: float) -> tuple[Figure, Any]:
    fig = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axes


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    return path


def _legend(ax: Axes, loc: LegendLoc = "best") -> None:
    ax.legend(loc=loc, fontsize=7, frameon=False, labelcolor=INK_SECONDARY)


def _expiry_title(result: ExpiryResult) -> str:
    return f"{result.days} days to expiry"


def _shade_extrapolated(ax: Axes, result: ExpiryResult) -> None:
    lo, hi = ax.get_xlim()
    d = result.density
    ax.axvspan(lo, d.strike_lo, color=EXTRAPOLATED, zorder=0, linewidth=0)
    ax.axvspan(d.strike_hi, hi, color=EXTRAPOLATED, zorder=0, linewidth=0)
    ax.set_xlim(lo, hi)


def _iv_error_bars(otm: pd.DataFrame) -> FloatArray:
    """Distance from mid to bid and to ask, in volatility points (a missing bid counts as 0)."""
    lower = otm["iv_mid"] - otm["iv_bid"].fillna(0.0)
    upper = otm["iv_ask"] - otm["iv_mid"]
    return np.vstack([lower.to_numpy(dtype=float), upper.to_numpy(dtype=float)]) * 100


def _parity_panel(ax: Axes, result: ExpiryResult) -> None:
    par = result.clean.parity
    resid = par["synth_mid"] - par["fitted"]
    # Bars span each pair's bid-ask band; a consistent pair's bar crosses zero.
    err = np.vstack([par["synth_mid"] - par["band_lo"], par["band_hi"] - par["synth_mid"]])
    ok = ~par["flagged"].to_numpy(dtype=bool)
    ax.axhline(0.0, color=AXIS, linewidth=1)
    for mask, marker, color, label in (
        (ok, "o", MUTED, "consistent pair (bid-ask band)"),
        (~ok, "x", DROPPED, "flagged: band misses the line"),
    ):
        ax.errorbar(
            par["strike"][mask],
            resid[mask],
            yerr=err[:, mask],
            fmt=marker,
            ms=5,
            mew=1.5,
            color=color,
            ecolor=color,
            elinewidth=1,
            label=label,
        )
    _style(ax, "Put-call parity check", "strike", "(call - put) minus fitted D(F - K)")
    _legend(ax)


def _quote_panel(ax: Axes, result: ExpiryResult) -> None:
    otm, used = result.otm, result.used
    bars = _iv_error_bars(otm)
    for mask, marker, color, label in (
        (used, "o", MUTED, "kept (bid-ask range)"),
        (~used, "x", DROPPED, "dropped: far outside own band"),
    ):
        if mask.any():
            ax.errorbar(
                otm["strike"][mask],
                otm["iv_mid"][mask] * 100,
                yerr=bars[:, mask],
                fmt=marker,
                ms=4,
                color=color,
                ecolor=color,
                elinewidth=1,
                label=label,
            )
    lab = result.clean.labelled
    parity = lab[(lab["drop_reason"] == "parity") & is_out_of_the_money(lab, result.clean.forward)]
    if len(parity):
        iv = implied_vol(
            parity["mid"].to_numpy(dtype=float),
            result.clean.forward,
            parity["strike"].to_numpy(dtype=float),
            result.maturity,
            result.clean.discount,
            parity["option_type"].to_numpy(),
        )
        ax.plot(
            parity["strike"],
            iv * 100,
            "x",
            ms=7,
            mew=2,
            color=DROPPED,
            label="dropped: failed parity",
        )
    n_zero = int((lab["drop_reason"] == "zero_bid").sum())
    n_other = int(((lab["drop_reason"] != KEEP) & (lab["drop_reason"] != "parity")).sum()) - n_zero
    title = f"Implied vols ({n_zero} zero-bid, {n_other} other quotes removed first)"
    _style(ax, title, "strike", "implied volatility (%)")
    _legend(ax)


def plot_quote_cleaning(result: ExpiryResult, path: Path) -> Path:
    """Two panels: parity residuals with flagged pairs, and quotes kept or dropped in vol terms.

    Args:
        result: One expiry's pipeline output.
        path: Output PNG path.

    Returns:
        ``path``.
    """
    fig, axes = _figure(1, 2, 10.0, 3.8)
    _parity_panel(axes[0, 0], result)
    _quote_panel(axes[0, 1], result)
    title = f"Quote cleaning, {_expiry_title(result)}"
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    return _save(fig, path)


def _smile_panel(
    ax: Axes,
    surfaces: Mapping[str, SurfaceResult],
    index: int,
    true_iv: Callable[[FloatArray, float], FloatArray] | None,
) -> None:
    base = next(iter(surfaces.values())).expiries[index]
    otm = base.otm
    ax.errorbar(
        otm["strike"],
        otm["iv_mid"] * 100,
        yerr=_iv_error_bars(otm),
        fmt="none",
        ecolor=MUTED,
        elinewidth=1.2,
        label="quotes (bid to ask)",
    )
    fwd = base.clean.forward
    span = base.smile.k_hi - base.smile.k_lo
    k_wide = np.linspace(base.smile.k_lo - 0.15 * span, base.smile.k_hi + 0.15 * span, 300)
    for method, surf in surfaces.items():
        sm = surf.expiries[index].smile
        k = k_wide if method == "svi" else np.linspace(sm.k_lo, sm.k_hi, 300)
        ax.plot(
            fwd * np.exp(k),
            sm.iv(k) * 100,
            color=METHOD_COLORS[method],
            linewidth=2,
            label=METHOD_LABELS[method],
        )
    if true_iv is not None:
        strikes = fwd * np.exp(k_wide)
        ax.plot(
            strikes,
            true_iv(strikes, base.maturity) * 100,
            "--",
            color=INK,
            linewidth=1.2,
            label="true smile",
        )
    _style(ax, _expiry_title(base), "strike", "implied volatility (%)" if index == 0 else "")
    if index == 0:
        _legend(ax, "upper right")


def plot_smiles(
    surfaces: Mapping[str, SurfaceResult],
    path: Path,
    true_iv: Callable[[FloatArray, float], FloatArray] | None = None,
) -> Path:
    """One panel per expiry: quotes, each fitted smile, and the true smile if known.

    Args:
        surfaces: Pipeline outputs keyed by method.
        path: Output PNG path.
        true_iv: Function of (strikes, maturity) giving the true implied vols, if known.

    Returns:
        ``path``.
    """
    n = len(next(iter(surfaces.values())).expiries)
    fig, axes = _figure(1, n, 3.2 * n, 3.4)
    for i in range(n):
        _smile_panel(axes[0, i], surfaces, i, true_iv)
    fig.suptitle("Implied-volatility smiles", color=INK, fontsize=11, x=0.01, ha="left")
    return _save(fig, path)


def _plot_window(result: ExpiryResult, truth: TruthCache | None) -> tuple[float, float]:
    if truth is not None:
        lo, hi = truth.quantiles(result.maturity, (0.0005, 0.9995))
        return float(lo), float(hi)
    lo, hi = result.density.quantile(np.array([0.0005, 0.9995]))
    return float(lo), float(hi)


def plot_densities(
    surfaces: Mapping[str, SurfaceResult], path: Path, truth: TruthCache | None = None
) -> Path:
    """One panel per expiry: recovered densities against the truth, extrapolated zones shaded.

    Args:
        surfaces: Pipeline outputs keyed by method.
        path: Output PNG path.
        truth: Cached truth, if known.

    Returns:
        ``path``.
    """
    first = next(iter(surfaces.values()))
    n = len(first.expiries)
    fig, axes = _figure(1, n, 3.2 * n, 3.4)
    for i, base in enumerate(first.expiries):
        ax = axes[0, i]
        lo, hi = _plot_window(base, truth)
        for method, surf in surfaces.items():
            d = surf.expiries[i].density
            ax.plot(
                d.strikes,
                d.pdf,
                color=METHOD_COLORS[method],
                linewidth=2,
                label=METHOD_LABELS[method],
            )
        if truth is not None:
            grid: FloatArray = np.linspace(lo, hi, 800)
            ax.plot(
                grid,
                truth.on(grid, base.maturity)[0],
                "--",
                color=INK,
                linewidth=1.2,
                label="true density",
            )
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        _shade_extrapolated(ax, base)
        _style(ax, _expiry_title(base), "price at expiry", "probability density" if i == 0 else "")
        if i == 0:
            ax.fill_between(
                [], [], facecolor=EXTRAPOLATED, edgecolor=AXIS, label="beyond quoted strikes"
            )
            _legend(ax, "upper left")
    fig.suptitle(
        "Risk-neutral density of the price at expiry", color=INK, fontsize=11, x=0.01, ha="left"
    )
    return _save(fig, path)


def _cdf_panels(
    axes: Any,
    surfaces: Mapping[str, SurfaceResult],
    index: int,
    truth: TruthCache | None,
) -> None:
    base = next(iter(surfaces.values())).expiries[index]
    lo, hi = _plot_window(base, truth)
    ax = axes[0, index]
    for method, surf in surfaces.items():
        d = surf.expiries[index].density
        ax.plot(
            d.strikes, d.cdf, color=METHOD_COLORS[method], linewidth=2, label=METHOD_LABELS[method]
        )
    if truth is not None:
        grid: FloatArray = np.linspace(lo, hi, 800)
        ax.plot(
            grid, truth.on(grid, base.maturity)[1], "--", color=INK, linewidth=1.2, label="true CDF"
        )
    ax.set_xlim(lo, hi)
    _shade_extrapolated(ax, base)
    _style(ax, _expiry_title(base), "", "P(price at expiry <= x)" if index == 0 else "")
    if index == 0:
        _legend(ax, "upper left")
    if truth is None:
        return
    ax = axes[1, index]
    ax.axhline(0.0, color=AXIS, linewidth=1)
    for method, surf in surfaces.items():
        d = surf.expiries[index].density
        err = d.cdf - truth.on(d.strikes, base.maturity)[1]
        ax.plot(
            d.strikes,
            err * 100,
            color=METHOD_COLORS[method],
            linewidth=2,
            label=METHOD_LABELS[method],
        )
    ax.set_xlim(lo, hi)
    _shade_extrapolated(ax, base)
    _style(ax, "", "price at expiry", "recovered - true (% points)" if index == 0 else "")


def plot_cdfs(
    surfaces: Mapping[str, SurfaceResult], path: Path, truth: TruthCache | None = None
) -> Path:
    """Top row: recovered and true CDFs. Bottom row (if truth known): recovered minus true.

    Args:
        surfaces: Pipeline outputs keyed by method.
        path: Output PNG path.
        truth: Cached truth, if known.

    Returns:
        ``path``.
    """
    n = len(next(iter(surfaces.values())).expiries)
    rows = 2 if truth is not None else 1
    fig, axes = _figure(rows, n, 3.2 * n, 3.0 * rows)
    for i in range(n):
        _cdf_panels(axes, surfaces, i, truth)
    fig.suptitle("Cumulative distribution and its error", color=INK, fontsize=11, x=0.01, ha="left")
    return _save(fig, path)


def plot_arbitrage(surface: SurfaceResult, path: Path) -> Path:
    """Left: total variance per expiry (curves must not cross). Right: ``g(k)`` (must stay >= 0).

    Args:
        surface: Pipeline output.
        path: Output PNG path.

    Returns:
        ``path``.
    """
    fig, axes = _figure(1, 2, 10.0, 3.6)
    k_lo = min(e.smile.k_lo for e in surface.expiries)
    k_hi = max(e.smile.k_hi for e in surface.expiries)
    k = np.linspace(k_lo, k_hi, 400)
    colors = (
        EXPIRY_RAMP[-len(surface.expiries) :] if len(surface.expiries) <= len(EXPIRY_RAMP) else None
    )
    for j, e in enumerate(surface.expiries):
        color = colors[j] if colors else None
        axes[0, 0].plot(
            k, e.smile.total_variance(k), color=color, linewidth=2, label=f"{e.days} days"
        )
        kq = np.linspace(e.smile.k_lo, e.smile.k_hi, 300)
        axes[0, 1].plot(kq, e.smile.g(kq), color=color, linewidth=2, label=f"{e.days} days")
    axes[0, 1].axhline(0.0, color=DROPPED, linewidth=1, linestyle=":")
    cal = "passes" if surface.calendar.passed else "FAILS"
    _style(
        axes[0, 0],
        f"Calendar: total variance rises with expiry ({cal})",
        "log-moneyness ln(K/F)",
        "total implied variance w = iv^2 T",
    )
    fly_ok = all(
        c.passed for e in surface.expiries for c in e.check_results if c.name == "smile butterfly"
    )
    fly = "passes" if fly_ok else "FAILS"
    _style(axes[0, 1], f"Butterfly: g(k) stays above zero ({fly})", "log-moneyness ln(K/F)", "g(k)")
    _legend(axes[0, 0], "upper center")
    _legend(axes[0, 1])
    fig.suptitle(
        f"No-arbitrage checks on the fitted {surface.method.upper()} surface",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    return _save(fig, path)


def plot_seed_study(study: pd.DataFrame, path: Path) -> Path:
    """Total-variation error for each method and expiry across many random chains.

    Args:
        study: Rows with ``seed``, ``days``, ``method`` and ``tv``.
        path: Output PNG path.

    Returns:
        ``path``.
    """
    fig, axes = _figure(1, 1, 7.0, 3.6)
    ax = axes[0, 0]
    days = sorted(study["days"].unique())
    methods = [m for m in METHOD_COLORS if m in set(study["method"])]
    width = 0.32
    jitter = np.random.default_rng(0)  # fixed seed: only spreads overlapping dots
    for j, method in enumerate(methods):
        offset = (j - (len(methods) - 1) / 2) * width
        for i, day in enumerate(days):
            vals = (
                study[(study["days"] == day) & (study["method"] == method)]["tv"].to_numpy(
                    dtype=float
                )
                * 100
            )
            x = i + offset + jitter.uniform(-0.07, 0.07, vals.size)
            ax.plot(
                x,
                vals,
                "o",
                ms=4,
                color=METHOD_COLORS[method],
                alpha=0.55,
                label=METHOD_LABELS[method] if i == 0 else None,
            )
            ax.plot(
                [i + offset - 0.12, i + offset + 0.12],
                [np.median(vals)] * 2,
                color=INK,
                linewidth=2,
                label="median" if (i == 0 and j == len(methods) - 1) else None,
            )
    ax.set_xticks(range(len(days)), [f"{d} days" for d in days])
    ax.set_ylim(bottom=0)
    n_seeds = study["seed"].nunique()
    _style(
        ax,
        f"Recovery error over {n_seeds} independently generated noisy chains",
        "",
        "total-variation distance to truth (%)",
    )
    _legend(ax, "upper left")
    return _save(fig, path)
