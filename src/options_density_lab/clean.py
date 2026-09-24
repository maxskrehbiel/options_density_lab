"""Quote cleaning: basic filters, forward and discount from put-call parity, OTM implied vols.

Parity ``C - P = D (F - K)`` is a straight line in strike, so a weighted line
fit recovers the forward ``F`` and discount factor ``D`` from the options alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._types import BoolArray, FloatArray
from .black76 import implied_vol
from .errors import CleaningError

logger = logging.getLogger(__name__)

KEEP = ""
MIN_BAND_WIDTH = 1e-4  # floor on a pair's band width, so zero-width quotes keep a finite weight


@dataclass(frozen=True)
class CleanConfig:
    """Settings for the cleaning step.

    Attributes:
        max_rel_spread: Drop quotes whose spread exceeds this multiple of the mid.
        min_iv_err: Floor on a quote's volatility uncertainty (0.001 = 0.1 vol point),
            so no single quote gets unlimited weight.
        max_parity_iter: Most pairs the parity check may flag (one per round).
        min_parity_pairs: Fewest call/put pairs the parity fit will accept.
        parity_tolerance: How far (in price units) the fitted line may miss a
            pair's band before the pair is flagged; absorbs rounding when
            quotes have zero width.
        max_discount: Largest believable discount factor. A fit above it (or
            at or below zero) means the quotes cannot support parity.
    """

    max_rel_spread: float = 1.0
    min_iv_err: float = 0.001
    max_parity_iter: int = 20
    min_parity_pairs: int = 3
    parity_tolerance: float = 1e-4
    max_discount: float = 1.5


@dataclass(frozen=True)
class CleanReport:
    """What the cleaning step decided for one expiry.

    Attributes:
        maturity: Time to expiry in years.
        n_raw: Number of quotes received.
        drop_counts: Count of quotes per outcome (``kept`` or a drop reason).
        forward: Forward price estimated from parity.
        discount: Discount factor estimated from parity.
        parity: One row per call/put pair with its band, fitted value and flag.
        labelled: Every quote with its ``mid`` and ``drop_reason``.
    """

    maturity: float
    n_raw: int
    drop_counts: dict[str, int]
    forward: float
    discount: float
    parity: pd.DataFrame
    labelled: pd.DataFrame

    @property
    def implied_rate(self) -> float:
        """Interest rate implied by the discount factor, from ``D = exp(-r T)``."""
        return float(-np.log(self.discount) / self.maturity)

    @property
    def n_parity_flagged(self) -> int:
        """Number of call/put pairs that failed the parity check."""
        return int(self.parity["flagged"].sum())


def is_out_of_the_money(quotes: pd.DataFrame, forward: float) -> pd.Series:
    """Puts struck below the forward and calls struck at or above it.

    Args:
        quotes: Quote table with ``strike`` and ``option_type`` columns.
        forward: Forward price for the expiry.

    Returns:
        Boolean mask aligned with ``quotes``.
    """
    puts = (quotes["option_type"] == "P") & (quotes["strike"] < forward)
    calls = (quotes["option_type"] == "C") & (quotes["strike"] >= forward)
    return puts | calls


def basic_filters(quotes: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    """Label each quote with a ``drop_reason``; an empty string means keep.

    Reasons, checked in order: ``bad`` (missing or non-positive ask), ``crossed``
    (bid above ask), ``zero_bid`` (nobody is willing to buy, so the value is
    unknown) and ``wide`` (spread larger than ``max_rel_spread`` times the mid).

    Args:
        quotes: Raw quotes for one expiry.
        config: Cleaning settings.

    Returns:
        A copy of ``quotes`` with ``mid`` and ``drop_reason`` columns added.
    """
    out = quotes.copy()
    bid, ask = out["bid"].to_numpy(dtype=float), out["ask"].to_numpy(dtype=float)
    mid = 0.5 * (bid + ask)
    reason = np.full(len(out), KEEP, dtype=object)
    rules = (
        ("bad", ~np.isfinite(bid) | ~np.isfinite(ask) | (ask <= 0)),
        ("crossed", bid > ask),
        ("zero_bid", bid <= 0),
        ("wide", (ask - bid) > config.max_rel_spread * mid),
    )
    for name, hit in rules:
        reason[(reason == KEEP) & hit] = name
    out["mid"] = mid
    out["drop_reason"] = reason
    return out


def _parity_pairs(quotes: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    """One row per strike quoted on both sides: ``call - put`` at mid and its bid-ask band."""
    calls = quotes[quotes["option_type"] == "C"].set_index("strike")
    puts = quotes[quotes["option_type"] == "P"].set_index("strike")
    common = calls.index.intersection(puts.index).sort_values()
    if len(common) < config.min_parity_pairs:
        raise CleaningError(
            f"need at least {config.min_parity_pairs} strikes quoted on both sides, "
            f"got {len(common)}"
        )
    c, p = calls.loc[common], puts.loc[common]
    return pd.DataFrame(
        {
            "strike": common.to_numpy(dtype=float),
            "synth_mid": (c["mid"] - p["mid"]).to_numpy(dtype=float),
            "band_lo": (c["bid"] - p["ask"]).to_numpy(dtype=float),
            "band_hi": (c["ask"] - p["bid"]).to_numpy(dtype=float),
        }
    )


def _weighted_line(
    x: FloatArray, y: FloatArray, width: FloatArray, use: BoolArray
) -> tuple[float, float]:
    """Intercept and slope of a line fitted with weights ``1 / width^2``."""
    weights = 1.0 / width[use] ** 2
    design = np.column_stack([np.ones(int(use.sum())), x[use]])
    lhs = design.T @ (design * weights[:, None])
    solution = np.linalg.solve(lhs, design.T @ (weights * y[use]))
    return float(solution[0]), float(solution[1])


def _fit_parity_line(pairs: pd.DataFrame, config: CleanConfig) -> tuple[float, float, BoolArray]:
    """Fit the parity line, removing the worst inconsistent pair each round."""
    strikes = pairs["strike"].to_numpy(dtype=float)
    synth_mid = pairs["synth_mid"].to_numpy(dtype=float)
    band_lo = pairs["band_lo"].to_numpy(dtype=float)
    band_hi = pairs["band_hi"].to_numpy(dtype=float)
    width = np.maximum(band_hi - band_lo, MIN_BAND_WIDTH)
    tol = config.parity_tolerance
    flagged = np.zeros(strikes.size, dtype=bool)
    intercept, slope = _weighted_line(strikes, synth_mid, width, ~flagged)
    # One bad pair can drag the line outside many good bands at once, so remove
    # only the single worst violator per round (miss measured in band widths).
    for _ in range(config.max_parity_iter):
        fitted = intercept + slope * strikes
        miss = np.maximum(band_lo - tol - fitted, fitted - band_hi - tol) / width
        miss[flagged] = 0.0
        worst = int(np.argmax(miss))
        if miss[worst] <= 0:
            break
        if (~flagged).sum() <= config.min_parity_pairs:
            logger.warning("parity check stopped early: too few consistent pairs remain")
            break
        flagged[worst] = True
        intercept, slope = _weighted_line(strikes, synth_mid, width, ~flagged)
    return intercept, slope, flagged


def estimate_forward_discount(
    quotes: pd.DataFrame, config: CleanConfig
) -> tuple[float, float, pd.DataFrame]:
    """Fit ``call - put = D F - D K`` across strikes by weighted least squares.

    Each pair is weighted by one over the squared width of its bid-ask band for
    ``call - put``, so tight near-the-money pairs dominate. A pair is flagged
    when the fitted line falls outside its band
    ``[call_bid - put_ask, call_ask - put_bid]``: no price inside the quotes
    could satisfy parity, which is the signature of a stale quote. Flagged
    pairs are excluded one at a time, worst first, refitting after each.

    Args:
        quotes: Usable quotes for one expiry (with a ``mid`` column).
        config: Cleaning settings.

    Returns:
        ``(forward, discount, pairs)`` where ``pairs`` holds one row per strike.

    Raises:
        CleaningError: If fewer than ``min_parity_pairs`` strikes have both
            quotes, or the fitted discount factor is not in ``(0, max_discount]``.
    """
    pairs = _parity_pairs(quotes, config)
    intercept, slope, flagged = _fit_parity_line(pairs, config)
    discount = -slope
    if not 0.0 < discount <= config.max_discount:  # also rejects NaN
        raise CleaningError(
            f"parity fit gives a discount factor of {discount:.4g}, outside "
            f"(0, {config.max_discount:g}]; the call and put quotes are inconsistent"
        )
    pairs["fitted"] = intercept + slope * pairs["strike"]
    pairs["flagged"] = flagged
    return intercept / discount, discount, pairs


def otm_implied_vols(
    quotes: pd.DataFrame, forward: float, discount: float, maturity: float, config: CleanConfig
) -> pd.DataFrame:
    """Out-of-the-money quotes converted to implied volatility and total variance.

    Out-of-the-money options are the liquid, tightly quoted ones; in-the-money
    options are mostly intrinsic value with wide spreads.

    Args:
        quotes: Usable quotes for one expiry.
        forward: Forward price.
        discount: Discount factor.
        maturity: Time to expiry in years.
        config: Cleaning settings.

    Returns:
        Rows sorted by log-moneyness ``k = ln(K/F)`` with columns ``iv_bid``,
        ``iv_mid``, ``iv_ask``, ``iv_err`` (half the bid-ask width in vol terms),
        ``k``, ``w`` (total variance ``iv^2 T``) and ``w_err``.
    """
    otm = quotes[is_out_of_the_money(quotes, forward)].copy()
    strikes = otm["strike"].to_numpy(dtype=float)
    kinds = otm["option_type"].to_numpy()
    for side in ("bid", "mid", "ask"):
        otm[f"iv_{side}"] = implied_vol(
            otm[side].to_numpy(dtype=float), forward, strikes, maturity, discount, kinds
        )
    otm = otm[np.isfinite(otm["iv_mid"])].copy()
    lo = otm["iv_bid"].fillna(0.0)
    hi = otm["iv_ask"].fillna(2.0 * otm["iv_mid"])
    otm["iv_err"] = np.maximum(0.5 * (hi - lo), config.min_iv_err)
    otm["k"] = np.log(otm["strike"] / forward)
    otm["w"] = otm["iv_mid"] ** 2 * maturity
    # First-order error propagation: dw = 2 sigma T d(sigma).
    otm["w_err"] = 2.0 * otm["iv_mid"] * maturity * otm["iv_err"]
    ordered: pd.DataFrame = otm.sort_values("k").reset_index(drop=True)
    return ordered


def clean_expiry(
    quotes: pd.DataFrame, config: CleanConfig | None = None
) -> tuple[pd.DataFrame, CleanReport]:
    """Run every cleaning step for one expiry.

    Args:
        quotes: Raw quotes for a single expiry.
        config: Cleaning settings; defaults to ``CleanConfig()``.

    Returns:
        ``(otm, report)``: the cleaned out-of-the-money quotes with implied
        vols, and an audit record of every decision.
    """
    config = config or CleanConfig()
    maturity = float(quotes["maturity"].iloc[0])
    labelled = basic_filters(quotes, config)
    forward, discount, pairs = estimate_forward_discount(
        labelled[labelled["drop_reason"] == KEEP], config
    )

    # Which leg of a failed pair is stale is unknowable from the pair alone, so
    # drop the out-of-the-money leg: the one the fit would otherwise use.
    bad_strikes = pairs.loc[pairs["flagged"], "strike"]
    parity_drop = (
        (labelled["drop_reason"] == KEEP)
        & is_out_of_the_money(labelled, forward)
        & labelled["strike"].isin(bad_strikes)
    )
    labelled.loc[parity_drop, "drop_reason"] = "parity"
    otm = otm_implied_vols(
        labelled[labelled["drop_reason"] == KEEP], forward, discount, maturity, config
    )
    counts = {
        str(k): int(v)
        for k, v in labelled["drop_reason"].replace(KEEP, "kept").value_counts().items()
    }
    logger.info(
        "T=%.3f: forward %.4f, discount %.5f, %d/%d parity pairs flagged, %d OTM quotes kept",
        maturity,
        forward,
        discount,
        int(pairs["flagged"].sum()),
        len(pairs),
        len(otm),
    )
    report = CleanReport(
        maturity=maturity,
        n_raw=len(quotes),
        drop_counts=counts,
        forward=forward,
        discount=discount,
        parity=pairs,
        labelled=labelled,
    )
    return otm, report
