"""Quote cleaning: filters, parity-based forward and discount, stale-quote removal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from options_density_lab.clean import (
    CleanConfig,
    basic_filters,
    clean_expiry,
    estimate_forward_discount,
)
from options_density_lab.errors import CleaningError
from options_density_lab.synthetic import MarketSetup, SyntheticChain


def _one_expiry(chain: SyntheticChain, index: int) -> pd.DataFrame:
    t = chain.maturities[index]
    return chain.quotes[chain.quotes["maturity"] == t].reset_index(drop=True)


def test_basic_filters_label_each_defect() -> None:
    quotes = pd.DataFrame(
        {
            "strike": [90.0, 95.0, 100.0, 105.0, 110.0],
            "option_type": ["P", "P", "C", "C", "C"],
            "bid": [1.00, 1.50, 0.00, 1.00, 0.10],
            "ask": [1.10, 1.40, 0.05, np.nan, 0.50],
        }
    )
    reasons = basic_filters(quotes, CleanConfig())["drop_reason"].tolist()
    assert reasons == ["", "crossed", "zero_bid", "bad", "wide"]


def test_forward_and_discount_recovered_from_noisy_quotes(
    noisy_chain: SyntheticChain, setup: MarketSetup
) -> None:
    for i, t in enumerate(noisy_chain.maturities):
        _, report = clean_expiry(_one_expiry(noisy_chain, i))
        assert report.forward / setup.forward(t) - 1 == pytest.approx(0.0, abs=5e-4)
        assert report.discount == pytest.approx(setup.discount(t), abs=3e-3)


def test_exact_quotes_give_exact_forward_and_no_flags(
    exact_chain: SyntheticChain, setup: MarketSetup
) -> None:
    for i, t in enumerate(exact_chain.maturities):
        _, report = clean_expiry(_one_expiry(exact_chain, i))
        assert report.forward == pytest.approx(setup.forward(t), rel=1e-6)
        assert report.implied_rate == pytest.approx(setup.rate, abs=1e-4)
        assert report.n_parity_flagged == 0


def test_stale_quote_fails_parity_and_is_dropped(exact_chain: SyntheticChain) -> None:
    quotes = _one_expiry(exact_chain, 0)
    stale = (quotes["strike"] == 97.0) & (quotes["option_type"] == "P")
    quotes.loc[stale, ["bid", "ask"]] += 0.40
    otm, report = clean_expiry(quotes)
    assert report.parity.loc[report.parity["flagged"], "strike"].tolist() == [97.0]
    assert 97.0 not in otm["strike"].to_numpy()
    assert report.drop_counts["parity"] == 1


def test_crossed_and_zero_bid_quotes_never_reach_the_fit(noisy_chain: SyntheticChain) -> None:
    quotes = _one_expiry(noisy_chain, 0)
    otm, report = clean_expiry(quotes)
    labelled = report.labelled
    assert set(labelled.loc[labelled["bid"] > labelled["ask"], "drop_reason"]) <= {"crossed"}
    assert (otm["bid"] > 0).all()
    assert ((otm["option_type"] == "P") == (otm["strike"] < report.forward)).all()
    assert (otm["iv_err"] >= CleanConfig().min_iv_err).all()
    assert np.all(np.diff(otm["k"]) > 0)


def test_too_few_pairs_raises() -> None:
    quotes = pd.DataFrame(
        {"strike": [100.0, 100.0], "option_type": ["C", "P"], "bid": [2.0, 1.9], "ask": [2.1, 2.0]}
    )
    labelled = basic_filters(quotes, CleanConfig())
    with pytest.raises(CleaningError, match="at least 3 strikes"):
        estimate_forward_discount(labelled, CleanConfig())


def test_implausible_discount_factor_raises() -> None:
    # Call minus put rising with strike implies a negative discount factor.
    strikes = [90.0, 100.0, 110.0]
    quotes = pd.DataFrame(
        {
            "strike": strikes * 2,
            "option_type": ["C"] * 3 + ["P"] * 3,
            "bid": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "ask": [1.1, 2.1, 3.1, 3.1, 2.1, 1.1],
        }
    )
    labelled = basic_filters(quotes, CleanConfig())
    with pytest.raises(CleaningError, match="discount factor"):
        estimate_forward_discount(labelled, CleanConfig())
