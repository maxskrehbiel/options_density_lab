"""Synthetic chain generation: reproducibility and the injected quote defects."""

from __future__ import annotations

import numpy as np
import pytest

from options_density_lab.synthetic import (
    QUOTE_COLUMNS,
    MarketNoise,
    MarketSetup,
    SyntheticChain,
    generate_chain,
    listed_strikes,
)


def test_same_seed_gives_same_chain(setup: MarketSetup) -> None:
    a = generate_chain(np.random.default_rng(3), setup)
    b = generate_chain(np.random.default_rng(3), setup)
    c = generate_chain(np.random.default_rng(4), setup)
    assert a.quotes.equals(b.quotes)
    assert not a.quotes.equals(c.quotes)


def test_quote_table_has_vendor_columns_only(noisy_chain: SyntheticChain) -> None:
    assert tuple(noisy_chain.quotes.columns) == QUOTE_COLUMNS
    assert len(noisy_chain.quotes) == len(noisy_chain.truth)


def test_defects_are_injected(noisy_chain: SyntheticChain) -> None:
    q, t = noisy_chain.quotes, noisy_chain.truth
    assert t["is_stale"].sum() > 0
    assert t["is_crossed"].sum() > 0
    assert (q.loc[t["is_crossed"], "bid"] > q.loc[t["is_crossed"], "ask"]).all()
    assert (q["bid"] == 0).sum() > 0  # deep wings are unquoted
    healthy = ~t["is_crossed"] & (q["bid"] > 0)
    assert (q.loc[healthy, "ask"] > q.loc[healthy, "bid"]).all()


def test_exact_chain_quotes_equal_true_prices(exact_chain: SyntheticChain) -> None:
    q, t = exact_chain.quotes, exact_chain.truth
    quoted = q["bid"] > 0
    np.testing.assert_allclose(q.loc[quoted, "bid"], q.loc[quoted, "ask"])
    np.testing.assert_allclose(q.loc[quoted, "bid"], t.loc[quoted, "true_price"], atol=1e-6)
    assert not t["is_stale"].any()


def test_forward_and_discount(setup: MarketSetup) -> None:
    assert setup.forward(1.0) == np.float64(100.0 * np.exp(0.025))
    assert setup.discount(0.5) == np.float64(np.exp(-0.02))


def test_listed_strikes_are_round_and_widen_with_maturity() -> None:
    short = listed_strikes(100.0, 30 / 365, 0.2)
    long = listed_strikes(100.0, 1.0, 0.2)
    assert np.allclose(np.diff(short), 1.0)
    assert np.allclose(np.diff(long), 5.0)
    assert long.min() < short.min() and long.max() > short.max()


@pytest.mark.parametrize(
    ("setup", "noise", "message"),
    [
        (MarketSetup(expiry_days=()), MarketNoise(), "expiry_days is empty"),
        (MarketSetup(expiry_days=(0, 30)), MarketNoise(), "at least one day"),
        (MarketSetup(spot=0.0), MarketNoise(), "spot must be positive"),
        (MarketSetup(), MarketNoise(stale_prob=1.5), "stale_prob"),
    ],
)
def test_unusable_settings_are_rejected_up_front(
    setup: MarketSetup, noise: MarketNoise, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_chain(np.random.default_rng(0), setup, noise)
