"""No-arbitrage checks catch injected violations and pass clean inputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from options_density_lab.black76 import black76_price
from options_density_lab.checks import (
    check_calendar,
    check_call_prices,
    check_density,
    check_quote_butterflies,
    check_smile_butterfly,
)
from options_density_lab.pipeline import SurfaceResult
from options_density_lab.smile import SVISmile

STRIKES = np.arange(80.0, 121.0, 1.0)


def inject_butterfly(strikes: np.ndarray, at: float, bump: float) -> np.ndarray:
    """Black-76 call prices (flat 20% vol) with the price at ``at`` pushed up by ``bump``."""
    calls = black76_price(100.0, strikes, 0.25, 0.2).copy()
    calls[int(np.argmin(np.abs(strikes - at)))] += bump
    return calls


def test_injected_butterfly_violation_is_caught() -> None:
    calls = inject_butterfly(STRIKES, at=100.0, bump=0.25)
    result = check_call_prices(STRIKES, calls, discount=1.0)
    assert not result.passed
    assert result.details is not None
    fly = result.details[result.details["type"] == "butterfly"]
    assert fly["strike"].tolist() == [100.0]
    assert "butterfly" in result.message


def test_clean_black_prices_pass() -> None:
    calls = black76_price(100.0, STRIKES, 0.25, 0.2, 0.99)
    result = check_call_prices(STRIKES, calls, discount=0.99)
    assert result.passed and result.n_violations == 0


def test_rising_call_prices_violate_monotonicity() -> None:
    calls = black76_price(100.0, STRIKES, 0.25, 0.2)
    calls[20] = calls[19] + 0.5
    result = check_call_prices(STRIKES, calls, discount=1.0)
    assert result.details is not None
    assert "monotonicity" in set(result.details["type"])


def test_calendar_violation_is_caught() -> None:
    short = SVISmile(0.01, 0.1, -0.5, 0.0, 0.1, maturity=0.25, forward=100.0, k_lo=-0.3, k_hi=0.3)
    longer_ok = replace(short, a=0.03, maturity=0.5)
    longer_bad = replace(short, a=0.005, maturity=0.5)
    assert check_calendar([short, longer_ok]).passed
    bad = check_calendar([longer_bad, short])  # order does not matter
    assert not bad.passed and bad.worst > 0


def test_known_arbitrageable_svi_is_caught() -> None:
    # Example 3.1 of Gatheral and Jacquier (2014), credited there to Axel Vogt.
    smile = SVISmile(
        -0.0410, 0.1331, 0.3060, 0.3586, 0.4153, maturity=1.0, forward=100.0, k_lo=-1.5, k_hi=1.5
    )
    result = check_smile_butterfly(smile)
    assert not result.passed and result.worst > 0.01


def test_tradable_quote_butterfly_is_caught() -> None:
    strikes = np.array([95.0, 100.0, 105.0])
    calls = black76_price(100.0, strikes, 0.25, 0.2)
    quotes = pd.DataFrame(
        {"strike": strikes, "option_type": "C", "bid": calls - 0.02, "ask": calls + 0.02}
    )
    assert check_quote_butterflies(quotes, 100.0, 1.0).passed
    quotes.loc[1, ["bid", "ask"]] += 1.0  # body now sells for more than the wings cost
    result = check_quote_butterflies(quotes, 100.0, 1.0)
    assert not result.passed and result.n_violations == 1


def test_density_checks_flag_a_broken_density(svi_surface: SurfaceResult) -> None:
    good = svi_surface.expiries[0].density
    assert all(c.passed for c in check_density(good))
    pdf = good.pdf.copy()
    pdf[len(pdf) // 2] = -1.0
    cdf = good.cdf.copy()
    cdf[len(cdf) // 2] = 0.0
    broken = replace(good, pdf=pdf, cdf=cdf, forward=good.forward * 1.1)
    failed = {c.name for c in check_density(broken) if not c.passed}
    assert failed == {
        "integrates to one",
        "non-negative density",
        "CDF monotone in [0, 1]",
        "mean equals forward",
    }
