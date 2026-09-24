"""Synthetic option chains with realistic quote defects and a known true distribution.

Prices come from the Heston model; spreads, noise, tick rounding, zero bids,
stale quotes and crossed quotes are then layered on. No market data is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import heston
from ._types import FloatArray
from .black76 import implied_vol
from .heston import HestonParams

QUOTE_COLUMNS = ("days", "maturity", "strike", "option_type", "bid", "ask", "underlying")
DAYS_PER_YEAR = 365.0
TICK_SWITCH_PRICE = 3.0
# Listed strikes reach this many at-the-money standard deviations below and above spot.
STRIKE_REACH_DOWN = 7.0
STRIKE_REACH_UP = 5.0


@dataclass(frozen=True)
class MarketNoise:
    """How messy the generated quotes are.

    Attributes:
        spread_abs: Minimum bid-ask spread in dollars.
        spread_rel: Extra spread as a fraction of the option price.
        mid_noise: Standard deviation of the quote centre's error, in spreads.
        stale_prob: Share of quotes priced off an out-of-date underlying.
        stale_move: Range of the underlying's move since a stale quote was set.
        crossed_prob: Share of quotes with bid and ask swapped (bad prints).
        tick_small: Price increment for options under $3.
        tick_large: Price increment for options at or above $3.
        min_bid: Options whose bid would fall below this get a zero bid.
        spread_jitter: Each spread is scaled by a uniform draw from this range.
    """

    spread_abs: float = 0.02
    spread_rel: float = 0.03
    mid_noise: float = 0.25
    stale_prob: float = 0.03
    stale_move: tuple[float, float] = (0.004, 0.012)
    crossed_prob: float = 0.01
    tick_small: float = 0.01
    tick_large: float = 0.05
    min_bid: float = 0.01
    spread_jitter: tuple[float, float] = (0.75, 1.25)

    @classmethod
    def none(cls) -> MarketNoise:
        """A perfect market: bid equals ask equals the true price.

        Options worth less than ``min_bid`` remain unquoted, so the usable
        strike range matches the noisy market's.

        Returns:
            A noise-free configuration.
        """
        return cls(
            spread_abs=0.0,
            spread_rel=0.0,
            mid_noise=0.0,
            stale_prob=0.0,
            crossed_prob=0.0,
            tick_small=0.0,
            tick_large=0.0,
        )


@dataclass(frozen=True)
class MarketSetup:
    """The fictional market a chain is generated from.

    Attributes:
        spot: Current price of the underlying.
        rate: Continuously compounded interest rate.
        div_yield: Continuous dividend yield.
        expiry_days: Calendar days to each expiry.
        heston: Parameters of the price-generating model.
    """

    spot: float = 100.0
    rate: float = 0.04
    div_yield: float = 0.015
    expiry_days: tuple[int, ...] = (30, 91, 182, 365)
    heston: HestonParams = field(default_factory=HestonParams)

    def forward(self, maturity: float) -> float:
        """Forward price ``S0 exp((r - q) T)``.

        Args:
            maturity: Time to expiry in years.

        Returns:
            The forward price.
        """
        return float(self.spot * np.exp((self.rate - self.div_yield) * maturity))

    def discount(self, maturity: float) -> float:
        """Discount factor ``exp(-r T)``.

        Args:
            maturity: Time to expiry in years.

        Returns:
            The discount factor.
        """
        return float(np.exp(-self.rate * maturity))

    def true_pdf(self, strikes: npt.ArrayLike, maturity: float) -> FloatArray:
        """True density of the price at expiry.

        Args:
            strikes: Price levels.
            maturity: Time to expiry in years.

        Returns:
            Density values.
        """
        return heston.price_pdf(strikes, self.forward(maturity), maturity, self.heston)

    def true_cdf(self, strikes: npt.ArrayLike, maturity: float) -> FloatArray:
        """True probability that the price ends at or below each level.

        Args:
            strikes: Price levels.
            maturity: Time to expiry in years.

        Returns:
            CDF values.
        """
        return heston.price_cdf(strikes, self.forward(maturity), maturity, self.heston)

    def true_iv(self, strikes: npt.ArrayLike, maturity: float) -> FloatArray:
        """True Black-76 implied volatility at each strike.

        Args:
            strikes: Strike prices.
            maturity: Time to expiry in years.

        Returns:
            Implied volatilities.
        """
        fwd, disc = self.forward(maturity), self.discount(maturity)
        k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
        calls = heston.call_prices(fwd, k, maturity, self.heston, disc)
        return implied_vol(calls, fwd, k, maturity, disc, "C")


def listed_strikes(spot: float, maturity: float, atm_vol: float) -> FloatArray:
    """Round-number strikes, more widely spaced for longer expiries.

    The list runs far into the wings on purpose; the zero-bid rule then decides
    which strikes carry usable quotes, as it does in practice.

    Args:
        spot: Current underlying price.
        maturity: Time to expiry in years.
        atm_vol: Rough at-the-money volatility used to size the range.

    Returns:
        Sorted strike prices.
    """
    step = 1.0 if maturity <= 0.1 else 2.5 if maturity <= 0.6 else 5.0
    scale = atm_vol * np.sqrt(maturity)
    lo = spot * np.exp(-STRIKE_REACH_DOWN * scale)
    hi = spot * np.exp(STRIKE_REACH_UP * scale)
    return np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)


def _round_to_tick(x: FloatArray, tick: FloatArray, up: bool) -> FloatArray:
    safe = np.where(tick > 0, tick, 1.0)
    steps = np.round(x / safe, 9)
    rounded = (np.ceil(steps) if up else np.floor(steps)) * safe
    return np.where(tick > 0, rounded, x)


@dataclass(frozen=True)
class SyntheticChain:
    """A generated chain: vendor-style quotes plus the hidden truth.

    Attributes:
        quotes: One row per quote with ``QUOTE_COLUMNS``; this is all the
            recovery pipeline sees.
        truth: Row-aligned with ``quotes``: true price and implied vol, the
            price the quote was built from, and stale/crossed flags.
        setup: The market that generated the chain.
        noise: The noise configuration used.
    """

    quotes: pd.DataFrame
    truth: pd.DataFrame
    setup: MarketSetup
    noise: MarketNoise

    @property
    def maturities(self) -> list[float]:
        """Distinct times to expiry, in years, ascending."""
        return sorted(float(t) for t in self.quotes["maturity"].unique())


def _one_side(
    rng: np.random.Generator,
    noise: MarketNoise,
    strikes: FloatArray,
    true_px: FloatArray,
    stale_px: FloatArray,
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_], npt.NDArray[np.bool_], FloatArray]:
    """Quote one option type across all strikes: bid, ask, stale and crossed flags."""
    n = strikes.size
    stale = rng.random(n) < noise.stale_prob
    quoted = np.where(stale, stale_px, true_px)
    spread = (noise.spread_abs + noise.spread_rel * quoted) * rng.uniform(*noise.spread_jitter, n)
    centre = quoted + rng.normal(0.0, 1.0, n) * noise.mid_noise * spread
    tick = np.where(quoted < TICK_SWITCH_PRICE, noise.tick_small, noise.tick_large)
    bid = _round_to_tick(centre - spread / 2, tick, up=False)
    ask = _round_to_tick(centre + spread / 2, tick, up=True)
    bid = np.where(bid < noise.min_bid, 0.0, bid)
    ask = np.maximum(ask, np.maximum(bid, noise.min_bid))
    crossed = (rng.random(n) < noise.crossed_prob) & (bid > 0) & (ask > bid)
    bid, ask = np.where(crossed, ask, bid), np.where(crossed, bid, ask)
    return bid, ask, stale, crossed, quoted


def validate_market(setup: MarketSetup, noise: MarketNoise) -> None:
    """Reject settings that cannot produce a usable chain, before any work is done.

    Args:
        setup: Market to simulate.
        noise: Quote defects to apply.

    Raises:
        ValueError: If there are no expiries, a non-positive expiry or spot, or
            a probability outside ``[0, 1]``.
    """
    if not setup.expiry_days:
        raise ValueError("MarketSetup.expiry_days is empty")
    if min(setup.expiry_days) <= 0:
        raise ValueError("every expiry must be at least one day away")
    if setup.spot <= 0:
        raise ValueError("spot must be positive")
    for name in ("stale_prob", "crossed_prob"):
        if not 0.0 <= getattr(noise, name) <= 1.0:
            raise ValueError(f"MarketNoise.{name} must be between 0 and 1")


def _expiry_quotes(
    rng: np.random.Generator, setup: MarketSetup, noise: MarketNoise, days: int
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Quotes and hidden truth for one expiry, calls then puts."""
    params = setup.heston
    maturity = days / DAYS_PER_YEAR
    fwd, disc = setup.forward(maturity), setup.discount(maturity)
    strikes = listed_strikes(setup.spot, maturity, float(np.sqrt(params.theta)))
    calls = heston.call_prices(fwd, strikes, maturity, params, disc)
    true_iv = implied_vol(calls, fwd, strikes, maturity, disc, "C")

    # A stale quote was set when the underlying was elsewhere; Heston prices
    # scale with the forward, so reprice at a shifted forward.
    move = rng.uniform(*noise.stale_move, strikes.size) * rng.choice([-1.0, 1.0], strikes.size)
    stale_fwd = fwd * (1.0 + move)
    stale_calls = heston.call_prices(1.0, strikes / stale_fwd, maturity, params, disc) * stale_fwd
    sides = {
        "C": (calls, stale_calls),
        "P": (calls - disc * (fwd - strikes), stale_calls - disc * (stale_fwd - strikes)),
    }
    quote_parts, truth_parts = [], []
    for option_type, (true_raw, stale_raw) in sides.items():
        true_px, stale_px = np.maximum(true_raw, 0.0), np.maximum(stale_raw, 0.0)
        bid, ask, stale, crossed, quoted = _one_side(rng, noise, strikes, true_px, stale_px)
        quote_parts.append(
            pd.DataFrame(
                {
                    "days": days,
                    "maturity": maturity,
                    "strike": strikes,
                    "option_type": option_type,
                    "bid": np.round(bid, 6),
                    "ask": np.round(ask, 6),
                    "underlying": setup.spot,
                }
            )
        )
        truth_parts.append(
            pd.DataFrame(
                {
                    "true_price": true_px,
                    "true_iv": true_iv,
                    "quoted_from": quoted,
                    "is_stale": stale,
                    "is_crossed": crossed,
                }
            )
        )
    return quote_parts, truth_parts


def generate_chain(
    rng: np.random.Generator,
    setup: MarketSetup | None = None,
    noise: MarketNoise | None = None,
) -> SyntheticChain:
    """Generate a synthetic option chain.

    Args:
        rng: Source of randomness; the same seed reproduces the same chain.
        setup: Market to simulate; defaults to ``MarketSetup()``.
        noise: Quote defects to apply; defaults to ``MarketNoise()``.

    Returns:
        The quotes and the hidden truth.

    Raises:
        ValueError: If the settings cannot produce a chain (see :func:`validate_market`).
    """
    setup = setup or MarketSetup()
    noise = noise or MarketNoise()
    validate_market(setup, noise)
    quote_parts, truth_parts = [], []
    for days in setup.expiry_days:
        quotes_one, truth_one = _expiry_quotes(rng, setup, noise, days)
        quote_parts += quotes_one
        truth_parts += truth_one
    quotes = pd.concat(quote_parts, ignore_index=True)
    truth = pd.concat(truth_parts, ignore_index=True)
    order = quotes.sort_values(["maturity", "strike", "option_type"]).index
    return SyntheticChain(
        quotes=quotes.loc[order].reset_index(drop=True),
        truth=truth.loc[order].reset_index(drop=True),
        setup=setup,
        noise=noise,
    )
