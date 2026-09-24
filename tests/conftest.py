"""Shared fixtures: noisy and exact synthetic chains, their pipeline outputs, and the truth."""

from __future__ import annotations

import numpy as np
import pytest

from options_density_lab.pipeline import PipelineConfig, SurfaceResult, run_pipeline
from options_density_lab.synthetic import MarketNoise, MarketSetup, SyntheticChain, generate_chain
from options_density_lab.validate import TruthCache

SEED = 7


@pytest.fixture(scope="session")
def setup() -> MarketSetup:
    return MarketSetup()


@pytest.fixture(scope="session")
def noisy_chain(setup: MarketSetup) -> SyntheticChain:
    return generate_chain(np.random.default_rng(SEED), setup)


@pytest.fixture(scope="session")
def exact_chain(setup: MarketSetup) -> SyntheticChain:
    return generate_chain(np.random.default_rng(SEED), setup, MarketNoise.none())


@pytest.fixture(scope="session")
def truth(setup: MarketSetup) -> TruthCache:
    return TruthCache(setup, n=1201)


@pytest.fixture(scope="session")
def svi_surface(noisy_chain: SyntheticChain) -> SurfaceResult:
    return run_pipeline(noisy_chain.quotes, PipelineConfig.for_method("svi"))


@pytest.fixture(scope="session")
def spline_surface(noisy_chain: SyntheticChain) -> SurfaceResult:
    return run_pipeline(noisy_chain.quotes, PipelineConfig.for_method("spline"))
