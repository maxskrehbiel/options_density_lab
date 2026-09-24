"""Recover market-implied (risk-neutral) probability distributions from option prices.

Covers synthetic chains, quote cleaning, arbitrage-aware smiles, Breeden-Litzenberger
recovery, no-arbitrage checks and validation against a known truth.
"""

from importlib.metadata import version

from .density import DensityConfig, RiskNeutralDensity, rnd_from_smile
from .pipeline import PipelineConfig, SurfaceResult, run_pipeline
from .synthetic import MarketNoise, MarketSetup, SyntheticChain, generate_chain

__version__ = version("options_density_lab")

__all__ = [
    "DensityConfig",
    "MarketNoise",
    "MarketSetup",
    "PipelineConfig",
    "RiskNeutralDensity",
    "SurfaceResult",
    "SyntheticChain",
    "__version__",
    "generate_chain",
    "rnd_from_smile",
    "run_pipeline",
]
