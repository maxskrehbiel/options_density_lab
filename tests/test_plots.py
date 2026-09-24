"""Every figure renders to a non-empty PNG, with and without the truth overlay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from options_density_lab import plots
from options_density_lab.pipeline import SurfaceResult
from options_density_lab.synthetic import MarketSetup
from options_density_lab.validate import TruthCache

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 10_000 and path.read_bytes()[:8] == PNG_SIGNATURE


def test_all_figures_render(
    tmp_path: Path,
    svi_surface: SurfaceResult,
    spline_surface: SurfaceResult,
    truth: TruthCache,
    setup: MarketSetup,
) -> None:
    surfaces = {"svi": svi_surface, "spline": spline_surface}
    study = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "days": [30, 30, 30, 30],
            "method": ["svi", "spline"] * 2,
            "tv": [0.01, 0.02, 0.015, 0.012],
        }
    )
    written = [
        plots.plot_quote_cleaning(svi_surface.expiries[0], tmp_path / "cleaning.png"),
        plots.plot_smiles(surfaces, tmp_path / "smiles.png", setup.true_iv),
        plots.plot_densities(surfaces, tmp_path / "densities.png", truth),
        plots.plot_cdfs(surfaces, tmp_path / "cdfs.png", truth),
        plots.plot_arbitrage(svi_surface, tmp_path / "arbitrage.png"),
        plots.plot_seed_study(study, tmp_path / "seeds.png"),
        plots.plot_densities({"svi": svi_surface}, tmp_path / "densities_no_truth.png"),
        plots.plot_cdfs({"svi": svi_surface}, tmp_path / "cdfs_no_truth.png"),
    ]
    assert all(_is_png(p) for p in written)
