"""Step 3: render figures for a processed sample into outputs/figures/.

Produces roads / buildings / green-spaces / combined images, plus a
vector-vs-raster comparison (requires the raw cache for the vector side).
"""

import sys

from pathlib import Path

# Make `src/` importable even when the editable install is not picked up
# (iCloud-synced folders get the macOS "hidden" flag, and Python skips hidden .pth files).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")  # file output only

from urbanmorph.config import BARCELONA_EIXAMPLE_001, FIGURES_DIR
from urbanmorph.data.dataset import load_sample
from urbanmorph.data.osm import is_cached, load_raw
from urbanmorph.preprocessing.pipeline import build_layout
from urbanmorph.visualization.layout_plot import plot_combined, plot_single_layer, plot_vector_vs_raster


def main() -> int:
    area = BARCELONA_EIXAMPLE_001
    layout, meta = load_sample(area.sample_id)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sid = area.sample_id

    written = [
        plot_single_layer(layout, meta, "roads", FIGURES_DIR / f"{sid}_roads.png"),
        plot_single_layer(layout, meta, "buildings", FIGURES_DIR / f"{sid}_buildings.png"),
        plot_single_layer(layout, meta, "green_spaces", FIGURES_DIR / f"{sid}_green_spaces.png"),
        plot_combined(layout, meta, FIGURES_DIR / f"{sid}_combined.png"),
    ]
    if is_cached(area):
        raw = load_raw(area)
        sample = build_layout(area, raw["edges"], raw["buildings"], raw["green"], road_areas=raw["road_areas"])
        written.append(plot_vector_vs_raster(sample, meta, FIGURES_DIR / f"{sid}_vector_vs_raster.png"))

    for p in written:
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
