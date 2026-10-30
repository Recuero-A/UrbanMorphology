"""Milestone 2, step D: exploratory comparison of Spain, Peru and South Korea.

Reads data/processed/m2/samples.csv (tiles in the dataset only) and writes
figures + a summary table to outputs/m2/analysis/. Descriptive only: no
significance tests (tiles are spatially clustered, and there are only 3 cities
per country).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")  # file output only

from urbanmorph.config import M2_OUTPUTS
from urbanmorph.data.m2_dataset import load_index, load_m2_sample
from urbanmorph.visualization import compare


def main() -> int:
    index = load_index()
    df = index[index["in_dataset"]].copy()
    out = M2_OUTPUTS / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    layouts, hists = {}, {}
    for sid in df["sample_id"]:
        layout, meta = load_m2_sample(sid, index)
        layouts[sid] = layout
        hists[sid] = meta["metrics"]["orientation_hist"]

    written = [
        compare.metric_strips(df, out / "metric_distributions.png"),
        compare.orientation_roses(df, hists, out / "orientation_roses.png"),
        compare.representative_tiles(df, layouts, out / "representative_tiles.png"),
        compare.completeness_scatter(df, out / "osm_vs_ghsl_buildings.png"),
    ]
    table = compare.summary_table(df)
    table.to_csv(out / "summary_by_city.csv", index=False)
    cols = ["country", "city", "n_tiles"] + [f"{m}_median" for m, _, _ in compare.METRICS] + [
        "share_buildings_incomplete_flag", "share_green_empty"]
    t = table[cols].round(3).astype(str)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in t.values]
    (out / "summary_by_city.md").write_text("\n".join(md) + "\n")
    for p in written:
        print(f"Wrote {p}")
    print(json.dumps(table[cols].round(3).to_dict(orient="records")[:1], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
