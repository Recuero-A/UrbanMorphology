"""Milestone 3, step 0: independent audit of the M2 dataset before training.

Re-derives everything from the saved arrays and the index instead of trusting
earlier tests: dimensions and values, split separation, exact and near
duplicates, balance, quality flags, channel statistics and the road-width
limitation. Writes outputs/m3/dataset_audit.md.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from urbanmorph.config import M2_DIR, PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index, load_m2_sample
from urbanmorph.splits.geographic import edge_distance

OUT = PROJECT_ROOT / "outputs" / "m3" / "dataset_audit.md"
CH = ("roads", "buildings", "green")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def main() -> int:
    index = load_index()
    ds = index[index["in_dataset"]].reset_index(drop=True)
    lines = ["# Dataset audit before Milestone 3 training", ""]

    # 1. dimensions and values
    arrays = {}
    problems = []
    for sid in ds["sample_id"]:
        a, _ = load_m2_sample(sid, index)
        if a.shape != (3, 128, 128) or a.dtype != np.uint8 or not set(np.unique(a)) <= {0, 1}:
            problems.append(sid)
        arrays[sid] = a
    lines += ["## 1. Dimensions and values", f"- {len(arrays)} tiles loaded; shape/dtype/binary problems: {problems or 'none'}", ""]

    # 2. geographic separation between splits
    lines += ["## 2. Separation between splits (edge-to-edge, same city)", "", "| pair | min distance (m) | tiles closer than 1000 m |", "|---|---|---|"]
    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
        dmin, close = np.inf, 0
        for _, g in ds.groupby("city_code"):
            a, b = g[g.split == s1], g[g.split == s2]
            for _, r1 in a.iterrows():
                for _, r2 in b.iterrows():
                    d = edge_distance(r1, r2)
                    dmin = min(dmin, d)
                    close += d < 1000
        lines.append(f"| {s1}–{s2} | {dmin:.0f} | {close} |")
    within = []
    for _, g in ds.groupby("city_code"):
        for (_, r1), (_, r2) in combinations(g.iterrows(), 2):
            within.append(edge_distance(r1, r2))
    lines += ["", f"- Minimum edge distance between ANY two tiles: {min(within):.0f} m (selection spacing 1536 m between centres).", ""]

    # 3. duplicates / near-duplicates
    ids = list(arrays)
    hashes = ds.set_index("sample_id")["layout_hash"]
    exact = hashes[hashes.duplicated(keep=False)]
    pair_iou = []
    for i, j in combinations(range(len(ids)), 2):
        a, b = arrays[ids[i]], arrays[ids[j]]
        pair_iou.append((iou(a[1], b[1]), iou(a[0], b[0]), ids[i], ids[j]))
    pair_iou.sort(reverse=True)
    lines += ["## 3. Duplicates and near-duplicates",
              f"- Exact duplicate rasters: {len(exact)}",
              "- Highest pairwise IoU (building channel, then road channel) between any two tiles:", ""]
    lines += ["| building IoU | road IoU | tile A | tile B |", "|---|---|---|---|"]
    lines += [f"| {b:.3f} | {r:.3f} | {x} | {y} |" for b, r, x, y in pair_iou[:5]]
    lines += ["", "Tiles are disjoint grid cells, so high IoU could only mean very regular repeated urban "
              "patterns (e.g. identical grid blocks), not shared territory.", ""]

    # 4. balance
    bal = ds.groupby(["country", "city"])["split"].value_counts().unstack(fill_value=0)
    lines += ["## 4. Balance", "", "| country | city | train | val | test |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {ci} | {r.get('train', 0)} | {r.get('val', 0)} | {r.get('test', 0)} |" for (c, ci), r in bal.iterrows()]
    strata = pd.crosstab(ds["density"], ds["split"])
    lines += ["", "Density strata by split:", "", "| density | " + " | ".join(strata.columns) + " |",
              "|---|" + "---|" * len(strata.columns)]
    lines += [f"| {k} | " + " | ".join(str(v) for v in r.values) + " |" for k, r in strata.iterrows()]
    lines.append("")

    # 5. quality flags
    lines += ["## 5. Quality flags by split", ""]
    for sp in ("train", "val", "test"):
        sub = ds[ds.split == sp]
        w = sub["flags_warning"].fillna("").str.split(";").explode()
        inf = sub["flags_info"].fillna("").str.split(";").explode()
        lines.append(f"- {sp} ({len(sub)}): warnings {w[w != ''].value_counts().to_dict()}, info {inf[inf != ''].value_counts().to_dict()}")
    lines.append("")

    # 6. channel statistics (for loss weighting)
    stats = []
    for sp in ("train", "val", "test"):
        stack = np.stack([arrays[s] for s in ds[ds.split == sp]["sample_id"]]).astype(float)
        stats.append((sp, *stack.mean(axis=(0, 2, 3))))
    lines += ["## 6. Positive-pixel fraction per channel (class imbalance)", "", "| split | roads | buildings | green |", "|---|---|---|---|"]
    lines += [f"| {s} | {r:.3f} | {b:.3f} | {g:.3f} |" for s, r, b, g in stats]
    lines.append("")

    # 7. road-width limitation
    rows = []
    for _, r in ds.iterrows():
        m = json.loads((M2_DIR / r.country / r.city / f"{r.sample_id}.json").read_text())
        s = m["road_buffer"]["segments_by_width_source"]
        tot = sum(s.values()) or 1
        rows.append({"country": r.country, "class_default_share": s.get("class_default", 0) / tot, "roads": r.roads_coverage})
    rw = pd.DataFrame(rows).groupby("country").mean()
    lines += ["## 7. Road-width limitation", "", "| country | share of road segments with class-default width | mean road coverage |", "|---|---|---|"]
    lines += [f"| {c} | {r.class_default_share:.2f} | {r.roads:.3f} |" for c, r in rw.iterrows()]
    lines += ["", "Road coverage is partly an artefact of default widths (largest effect in Japan). The model will "
              "learn these widths as if they were real; road-coverage errors are reported but not interpreted "
              "as cross-country differences.", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
