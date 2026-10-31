"""Milestone 8, Phase 1 (research and diagnosis; no training).

    python scripts/m8_analysis.py

1. Urban blocks from real street graphs (layout/blocks.py): counts, sizes, shapes, open blocks.
2. Building-size relationships per country (real OSM buildings, TRAINING tiles): size / shape
   statistics and size similarity under 4 neighbourhood definitions (nearest neighbours, same
   frontage, same block, 50 m radius), with Moran's I of log area and a within-tile permutation
   baseline.
3. Diagnosis of the Milestone 7 frontage phase: why proposed buildings are rejected, and what
   the fallback (interior) placer contributes.
Writes outputs/m8/analysis/*.json, *.csv, *.png. Test tiles are never read.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import torch

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import road_polygons
from urbanmorph.layout.blocks import block_of, extract_blocks
from urbanmorph.layout.export import BUILDING_FILL, GROUND, ROAD_FILL, _fill
from urbanmorph.layout.frontage import assign, interior_angles
from urbanmorph.layout.objects import MIN_BUILDING_M2, mrr_params
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m8" / "analysis"
COUNTRIES = ("spain", "united_states", "japan")


def morans_i(x, W):
    """Moran's I of x for a (row-standardised) weight matrix W."""
    z = x - x.mean()
    if (z ** 2).sum() == 0 or W.sum() == 0:
        return np.nan
    return float(len(x) / W.sum() * (z @ W @ z) / (z @ z))


def row_norm(W):
    s = W.sum(1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)


def tile_size_stats(polys, graph, blocks, rng):
    n = len(polys)
    area = np.array([p.area for p in polys])
    la = np.log(area)
    enc = [mrr_params(p) for p in polys]
    A = np.array([e[0] for e in enc])
    L = np.array([e[2] for e in enc])
    Wd = np.array([e[3] for e in enc])
    per = np.array([p.length for p in polys])
    D = np.hypot(*(A[:, None] - A[None]).transpose(2, 0, 1))
    np.fill_diagonal(D, np.inf)
    weights = {}
    k = min(5, n - 1)
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), np.argsort(D, 1)[:, :k].ravel()] = 1
    weights["knn5"] = W
    W = np.zeros((n, n))
    W[np.arange(n), D.argmin(1)] = 1
    weights["nn1"] = W
    weights["radius50"] = (D < 50).astype(float)
    fr, _ = assign(polys, graph, A)
    W = np.zeros((n, n))
    for f in fr:
        m = f["members"]
        for a, b in zip(m[:-1], m[1:]):
            W[a, b] = W[b, a] = 1
    weights["same_frontage_consecutive"] = W
    bid = block_of(A, blocks)
    W = ((bid[:, None] == bid[None]) & (bid[:, None] >= 0)).astype(float)
    np.fill_diagonal(W, 0)
    weights["same_block"] = W
    out = {"n": n, "log_area": la, "width": Wd, "length": L, "aspect": L / Wd, "perimeter": per,
           "compactness": 4 * np.pi * area / per ** 2, "pairs": {}}
    for name, W in weights.items():
        i, j = np.nonzero(np.triu(W + W.T, 1))
        if len(i) < 5:
            continue
        perm = rng.permutation(n)
        out["pairs"][name] = {"abs_log_ratio": np.abs(la[i] - la[j]),
                              "baseline_abs_log_ratio": np.abs(la[i] - la[perm[j]]),
                              "morans_i": morans_i(la, row_norm(W))}
    # local size variance within blocks (closed and open), only blocks with >= 3 buildings
    out["block_log_area_std"] = [float(la[bid == b].std()) for b in np.unique(bid[bid >= 0]) if (bid == b).sum() >= 3]
    return out


def diagnose_frontage(data, cw, device):
    """Instrument Milestone 7 (frontage seed 0 + P2 seed 0 interior, strict rules) on validation streets."""
    import urbanmorph.layout.place as P
    from m6_evaluate import load_placer
    from m7_evaluate import load_frontage
    log = []
    orig = P.LayoutState.try_add

    def ta(self, poly, typ):
        n_before = len(self.buildings)
        before = dict(self.rules)
        ok = orig(self, poly, typ)
        if typ == 0 and getattr(self, "_phase_tag", "frontage") == "frontage":
            changed = [k for k in self.rules if self.rules[k] != before[k]]
            rec = {"outcome": "accepted" if ok else (changed[0] if changed else "unknown"), "trimmed": "building_trimmed" in changed}
            if not ok and changed and changed[0] == "rejected_overlap" and n_before:
                arr = np.array(self.buildings, dtype=object)
                hit = shapely.intersects(arr, poly)
                ov = shapely.area(shapely.intersection(arr[hit], poly)) / np.minimum(shapely.area(arr[hit]), poly.area)
                worst = np.flatnonzero(hit)[np.argmax(ov)]
                rec["overlap_with"] = "previous building on same frontage" if worst == n_before - 1 else "earlier building (other frontage / earlier in row)"
                rec["overlap_share"] = float(ov.max())
            if not ok and changed and changed[0] == "rejected_on_road":
                land = poly.difference(self.surface)
                rec["share_on_road"] = float(1 - land.area / poly.area) if poly.area else np.nan
            log.append(rec)
        return ok

    P.LayoutState.try_add = ta
    graphs = [t["graph"] for t in data["val"]]
    states = P.place_frontage(load_frontage(0, device), graphs, cw, device=device, seed=21, strict=True)
    n_front = [len(s.buildings) for s in states]
    for s in states:
        s._phase_tag = "interior"
    states = P.place_neural(load_placer(0, device), None, cw, device=device, seed=21, states=states)
    P.LayoutState.try_add = orig
    df = pd.DataFrame(log)
    res = {"proposed_frontage_buildings": len(df), "outcomes": df["outcome"].value_counts(normalize=True).round(4).to_dict(),
           "share_trimmed_of_accepted": float(df.loc[df.outcome == "accepted", "trimmed"].mean()),
           "overlap_partner": df["overlap_with"].value_counts(normalize=True).round(4).to_dict() if "overlap_with" in df else {},
           "overlap_share_median": float(df["overlap_share"].median()) if "overlap_share" in df else np.nan,
           "on_road_share_median_of_rejected": float(df["share_on_road"].median()) if "share_on_road" in df else np.nan}
    # alignment: frontage-phase buildings vs interior-phase buildings
    al_f, al_i, n_int = [], [], 0
    for s, k in zip(states, n_front):
        b = s.buildings
        if len(b) < 3:
            continue
        th = np.array([mrr_params(p)[1] for p in b])
        A = np.array([mrr_params(p)[0] for p in b])
        D = np.hypot(*(A[:, None] - A[None]).transpose(2, 0, 1))
        np.fill_diagonal(D, np.inf)
        nn = D.argmin(1)
        d = np.degrees(np.abs((th - th[nn]) % (np.pi / 2)))
        d = np.minimum(d, 90 - d) < 5
        al_f += list(d[:k])
        al_i += list(d[k:])
        n_int += len(b) - k
    res["final_buildings"] = int(sum(len(s.buildings) for s in states))
    res["share_from_interior_fallback"] = n_int / max(res["final_buildings"], 1)
    res["nn_aligned_5deg_frontage_buildings"] = float(np.mean(al_f))
    res["nn_aligned_5deg_interior_buildings"] = float(np.mean(al_i)) if al_i else np.nan
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    data = load_objects()
    cw = data["class_widths"]
    idx = load_index_with_expansion().set_index("sample_id")

    # ---------------------------------------------------------------- 1 + 2: blocks and building sizes (train tiles)
    block_rows, size = [], {c: {"stats": [], "tiles": 0} for c in COUNTRIES}
    examples = []
    for t in data["train"][::3]:                       # every third training tile (~190 tiles) for speed
        r = idx.loc[t["sample_id"]]
        f = sample_dir(r.country, r.city) / f"{t['sample_id']}_vectors.gpkg"
        polys = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
        blocks = extract_blocks(t["graph"], cw)
        for b in blocks:
            block_rows.append({"country": t["country"], "city": r.city, "sample_id": t["sample_id"], **{k: v for k, v in b.items() if k != "polygon"}})
        if len(examples) < 6 and len(polys) > 30 and sum(e[0] == t["country"] for e in examples) < 2:
            examples.append((t["country"], t, polys, blocks))
        if len(polys) >= 10:
            size[t["country"]]["stats"].append(tile_size_stats(polys, t["graph"], blocks, rng))
            size[t["country"]]["tiles"] += 1
    bdf = pd.DataFrame(block_rows)
    bdf.to_csv(OUT / "blocks_train_sample.csv", index=False)
    blocks_summary = {}
    for c in COUNTRIES:
        d = bdf[bdf.country == c]
        closed = d[~d.open]
        blocks_summary[c] = {"tiles": int(d.sample_id.nunique()), "blocks_per_tile_median": float(d.groupby("sample_id").size().median()),
                             "share_open_blocks": float(d.open.mean()),
                             "closed_block_area_m2_p10_p50_p90": [float(np.percentile(closed.area_m2, q)) for q in (10, 50, 90)] if len(closed) else None,
                             "closed_block_rectangularity_median": float(closed.rectangularity.median()) if len(closed) else None,
                             "closed_block_compactness_median": float(closed.compactness.median()) if len(closed) else None}
    size_summary = {}
    for c in COUNTRIES:
        S = size[c]["stats"]
        cat = lambda k: np.concatenate([s[k] for s in S])
        e = {"tiles": size[c]["tiles"], "buildings": int(sum(s["n"] for s in S)),
             "area_m2_p10_p50_p90": np.round(np.exp(np.percentile(cat("log_area"), [10, 50, 90])), 1).tolist(),
             "width_m_median": float(np.median(cat("width"))), "length_m_median": float(np.median(cat("length"))),
             "aspect_median": float(np.median(cat("aspect"))), "compactness_median": float(np.median(cat("compactness"))),
             "within_block_log_area_std_median": float(np.median(np.concatenate([s["block_log_area_std"] for s in S]))),
             "neighbourhoods": {}}
        for nb in ("nn1", "knn5", "same_frontage_consecutive", "same_block", "radius50"):
            pr = [s["pairs"][nb] for s in S if nb in s["pairs"]]
            if not pr:
                continue
            obs = np.concatenate([p["abs_log_ratio"] for p in pr])
            base = np.concatenate([p["baseline_abs_log_ratio"] for p in pr])
            mi = [p["morans_i"] for p in pr if np.isfinite(p["morans_i"])]
            e["neighbourhoods"][nb] = {"median_abs_log_area_ratio": float(np.median(obs)),
                                       "random_pairs_same_tile": float(np.median(base)),
                                       "share_within_factor_1.5": float((obs < np.log(1.5)).mean()),
                                       "random_share_within_factor_1.5": float((base < np.log(1.5)).mean()),
                                       "morans_i_log_area_median_over_tiles": float(np.median(mi)),
                                       "share_tiles_morans_i_positive": float(np.mean(np.array(mi) > 0))}
        size_summary[c] = e

    # ---------------------------------------------------------------- 3: frontage diagnosis (validation streets)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    diag = diagnose_frontage(data, cw, device)
    res = {"blocks": blocks_summary, "building_sizes": size_summary, "frontage_diagnosis": diag}
    (OUT / "phase1_analysis.json").write_text(json.dumps(res, indent=2, default=float))

    # ---------------------------------------------------------------- figure: streets, blocks, buildings
    fig, axes = plt.subplots(2, len(examples), figsize=(3.2 * len(examples), 6.6), facecolor=SURFACE)
    cmap = plt.get_cmap("tab20")
    for j, (c, t, polys, blocks) in enumerate(examples):
        for row in (0, 1):
            ax = axes[row, j]
            ax.set_facecolor(GROUND)
            _fill(ax, road_polygons(t["graph"], class_widths=cw), ROAD_FILL, z=1)
            for k, b in enumerate(blocks):
                if row == 0:
                    x, y = b["polygon"].exterior.xy
                    ax.fill(x, y, color=cmap(k % 20), alpha=0.35 if not b["open"] else 0.15, lw=0.6, ec="#555555", zorder=2,
                            ls="-" if not b["open"] else "--")
            if row == 1:
                for p in polys:
                    _fill(ax, p, BUILDING_FILL, z=3)
            ax.set_xlim(0, 512), ax.set_ylim(0, 512), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        axes[0, j].set_title(f"{c.replace('_', ' ')}: {sum(not b['open'] for b in blocks)} closed / {sum(b['open'] for b in blocks)} open blocks",
                             fontsize=7.5, loc="left", color=INK)
    fig.suptitle("Milestone 8 Stage A (analysis): blocks = tile minus road surface (dashed = open at the tile edge); real OSM buildings below",
                 fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "blocks_examples.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
