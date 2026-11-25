"""Milestone 9 preparation: how are real blocks subdivided? (OBSERVED parcels, TRAIN tiles, Spain + US.)

    python scripts/m9_subdivision_analysis.py   # -> outputs/m9/analysis/subdivision.json, subdivision.png

Per CLOSED block (not touching the tile edge; layout/blocks.py), parcels assigned by representative point:
  * parcelled share: area of the block covered by parcels (1 - leftover);
  * type: none (no parcel), single (1), few (2-3), rows (>= 4 parcels);
  * interior share: area of parcels without street frontage / parcelled area;
  * street sides: each street parcel goes to the simplified block edge (1 m) its boundary runs along most
    (within 8 m: US parcel lines sit ~5 m behind the modelled road surface, Spain ~0.5 m);
    per side with >= 3 uncut street parcels: CV and robust spread (IQR / median) of frontage width, share
    of lots within +-15 % of the side median, and the same measures for widths drawn at random from the
    whole country (null; corner lots inflate the CV, hence the robust spread);
  * depth ratio: lot depth / block depth along the inward normal from the side's midpoint
    (~0.5 = back-to-back rows meeting in the middle, ~1 = lot runs through the block).
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely

from m6_build_objects import load_objects  # noqa: E402
from m8_parcel_records import OBS, load_records  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.layout.blocks import block_of, extract_blocks
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m9" / "analysis"
COUNTRIES = ("spain", "united_states")


def robust(w):
    return float(np.subtract(*np.percentile(w, [75, 25])) / np.median(w))


def block_depth(block, a, b):
    """Depth of the block along the inward normal from the midpoint of edge (a, b)."""
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([-t[1], t[0]])
    mid = (a + b) / 2
    if not block.contains(shapely.Point(mid + n)):
        n = -n
    r = np.arange(0.5, 600, 1.0)
    pts = mid[None] + r[:, None] * n[None]
    out = np.flatnonzero(~shapely.contains_xy(block, pts[:, 0], pts[:, 1]))
    return float(r[out[0]]) if len(out) else 600.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = pickle.loads(OBS.read_bytes())
    rec = load_records()
    rec = rec[rec.split == "train"]
    objs = load_objects()
    graphs = {t["sample_id"]: t["graph"] for t in objs["train"]}
    cw = objs["class_widths"]
    B = {c: {"share": [], "type": [], "interior": [], "n": [], "area": []} for c in COUNTRIES}
    S = {c: {"cv": [], "rs": [], "within15": [], "n": [], "width": [], "ratio": []} for c in COUNTRIES}
    for sid, d in rec.groupby("sample_id", sort=False):
        c = d.country.iat[0]
        polys = tiles[sid]["parcels"]
        blocks = [b for b in extract_blocks(graphs[sid], cw)]
        pts = np.array([p.representative_point().coords[0] for p in polys])
        bid = block_of(pts, blocks)
        for b in blocks:
            if b["open"]:
                continue
            poly = b["polygon"]
            idx = np.flatnonzero(bid == b["block_id"])
            n = len(idx)
            B[c]["n"].append(n)
            B[c]["area"].append(poly.area)
            B[c]["type"].append("none" if n == 0 else "single" if n == 1 else "few" if n <= 3 else "rows")
            if n == 0:
                B[c]["share"].append(0.0)
                continue
            parc = shapely.union_all([polys[i] for i in idx])
            B[c]["share"].append(min(parc.intersection(poly).area / poly.area, 1.0))
            dd = d.iloc[idx]
            ia = sum(polys[i].area for i, it in zip(idx, dd.interior) if it)
            B[c]["interior"].append(ia / max(parc.area, 1e-9))
            if n < 4:
                continue
            co = np.asarray(shapely.simplify(poly.exterior, 1.0).coords)
            edges = [(a_, b_) for a_, b_ in zip(co[:-1], co[1:]) if np.linalg.norm(b_ - a_) >= 6]
            if not edges:
                continue
            segs = [shapely.LineString([a_, b_]) for a_, b_ in edges]
            side = {}
            for i, row in zip(idx, dd.itertuples()):
                if row.interior or row.cut or not row.frontage_m > 1:
                    continue
                buf = polys[i].buffer(8.0)   # US parcel lines sit ~5 m behind the road surface (sidewalk)
                ov = [buf.intersection(s).length for s in segs]
                k = int(np.argmax(ov))
                if ov[k] > 1:
                    side.setdefault(k, []).append((row.frontage_m, row.depth_m))
            for k, lots in side.items():
                w = np.array([x for x, _ in lots])
                dep = np.array([y for _, y in lots], float)
                bd = block_depth(poly, *edges[k])
                S[c]["ratio"] += list(dep[np.isfinite(dep)] / bd)
                if len(w) >= 3:
                    S[c]["cv"].append(float(w.std() / w.mean()))
                    S[c]["rs"].append(robust(w))
                    S[c]["within15"].append(float(np.mean(np.abs(w / np.median(w) - 1) <= 0.15)))
                    S[c]["n"].append(len(w))
                    S[c]["width"].append(w)
        print(sid, flush=True)

    res = {}
    rng = np.random.default_rng(0)
    for c in COUNTRIES:
        b, s = B[c], S[c]
        types = np.array(b["type"])
        area = np.array(b["area"])
        allw = np.concatenate(s["width"])
        null = [rng.choice(allw, len(w)) for w in s["width"]]
        null_cv = [float(x.std() / x.mean()) for x in null]
        ratio = np.clip(np.array(s["ratio"]), 0, 1.5)
        res[c] = {
            "closed_blocks": len(types),
            "block_type_share": {t: float(np.mean(types == t)) for t in ("none", "single", "few", "rows")},
            "block_type_area_share": {t: float(area[types == t].sum() / area.sum()) for t in ("none", "single", "few", "rows")},
            "median_block_area_m2_by_type": {t: float(np.median(area[types == t])) if (types == t).any() else None
                                             for t in ("none", "single", "few", "rows")},
            "parcelled_share_median": float(np.median(b["share"])),
            "parcelled_share_median_rows": float(np.median(np.array(b["share"])[types == "rows"])),
            "interior_area_share_median": float(np.median(b["interior"])),
            "street_sides_ge3": len(s["cv"]),
            "lots_per_side_median": float(np.median(s["n"])),
            "width_cv_within_side_median": float(np.median(s["cv"])),
            "width_cv_null_random_median": float(np.median(null_cv)),
            "width_robust_spread_within_side_median": float(np.median(s["rs"])),
            "width_robust_spread_null_median": float(np.median([robust(x) for x in null])),
            "share_lots_within15pct_of_side_median": float(np.median(s["within15"])),
            "share_lots_within15pct_null_median": float(np.median([np.mean(np.abs(x / np.median(x) - 1) <= 0.15) for x in null])),
            "depth_ratio_quantiles_10_25_50_75_90": [float(q) for q in np.quantile(ratio, [.1, .25, .5, .75, .9])],
            "depth_ratio_share_0.35_0.65": float(np.mean((ratio >= .35) & (ratio <= .65))),
            "depth_ratio_share_ge_0.85": float(np.mean(ratio >= .85)),
        }
        res[c]["_raw"] = {"cv": s["cv"], "null_cv": null_cv, "ratio": ratio.tolist()}
    (OUT / "subdivision.json").write_text(json.dumps({c: {k: v for k, v in r.items() if k != "_raw"}
                                                      for c, r in res.items()}, indent=1))

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), facecolor=SURFACE)
    for r_, c in enumerate(COUNTRIES):
        x = res[c]
        ax = axes[r_, 0]
        t = list(x["block_type_share"])
        ax.barh(t, [x["block_type_share"][k] for k in t], color="#3d6fb6", height=0.6)
        ax.set_title(f"{c}: closed blocks by type (n = {x['closed_blocks']})", fontsize=9, loc="left", color=INK)
        ax.set_xlim(0, 1)
        ax = axes[r_, 1]
        bins = np.linspace(0, 1.2, 25)
        ax.hist(x["_raw"]["null_cv"], bins, color="#bdbdbd", label="random lots (null)")
        ax.hist(x["_raw"]["cv"], bins, color="#3d6fb6", alpha=0.85, label="same street side")
        ax.set_title("frontage-width CV per street side (≥ 3 lots)", fontsize=9, loc="left", color=INK)
        ax.legend(fontsize=7, frameon=False)
        ax = axes[r_, 2]
        ax.hist(x["_raw"]["ratio"], np.linspace(0, 1.5, 31), color="#3d6fb6")
        ax.axvline(0.5, color=INK, lw=0.8, ls="--")
        ax.set_title("lot depth / block depth (0.5 = back-to-back)", fontsize=9, loc="left", color=INK)
        for a in axes[r_]:
            a.set_facecolor(SURFACE)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
    fig.suptitle("How real blocks are subdivided (observed parcels, train tiles)", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "subdivision.png", dpi=130, facecolor=SURFACE)
    print(json.dumps({c: {k: v for k, v in r.items() if k != "_raw"} for c, r in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
