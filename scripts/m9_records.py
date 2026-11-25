"""Milestone 9: training records for BlockNet, zoning and template choice, from OBSERVED parcels.

    python scripts/m9_records.py      # -> data/processed/m9/records.pkl (git-ignored), outputs/m9/analysis/jitter.json

Train + val tiles of Spain and the US (split column kept; models fit on train, select on val).
* blocks (CLOSED blocks only, layout/blocks.py): features + type (none / single / few / rows by parcel count);
* sides: every street edge of a closed block (parcel_gen.street_edges, the same edges generation uses):
  N = street parcels assigned to it (parcel boundary within 8 m of the edge, longest overlap; US parcel
  lines sit ~5 m behind the modelled road surface), rear ratio = median over its lots of
  (distance from the edge to the lot's rear) / (block depth along the inward normal);
* zoning: every parcel; green = >= 50 % covered by real green; block type of its block;
* shapes: every real building alone on a parcel: best template (layout/templates.py) in the lot frame
  (street parcels: x along the street edge, y inwards; interior parcels: the parcel's minimum rotated
  rectangle) + lot features.
Width jitter: per country, robust SD (1.4826 MAD) of log(width / side mean) over sides with >= 3 lots.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import shapely

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from m8_parcel_records import OBS, load_records  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.layout.blocks import block_of, extract_blocks
from urbanmorph.layout.objects import MIN_BUILDING_M2, MIN_GREEN_M2
from urbanmorph.layout.parcel_gen import street_class_at, street_edges
from urbanmorph.layout.parcels import FRONT_TOL_M
from urbanmorph.layout.templates import best_template, to_unit

OUT = PROJECT_ROOT / "data" / "processed" / "m9" / "records.pkl"
JIT = PROJECT_ROOT / "outputs" / "m9" / "analysis" / "jitter.json"
TYPES = ("none", "single", "few", "rows")
SIDE_TOL_M = 8.0


def block_type(n):
    return 0 if n == 0 else 1 if n == 1 else 2 if n <= 3 else 3


def full_depth(block, a, t, n, L):
    mid = a + t * L / 2
    r = np.arange(0.5, 600, 1.0)
    pts = mid[None] + r[:, None] * n[None]
    out = np.flatnonzero(~shapely.contains_xy(block, pts[:, 0], pts[:, 1]))
    return float(r[out[0]]) if len(out) else 600.0


def block_features(b, edges, graph):
    poly = b["polygon"]
    cls = street_class_at(np.array([a + t * L / 2 for L, a, t, n in edges]), graph) if edges else np.array([2])
    return {"log_area": float(np.log(poly.area)), "rect": float(b["rectangularity"]), "compact": float(b["compactness"]),
            "n_edges": len(edges), "cls": int(np.bincount(cls, minlength=4).argmax())}


def assign_sides(polys, idx, edges):
    """Street-edge index for each parcel in idx (-1 = none)."""
    segs = [shapely.LineString([a, a + t * L]) for L, a, t, n in edges]
    out = {}
    for i in idx:
        if not segs:
            out[i] = -1
            continue
        buf = polys[i].buffer(SIDE_TOL_M)
        ov = [buf.intersection(s).length for s in segs]
        k = int(np.argmax(ov))
        out[i] = k if ov[k] > 1.0 else -1
    return out


def tile_records(sid, d, polys, graph, cw, green, buildings):
    c, split = d.country.iat[0], d.split.iat[0]
    seg = graph.segments()
    near = shapely.MultiLineString([s.reshape(2, 2) for s in seg]).buffer(FRONT_TOL_M) if len(seg) else shapely.Polygon()
    blocks = extract_blocks(graph, cw)
    rp = np.array([p.representative_point().coords[0] for p in polys])
    bid = block_of(rp, blocks)
    G = shapely.union_all(green) if green else shapely.Polygon()
    B, S, Z, H, W = [], [], [], [], []
    btype = {}
    side_of = {}
    edges_of = {}
    for b in blocks:
        idx = np.flatnonzero(bid == b["block_id"])
        edges = street_edges(b["polygon"], near)
        edges_of[b["block_id"]] = edges
        btype[b["block_id"]] = block_type(len(idx))
        side_of.update(assign_sides(polys, idx, edges))
        if b["open"]:
            continue
        f = block_features(b, edges, graph)
        B.append({"sample_id": sid, "country": c, "split": split, **f, "type": btype[b["block_id"]]})
        for k, (L, a, t, n) in enumerate(edges):
            lots = [i for i in idx if side_of[i] == k and not d.interior.iat[i] and not d.cut.iat[i]]
            D = full_depth(b["polygon"], a, t, n, L)
            ratios = [float(((np.asarray(polys[i].exterior.coords) - a) @ n).max() / D) for i in lots]
            S.append({"sample_id": sid, "country": c, "split": split, **f, "type": btype[b["block_id"]],
                      "log_len": float(np.log(L)), "side_cls": int(street_class_at((a + t * L / 2)[None], graph)[0]),
                      "N": len(lots), "rear_ratio": float(np.median(ratios)) if ratios else np.nan})
            w = d.frontage_m.to_numpy()[lots]
            if len(w) >= 3:
                W.append(np.log(w / w.mean()))
    bpts = np.array([p.representative_point().coords[0] for p in buildings]) if buildings else np.zeros((0, 2))
    owner = -np.ones(len(buildings), int)
    for i, p in enumerate(polys):
        if len(bpts):
            owner[shapely.contains_xy(p, bpts[:, 0], bpts[:, 1]) & (owner < 0)] = i
    for i, p in enumerate(polys):
        gfrac = p.intersection(G).area / p.area if not G.is_empty and p.intersects(G) else 0.0
        bt = btype.get(int(bid[i]), 3)
        Z.append({"sample_id": sid, "country": c, "split": split, "log_area": float(np.log(p.area)),
                  "interior": bool(d.interior.iat[i]), "btype": bt, "green": gfrac >= 0.5, "nb": d.nb.iat[i]})
        mine = np.flatnonzero(owner == i)
        if len(mine) != 1 or d.cut.iat[i]:
            continue
        bpoly = buildings[mine[0]]
        k_side = side_of.get(i, -1)
        if not d.interior.iat[i] and k_side >= 0 and d.frontage_m.iat[i] > 1:
            L, a, t, n = edges_of[int(bid[i])][k_side]
            u = to_unit(bpoly, a, t, n)[0]
            w_, d_ = float(d.frontage_m.iat[i]), float(d.depth_m.iat[i]) if np.isfinite(d.depth_m.iat[i]) else np.nan
            inter = False
        else:
            rect = np.asarray(shapely.minimum_rotated_rectangle(p).exterior.coords)[:4]
            e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
            if np.hypot(*e1) < np.hypot(*e2):
                e1, e2 = e2, e1
            t, n = e1 / np.hypot(*e1), e2 / np.hypot(*e2)
            u = to_unit(bpoly, rect[0], t, n)[0]
            w_, d_ = float(np.hypot(*e1)), float(np.hypot(*e2))
            inter = True
        if not np.isfinite(d_) or not u.is_valid:
            continue
        k, iou = best_template(u)
        H.append({"sample_id": sid, "country": c, "split": split, "log_w": float(np.log(max(w_, 1))), "log_d": float(np.log(max(d_, 1))),
                  "coverage": float(d.coverage.iat[i]), "interior": inter, "cls": int(d.cls.iat[i]), "template": k, "iou": iou})
    return B, S, Z, H, W


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tiles = pickle.loads(OBS.read_bytes())
    rec = load_records()
    objs = load_objects()
    graphs = {t["sample_id"]: t["graph"] for t in objs["train"] + objs["val"]}
    cw = objs["class_widths"]
    idx = load_index_with_expansion().set_index("sample_id")
    out = {k: [] for k in ("blocks", "sides", "zoning", "shapes")}
    W = {"spain": [], "united_states": []}
    offset = 0
    for sid, d in rec.groupby("sample_id", sort=False):
        d = d.reset_index(drop=True)
        r = idx.loc[sid]
        f = sample_dir(r.country, r.city) / f"{sid}_vectors.gpkg"
        green = tile_polygons(f, r.xmin, r.ymin, "green", MIN_GREEN_M2)
        bld = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
        B, S, Z, H, w = tile_records(sid, d, tiles[sid]["parcels"], graphs[sid], cw, green, bld)
        for z in Z:                                  # neighbour indices stay global (records order)
            z["row"] = offset
            offset += 1
        out["blocks"] += B
        out["sides"] += S
        out["zoning"] += Z
        out["shapes"] += H
        if d.split.iat[0] == "train":
            W[d.country.iat[0]] += w
        print(sid, len(B), len(S), len(H), flush=True)
    out = {k: pd.DataFrame(v) for k, v in out.items()}
    OUT.write_bytes(pickle.dumps(out))
    jit = {c: float(1.4826 * np.median(np.abs(np.concatenate(v) - np.median(np.concatenate(v))))) for c, v in W.items()}
    JIT.write_text(json.dumps({"width_jitter_log_sd": jit, "sides_used": {c: len(v) for c, v in W.items()}}, indent=1))
    print({k: len(v) for k, v in out.items()}, jit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
