"""Milestone 3b: plan additional TRAINING tiles from cached data (no network).

Rules (val/test untouched, splits unchanged):
  1. Same population as Milestone 2: eligible GHSL cells in the same study areas.
  2. Exclude every cell in a spatial block (3,072 m) that holds a val or test tile.
  3. Exclude every cell closer than the split buffer (1,000 m edge to edge) to any val/test tile.
  4. Exclude cells already used by any dataset tile.
  5. Spacing: new tile centres >= --spacing from every train tile (existing and new),
     to avoid near-identical neighbours.
  6. Stratified random selection (same density x era strata, same seeded procedure as M2),
     capped per city for balance.

    python scripts/m3b_expansion_plan.py                 # feasibility report only
    python scripts/m3b_expansion_plan.py --write --per-city 60 --spacing 1024
"""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from pyproj import Transformer

from urbanmorph.config import M2_DIR, PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index
from urbanmorph.sampling import strata as st
from urbanmorph.sampling.grid import sample_id
from urbanmorph.sampling.select import city_rng, select_tiles
from urbanmorph.sampling.study_area import cities, load_config
from urbanmorph.splits.geographic import block_ids

PLAN_OUT = M2_DIR / "sampling" / "expansion_m3b.csv"
SEED_OFFSET = 3  # a different random stream from the M2 draw


def _plan_module():
    spec = importlib.util.spec_from_file_location("plan", PROJECT_ROOT / "scripts" / "m2_plan_sampling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def edge_dist(cands: pd.DataFrame, tiles: pd.DataFrame) -> np.ndarray:
    """Min edge-to-edge distance from each candidate to any tile (same CRS), shape (n,)."""
    if tiles.empty:
        return np.full(len(cands), np.inf)
    dx = np.maximum(0, np.maximum(cands.xmin.values[:, None], tiles.xmin.values[None]) -
                    np.minimum(cands.xmax.values[:, None], tiles.xmax.values[None]))
    dy = np.maximum(0, np.maximum(cands.ymin.values[:, None], tiles.ymin.values[None]) -
                    np.minimum(cands.ymax.values[:, None], tiles.ymax.values[None]))
    return np.hypot(dx, dy).min(axis=1)


def centre_dist(cands: pd.DataFrame, tiles: pd.DataFrame) -> np.ndarray:
    if tiles.empty:
        return np.full(len(cands), np.inf)
    cx, cy = (cands.xmin + cands.xmax).values / 2, (cands.ymin + cands.ymax).values / 2
    tx, ty = (tiles.xmin + tiles.xmax).values / 2, (tiles.ymin + tiles.ymax).values / 2
    return np.hypot(cx[:, None] - tx[None], cy[:, None] - ty[None]).min(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spacing", type=float, default=1024.0, help="min centre distance between train tiles (m)")
    ap.add_argument("--per-city", type=int, default=0, help="cap of new tiles per city (0 = report maximum)")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    plan_mod = _plan_module()
    index = load_index()
    ds = index[index["in_dataset"]]
    sp = cfg["splits"]
    rows, report = [], []
    for city in cities(cfg):
        _, crs, cands = plan_mod.candidates_for(city, cfg)
        cands["density"] = st.density_class(cands["ghsl_built_2020"], cfg["strata"]["density_thresholds"])
        cands["era"] = st.era_class(cands, cfg["study_area"]["ghsl_epochs"], cfg["strata"]["era_builtup_fraction"])
        cands["stratum"] = cands["density"] + "|" + cands["era"]
        cands["city_code"] = city.code
        el = cands[cands["eligible"]].reset_index(drop=True)
        mine = ds[ds.city_code == city.code]
        held = mine[mine.split.isin(["val", "test"])]
        train = mine[mine.split == "train"]
        n0 = len(el)
        el["block_id"] = block_ids(el, sp["block_size_m"])
        el = el[~el["block_id"].isin(set(block_ids(held, sp["block_size_m"])))]
        n_block = len(el)
        el = el[edge_dist(el, held) >= sp["buffer_m"]]
        n_buffer = len(el)
        used = set(zip(mine.grid_e, mine.grid_n))
        el = el[[(e, n) not in used for e, n in zip(el.grid_e, el.grid_n)]]
        el = el[centre_dist(el, train) >= args.spacing].reset_index(drop=True)
        n_pool = len(el)
        rng = city_rng(cfg["seed"] + SEED_OFFSET, city.code)
        target = args.per_city if args.per_city else 10_000
        sel = select_tiles(el, target, 0, args.spacing, rng)
        report.append({"city": city.code, "eligible": n0, "after_block_excl": n_block, "after_buffer": n_buffer,
                       "pool_after_spacing_to_train": n_pool, "selectable": len(sel),
                       **{f"sel_{d}": int((sel.density == d).sum()) for d in ("low", "medium", "high")}})
        if args.write and len(sel):
            to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = to_ll.transform(((sel.xmin + sel.xmax) / 2).to_numpy(), ((sel.ymin + sel.ymax) / 2).to_numpy())
            sel = sel.drop(columns="geometry").assign(
                sample_id=[sample_id(city.code, e, n) for e, n in zip(sel.grid_e, sel.grid_n)],
                country=city.country, city=city.city, crs=crs, center_lat=lat, center_lon=lon, role="expansion_m3b")
            rows.append(sel)
    rep = pd.DataFrame(report)
    print(f"spacing {args.spacing:.0f} m, cap {args.per_city or 'none'}\n" + rep.to_string(index=False))
    print(f"TOTAL selectable new train tiles: {rep.selectable.sum()}")
    if args.write:
        cols = ["sample_id", "country", "city", "city_code", "role", "rank", "stratum", "density", "era", "p_inclusion",
                "grid_e", "grid_n", "crs", "xmin", "ymin", "xmax", "ymax", "center_lat", "center_lon",
                "ghsl_built_1975", "ghsl_built_2000", "ghsl_built_2020", "ghsl_nres_share_2020", "block_id"]
        out = pd.concat(rows)[cols]
        out.to_csv(PLAN_OUT, index=False)
        print(f"wrote {PLAN_OUT} ({len(out)} tiles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
