"""Geographic train / validation / test splitting.

Why not split tiles at random? Nearby tiles share streets, blocks and building
types (spatial autocorrelation). A random split puts near-copies of test tiles
in the training set, so test scores measure memorisation of places, not
generalisation to new ones.

Procedure (per city, deterministic):
1. Group tiles into square spatial BLOCKS on the city's UTM grid
   (``block_size_m``, default 3072 m = 6 x 6 tiles). A block is assigned whole.
2. Shuffle blocks with a seeded RNG and assign each to the split whose sample
   count is furthest below its target fraction (val/test are filled first
   when tied, so they are never empty if there are >= 3 blocks).
3. BUFFER: any two tiles in different splits must be at least ``buffer_m``
   apart edge to edge. Violations are resolved by excluding the tile of the
   lower-priority split (train < val < test), so evaluation sets stay intact.

A second, independent scheme supports CITY-HELD-OUT evaluation: train on all
cities except one, test on the held-out city (``city_holdout``). That measures
geographic generalisation to an unseen city; it is a different question from
classifying the country of a tile, where the country label could be predicted
from city-specific cues.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
PRIORITY = {"train": 0, "val": 1, "test": 2}
EXCLUDED = "excluded_buffer"


def block_ids(df: pd.DataFrame, block_size_m: float) -> pd.Series:
    cx = (df["xmin"] + df["xmax"]) / 2
    cy = (df["ymin"] + df["ymax"]) / 2
    bx = np.floor(cx / block_size_m).astype(int)
    by = np.floor(cy / block_size_m).astype(int)
    return df["city_code"] + "_b" + bx.astype(str) + "_" + by.astype(str)


def edge_distance(a: pd.Series, b: pd.Series) -> float:
    """Edge-to-edge distance between two axis-aligned tiles in the same CRS."""
    dx = max(0.0, max(a["xmin"], b["xmin"]) - min(a["xmax"], b["xmax"]))
    dy = max(0.0, max(a["ymin"], b["ymin"]) - min(a["ymax"], b["ymax"]))
    return float(np.hypot(dx, dy))


def assign_splits(df: pd.DataFrame, fractions: dict, block_size_m: float, seed: int) -> pd.DataFrame:
    out = df.copy()
    out["block_id"] = block_ids(out, block_size_m)
    out["split"] = ""
    for code, grp in out.groupby("city_code"):
        rng = np.random.default_rng([seed, zlib.crc32(code.encode())])
        blocks = sorted(grp["block_id"].unique())
        blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        sizes = grp["block_id"].value_counts()
        total = len(grp)
        counts = {s: 0 for s in SPLITS}
        for b in blocks:
            deficit = {s: fractions[s] * total - counts[s] for s in SPLITS}
            # val/test first on ties so they get at least one block
            target = max(SPLITS, key=lambda s: (deficit[s], PRIORITY[s]))
            out.loc[out["block_id"] == b, "split"] = target
            counts[target] += int(sizes[b])
    return out


def enforce_buffer(df: pd.DataFrame, buffer_m: float) -> pd.DataFrame:
    out = df.copy()
    for code, grp in out.groupby("city_code"):
        idx = list(grp.index)
        for i_pos, i in enumerate(idx):
            for j in idx[i_pos + 1:]:
                si, sj = out.at[i, "split"], out.at[j, "split"]
                if si == sj or EXCLUDED in (si, sj):
                    continue
                if edge_distance(out.loc[i], out.loc[j]) < buffer_m:
                    loser = i if PRIORITY[si] < PRIORITY[sj] else j
                    out.at[loser, "split"] = EXCLUDED
    return out


def split_violations(df: pd.DataFrame, buffer_m: float) -> list[tuple[str, str, float]]:
    """Pairs of tiles in different (non-excluded) splits closer than ``buffer_m``."""
    bad = []
    active = df[df["split"].isin(SPLITS)]
    for code, grp in active.groupby("city_code"):
        rows = list(grp.iterrows())
        for k, (_, a) in enumerate(rows):
            for _, b in rows[k + 1:]:
                if a["split"] != b["split"]:
                    d = edge_distance(a, b)
                    if d < buffer_m:
                        bad.append((a["sample_id"], b["sample_id"], d))
    return bad


def city_holdout(df: pd.DataFrame, held_out_city_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) for a leave-one-city-out experiment over usable samples."""
    usable = df[df["usable"]] if "usable" in df else df
    return usable[usable["city_code"] != held_out_city_code], usable[usable["city_code"] == held_out_city_code]
