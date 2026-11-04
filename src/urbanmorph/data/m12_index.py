"""Milestone 12 tile index: the M2 + M3b tiles plus the terrain-diverse expansion (scripts/m12_build_tiles.py).

Splits:
* ``train`` / ``val`` / ``test``: the M2 splits, unchanged (``test`` is never returned unless asked for);
* ``train`` (M12): steep Los Angeles tiles selected by measured landform slope;
* ``val_t``: San Diego terrain-validation tiles (a held-out city, development use);
* ``test_t``: San Diego terrain-test tiles. They are not built before the final, authorised evaluation and are
  never returned unless ``include_test=True``.
"""

from __future__ import annotations

import pandas as pd

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion

M12_INDEX = PROJECT_ROOT / "data" / "processed" / "m12" / "samples_m12.csv"
DEV_SPLITS = ("train", "val", "val_t")


def load_index_m12(include_test: bool = False) -> pd.DataFrame:
    base = load_index_with_expansion()
    parts = [base[base.in_dataset]]
    if M12_INDEX.exists():
        parts.append(pd.read_csv(M12_INDEX))
    idx = pd.concat(parts, ignore_index=True)
    if not include_test:
        idx = idx[idx.split.isin(DEV_SPLITS)]
    return idx.reset_index(drop=True)
