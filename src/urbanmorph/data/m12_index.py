"""Milestone 12 tile index: the M2 + M3b tiles plus the terrain-diverse expansion (scripts/m12_build_tiles.py).

Splits:
* ``train`` / ``val`` / ``test``: the M2 splits, unchanged (``test`` is never returned unless asked for);
* ``train`` (M12): steep Los Angeles tiles selected by measured landform slope;
* ``val_t``: San Diego terrain-validation tiles (a held-out city, development use);
* ``test_t``: San Diego terrain-test tiles. They are not built before the final, authorised evaluation and are
  never returned unless ``include_test=True``.
"""

from __future__ import annotations

import os

import pandas as pd

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion

M12_INDEX = PROJECT_ROOT / "data" / "processed" / "m12" / "samples_m12.csv"
DEV_SPLITS = ("train", "val", "val_t")
TEST_SPLITS = ("test", "test_t")


def test_authorised() -> bool:
    """True only inside the authorised final test run (scripts/m12_final_test.sh sets M12_INCLUDE_TEST=yes)."""
    return os.environ.get("M12_INCLUDE_TEST") == "yes"


def active_splits() -> tuple:
    return DEV_SPLITS + (TEST_SPLITS if test_authorised() else ())


def load_index_m12(include_test: bool | None = None) -> pd.DataFrame:
    include_test = test_authorised() if include_test is None else include_test
    base = load_index_with_expansion()
    parts = [base[base.in_dataset]]
    if M12_INDEX.exists():
        parts.append(pd.read_csv(M12_INDEX))
    idx = pd.concat(parts, ignore_index=True)
    if not include_test:
        idx = idx[idx.split.isin(DEV_SPLITS)]
    return idx.reset_index(drop=True)
