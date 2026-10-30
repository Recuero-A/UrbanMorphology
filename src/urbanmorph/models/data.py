"""PyTorch access to the Milestone 2 dataset (uses the existing loaders and splits)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from urbanmorph.data.m2_dataset import load_index, load_m2_sample


def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """One of the 8 symmetries of the square: k in 0..7 (rotation k%4 x 90 deg, mirrored if k>=4).

    Works on (..., H, W) tensors. A rotated or mirrored neighbourhood is still a
    physically valid layout, but absolute (north-up) orientation is lost.
    """
    if k >= 4:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, k % 4, dims=(-2, -1))


class LayoutDataset(Dataset):
    """All tiles of one split, loaded into memory once (135 tiles x 48 KB)."""

    def __init__(self, split: str, index: pd.DataFrame | None = None, augment: bool = False, seed: int = 0):
        index = load_index() if index is None else index
        rows = index[(index["in_dataset"]) & (index["split"] == split)].sort_values("sample_id")
        if rows.empty:
            raise ValueError(f"no tiles in split {split!r}")
        self.rows = rows.reset_index(drop=True)
        arrays = [load_m2_sample(sid, index)[0] for sid in self.rows["sample_id"]]  # validated
        self.x = torch.from_numpy(np.stack(arrays).astype(np.float32))
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        x = self.x[i]
        if self.augment:
            x = dihedral(x, int(self.rng.integers(8)))
        return x, i

    def pos_fraction(self) -> np.ndarray:
        """Fraction of positive pixels per channel (used for loss weighting)."""
        return self.x.mean(dim=(0, 2, 3)).numpy()
