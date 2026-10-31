"""Milestone 8, Experiment S: a library of REAL building footprint shapes.

A footprint is (minimum rotated rectangle, 20 ray fractions; layout/objects.py). Instead of sampling
the 20 ray fractions independently (which produced spiky shapes: 21 % of B1 frontage proposals were
rejected for an angle < 45 deg), a building picks one of K library shapes, which is then stretched to
its own length and width (scale + aspect).

Library (fitted on TRAIN footprints only; PROGRAMMED choices documented here):
  * index 0 = the plain rectangle (22-53 % of real footprints are exactly their rectangle);
  * the other K - 1 = medoids (real footprints, not averages) of k-means clusters of the log ray
    fractions of non-rectangular footprints, k = 2 (K - 1); clusters whose medoid has an interior angle
    < 45 deg at aspect 1, 1.5 or 2.5 (or is within 0.95 of a rectangle) are dropped, and the K - 1 most
    populous remaining clusters are kept;
  * each shape is usable in its 4 frame symmetries (identity, mirror, half turn, both), so the model
    chooses among 4 K classes; a real footprint is labelled with its nearest class (squared log-ratio
    distance) in the street-canonical frame.
"""

from __future__ import annotations

import numpy as np
import torch

from urbanmorph.layout.frontage import interior_angles
from urbanmorph.layout.objects import decode
from urbanmorph.layout.placer_data import _HALF_TURN_RAYS, _MIRROR_RAYS

K_DEFAULT = 100
RECT_MIN_RATIO = 0.95
_VARIANTS = [np.arange(20), _MIRROR_RAYS, _HALF_TURN_RAYS, _MIRROR_RAYS[_HALF_TURN_RAYS]]


def _log(r):
    return np.log(np.clip(np.asarray(r, float), 1e-3, None))


def kmeans(x, k, seed=0, iters=50):
    """Plain k-means with k-means++ initialisation. Returns (centres, labels)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32)
    c = x[torch.randint(len(x), (1,), generator=g)]
    for _ in range(k - 1):
        d = torch.cdist(x, c).min(1).values ** 2
        c = torch.cat([c, x[torch.multinomial(d / d.sum(), 1, generator=g)]])
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        c = torch.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c.numpy(), torch.cdist(x, c).argmin(1).numpy()


def acute_free(ratios, aspect) -> bool:
    return interior_angles(decode(np.zeros(2), 0.0, 20.0, 20.0 / aspect, ratios)).min() >= 45


class ShapeLibrary:
    def __init__(self, shapes: np.ndarray, counts: np.ndarray):
        self.shapes = np.asarray(shapes, np.float32)          # (K, 20) ray fractions
        self.counts = np.asarray(counts)
        self.variants = np.stack([self.shapes[:, v] for v in _VARIANTS], 1).reshape(-1, 20)   # (4K, 20); class = 4 k + v
        self._lv = _log(self.variants)

    @property
    def n_classes(self) -> int:
        return len(self.variants)

    @classmethod
    def fit(cls, ratios: np.ndarray, aspect: np.ndarray, k: int = K_DEFAULT, seed: int = 0) -> "ShapeLibrary":
        """``aspect``: length / width of each footprint."""
        ratios = np.asarray(ratios)
        rect = ratios.min(1) >= RECT_MIN_RATIO
        nr, asp = ratios[~rect], np.asarray(aspect)[~rect]
        lt = _log(nr)
        cen, lab = kmeans(lt, 2 * (k - 1), seed)
        cand = []
        for c in range(len(cen)):
            m = np.flatnonzero(lab == c)
            if not len(m):
                continue
            j = m[np.argmin(((lt[m] - cen[c]) ** 2).sum(1))]
            med = nr[j]
            if med.min() < RECT_MIN_RATIO and acute_free(med, asp[j]):
                cand.append((len(m), med))
        cand.sort(key=lambda x: -x[0])
        cand = cand[:k - 1]
        shapes = np.r_[np.ones((1, 20)), np.stack([s for _, s in cand])]
        counts = np.r_[int(rect.sum()), [n for n, _ in cand]]
        return cls(shapes, counts)

    def nearest(self, ratios: np.ndarray) -> np.ndarray:
        """Class index (4 k + variant) of the nearest library shape for each row of ray fractions."""
        r = np.atleast_2d(ratios)
        lr = _log(r)
        d = (lr ** 2).sum(1)[:, None] - 2 * lr @ self._lv.T + (self._lv ** 2).sum(1)[None]
        out = d.argmin(1)
        out[r.min(1) >= RECT_MIN_RATIO] = 0
        return out

    def ratios(self, cls_idx) -> np.ndarray:
        return self.variants[np.asarray(cls_idx)]

    def unbounded(self, cls_idx) -> np.ndarray:
        """Ray logits (placer_data.shape_from_unbounded inverse) that decode exactly to the library shape."""
        r = np.clip((self.ratios(cls_idx) - 0.05) / 0.95, 1e-6, 1 - 1e-6)
        return np.log(r / (1 - r)).astype(np.float32)

    def save(self, path):
        np.savez(path, shapes=self.shapes, counts=self.counts)

    @classmethod
    def load(cls, path) -> "ShapeLibrary":
        z = np.load(path)
        return cls(z["shapes"], z["counts"])
