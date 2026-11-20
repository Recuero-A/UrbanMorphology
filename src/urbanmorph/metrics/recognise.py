"""Milestone 16 recognisability: a nearest-centroid style classifier fitted on REAL reference windows only
(docs/M16_PROTOCOL.md §5, docs/M16_METRICS.md §5).

Features = the union of the protocol's primary outcomes (graphs/network_outcomes), none of which a style rule optimises.
Each feature is standardised by the pooled within-style SD of the references; a missing value (e.g. no blocks) is
imputed with the reference median of that feature, so a graph is never classified on fewer features. A generated
network is "recognised" when its nearest style centroid is the requested style.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("block_rect_share", "stroke_xjunc_lw", "share_4way", "circuity_200_350", "block_logA_sd",
            "jangle_right_share", "share_3way", "block_phi_med", "spokes_long", "ring_cover_max", "sector_circ",
            "tree_len_share")
STYLES = ("grid", "organic", "radial", "culdesac")


@dataclass
class Centroids:
    features: tuple
    centre: dict          # style -> standardised centroid
    med: np.ndarray       # imputation values
    sd: np.ndarray        # pooled within-style SD

    def _z(self, X):
        X = np.where(np.isfinite(X), X, self.med)
        return X / self.sd

    def predict(self, rows) -> list[str]:
        X = self._z(matrix(rows, self.features))
        st = list(self.centre)
        C = np.stack([self.centre[s] for s in st])
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        return [st[k] for k in d.argmin(1)]


def matrix(rows, features=FEATURES) -> np.ndarray:
    return np.array([[float(r.get(f)) if r.get(f) is not None else np.nan for f in features] for r in rows], float)


def fit(rows, styles, features=FEATURES) -> Centroids:
    X = matrix(rows, features)
    styles = np.asarray(styles)
    med = np.nanmedian(X, 0)
    X = np.where(np.isfinite(X), X, med)
    res = np.concatenate([X[styles == s] - X[styles == s].mean(0) for s in STYLES if (styles == s).any()])
    sd = np.sqrt((res ** 2).sum(0) / max(len(res) - len(set(styles)), 1))
    sd = np.where(sd > 1e-9, sd, 1.0)
    return Centroids(tuple(features), {s: (X[styles == s] / sd).mean(0) for s in STYLES if (styles == s).any()}, med, sd)


def loo_accuracy(rows, styles, features=FEATURES) -> float:
    """Leave-one-out accuracy on the references (the classifier's own sanity check)."""
    ok = []
    for k in range(len(rows)):
        keep = [j for j in range(len(rows)) if j != k]
        c = fit([rows[j] for j in keep], [styles[j] for j in keep], features)
        ok.append(c.predict([rows[k]])[0] == styles[k])
    return float(np.mean(ok))
