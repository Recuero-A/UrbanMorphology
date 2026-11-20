"""Milestone 15: sample-based fidelity and coverage of generated neighbourhoods against real ones.

Per-tile feature vectors (features OUTSIDE the generator's objective, docs/M15_METRICS.md) are standardised with the
REAL set's mean and standard deviation. With k-nearest-neighbour balls around real samples (radius = distance to the
k-th nearest real neighbour):
* precision (Kynkaenniemi et al., NeurIPS 2019): share of generated samples inside at least one real ball;
* recall:    share of real samples inside at least one generated ball (balls around generated samples);
* density (Naeem et al., ICML 2020): mean number of real balls containing a generated sample, divided by k;
* coverage (Naeem et al., ICML 2020): share of real samples whose ball contains at least one generated sample.
``real_vs_real`` gives the same numbers for two disjoint halves of the real set (the benchmark a generator is read
against).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def standardise(real: np.ndarray, *others: np.ndarray):
    mu, sd = np.nanmean(real, 0), np.nanstd(real, 0)
    sd = np.where(sd > 0, sd, 1.0)
    return [(x - mu) / sd for x in (real, *others)]


def _radii(X: np.ndarray, k: int) -> np.ndarray:
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, k - 1]


def prdc(real: np.ndarray, gen: np.ndarray, k: int = 5) -> dict:
    """Precision, recall, density and coverage. Rows with a non-finite feature are dropped (and counted)."""
    rk, gk = np.isfinite(real).all(1), np.isfinite(gen).all(1)
    R, G = standardise(real[rk], gen[gk])
    if len(R) <= k or len(G) <= k:
        return {"precision": np.nan, "recall": np.nan, "density": np.nan, "coverage": np.nan,
                "n_real": int(len(R)), "n_gen": int(len(G)), "dropped_real": int((~rk).sum()), "dropped_gen": int((~gk).sum())}
    rr, gr = _radii(R, k), _radii(G, k)
    D = cdist(R, G)                                   # real x generated
    inside = D <= rr[:, None]
    return {"precision": float(inside.any(0).mean()), "recall": float((D <= gr[None, :]).any(1).mean()),
            "density": float(inside.sum(0).mean() / k), "coverage": float(inside.any(1).mean()),
            "n_real": int(len(R)), "n_gen": int(len(G)), "dropped_real": int((~rk).sum()), "dropped_gen": int((~gk).sum())}


def real_vs_real(real: np.ndarray, groups: np.ndarray, k: int = 5, n_splits: int = 20, seed: int = 0) -> dict:
    """PRDC between random halves of the real set, split by GROUP (city or block) so near-duplicate neighbours do not
    sit on both sides; mean and range over ``n_splits`` splits."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    res = []
    for _ in range(n_splits):
        a = set(rng.choice(ug, len(ug) // 2, replace=False).tolist())
        m = np.array([x in a for x in groups])
        res.append(prdc(real[m], real[~m], k))
    keys = ("precision", "recall", "density", "coverage")
    return {k_: float(np.nanmean([r[k_] for r in res])) for k_ in keys} | \
           {f"{k_}_range": [float(np.nanmin([r[k_] for r in res])), float(np.nanmax([r[k_] for r in res]))] for k_ in keys}
