"""Stratified, spatially separated random selection of tiles within a city.

Procedure (deterministic given the seed and the city code):

1. Allocation: every non-empty stratum gets one sample (if there are no more
   strata than samples); the rest is allocated proportionally to stratum size
   (largest-remainder rounding), never exceeding the stratum size.
2. Each stratum's candidates are put in a seeded random order.
3. Round-robin over strata (fixed order): take the next candidate whose centre
   is at least ``min_dist`` from every already accepted tile. If a stratum runs
   out of admissible candidates, its remaining quota is reported as shortfall
   and moved to the largest stratum that still has candidates.
4. Reserves are drawn the same way, after the primaries, and are used in rank
   order (same stratum first) to replace primaries that fail quality checks.

Inclusion probability is approximated by n_h / N_h. The distance constraint
makes the true probabilities slightly non-uniform (tiles in sparse areas are
favoured); this is recorded as a limitation.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd


def city_rng(seed: int, code: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(code.encode())])


def allocate(stratum_sizes: dict[str, int], n: int) -> dict[str, int]:
    sizes = {k: v for k, v in sorted(stratum_sizes.items()) if v > 0}
    if not sizes:
        return {}
    total = min(n, sum(sizes.values()))
    alloc = {k: 0 for k in sizes}
    if len(sizes) <= total:
        alloc = {k: 1 for k in sizes}
    remaining = total - sum(alloc.values())
    if remaining > 0:
        cap = {k: sizes[k] - alloc[k] for k in sizes}
        weight = sum(sizes.values())
        raw = {k: remaining * sizes[k] / weight for k in sizes}
        add = {k: min(int(np.floor(raw[k])), cap[k]) for k in sizes}
        left = remaining - sum(add.values())
        for k in sorted(sizes, key=lambda k: (-(raw[k] - np.floor(raw[k])), k)):
            if left == 0:
                break
            if add[k] < cap[k]:
                add[k] += 1
                left -= 1
        for k in sorted(sizes, key=lambda k: -cap[k]):  # leftovers caused by caps
            while left > 0 and add[k] < cap[k]:
                add[k] += 1
                left -= 1
        alloc = {k: alloc[k] + add[k] for k in sizes}
    return alloc


def select_tiles(
    cands: pd.DataFrame,
    n: int,
    n_reserve: int,
    min_dist: float,
    rng: np.random.Generator,
    stratum_col: str = "stratum",
) -> pd.DataFrame:
    """Return the selected rows with columns role, rank, p_inclusion."""
    cx = ((cands["xmin"] + cands["xmax"]) / 2).to_numpy()
    cy = ((cands["ymin"] + cands["ymax"]) / 2).to_numpy()
    strata = cands[stratum_col].to_numpy()
    sizes = pd.Series(strata).value_counts().to_dict()
    alloc = allocate(sizes, n)

    queues = {h: list(rng.permutation(np.flatnonzero(strata == h))) for h in sorted(sizes)}
    accepted: list[int] = []
    roles: list[str] = []

    def admissible(i: int) -> bool:
        if not accepted:
            return True
        a = np.asarray(accepted)
        return bool(np.min(np.hypot(cx[a] - cx[i], cy[a] - cy[i])) >= min_dist)

    def take(h: str) -> bool:
        q = queues[h]
        while q:
            i = q.pop(0)
            if admissible(i):
                accepted.append(i)
                return True
        return False

    quota = dict(alloc)
    shortfall = {}
    while any(quota.values()):
        progressed = False
        for h in sorted(quota):
            if quota[h] == 0:
                continue
            if take(h):
                quota[h] -= 1
                roles.append("primary")
                progressed = True
            else:
                shortfall[h] = shortfall.get(h, 0) + quota[h]
                moved, quota[h] = quota[h], 0
                donors = [k for k in sorted(queues, key=lambda k: -len(queues[k])) if queues[k]]
                if donors:
                    quota[donors[0]] = quota.get(donors[0], 0) + moved
                    progressed = True
        if not progressed:
            break

    n_primary = len(accepted)
    order = sorted(queues, key=lambda h: -sizes[h])
    while len(accepted) - n_primary < n_reserve and any(queues.values()):
        for h in order:
            if len(accepted) - n_primary >= n_reserve:
                break
            if take(h):
                roles.append("reserve")

    out = cands.iloc[accepted].copy()
    out["role"] = roles
    out["rank"] = np.arange(len(out))
    realised = pd.Series(out.loc[out["role"] == "primary", stratum_col]).value_counts().to_dict()
    out["p_inclusion"] = [realised.get(h, 0) / sizes[h] for h in out[stratum_col]]
    out.attrs["allocation"] = alloc
    out.attrs["shortfall"] = shortfall
    out.attrs["stratum_sizes"] = sizes
    return out
