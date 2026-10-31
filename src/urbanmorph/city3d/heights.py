"""Building heights with provenance (docs/M11_3D_ARCHITECTURE.md §4.1).

Priority: observed OSM ``height`` -> derived from ``building:levels`` x city storey height -> synthetic.
Synthetic floors are drawn per BLOCK so neighbours are correlated: each block gets a quantile u (the mean
quantile of its buildings with known floors when there are any, else uniform), each building u + N(0, 0.08),
read off the empirical floor quantiles of its footprint-area bin (outputs/m11/audit/height_priors.json,
training tiles only).
"""

from __future__ import annotations

import numpy as np

AREA_BINS = [60.0, 150.0, 400.0, 1500.0]          # m2: bins <60, 60-150, 150-400, 400-1500, >=1500
QS = [round(x, 3) for x in np.linspace(0, 1, 101)]   # percentiles: coarser grids stretch the tail (see U_CLIP)
DEFAULT_STOREY_M = 3.0                            # used where a city has < MIN_BOTH buildings with both tags
MIN_BOTH = 500
U_SD = 0.08                                       # within-block spread of the building quantile (default; the
                                                  # value calibrated on training tiles is priors["u_sd"])
U_CLIP = (0.01, 0.99)                             # never sample the extreme tails: a single 29-floor outlier in
                                                  # the training tiles would otherwise set whole blocks


def storey_table(audit: dict) -> dict:
    """City -> storey height (m) with its source, from outputs/m11/audit/height_audit.json."""
    out = {}
    for city, v in audit["cities"].items():
        s = v.get("storey_height_m_from_both")
        if s is not None and v["n_both"] >= MIN_BOTH:
            out[city] = {"value": round(s["p50"], 2), "source": f"median height/levels of {v['n_both']} {city} buildings"}
        else:
            out[city] = {"value": DEFAULT_STOREY_M, "source": f"default {DEFAULT_STOREY_M} m ({v['n_both']} {city} buildings with both tags < {MIN_BOTH})"}
    return out


def floor_quantiles(priors: dict, key: str, area: float) -> np.ndarray:
    b = str(int(np.searchsorted(AREA_BINS, area, side="right")))
    p = priors["priors"][key]
    if b not in p or p[b]["n"] < 30:                   # sparse bin: nearest populated bin
        cand = sorted((abs(int(k) - int(b)), k) for k, v in p.items() if v["n"] >= 30)
        b = cand[0][1]
    return np.asarray(p[b]["q"])


def quantile_of(q: np.ndarray, floors: float) -> float:
    return float(np.interp(floors, q, np.linspace(0, 1, len(q)), left=0.0, right=1.0))


def assign_heights(buildings: list[dict], priors: dict, key: str, storey_m: float, rng, noise: bool = True) -> None:
    """In place. Each building dict has ``_poly``, ``block``, optional ``_obs_height`` / ``_obs_levels``
    (NaN when absent); sets height_m, height_source, floors, floors_source."""
    for b in buildings:
        h, lv = b.get("_obs_height", np.nan), b.get("_obs_levels", np.nan)
        if np.isfinite(h) and 2.0 <= h <= 400:
            b.update(height_m=float(h), height_source="observed",
                     floors=int(lv) if np.isfinite(lv) else max(1, int(round(h / storey_m))),
                     floors_source="observed" if np.isfinite(lv) else "derived")
        elif np.isfinite(lv):
            b.update(height_m=float(lv * storey_m), height_source="derived", floors=int(lv), floors_source="observed")
    blocks = {}
    for b in buildings:
        blocks.setdefault(b["block"], []).append(b)
    for members in blocks.values():
        known = [quantile_of(floor_quantiles(priors, key, b["_poly"].area), b["floors"]) for b in members if "floors" in b]
        u0 = float(np.mean(known)) if known else float(rng.uniform(*U_CLIP))
        for b in members:
            if "floors" in b:
                continue
            u = float(np.clip(u0 + (rng.normal(0, priors.get("u_sd", U_SD)) if noise else 0.0), *U_CLIP))
            q = floor_quantiles(priors, key, b["_poly"].area)
            fl = max(1, int(round(np.interp(u, np.linspace(0, 1, len(q)), q))))
            if b["_poly"].area < 40:
                fl = min(fl, 2)
            b.update(floors=fl, floors_source="synthetic", height_m=float(fl * storey_m), height_source="synthetic")
