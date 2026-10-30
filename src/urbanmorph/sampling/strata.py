"""Operational strata from GHSL (no subjective categories).

* density:  GHSL 2020 built-up fraction of the tile, cut at absolute
            thresholds shared by all cities -> "low" | "medium" | "high".
* era:      first GHSL epoch at which the tile's built-up fraction reached
            ``era_builtup_fraction`` -> "by1975" | "1975_2000" | "2000_2020".
            This is the era of *urbanisation of the tile* as seen from
            satellites, NOT the age of individual buildings, and GHSL starts in
            1975, so "by1975" lumps medieval cores with 1960s estates.
* nres_share: non-residential share of built-up surface in 2020 (covariate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from urbanmorph.sampling import ghsl

DENSITY_LABELS = ("low", "medium", "high")
ERA_LABELS = ("by1975", "1975_2000", "2000_2020")


def add_ghsl_columns(cands: pd.DataFrame, code: str, bounds_wgs84, crs: str, cfg: dict) -> pd.DataFrame:
    out = cands.copy()
    bb = out[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    for ep in cfg["study_area"]["ghsl_epochs"]:
        out[f"ghsl_built_{ep}"] = ghsl.builtup_fraction(ghsl.city_extract(code, bounds_wgs84, ep), bb, crs)
    ep = cfg["study_area"]["ghsl_nres_epoch"]
    nres = ghsl.builtup_fraction(ghsl.city_extract(code, bounds_wgs84, ep, nres=True), bb, crs)
    total = out[f"ghsl_built_{ep}"].to_numpy()
    out[f"ghsl_nres_share_{ep}"] = np.where(total > 0, np.clip(nres / np.maximum(total, 1e-9), 0, 1), np.nan)
    return out


def density_class(frac, thresholds) -> np.ndarray:
    lo, hi = thresholds
    frac = np.asarray(frac, dtype=float)
    return np.where(frac < lo, "low", np.where(frac < hi, "medium", "high"))


def era_class(df: pd.DataFrame, epochs, threshold: float) -> np.ndarray:
    """First epoch reaching ``threshold``; epochs must be e.g. [1975, 2000, 2020]."""
    e0, e1, e2 = epochs
    by0 = df[f"ghsl_built_{e0}"].to_numpy() >= threshold
    by1 = df[f"ghsl_built_{e1}"].to_numpy() >= threshold
    return np.where(by0, f"by{e0}", np.where(by1, f"{e0}_{e1}", f"{e1}_{e2}"))


def weighted_quantiles(values, weights, qs) -> list[float]:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(q, cw, v)) for q in qs]


def calibrate_density_thresholds(eligible_by_city: dict[str, pd.DataFrame], col: str) -> list[float]:
    """Pooled terciles of eligible cells with every city weighted equally.

    Equal city weighting stops large cities (Lima: thousands of cells) from
    defining "high density" for everyone.
    """
    vals, wts = [], []
    for df in eligible_by_city.values():
        vals.append(df[col].to_numpy())
        wts.append(np.full(len(df), 1.0 / len(df)))
    return [round(q, 3) for q in weighted_quantiles(np.concatenate(vals), np.concatenate(wts), [1 / 3, 2 / 3])]
