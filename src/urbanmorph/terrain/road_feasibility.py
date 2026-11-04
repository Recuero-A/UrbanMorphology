"""Terrain feasibility of a proposed street segment (Milestone 12). Independent of any neural network.

``evaluate(terrain, a, b, cls, th)`` samples the terrain along the WHOLE segment (every 2 m) and returns a
``Feasibility`` with one of:

* ``ground``                 every 30 m sub-span grade <= the class limit, no water (30 m: see the calibration);
* ``excessive_grade``        too steep at ground level and no viaduct is possible;
* ``possible_viaduct``       too steep (or a dip) at ground level, but a straight deck between the end points has an
                             acceptable grade, clears the lowest ground by >= viaduct_clearance_min and spans
                             <= viaduct_span_max;
* ``possible_bridge``        crosses a river / canal; wet length <= the class span limit, crossing angle >= limit,
                             deck grade between the banks acceptable;
* ``impossible_crossing``    crosses a river but fails a bridge condition;
* ``water_obstruction``      runs into a lake or the sea beyond the (short) lake span limit;
* ``ends_in_water``          the end point is in water (the arm needs a longer, bridging step);
* ``insufficient_information`` no terrain under part of the segment.

Thresholds (``Thresholds``) come from TRAINING data where the data contain the structure (street grades per class,
bridge spans; scripts/m12_calibrate_feasibility.py) and from stated engineering assumptions where they do not
(viaducts: our data contain no valley-spanning bridges). Their source is recorded with them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from urbanmorph.terrain.heightfield import Terrain


@dataclass
class Thresholds:
    grade_max: dict = field(default_factory=lambda: {0: 0.08, 1: 0.10, 2: 0.15, 3: 0.20})
    bridge_span_max: dict = field(default_factory=lambda: {0: 400.0, 1: 200.0, 2: 80.0, 3: 40.0})
    bridge_angle_min_deg: float = 30.0
    lake_span_max: float = 30.0
    viaduct_clearance_min: float = 10.0
    viaduct_span_max: float = 300.0
    source: dict = field(default_factory=lambda: {"all": "defaults (not calibrated)"})

    @classmethod
    def load(cls, path) -> "Thresholds":
        d = json.loads(open(path).read())
        for k in ("grade_max", "bridge_span_max"):
            d[k] = {int(c): float(v) for c, v in d[k].items()}
        return cls(**d)

    def save(self, path) -> None:
        open(path, "w").write(json.dumps(asdict(self), indent=1))


@dataclass
class Feasibility:
    kind: str
    max_grade: float = np.nan
    mean_grade: float = np.nan
    dz: float = np.nan
    roughness: float = np.nan        # std of the second difference of the profile (profile smoothness)
    wet_length: float = 0.0
    water_kinds: tuple = ()
    crossing_angle_deg: float = np.nan
    clearance: float = np.nan
    deck_grade: float = np.nan

    @property
    def buildable(self) -> bool:
        return self.kind in ("ground", "possible_bridge", "possible_viaduct")

    @property
    def structure(self) -> int:
        """0 ground, 1 bridge (river bridge or viaduct): what the street would be built as."""
        return 1 if self.kind in ("possible_bridge", "possible_viaduct") else 0


def water_kind_at(t: Terrain, x, y) -> np.ndarray:
    kind = getattr(t, "water_kind", None)
    if kind is None:
        kind = t.water.astype(np.int8) if t.water is not None else None
    if kind is None:
        return np.zeros(np.shape(x), np.int8)
    r, c = t.rc(x, y)
    r, c = np.round(r).astype(int), np.round(c).astype(int)
    ok = (r >= 0) & (c >= 0) & (r < kind.shape[0]) & (c < kind.shape[1])
    out = np.zeros(np.shape(x), np.int8)
    out[ok] = kind[r[ok], c[ok]]
    return out


def river_half_width(t: Terrain) -> np.ndarray:
    """Distance (m) from each water cell to the nearest land: twice its maximum across a crossing ~ river width."""
    if getattr(t, "_half_width", None) is None:
        w = water_kind_at_grid(t) > 0
        t._half_width = ndimage.distance_transform_edt(w) * t.res
    return t._half_width


def water_kind_at_grid(t: Terrain) -> np.ndarray:
    k = getattr(t, "water_kind", None)
    return k if k is not None else (t.water.astype(np.int8) if t.water is not None else np.zeros(t.shape, np.int8))


def evaluate(t: Terrain, a, b, cls: int, th: Thresholds, step: float = 2.0, window: float = 30.0) -> Feasibility:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d, z = t.profile(a, b, step)
    if not np.isfinite(z).all() or len(d) < 2:
        return Feasibility("insufficient_information")
    L = d[-1]
    frac = d / max(L, 1e-9)
    p = a + frac[:, None] * (b - a)
    wk = water_kind_at(t, p[:, 0], p[:, 1])
    k = max(1, int(round(window / step)))
    g = np.abs(z[k:] - z[:-k]) / (d[k:] - d[:-k]) if len(d) > k else np.abs(np.diff(z)) / np.diff(d)
    rough = float(np.std(np.diff(z, 2))) if len(z) > 2 else 0.0
    f = Feasibility("ground", float(g.max()), float(abs(z[-1] - z[0]) / max(L, 1e-9)), float(z[-1] - z[0]), rough)
    if wk.any():
        wet = wk > 0
        f.wet_length = float(wet.sum() * step)
        f.water_kinds = tuple(sorted(set(int(v) for v in wk[wet])))
        if wet[-1]:
            f.kind = "ends_in_water"
            return f
        land = np.flatnonzero(~wet)
        i0 = land[land < np.flatnonzero(wet)[0]].max() if (land < np.flatnonzero(wet)[0]).any() else 0
        i1 = land[land > np.flatnonzero(wet)[-1]].min()
        f.deck_grade = float(abs(z[i1] - z[i0]) / max(d[i1] - d[i0], 1e-9))
        if any(v in (2, 3) for v in f.water_kinds) and f.wet_length > th.lake_span_max:
            f.kind = "water_obstruction"
            return f
        hw = river_half_width(t)
        r, c = t.rc(p[wet, 0], p[wet, 1])
        r = np.clip(np.round(r).astype(int), 0, hw.shape[0] - 1)
        c = np.clip(np.round(c).astype(int), 0, hw.shape[1] - 1)
        width = 2 * float(hw[r, c].max())
        f.crossing_angle_deg = float(np.degrees(np.arcsin(np.clip(width / max(f.wet_length, 1e-9), 0, 1))))
        ok = (f.wet_length <= th.bridge_span_max[cls] and f.crossing_angle_deg >= th.bridge_angle_min_deg
              and f.deck_grade <= th.grade_max[cls])
        f.kind = "possible_bridge" if ok else "impossible_crossing"
        return f
    if f.max_grade <= th.grade_max[cls]:
        return f
    deck = z[0] + frac * (z[-1] - z[0])
    f.clearance = float((deck - z).max())
    f.deck_grade = f.mean_grade
    if f.deck_grade <= th.grade_max[cls] and f.clearance >= th.viaduct_clearance_min and L <= th.viaduct_span_max:
        f.kind = "possible_viaduct"
    else:
        f.kind = "excessive_grade"
    return f


def far_bank(t: Terrain, a, direction, max_len: float, step: float = 2.0):
    """First dry point beyond the water ahead of ``a`` along ``direction`` (unit), or None within ``max_len``."""
    a, u = np.asarray(a, float), np.asarray(direction, float)
    s = np.arange(step, max_len + step, step)
    p = a + s[:, None] * u
    wk = water_kind_at(t, p[:, 0], p[:, 1])
    wet = np.flatnonzero(wk > 0)
    if not len(wet):
        return None
    dry = np.flatnonzero((wk == 0) & (np.arange(len(s)) > wet[0]))
    return p[dry[0]] + 2 * step * u if len(dry) else None


def far_rim(t: Terrain, a, direction, cls: int, th: Thresholds, min_len: float = 40.0, step: float = 4.0):
    """For a step that drops into a dry valley: the first point ahead (up to th.viaduct_span_max) that a straight
    deck from ``a`` can reach within the class grade while clearing the ground below by viaduct_clearance_min.
    Rule-based (Milestone 12 exploratory HRV policy); None if there is none."""
    a, u = np.asarray(a, float), np.asarray(direction, float)
    s = np.arange(step, th.viaduct_span_max + step, step)
    p = a + s[:, None] * u
    z = t.sample(p[:, 0], p[:, 1])
    z0 = t.sample(a[:1], a[1:2])[0]
    if not np.isfinite(z0):
        return None
    wk = water_kind_at(t, p[:, 0], p[:, 1])
    dipped = False
    for i in range(len(s)):
        if not np.isfinite(z[i]) or wk[i] > 0:
            return None
        dipped |= z0 - z[i] >= th.viaduct_clearance_min           # the ground has fallen away: a valley
        if dipped and s[i] >= min_len and abs(z[i] - z0) / s[i] <= th.grade_max[cls]:
            deck = z0 + (z[i] - z0) * s[:i + 1] / s[i]
            if np.nanmax(deck - z[:i + 1]) >= th.viaduct_clearance_min:
                return p[i] + step * u                            # land just beyond the rim
    return None
