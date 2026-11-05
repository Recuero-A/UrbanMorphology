"""Milestone 13: hard constraints and soft log-potentials for terrain-constrained sampling (docs/M13_METHOD.md §3).

Everything is evaluated on the COMMITTED edge, i.e. the geometry ``GenState.resolve_step`` (ground) or
``resolve_structure`` (bridge / tunnel) says would be inserted, never on the raw sampled step.

Hard constraints (log-potential -inf) - genuine impossibilities or explicit representation restrictions only:
* ``unrepresentable``     the geometric rules drop the step (shorter than 8 m after clipping / merging, node budget);
* ``ground_in_water``     a ground edge whose profile passes over water (river, lake or sea): it needs a structure;
* ``ends_in_water``       any edge ending in water;
* ``unsupported_structure`` a bridge over neither water nor a street (viaducts only in the exploratory mode), a tunnel
                          without >= 3 m of cover over at least half its length;
* ``sea_crossing``        a bridge over sea (no sea bridges in the training data);
* ``span_exceeds_max``    a structure longer than the longest real one of its kind in the training data;
* ``degenerate_angle``    a water crossing at an angle below the smallest real one.
Steep grades are NOT hard: real streets are built on steep land (switchbacks, cut and fill).

Soft log-potentials (log w <= 0 only for structures; ratio potentials can be > 0):
* ground edges: log r_grade(g | class, slope bin) + log r_align(|cos| | slope bin), density ratios of the REAL training
  distribution to the model's own unconstrained proposals on training tiles (configs/m13_potentials/<model>.json);
  E_p[r] = 1 under the proposal, so the correction has no systematic bias toward shorter networks;
* bridges / tunnels: log min(1, S(span) / S(median span)) with S the real span survival function (class-specific,
  pooled when sparse), plus log min(1, F(angle) / F(median angle)) for water crossings;
* geometry (optional): log min(1, F_J(phi) / F_J(median)) for the smallest junction angle a T / crossing / merge
  creates, F_J the real CDF.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from urbanmorph.terrain.road_feasibility import river_half_width, water_kind_at

GRADE_BINS = np.r_[0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.27, 0.33, 0.40, np.inf]
SLOPE_BINS = np.r_[0.0, 0.05, 0.10, 0.15, 0.25, np.inf]           # landform slope (32 m), training-documented bins
ALIGN_BINS = np.linspace(0, 1, 11)
WINDOW_M, STEP_M = 30.0, 2.0
NEG_INF = -np.inf


def slope_bin(s: float) -> int:
    return int(np.clip(np.searchsorted(SLOPE_BINS, s, side="right") - 1, 0, len(SLOPE_BINS) - 2))


def grade_bin(g: float) -> int:
    return int(np.clip(np.searchsorted(GRADE_BINS, g, side="right") - 1, 0, len(GRADE_BINS) - 2))


def align_bin(a: float) -> int:
    return int(np.clip(np.searchsorted(ALIGN_BINS, a, side="right") - 1, 0, len(ALIGN_BINS) - 2))


class TerrainField:
    """A tile's terrain with the derived rasters the potentials need (built once per tile)."""

    def __init__(self, t, landform=None):
        from urbanmorph.graphs.terrain_metrics import landform_gradient
        self.t = t
        if landform is None:
            gx, gy = landform_gradient(t) if np.isfinite(t.elev).any() else (np.zeros(t.shape), np.zeros(t.shape))
        else:
            gx, gy = landform
        self.gx, self.gy = np.nan_to_num(gx), np.nan_to_num(gy)
        self.has_dem = bool(np.isfinite(t.elev).all())
        self.hw = river_half_width(t) if getattr(t, "water_kind", None) is not None and (t.water_kind == 1).any() else None

    def grad_at(self, x, y):
        r, c = self.t.rc(np.atleast_1d(x), np.atleast_1d(y))
        r = np.clip(np.round(r).astype(int), 0, self.gx.shape[0] - 1)
        c = np.clip(np.round(c).astype(int), 0, self.gx.shape[1] - 1)
        return self.gx[r, c], self.gy[r, c]


def edge_features(tf: TerrainField, a, q, struct: int = 0, over_street: bool = False) -> dict:
    """Profile features of the committed edge a -> q (world = tile metres)."""
    a, q = np.asarray(a, float), np.asarray(q, float)
    L = float(np.hypot(*(q - a)))
    n = max(1, int(np.ceil(L / STEP_M)))
    tt = np.linspace(0, 1, n + 1)
    p = a + tt[:, None] * (q - a)
    d = tt * L
    t = tf.t
    z = t.sample(p[:, 0], p[:, 1]) if tf.has_dem else np.full(len(p), np.nan)
    wk = water_kind_at(t, p[:, 0], p[:, 1]) if getattr(t, "water_kind", None) is not None else np.zeros(len(p), np.int8)
    f = {"length": L, "struct": int(struct), "over_street": bool(over_street), "finite": bool(np.isfinite(z).all())}
    if f["finite"] and len(z) > 1:
        k = max(1, int(round(WINDOW_M / STEP_M)))
        g = np.abs(z[k:] - z[:-k]) / (d[k:] - d[:-k]) if len(d) > k else np.abs(np.diff(z)) / np.maximum(np.diff(d), 1e-9)
        f["grade"] = float(g.max())
        f["mean_grade"] = float(abs(z[-1] - z[0]) / max(L, 1e-9))
        deck = z[0] + tt * (z[-1] - z[0])
        f["clearance"] = float((deck - z).max())
        f["cover"] = float(np.mean((z - deck) >= 3.0))                 # share of the chord with >= 3 m of ground above
    else:
        f.update(grade=np.nan, mean_grade=np.nan, clearance=np.nan, cover=np.nan)
    mid = 0.5 * (a + q)
    gx, gy = tf.grad_at(mid[0], mid[1])
    s = float(np.hypot(gx[0], gy[0]))
    f["slope"] = s
    u = (q - a) / max(L, 1e-9)
    f["align"] = float(abs(gx[0] * u[0] + gy[0] * u[1]) / s) if s > 1e-9 else np.nan
    wet = wk > 0
    f["wet_length"] = float(wet.sum() * STEP_M)
    f["water_kinds"] = tuple(sorted(set(int(v) for v in wk[wet])))
    f["ends_in_water"] = bool(wet[-1])
    if wet.any() and tf.hw is not None and 1 in f["water_kinds"]:
        r, c = t.rc(p[wet, 0], p[wet, 1])
        r = np.clip(np.round(r).astype(int), 0, tf.hw.shape[0] - 1)
        c = np.clip(np.round(c).astype(int), 0, tf.hw.shape[1] - 1)
        width = 2 * float(tf.hw[r, c].max())
        f["river_width"] = width
        f["angle"] = float(np.degrees(np.arcsin(np.clip(width / max(f["wet_length"], 1e-9), 0, 1))))
    else:
        f["river_width"], f["angle"] = np.nan, np.nan
    return f


@dataclass
class Potentials:
    """Calibrated potential tables (configs/m13_potentials/<model>.json) plus weights."""
    r_grade: np.ndarray | None = None           # (4 classes, n_slope_bins, n_grade_bins) log-ratio
    r_align: np.ndarray | None = None           # (n_slope_bins, n_align_bins) log-ratio
    span_q: dict = field(default_factory=dict)  # struct -> class -> sorted real spans
    angle_q: np.ndarray | None = None           # sorted real crossing angles (deg)
    junction_q: np.ndarray | None = None        # sorted real smallest junction angles (deg)
    span_max: dict = field(default_factory=dict)
    angle_min: float = 0.0
    lam_ratio: float = 1.0
    lam_infra: float = 1.0
    lam_geom: float = 0.0
    viaducts: bool = False
    source: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path, **kw) -> "Potentials":
        d = json.loads(Path(path).read_text())
        p = cls(r_grade=np.array(d["log_r_grade"]), r_align=np.array(d["log_r_align"]),
                span_q={int(s): {int(c): np.array(v) for c, v in x.items()} for s, x in d["span_q"].items()},
                angle_q=np.array(d["angle_q"]), junction_q=np.array(d["junction_q"]),
                span_max={int(k): float(v) for k, v in d["span_max"].items()}, angle_min=float(d["angle_min"]),
                source=d.get("source", {}))
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    # -- survival helpers
    @staticmethod
    def _surv_ratio(q: np.ndarray, x: float) -> float:
        """log min(1, S(x) / S(median)) from sorted real values q (S = empirical survival, +1 smoothing)."""
        if q is None or len(q) == 0 or not np.isfinite(x):
            return 0.0
        S = (len(q) - np.searchsorted(q, x, side="left") + 1) / (len(q) + 1)
        return float(min(0.0, np.log(S / 0.5)))

    @staticmethod
    def _cdf_ratio(q: np.ndarray, x: float) -> float:
        """log min(1, F(x) / F(median)): penalises unusually SMALL values (acute angles)."""
        if q is None or len(q) == 0 or not np.isfinite(x):
            return 0.0
        F = (np.searchsorted(q, x, side="right") + 1) / (len(q) + 1)
        return float(min(0.0, np.log(F / 0.5)))

    def span_list(self, struct: int, cls: int):
        tab = self.span_q.get(struct, {})
        return tab.get(cls, tab.get(-1))

    # -- the potential of one committed edge
    def log_potential(self, f: dict, cls: int, kind: str, junction_angle: float = np.nan) -> tuple[float, str, dict]:
        """(log w, reason, parts). reason is "" for a feasible edge, else the violated hard constraint."""
        if kind == "dropped":
            return NEG_INF, "unrepresentable", {}
        st = f["struct"]
        kinds = f["water_kinds"]
        if f["ends_in_water"]:
            return NEG_INF, "ends_in_water", {}
        parts = {}
        if st == 0:
            if f["wet_length"] > 0:
                return NEG_INF, "ground_in_water", {}
            lr = 0.0
            if f["finite"] and np.isfinite(f["grade"]) and self.r_grade is not None:
                lr += float(self.r_grade[cls, slope_bin(f["slope"]), grade_bin(f["grade"])])
                if np.isfinite(f["align"]) and f["slope"] >= SLOPE_BINS[1] and self.r_align is not None:
                    lr += float(self.r_align[slope_bin(f["slope"]), align_bin(f["align"])])
            parts["ratio"] = lr
        elif st == 1:                                                     # bridge
            over_water = f["wet_length"] > 0
            if 3 in kinds:
                return NEG_INF, "sea_crossing", {}
            dry_valley = (not over_water and not f["over_street"] and np.isfinite(f["clearance"])
                          and f["clearance"] >= 10.0)
            if not over_water and not f["over_street"] and not (self.viaducts and dry_valley):
                return NEG_INF, "unsupported_structure", {}
            span = f["wet_length"] if over_water else f["length"]
            if span > self.span_max.get(1, np.inf):
                return NEG_INF, "span_exceeds_max", {}
            if over_water and np.isfinite(f["angle"]) and f["angle"] < self.angle_min:
                return NEG_INF, "degenerate_angle", {}
            e = self._surv_ratio(self.span_list(1, cls), span)
            if over_water:
                e += self._cdf_ratio(self.angle_q, f["angle"])
            parts["infra"] = e
        else:                                                             # tunnel
            if not (np.isfinite(f["cover"]) and f["cover"] >= 0.5):
                return NEG_INF, "unsupported_structure", {}
            if f["length"] > self.span_max.get(2, np.inf):
                return NEG_INF, "span_exceeds_max", {}
            parts["infra"] = self._surv_ratio(self.span_list(2, cls), f["length"])
        if self.lam_geom and np.isfinite(junction_angle):
            parts["geom"] = self._cdf_ratio(self.junction_q, junction_angle)
        lw = self.lam_ratio * parts.get("ratio", 0.0) + self.lam_infra * parts.get("infra", 0.0) + self.lam_geom * parts.get("geom", 0.0)
        return float(lw), "", parts


HARD_REASONS = ("unrepresentable", "ground_in_water", "ends_in_water", "unsupported_structure", "sea_crossing",
                "span_exceeds_max", "degenerate_angle")
