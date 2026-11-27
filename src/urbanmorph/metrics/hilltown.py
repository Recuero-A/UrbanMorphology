"""Milestone 20 D3: the hill-town metric suite (docs/M20_LOG.md D3 with its sign-off amendments, D4 "Consequences
for D3", and the D3a / D3b implementation rulings). Real and generated networks go through exactly this code.

Inputs. A vehicular RoadGraph in unit metres (``nodes`` (N, 2), ``edges`` (E, 2), optional ``edge_struct`` /
``edge_level``; graphs/road_graph, experiments/graph_io); an optional pedestrian layer (polylines and a steps flag per
line); a ``HillField``: ``T`` samples elevation (``T.sample(x, y)``, heightfield.Terrain, FABDEM averaged to 32 m),
``S`` is the slope grid (a Terrain whose gradient gives slope and fall line; ``T`` by default), ``land`` / ``unit``
are masks on ``S``'s grid (unit polygon; land = dry and valid inside it).

Geometry (D3a). native = every junction-to-junction polyline densified to <= 5 m, then Douglas-Peucker at 1 m,
identically for real and generated networks (``native``); convention = road_graph.build_graph of the polylines
(m18_terrain_reference.ground_polylines per (level, structure)), step 30 m, merge 8 m. The convention is primary,
except: M3 hairpins, G2 and G3 use native (the convention is the ``*_conv`` sensitivity), and M7 = native length /
convention length. Each edge is cut into equal pieces of at most 10 m (m18_terrain_reference.tile_sums): grade =
|dz| / length on ``T``; class, land and fall line at the piece midpoint on ``S`` (nearest pixel).

Cells: the classes 0-5 / 5-10 / 10-20 / 20-30 / > 30 % (``KEYS``), ``20-inf`` (20-30 and > 30 pooled, D3a),
``gt15`` (land steeper than 15 %) and ``unit``. Metrics (flat dict, key ``metric@cell``):
M1 density, coverage. M2 contour_index (classes >= 5 %), grade_p90, share_gt15.
M3 (targeted; never realism evidence for the hybrid): run_p50 / run_p90 (D3a: signed grade over a 32 m window along
   strokes through junctions, network_outcomes.strokes; stretches above 15 % of one sign, >= 32 m long);
   hairpin_rate@gt15 (native, 60 m window; convention: hairpin_rate_conv).
M4 deadend_per_km, share_3way, share_4way, close15 (after junctions.consolidate at 10 m), lcc_share, loops_per_km2
   (D3a: bounded faces whose representative point is on class land, per km2 of class land), lens_share_2ap
   (primary since D3c: 2A/P < 20 m and elongation > 6; the topological lens_share is the sensitivity).
M5 face_area_med / p90, phi_med, block_contour_cos (classes >= 10 %; D3a: faces with elongation >= 1.5 only).
Mesh diagnostics (D3c, descriptive only: outside P2, Holm and the gate; their own frozen plan, ``MESH_CELLS``):
   face_area_p10, small_face_share (faces < 0.1 ha), acute_share (junctions with two angularly adjacent incident
   edges under 30 deg), jj_len_med (median junction-to-junction chain length, class at the chain's midpoint).
M7 tortuosity. G1 g1_15 / g1_12. G2 g2_tight_share@gt15 (native). G3 grade_change_p95 (native). Pedestrian
   ped_stairs / ped_ramp8. Structures (D3a): struct_share, the bridge + tunnel share of street length
   (``structure_flags`` flags a network above the real 95th percentile).
M6 m6_15 / m6_12 / m6_10 (D4), and descriptively m6_<t>_steepcls@<class>: the served share by the class of the
   steeper endpoint (D3a). M6 IS CONSERVATIVE: the 10 % steep share is checked on the lexicographic route only, so a
   pair that some longer route would serve within 10 % is counted as not served.

Scoring (D3a, D3b). z = (t(x) - median of t(reference)) / scale, |z| capped at 5, sign kept. t is fixed per metric
family (``METRICS``): log(x + c) for unbounded positive metrics, logit((x + c) / (1 + 2c)) for proportions, identity
for cosines, phi and tortuosity; c = 0.1 x the median of the positive reference values, per metric x cell. A real town
enters a cell's reference with >= 0.05 km2 of class land and, for street-shape metrics only, >= 1 km of street there
(D3b: M1 and the access shares need land only, so a density of 0 is scored through log(x + c)); a cell needs >= 8
such towns (``REF_MIN_N``), otherwise 20-30 and > 30 are pooled into 20-inf, otherwise it is descriptive.
Declared deviation, logged per cell (``RefCell.binding``; ``floor_bound`` counts them): scale = max(1.4826 MAD,
(q90 - q10) / 2.563, floor), floor (D3b) = the span on the t scale of one measurement unit at the reference-median
cell / sqrt(12), the units of the D3b table (``METRICS``: a constant, or a constant / the median of the cell's
denominator over the reference towns). c, the units and the floors are frozen together from the real dev-H towns
(``make_plan``). Definedness uses the unit's own real network (>= 0.05 km2 of land; street metrics: >= 1 km); a
generator under 1 km on a defined street cell scores |z| = 5.

Implementation choices (left open by D3 / D3a; for the committee):
 1. Bridges and tunnels count as street (M1, M4, M5, M7, routing) but carry no grade: never steep, always drivable.
    Pieces <= 0.5 m and no-data pieces are left out of grade statistics (tile_sums).
 2. grade_p90 is the exact length-weighted quantile (style_metrics.grade_exposure), not M18's 0.5 % histogram.
 3. Coverage: land pixel centres of ``S`` within 60 m of the street, sampled every 2 m.
 4. Runs: the 32 m window is centred on each piece along the stroke (shifted inside near its ends; the whole stroke if
    shorter); a piece off land, off ground or without data ends a run; a run's class holds most of its length; a
    class with streets and no run scores 0.
 5. Hairpins: SIGNED net heading change > 120 deg within 60 m of path; the minimal window from each start, greedy
    left to right, non-overlapping (counted once). Chains end at junctions and dead ends. Apex = the vertex of largest
    turn; counted on land > 15 % at the apex; rate per km of convention street on that land.
 6. G2 radius = path length between the window's first and last turning vertex / (net turn - half the turns at
    those two vertices): exact on a sampled circle, blind to the straight legs; a one-vertex hairpin has radius 0.
 7. M4 node metrics count nodes >= 40 m inside the unit (junctions.EDGE_M), class at the node; dead ends per km of
    class street whose midpoint is that far inside. 3- and 4-way shares on the cleaned convention junctions.
    close15: consolidated clusters replaced by their centroid, nearest other centroid < 15 m, NaN under 2 focal.
 8. lcc_share: the share of class street length in the largest component (by length) of the WHOLE unit network.
 9. Faces: polygonize of the convention graph, > 1 m2, class at a point on the surface (off land: dropped); no
    edge censoring (networks are cropped to the unit). Elongation = long / short side of the minimum rotated
    rectangle, whose long side is the block axis. Lens: exterior-ring nodes of graph degree >= 3 == 2 (holes
    ignored). 2A/P uses the full perimeter. Louf phi = A / (pi r^2), r of the minimum bounding circle (M16).
    M5 summaries are medians per cell.
10. G3: native ground chains resampled every 10 m (m18 resample), pieces < 5 m dropped, |grade change| between
    consecutive pieces (percentage points), class at their shared vertex, unweighted p95.
11. Pairs: m19_eval.od_points with pixel area res^2 instead of 16 m2 (identical on 4 m grids, tested). M6, G1 and
    pedestrian reach are unit-level (share of pairs served).
12. M6 route: steep length quantised to 0.1 m, combined weight steep_dm * BIG + length (exact lexicographic order at
    0.1 m); partial edges at the joins prorated. Steep share = steep / street length of the route (the <= 60 m access
    legs count in the detour only, as D5). Outcomes: 1 not joined, 2 disconnected, 3 detour > 4, 4 steep share.
13. G1: grade over a 30 m window centred on each piece along its ground chain; edges are cut where drivability
    changes; routing = D5 shortest length. Pedestrian: one convention build of vehicular + pedestrian lines; "with
    stairs" = all of it; "ramps" = non-steps pieces with G1's window grade <= 8 %. Without a layer: vehicular only.
14. Floor units: each metric's own denominator gives "the median count" (3- and 4-way shares: interior junctions;
    close15: focal consolidated junctions; rates: the cell's street km; G2: its hairpins); one unit spans
    [median - u/2, median + u/2], shifted inside [0, 1] or [0, inf) at a bound. Metrics outside the D3b table follow
    their family (lens 2A/P as lens; convention sensitivities as their primary; per-class hairpin rate as a rate).
    Pooling replaces BOTH steep cells when either is short (D3b confirmed); a generator NaN on a defined cell scores 5.
15. Gate 1: pooled coverage from the plain leave-one-out; percentiles (D3a): every pair (r, q) is scored against a
    reference without both, p_r = (#{q: C_q < C_r} + ties / 2 + 1/2) / n; the KS p is APPROXIMATE (the p_r are
    dependent). Gate 2: suite composite over ``SUITE_CELLS`` (two-sided mean |z|); a targeted group fails when its
    mean SIGN-ORIENTED z (``TARGET_SIGN``, D3c; cells scored by rule left out) is above the 95th percentile of the
    real LOO sign-oriented group means; the zigzag's M7 is its sign-oriented M7 mean, compared with the largest real
    LOO value; a case passes when >= 90 % of towns fail it (``gate_adversarial_towns``). The single-site
    leastcostpath case is checked against its real site network as control (``gate_site_control``, D3c).
16. C2ST: raw features, training-fold mean imputation and standardisation, one no-streets indicator; pooled
    out-of-fold AUC; town bootstrap without refitting; p = (1 + #perm >= obs) / (1 + n_perm), n_perm >= 1999 (D3a).
    L2 logistic regression with scipy (sklearn's objective, C = 1), tested against a second scipy solver. The
    boosted-trees sensitivity is dropped (D3a).
17. Adversarial frame: land centroid and mean gradient over land. Switchback: 150 m contour legs 30 m apart (15 m
    hairpins) up the whole land extent, longest clipped piece kept. Contour streets: contours of ``S`` every 60 m x
    the median land slope. Fall-line grid: 80 m. Raster zigzag: each segment as an axis-aligned staircase, major
    legs U(4, 9) m.

The suite reuses scripts/m18_terrain_reference.py and scripts/m19_eval.py (scripts/ is appended to sys.path): the
freeze hash must include them.
"""
from __future__ import annotations

import importlib
import math
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from urbanmorph.graphs.junctions import EDGE_M, MIN_FOCAL, clean, consolidate
from urbanmorph.graphs.network_outcomes import strokes
from urbanmorph.graphs.road_graph import MERGE_M, STEP_M, RoadGraph, build_graph

ROOT = Path(__file__).resolve().parents[3]


def _script(name: str):
    """A milestone script as a module (scripts/ appended to sys.path, so it never shadows a package)."""
    p = str(ROOT / "scripts")
    if p not in sys.path:
        sys.path.append(p)
    return importlib.import_module(name)


_M18 = _script("m18_terrain_reference")
KEYS = tuple(_M18.KEYS)                                    # "0-5", "5-10", "10-20", "20-30", "30-inf"
CLASS_EDGES = (5, 10, 20, 30)
PIECE_M, COVER_M, SAMPLE_M = 10.0, float(_M18.COVER_M), 2.0
STEEP_PCT = 15.0
CELL_LAND_KM2, STREET_KM_MIN, C2ST_LAND_KM2 = 0.05, 1.0, 0.02
Z_CAP, Z_FIT, REF_MIN_N = 5.0, 2.0, 8                     # D3a: >= 8 reference towns with >= 1 km in the cell
POOLED = "20-inf"                                          # D3a: 20-30 and > 30 pooled
CELL_DEF = tuple((k, (q,)) for q, k in enumerate(KEYS)) + ((POOLED, (3, 4)),)
TAU_CONS, TAU_CLOSE = 10.0, 15.0
LENS_ELONG, LENS2AP_W, LENS2AP_ELONG, ALIGN_ELONG = 4.0, 20.0, 6.0, 1.5
SMALL_FACE_M2, ACUTE_DEG = 1000.0, 30.0                      # D3c mesh diagnostics: 0.1 ha, 30 deg
HAIRPIN_DEG, HAIRPIN_M, TIGHT_R_M = 120.0, 60.0, 12.0     # D3a: 60 m window
NATIVE_DENSIFY_M, NATIVE_DP_M = 5.0, 1.0                   # D3a native geometry
RUN_WINDOW_M, RUN_MIN_M = 32.0, 32.0                       # D3a M3 runs
M6_STEEP, M6_SHARE = (15.0, 12.0, 10.0), 0.10
G1_WINDOW_M, G1_CAPS, RAMP_PCT = 30.0, (15.0, 12.0), 8.0
G3_STEP = 10.0
_ALL = (-1e18, -1e18, 1e18, 1e18)                          # a window in which every junction is focal


# ---------------------------------------------------------------- terrain and land
class HillField:
    """Terrain, slope classes and land of one unit (attributes as m18_terrain_reference.Field, so m19_eval can read it).

    ``S``: slope grid (heightfield.Terrain); ``land`` / ``unit``: bool masks on its grid; ``T``: elevation sampler."""

    def __init__(self, S, land, T=None, unit=None, uid: str = "unit"):
        self.S, self.T, self.uid, self.res = S, (S if T is None else T), uid, float(S.res)
        self.gx, self.gy = S.gradient()
        self.slope = np.hypot(self.gx, self.gy) * 100
        self.land = np.asarray(land, bool) & np.isfinite(self.slope)
        self.unit = self.land.copy() if unit is None else np.asarray(unit, bool)
        self.cls = np.digitize(np.nan_to_num(self.slope), CLASS_EDGES)
        self.interior = ndimage.distance_transform_edt(np.pad(self.unit, 1))[1:-1, 1:-1] * self.res >= EDGE_M

    @classmethod
    def from_polygon(cls, S, polygon, T=None, water=None, uid: str = "unit"):
        """Unit = pixel centres of ``S`` inside ``polygon``; land = unit minus ``water`` (bool, ``S``'s grid)."""
        H, W = S.elev.shape
        X, Y = np.meshgrid(S.x0 + (np.arange(W) + 0.5) * S.res, S.y1 - (np.arange(H) + 0.5) * S.res)
        unit = shapely.contains_xy(polygon, X, Y)
        land = unit if water is None else unit & ~np.asarray(water, bool)
        return cls(S, land, T=T, unit=unit, uid=uid)

    def rc(self, P):
        P = np.asarray(P, float).reshape(-1, 2)
        r, c = self.S.rc(P[:, 0], P[:, 1])
        r, c = np.round(r).astype(int), np.round(c).astype(int)
        H, W = self.land.shape
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        return np.clip(r, 0, H - 1), np.clip(c, 0, W - 1), ok

    def lookup(self, P):
        """(class, -1 off land), slope %, interior, gx, gy at points (nearest pixel)."""
        r, c, ok = self.rc(P)
        land = ok & self.land[r, c]
        return (np.where(land, self.cls[r, c], -1), np.where(ok, self.slope[r, c], np.nan), ok & self.interior[r, c],
                self.gx[r, c], self.gy[r, c])

    def inside(self, P) -> np.ndarray:
        r, c, ok = self.rc(P)
        return ok & self.unit[r, c]

    def class_land_km2(self) -> np.ndarray:
        return np.array([(self.land & (self.cls == q)).sum() for q in range(5)]) * self.res ** 2 / 1e6

    def centres(self, mask):
        r, c = np.nonzero(mask)
        return np.c_[self.S.x0 + (c + 0.5) * self.res, self.S.y1 - (r + 0.5) * self.res]


# ---------------------------------------------------------------- geometry
def _view(nodes, edges):
    """A graph view whose edges all count as ground (for ground_polylines over every structure)."""
    return SimpleNamespace(nodes=nodes, edges=edges, structs=lambda: np.zeros(len(edges), int))


def cleaned(g) -> RoadGraph:
    nodes, e, st = clean(g)
    return RoadGraph(nodes, e, np.zeros(len(nodes), bool), edge_struct=st)


def chains(g, ground_only: bool = False) -> list:
    """[(coords, level, struct)]: junction-to-junction polylines per (level, structure) (ground_polylines)."""
    if len(g.edges) == 0:
        return []
    lv, st = np.asarray(g.levels(), int), np.asarray(g.structs(), int)
    out = []
    for L_, S_ in sorted(set(zip(lv.tolist(), st.tolist()))):
        if ground_only and S_ != 0:
            continue
        nodes, e, _ = clean(RoadGraph(g.nodes, np.asarray(g.edges)[(lv == L_) & (st == S_)], np.zeros(g.n, bool)))
        out += [(p, L_, S_) for p in _M18.ground_polylines(_view(nodes, e))]
    return out


def convention(g, step: float = STEP_M, merge: float = MERGE_M) -> RoadGraph:
    """The common convention (resample <= 30 m, merge 8 m; 3D-aware when there are structures), cleaned."""
    ch = chains(g)
    lines = [c for c, _, _ in ch]
    if all(l_ == 0 and s_ == 0 for _, l_, s_ in ch):
        return cleaned(build_graph(lines, step=step, merge=merge))
    return cleaned(build_graph(lines, step=step, merge=merge, levels=[l_ for _, l_, _ in ch],
                               structs=[s_ for _, _, s_ in ch]))


def lines_graph(lines, node: bool = True, structs=None) -> RoadGraph:
    """A graph from polylines: vertices are nodes (equal points merged); ``node``: crossings become nodes (structures
    are then lost); ``structs``: one structure code per line (``node`` False)."""
    items = [(np.asarray(l_, float)[:, :2], 0 if structs is None else int(structs[i])) for i, l_ in enumerate(lines)
             if len(l_) >= 2]
    if node and items:
        u = shapely.unary_union([shapely.LineString(p) for p, _ in items])
        items = [(np.asarray(x.coords)[:, :2], 0) for x in getattr(u, "geoms", [u]) if x.geom_type == "LineString"]
    items = [(p, s_) for p, s_ in items if shapely.LineString(p).length > 0]
    if not items:
        return RoadGraph(np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, bool), edge_class=np.zeros(0, int),
                         edge_struct=np.zeros(0, int))
    allp = np.concatenate([p for p, _ in items])
    _, rep, inv = np.unique(np.round(allp * 1e6).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    inv, off = inv.ravel(), np.cumsum([0] + [len(p) for p, _ in items])
    E = np.sort(np.concatenate([np.c_[inv[a:b - 1], inv[a + 1:b]] for a, b in zip(off[:-1], off[1:])]), 1)
    st = np.concatenate([np.full(len(p) - 1, s_) for p, s_ in items])
    ok = E[:, 0] != E[:, 1]
    E, st = E[ok], st[ok]
    E, first = np.unique(E, axis=0, return_index=True)
    return RoadGraph(allp[rep], E, np.zeros(len(rep), bool), edge_class=np.full(len(E), 2), edge_struct=st[first])


def native(g) -> RoadGraph:
    """D3a native geometry: every junction-to-junction polyline densified to <= 5 m, then Douglas-Peucker at 1 m
    (end points kept; a polyline that would collapse is kept as it is), the same for real and generated networks."""
    lines, sts = [], []
    for poly, _, s_ in chains(g):
        q = shapely.simplify(shapely.segmentize(shapely.LineString(poly), NATIVE_DENSIFY_M), NATIVE_DP_M,
                             preserve_topology=False)
        c = shapely.get_coordinates(q)
        lines.append(c if len(c) >= 2 and shapely.length(q) > 0 else poly)
        sts.append(s_)
    return lines_graph(lines, node=False, structs=sts)


@dataclass
class Pieces:
    edge: np.ndarray       # piece -> edge
    k: np.ndarray          # index of the piece along its edge (from edges[:, 0])
    n: np.ndarray          # pieces per EDGE (E,)
    p0: np.ndarray
    p1: np.ndarray
    length: np.ndarray
    dz: np.ndarray         # signed rise p0 -> p1; NaN off ground or without data
    grade: np.ndarray      # |dz| / length, %; NaN where not measured
    cls: np.ndarray        # slope class at the midpoint, -1 off land
    slope: np.ndarray
    interior: np.ndarray
    ci: np.ndarray         # contour index |cos(piece, fall line)|
    ground: np.ndarray     # structure 0 (not a bridge or tunnel)

    @property
    def off(self) -> np.ndarray:
        return np.r_[0, np.cumsum(self.n)]


def pieces(g, F: HillField, step: float = PIECE_M) -> Pieces:
    E = np.asarray(g.edges, int).reshape(-1, 2)
    A, B = g.nodes[E[:, 0]], g.nodes[E[:, 1]]
    L = np.hypot(*(B - A).T)
    n = np.maximum(1, np.ceil(L / step)).astype(int)
    e = np.repeat(np.arange(len(L)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    p0 = A[e] + (k / n[e])[:, None] * (B - A)[e]
    p1 = A[e] + ((k + 1) / n[e])[:, None] * (B - A)[e]
    pl = L[e] / n[e]
    ground = np.asarray(g.structs(), int)[e] == 0
    dz = np.full(len(e), np.nan)
    if ground.any():
        dz[ground] = F.T.sample(p1[ground, 0], p1[ground, 1]) - F.T.sample(p0[ground, 0], p0[ground, 1])
    dz[pl <= 0.5] = np.nan
    cls, slope, inner, gx, gy = F.lookup((p0 + p1) / 2)
    u = (p1 - p0) / np.maximum(pl, 1e-9)[:, None]
    ci = np.clip(np.abs(u[:, 0] * gx + u[:, 1] * gy) / np.maximum(np.hypot(gx, gy), 1e-9), 0, 1)
    return Pieces(e, k, n, p0, p1, pl, dz, np.abs(dz) / np.maximum(pl, 1e-9) * 100, cls, slope, inner, ci, ground)


def _window_grade(P, s, smid, F: HillField, window: float) -> np.ndarray:
    """Signed grade (%) along the polyline ``P`` (arc positions ``s``) over a ``window`` m window centred at ``smid``
    (shifted inside the polyline near its ends; the whole polyline if shorter)."""
    lo = np.clip(smid - window / 2, 0, max(s[-1] - window, 0))
    hi = np.minimum(lo + window, s[-1])
    za = F.T.sample(np.interp(lo, s, P[:, 0]), np.interp(lo, s, P[:, 1]))
    zb = F.T.sample(np.interp(hi, s, P[:, 0]), np.interp(hi, s, P[:, 1]))
    return (zb - za) / np.maximum(hi - lo, 1e-9) * 100


def wquantile(x, w, q: float) -> float:
    """Lowest x whose cumulative weight share reaches q (style_metrics.grade_exposure)."""
    if len(x) == 0 or np.sum(w) <= 0:
        return np.nan
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / np.sum(w)
    return float(x[o][min(int(np.searchsorted(cw, q)), len(o) - 1)])


def _samples(nodes, edges, step: float = SAMPLE_M):
    """Points every <= ``step`` m along the edges: (points, edge, parameter t)."""
    a, b = nodes[edges[:, 0]], nodes[edges[:, 1]]
    L = np.hypot(*(b - a).T)
    m = np.maximum(1, np.ceil(L / step)).astype(int)
    e = np.repeat(np.arange(len(L)), m + 1)
    t = (np.arange((m + 1).sum()) - np.repeat(np.cumsum(m + 1) - (m + 1), m + 1)) / m[e]
    return a[e] + t[:, None] * (b - a)[e], e, t


# ---------------------------------------------------------------- M1-M5, M7, G2, G3
def _coverage(g, F: HillField):
    """(land pixels within 60 m of a street, land pixels) per class."""
    cl = F.cls[F.land]
    tot = np.bincount(cl, minlength=5).astype(float)
    if len(g.edges) == 0:
        return np.zeros(5), tot
    pts, _, _ = _samples(g.nodes, g.edges)
    d, _ = cKDTree(pts).query(F.centres(F.land), distance_upper_bound=COVER_M + 1e-9)
    return np.bincount(cl[d <= COVER_M], minlength=5).astype(float), tot


def stroke_runs(g, pc: Pieces, F: HillField):
    """(lengths m, classes) of the M3 runs (D3a): along strokes through junctions, the signed grade over a 32 m window
    centred on each piece; maximal stretches above 15 % of one sign, at least 32 m long (choice 4)."""
    L_, C_ = [], []
    if len(g.edges) == 0:
        return np.zeros(0), np.zeros(0, int)
    S, _ = strokes(g)
    eid = {(int(a), int(b)): i for i, (a, b) in enumerate(g.edges)}
    off = pc.off
    for path in S:
        if len(path) < 2:
            continue
        P = g.nodes[path]
        seg = np.hypot(*np.diff(P, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        idx, smid = [], []
        for t_, (u, v) in enumerate(zip(path[:-1], path[1:])):
            e = eid[(min(u, v), max(u, v))]
            r = np.arange(off[e], off[e + 1])
            f = (pc.k[r] + 0.5) / pc.n[e]
            if g.edges[e, 0] != u:
                r, f = r[::-1], 1 - f[::-1]
            idx.append(r)
            smid.append(s[t_] + f * seg[t_])
        idx, smid = np.concatenate(idx), np.concatenate(smid)
        wg = _window_grade(P, s, smid, F, RUN_WINDOW_M)
        ln, cl = pc.length[idx], pc.cls[idx]
        ok = np.isfinite(pc.dz[idx]) & (cl >= 0) & np.isfinite(wg)
        sg = np.where(ok & (np.abs(np.nan_to_num(wg)) > STEEP_PCT), np.sign(np.nan_to_num(wg)), 0)
        start = np.flatnonzero(np.r_[True, sg[1:] != sg[:-1]])
        for a, b in zip(start, np.r_[start[1:], len(sg)]):
            if sg[a] != 0 and ln[a:b].sum() >= RUN_MIN_M:
                L_.append(float(ln[a:b].sum()))
                C_.append(int(np.argmax(np.bincount(cl[a:b], weights=ln[a:b], minlength=5))))
    return np.array(L_), np.array(C_, int)


def hairpins(g) -> list:
    """[(apex xy, net turn of the minimal window deg, centreline radius m)] along chains of degree-2 nodes
    (choices 5, 6)."""
    out = []
    if len(g.edges) == 0:
        return out
    lim = math.radians(HAIRPIN_DEG)
    for P in _M18.ground_polylines(_view(g.nodes, g.edges)):
        d = np.diff(P, axis=0)
        ln = np.hypot(*d.T)
        P = np.vstack([P[:1], P[1:][ln > 1e-6]])
        d = np.diff(P, axis=0)
        ln = np.hypot(*d.T)
        if len(P) < 3:
            continue
        h = np.arctan2(d[:, 1], d[:, 0])
        turn = (np.diff(h) + np.pi) % (2 * np.pi) - np.pi        # at vertex j + 1
        s = np.cumsum(ln)[:-1]
        i = 0
        while i < len(turn):
            acc, j, found = 0.0, i, -1
            while j < len(turn) and s[j] - s[i] <= HAIRPIN_M:
                acc += turn[j]
                if abs(acc) > lim:
                    found = j
                    break
                j += 1
            if found < 0:
                i += 1
                continue
            apex = i + int(np.argmax(np.abs(turn[i:found + 1])))
            inner = abs(acc) - (abs(turn[i]) + abs(turn[found])) / 2     # turn between the end vertices
            out.append((P[apex + 1].copy(), math.degrees(abs(acc)), (s[found] - s[i]) / inner if inner > 1e-9 else 0.0))
            i = found + 1
    return out


def _faces(g, F: HillField):
    """Faces of the planar street graph with their class, shape and lens flags (choice 9)."""
    from shapely.ops import polygonize
    empty = None
    if len(g.edges) < 3:
        return empty
    segs = shapely.linestrings(np.stack([g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]], 1))
    polys = np.array(list(polygonize(shapely.unary_union(segs))), dtype=object)
    if len(polys) == 0:
        return empty
    A = shapely.area(polys)
    polys, A = polys[A > 1.0], A[A > 1.0]
    if len(polys) == 0:
        return empty
    per = shapely.length(polys)
    with np.errstate(all="ignore"):                       # GEOS warns on near-degenerate faces
        R = shapely.oriented_envelope(polys)
    rect = np.full((len(R), 4, 2), np.nan)
    for i_, r_ in enumerate(R):
        cc = shapely.get_coordinates(r_)
        if len(cc) >= 4:
            rect[i_] = cc[:4]
    s1, s2 = rect[:, 1] - rect[:, 0], rect[:, 2] - rect[:, 1]
    l1, l2 = np.hypot(*s1.T), np.hypot(*s2.T)
    axis = np.where((l1 >= l2)[:, None], s1, s2) / np.maximum(np.maximum(l1, l2), 1e-9)[:, None]
    elong = np.maximum(l1, l2) / np.maximum(np.minimum(l1, l2), 1e-9)
    phi = A / (math.pi * shapely.minimum_bounding_radius(polys) ** 2)
    rp = shapely.get_coordinates(shapely.point_on_surface(polys))
    cls, _, _, gx, gy = F.lookup(rp)
    gn = np.maximum(np.hypot(gx, gy), 1e-12)
    cos_c = np.abs(axis[:, 0] * (-gy / gn) + axis[:, 1] * (gx / gn))
    # ring nodes -> graph nodes (degree), consecutive mapped nodes -> edges
    coords, fi = shapely.get_coordinates(shapely.get_exterior_ring(polys), return_index=True)
    dist, nid = cKDTree(g.nodes).query(coords, distance_upper_bound=1e-6)
    mapped = np.isfinite(dist)
    deg = np.bincount(g.edges.ravel(), minlength=g.n)
    pairs = np.unique(np.c_[fi[mapped], nid[mapped]], axis=0)
    njunc = np.bincount(pairs[:, 0], weights=(deg[pairs[:, 1]] >= 3).astype(float), minlength=len(polys))
    lens = (njunc == 2) & (elong >= LENS_ELONG)
    lens2ap = (2 * A / per < LENS2AP_W) & (elong > LENS2AP_ELONG)
    eid = {(int(a), int(b)): i for i, (a, b) in enumerate(g.edges)}
    on_lens, on_2ap = np.zeros(len(g.edges), bool), np.zeros(len(g.edges), bool)
    bnd = np.searchsorted(fi, np.arange(len(polys) + 1))
    for f in np.flatnonzero(lens | lens2ap):
        ring = nid[bnd[f]:bnd[f + 1]][mapped[bnd[f]:bnd[f + 1]]]
        for u, v in zip(ring[:-1], ring[1:]):
            e = eid.get((min(u, v), max(u, v)))
            if e is not None:
                on_lens[e] |= lens[f]
                on_2ap[e] |= lens2ap[f]
    return {"cls": cls, "area": A, "phi": phi, "cos": cos_c, "elong": elong, "on_lens": on_lens, "on_2ap": on_2ap}


def _grade_change(g, F: HillField):
    """(|grade change| pp per 10 m, class) between consecutive 10 m pieces of the ground chains (choice 10)."""
    V, C = [], []
    for poly, _, _ in chains(g, ground_only=True):
        q = _M18.resample(poly, G3_STEP)
        if q is None or len(q) < 3:
            continue
        ln = np.hypot(*np.diff(q, axis=0).T)
        gr = np.diff(F.T.sample(q[:, 0], q[:, 1])) / np.maximum(ln, 1e-9) * 100
        ok = (ln >= G3_STEP / 2) & np.isfinite(gr)
        v = ok[:-1] & ok[1:]
        if v.any():
            V.append(np.abs(np.diff(gr))[v])
            C.append(F.lookup(q[1:-1][v])[0])
    return (np.concatenate(V), np.concatenate(C)) if V else (np.zeros(0), np.zeros(0, int))


def _hairpin_table(hp, F: HillField):
    """(class, slope %, radius) of each hairpin's apex."""
    cls, slope, _, _, _ = F.lookup(np.array([h[0] for h in hp]).reshape(-1, 2))
    return cls, slope, np.array([h[2] for h in hp])


def jj_chains(g):
    """Junction-to-junction chains of ``g`` (through degree-2 nodes; pure degree-2 cycles ignored): length, midpoint
    (the point at half the chain's length) and whether both ends are junctions (degree >= 3)."""
    if len(g.edges) == 0:
        return np.zeros(0), np.zeros((0, 2)), np.zeros(0, bool)
    deg = np.bincount(g.edges.ravel(), minlength=g.n)
    adj = [[] for _ in range(g.n)]
    for i, (a, b) in enumerate(g.edges):
        adj[a].append((b, i))
        adj[b].append((a, i))
    seen = np.zeros(len(g.edges), bool)
    L, M, J = [], [], []
    for u in np.flatnonzero(deg != 2):
        for v, e in adj[u]:
            if seen[e]:
                continue
            path, cur = [u, v], v
            seen[e] = True
            while deg[cur] == 2:
                (n1, e1), (n2, e2) = adj[cur]
                nxt, en = (n2, e2) if e1 == e else (n1, e1)          # the other incident edge
                if seen[en]:
                    break
                seen[en] = True
                cur, e = nxt, en
                path.append(cur)
            P = g.nodes[path]
            seg = np.hypot(*np.diff(P, axis=0).T)
            s_ = np.r_[0, np.cumsum(seg)]
            h = s_[-1] / 2
            k = min(int(np.searchsorted(s_, h, side="right")) - 1, len(seg) - 1)
            t = (h - s_[k]) / seg[k] if seg[k] > 0 else 0.0
            L.append(s_[-1])
            M.append(P[k] + t * (P[k + 1] - P[k]))
            J.append(deg[path[0]] >= 3 and deg[path[-1]] >= 3)
    return np.array(L), np.array(M).reshape(-1, 2), np.array(J, bool)


def acute_nodes(g, deg_min: int = 3, angle: float = 30.0) -> np.ndarray:
    """Per node: True when two angularly adjacent incident edges meet at under ``angle`` deg (nodes of degree >=
    ``deg_min`` only)."""
    out = np.zeros(g.n, bool)
    if len(g.edges) == 0:
        return out
    E = np.asarray(g.edges)
    src, dst = np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]]
    d = g.nodes[dst] - g.nodes[src]
    th = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    order = np.lexsort((th, src))
    src, th = src[order], th[order]
    bnd = np.r_[0, np.flatnonzero(np.diff(src)) + 1, len(src)]
    for a, b in zip(bnd[:-1], bnd[1:]):
        if b - a >= deg_min:
            t = th[a:b]
            gaps = np.diff(np.r_[t, t[0] + 360.0])
            out[src[a]] = bool(gaps.min() < angle)
    return out


def unit_metrics(g, F: HillField, ped=None, access: bool = True) -> dict:
    """Every D3 metric of one network on one unit, as a flat dict ``metric@cell`` (module docstring).

    ``ped``: (polylines, steps flags) or None. ``access``: compute M6, G1 and pedestrian reach (the pairs)."""
    gn = native(g)
    cv = convention(g)
    pc, pn = pieces(cv, F), pieces(gn, F)
    land = F.class_land_km2()
    out = {}
    on = pc.cls >= 0
    bins = lambda c, w: np.bincount(c, weights=w, minlength=5)
    Lc, Ln = bins(pc.cls[on], pc.length[on]), bins(pn.cls[pn.cls >= 0], pn.length[pn.cls >= 0])
    Ls = bins(pc.cls[on & ~pc.ground], pc.length[on & ~pc.ground])
    Lin = bins(pc.cls[on & pc.interior], pc.length[on & pc.interior])
    cov_n, cov_d = _coverage(cv, F)
    graded = np.isfinite(pc.grade) & on
    run_L, run_c = stroke_runs(cv, pc, F)
    hp_n, hp_c = hairpins(gn), hairpins(cv)
    hn, hc = _hairpin_table(hp_n, F), _hairpin_table(hp_c, F)
    deg = np.bincount(cv.edges.ravel(), minlength=cv.n) if len(cv.edges) else np.zeros(cv.n, int)
    ncls, _, ninner, _, _ = F.lookup(cv.nodes)
    _, jcls, jfoc, nn = _consolidated(cv, F)
    if len(cv.edges):
        A = coo_matrix((np.ones(len(cv.edges)), (cv.edges[:, 0], cv.edges[:, 1])), shape=(cv.n, cv.n))
        lab = connected_components(A, directed=False)[1]
        eL = np.hypot(*(cv.nodes[cv.edges[:, 0]] - cv.nodes[cv.edges[:, 1]]).T)
        in_lcc = lab[cv.edges[:, 0]] == np.argmax(np.bincount(lab[cv.edges[:, 0]], weights=eL))
    else:
        in_lcc = np.zeros(0, bool)
    fc = _faces(cv, F)
    g3n, g3c = _grade_change(gn, F), _grade_change(cv, F)
    acute = acute_nodes(cv)
    jl, jm, jj = jj_chains(cv)
    jcls_ = F.lookup(jm)[0] if len(jm) else np.zeros(0, int)
    for key, qs in CELL_DEF:
        qs = list(qs)
        isq = lambda a: np.isin(a, qs)
        put = lambda m, v: out.__setitem__(f"{m}@{key}", float(v) if v is not None else np.nan)
        la, L, km = land[qs].sum(), Lc[qs].sum(), Lc[qs].sum() / 1000
        has = L > 0
        put("land_km2", la)
        put("street_km", km)
        put("density", km / la if la > 0 else np.nan)
        put("coverage", cov_n[qs].sum() / cov_d[qs].sum() if cov_d[qs].sum() > 0 else np.nan)
        put("struct_share", Ls[qs].sum() / L if has else np.nan)
        s = graded & isq(pc.cls)
        w, gr = pc.length[s], pc.grade[s]
        put("contour_index", (w * pc.ci[s]).sum() / w.sum() if min(qs) >= 1 and w.sum() > 0 else np.nan)
        put("grade_p90", wquantile(gr, w, 0.9))
        put("share_gt15", w[gr > STEEP_PCT].sum() / w.sum() if w.sum() > 0 else np.nan)
        r = run_L[isq(run_c)]
        put("run_p50", (np.percentile(r, 50) if len(r) else 0.0) if has else np.nan)
        put("run_p90", (np.percentile(r, 90) if len(r) else 0.0) if has else np.nan)
        put("hairpin_rate_cls", isq(hn[0]).sum() / km if has else np.nan)
        lin = Lin[qs].sum() / 1000
        put("deadend_per_km", ((deg == 1) & ninner & isq(ncls)).sum() / lin if lin > 0 else np.nan)
        j = deg[(deg >= 3) & ninner & isq(ncls)]
        put("junctions", len(j))
        put("share_3way", (j == 3).mean() if len(j) else np.nan)
        put("share_4way", (j == 4).mean() if len(j) else np.nan)
        f = jfoc & isq(jcls)
        put("junctions_focal", f.sum())
        put("close15", (nn[f] < TAU_CLOSE).mean() if f.sum() >= MIN_FOCAL else np.nan)
        sq = isq(pc.cls)
        put("lcc_share", pc.length[sq & in_lcc[pc.edge]].sum() / L if has else np.nan)
        fa = isq(fc["cls"]) if fc is not None else np.zeros(0, bool)
        put("loops_per_km2", fa.sum() / la if la > 0 and (has or fa.any()) else np.nan)
        for m_, flag in (("lens_share", "on_lens"), ("lens_share_2ap", "on_2ap")):
            put(m_, (pc.length[sq & fc[flag][pc.edge]].sum() / L if fc is not None else 0.0) if has else np.nan)
        put("face_area_med", np.median(fc["area"][fa]) if fa.any() else np.nan)
        put("face_area_p90", np.percentile(fc["area"][fa], 90) if fa.any() else np.nan)
        put("faces", fa.sum())                                                   # D3c mesh diagnostics
        put("face_area_p10", np.percentile(fc["area"][fa], 10) if fa.any() else np.nan)
        put("small_face_share", np.mean(fc["area"][fa] < SMALL_FACE_M2) if fa.any() else np.nan)
        jm_ = (deg >= 3) & ninner & isq(ncls)
        put("acute_share", acute[jm_].mean() if jm_.any() else np.nan)
        jq = jj & isq(jcls_)
        put("jj_len_med", np.median(jl[jq]) if has and jq.any() else np.nan)
        put("phi_med", np.median(fc["phi"][fa]) if fa.any() else np.nan)
        al = fa & (fc["elong"] >= ALIGN_ELONG) & np.isfinite(fc["cos"]) if fa.any() else fa
        put("block_contour_cos", np.median(fc["cos"][al]) if min(qs) >= 2 and al.any() else np.nan)
        put("tortuosity", Ln[qs].sum() / L if has else np.nan)
        for m_, (v_, c_) in (("grade_change_p95", g3n), ("grade_change_p95_conv", g3c)):
            put(m_, np.percentile(v_[isq(c_)], 95) if isq(c_).any() else np.nan)
    # land steeper than 15 %: hairpins (native primary, convention sensitivity) and G2
    a = F.res ** 2 / 1e6
    st_km = pc.length[on & (pc.slope > STEEP_PCT)].sum() / 1000
    on_n, on_c = (hn[0] >= 0) & (hn[1] > STEEP_PCT), (hc[0] >= 0) & (hc[1] > STEEP_PCT)
    rate = lambda m: float(m.sum() / st_km) if st_km > 0 else np.nan
    out.update({"land_km2@gt15": float((F.land & (F.slope > STEEP_PCT)).sum() * a), "street_km@gt15": float(st_km),
                "hairpin_rate@gt15": rate(on_n), "hairpin_rate_conv@gt15": rate(on_c),
                "g2_tight_share@gt15": float((hn[2][on_n] < TIGHT_R_M).mean()) if on_n.any() else np.nan,
                "g2_tight_share_conv@gt15": float((hc[2][on_c] < TIGHT_R_M).mean()) if on_c.any() else np.nan,
                "hairpins@gt15": float(on_n.sum()), "hairpins_conv@gt15": float(on_c.sum()),
                "land_km2@unit": float(F.land.sum() * a), "street_km@unit": float(pc.length.sum() / 1000),
                "struct_share@unit": float(pc.length[~pc.ground].sum() / pc.length.sum()) if len(pc.length) else np.nan})
    if access:
        out.update(access_metrics(cv, pc, F, g, ped))
    return out


def _consolidated(g, F: HillField):
    """Junction clusters of junctions.consolidate at 10 m as centroids: (points, class, focal, nearest other)."""
    deg = np.bincount(g.edges.ravel(), minlength=g.n) if len(g.edges) else np.zeros(g.n, int)
    jid = np.flatnonzero(deg >= 3)
    if len(jid) == 0:
        z = np.zeros(0)
        return np.zeros((0, 2)), z.astype(int), z.astype(bool), z
    root = {int(j): int(j) for j in jid}
    for grp in consolidate(g, TAU_CONS, _ALL)["groups"]:
        for v in grp:
            root[v] = grp[0]
    keys = np.array([root[int(j)] for j in jid])
    _, inv = np.unique(keys, return_inverse=True)
    J = np.c_[np.bincount(inv, g.nodes[jid, 0]), np.bincount(inv, g.nodes[jid, 1])] / np.bincount(inv)[:, None]
    cls, _, inner, _, _ = F.lookup(J)
    nn = cKDTree(J).query(J, 2)[0][:, 1] if len(J) >= 2 else np.full(len(J), np.inf)
    return J, cls, inner, nn


# ---------------------------------------------------------------- M6, G1, pedestrian reach
def od_pairs(F: HillField, uid: str | None = None, tile: bool | None = None):
    """The D5 pairs: m19_eval.od_points with pixel area res^2 (it assumes 4 m pixels); pairs found with a KD tree in
    the same order. ``tile`` None: a 256-pixel grid is a tile (scale 1), as m19_eval."""
    ev = _script("m19_eval")
    elig = F.land & (F.slope < ev.STEEP_HI)
    steep = elig & (F.slope >= ev.STEEP_LO)
    reg, _ = ndimage.label(F.land, structure=np.ones((3, 3)))
    rng = np.random.default_rng(zlib.crc32((F.uid if uid is None else uid).encode()))
    res = F.S.res
    tile = F.land.shape[0] == 256 if tile is None else tile
    scale = 1.0 if tile else float(elig.sum() * res * res / 1e6) / 0.262144

    def pick(m, n):
        r, c = np.nonzero(m)
        if not len(r):
            return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, bool)
        k = rng.choice(len(r), size=min(int(round(n)), len(r)), replace=False)
        x = F.S.x0 + (c[k] + 0.5) * res + rng.uniform(-res / 2, res / 2, len(k))
        y = F.S.y1 - (r[k] + 0.5) * res + rng.uniform(-res / 2, res / 2, len(k))
        return np.c_[x, y], reg[r[k], c[k]], steep[r[k], c[k]]
    P1, R1, S1 = pick(steep, ev.N_STEEP * scale)
    P2, R2, S2 = pick(elig, ev.N_ANY * scale)
    P, R, S = np.vstack([P1, P2]), np.r_[R1, R2], np.r_[S1, S2]
    if len(P) < 2:
        return P, np.zeros(0, int), np.zeros(0, int)
    pr = cKDTree(P).query_pairs(ev.D_HI + 1e-6, output_type="ndarray")
    if len(pr) == 0:
        return P, np.zeros(0, int), np.zeros(0, int)
    pr = pr[np.lexsort((pr[:, 1], pr[:, 0]))]
    I, J = pr[:, 0], pr[:, 1]
    D = np.hypot(*(P[I] - P[J]).T)
    ok = (D >= ev.D_LO) & (D <= ev.D_HI) & (R[I] == R[J]) & (R[I] > 0) & (S[I] | S[J])
    return P, I[ok], J[ok]


def route_pairs(nodes, edges, P, I, J, steep=None, lex: bool = False, max_share: float = M6_SHARE, chunk: int = 256):
    """Per pair: outcome (0 served, 1 an endpoint not joined, 2 disconnected, 3 detour > 4, 4 steep share above
    ``max_share``), route street length, route steep length. Plain: shortest length (m19_eval.served, open graph).
    ``lex``: least steep length, ties by length (choice 13)."""
    ev = _script("m19_eval")
    nodes, edges = np.asarray(nodes, float), np.asarray(edges, int).reshape(-1, 2)
    nI = len(I)
    out, rl, rs = np.ones(nI, int), np.full(nI, np.nan), np.full(nI, np.nan)
    if nI == 0 or len(edges) == 0:
        return out, rl, rs
    L = np.hypot(*(nodes[edges[:, 1]] - nodes[edges[:, 0]]).T)
    Q = np.round(np.zeros(len(L)) if steep is None else np.asarray(steep, float) * 10)       # decimetres
    BIG = float(L.sum()) + 1.0 if lex else 0.0
    w = Q * BIG + L
    n = len(nodes)
    A = coo_matrix((np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    lab = connected_components(A, directed=False)[1]
    pts, e, t = _samples(nodes, edges)
    dist, idx = cKDTree(pts).query(P)
    joined = dist <= ev.JOIN_M
    ea, te = e[idx], t[idx]
    na, nb = edges[ea, 0], edges[ea, 1]
    both = joined[I] & joined[J]
    conn = both & (lab[na[I]] == lab[na[J]])
    out[both & ~conn] = 2
    qs = np.flatnonzero(conn)
    if not len(qs):
        return out, rl, rs
    straight = np.hypot(*(P[I] - P[J]).T)
    limit = np.inf if lex else float(ev.DETOUR_MAX * straight[qs].max() + 1.0)
    src_e = np.unique(ea[I[qs]])
    rowof = np.full(n, -1)
    for c0 in range(0, len(src_e), chunk):
        ce = src_e[c0:c0 + chunk]
        src = np.unique(edges[ce].ravel())
        rowof[src] = np.arange(len(src))
        Dn = dijkstra(A, indices=src, limit=limit)
        ks = qs[np.isin(ea[I[qs]], ce)]
        i, j = I[ks], J[ks]
        fi = np.c_[te[i], 1 - te[i]]
        fj = np.c_[te[j], 1 - te[j]]
        ui, vj = np.c_[na[i], nb[i]], np.c_[na[j], nb[j]]
        Li, Lj, Qi, Qj = L[ea[i]], L[ea[j]], Q[ea[i]], Q[ea[j]]
        best = np.full(len(ks), np.inf)
        blen, bq = np.full(len(ks), np.nan), np.full(len(ks), np.nan)
        for x in (0, 1):
            for y in (0, 1):
                D = Dn[rowof[ui[:, x]], vj[:, y]]
                fin = np.isfinite(D)
                D = np.where(fin, D, 0.0)
                qD = np.floor(D / BIG) if lex else np.zeros(len(D))
                lD = D - qD * BIG if lex else D
                if lex:                                   # float guard of the decoding (length < BIG)
                    neg = lD < 0
                    qD, lD = np.where(neg, qD - 1, qD), np.where(neg, lD + BIG, lD)
                qt = qD + np.round(fi[:, x] * Qi) + np.round(fj[:, y] * Qj)
                lt = lD + fi[:, x] * Li + fj[:, y] * Lj
                W = qt * BIG + lt if lex else lt
                W = np.where(fin, W, np.inf)
                b = W < best
                best[b], blen[b], bq[b] = W[b], lt[b], qt[b]
        same = ea[i] == ea[j]
        if same.any():
            dt = np.abs(te[i] - te[j])[same]
            qt, lt = np.round(dt * Qi[same]), dt * Li[same]
            W = qt * BIG + lt if lex else lt
            b = W <= best[same]
            sel = np.flatnonzero(same)[b]
            best[sel], blen[sel], bq[sel] = W[b], lt[b], qt[b]
        rl[ks], rs[ks] = blen, bq / 10
        detour = (blen + dist[i] + dist[j]) / straight[ks]
        o = np.where(~np.isfinite(best) | ~(detour <= ev.DETOUR_MAX), 3, 0)
        if lex:
            o = np.where((o == 0) & (bq / 10 > max_share * blen + 1e-9), 4, o)
        out[ks] = o
        rowof[src] = -1
    return out, rl, rs


def window_grades(g, F: HillField, pc: Pieces, window: float = G1_WINDOW_M) -> np.ndarray:
    """Grade (%) over a ``window`` m window centred on each piece along its ground chain (choice 14); NaN off ground."""
    wg = np.full(len(pc.length), np.nan)
    if len(g.edges) == 0:
        return wg
    nid = {(float(x), float(y)): i for i, (x, y) in enumerate(g.nodes)}
    eid = {(int(a), int(b)): i for i, (a, b) in enumerate(g.edges)}
    off = pc.off
    for poly in _M18.ground_polylines(g):
        ids = [nid.get((float(x), float(y))) for x, y in poly]
        es = [eid.get((min(u, v), max(u, v))) if u is not None and v is not None else None
              for u, v in zip(ids[:-1], ids[1:])]
        if any(e is None for e in es):                    # coincident nodes on two levels: left unmeasured
            continue
        seg = np.hypot(*np.diff(poly, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        idx, smid = [], []
        for k_, (u, e) in enumerate(zip(ids[:-1], es)):
            r = np.arange(off[e], off[e + 1])
            f = (pc.k[r] + 0.5) / pc.n[e]
            idx.append(r)
            smid.append(s[k_] + (f if g.edges[e, 0] == u else 1 - f) * seg[k_])
        idx, smid = np.concatenate(idx), np.concatenate(smid)
        wg[idx] = np.abs(_window_grade(poly, s, smid, F, window))
    return wg


def split_graph(g, pc: Pieces, keep: np.ndarray):
    """(nodes, edges) of the kept pieces; an edge is cut where ``keep`` changes along it (new nodes there)."""
    nodes, E = [g.nodes], []
    off, n_new = pc.off, g.n
    for e in range(len(g.edges)):
        kk = keep[off[e]:off[e + 1]]
        if kk.all():
            E.append(g.edges[e])
            continue
        if not kk.any():
            continue
        a, b = g.nodes[g.edges[e]]
        m = len(kk)
        st = np.flatnonzero(np.diff(np.r_[0, kk.astype(int), 0]) == 1)
        en = np.flatnonzero(np.diff(np.r_[0, kk.astype(int), 0]) == -1)
        for k0, k1 in zip(st, en):
            ends = []
            for kb in (k0, k1):
                if kb == 0:
                    ends.append(g.edges[e, 0])
                elif kb == m:
                    ends.append(g.edges[e, 1])
                else:
                    nodes.append((a + kb / m * (b - a))[None])
                    ends.append(n_new)
                    n_new += 1
            E.append(np.array(ends))
    return np.vstack(nodes), np.array(E, int).reshape(-1, 2)


def access_metrics(cv, pc: Pieces, F: HillField, g_native, ped=None) -> dict:
    """M6 at 15 / 12 / 10 % (and, descriptively, by the class of the steeper endpoint), G1 at 15 / 12 % and pedestrian
    reach, as shares of the D5 pairs (unit level). M6 is conservative (module docstring)."""
    P, I, J = od_pairs(F)
    out = {"pairs@unit": float(len(I)), "_outcomes": {}}
    share = lambda o: float((o == 0).mean()) if len(o) else np.nan
    pcls, pslope = F.lookup(P)[:2]
    steep_end = np.where(pslope[I] >= pslope[J], I, J) if len(I) else np.zeros(0, int)
    ecls = pcls[steep_end]
    for th in M6_STEEP:
        st = np.bincount(pc.edge, weights=pc.length * (np.nan_to_num(pc.grade) > th), minlength=len(cv.edges))
        o = route_pairs(cv.nodes, cv.edges, P, I, J, steep=st, lex=True)[0]
        out[f"m6_{th:g}@unit"] = share(o)
        for q in (2, 3, 4):                                # steep endpoints lie on 15-35 % land
            out[f"m6_{th:g}_steepcls@{KEYS[q]}"] = share(o[ecls == q])
        out["_outcomes"][f"m6_{th:g}"] = np.bincount(o, minlength=5).tolist()
    wg = window_grades(cv, F, pc)
    for cap in G1_CAPS:
        o = route_pairs(*split_graph(cv, pc, ~(wg > cap)), P, I, J)[0]
        out[f"g1_{cap:g}@unit"] = share(o)
        out["_outcomes"][f"g1_{cap:g}"] = np.bincount(o, minlength=5).tolist()
    pg, kind = pedestrian_graph(g_native, ped)
    pp = pieces(pg, F)
    o = route_pairs(pg.nodes, pg.edges, P, I, J)[0]
    out["ped_stairs@unit"] = share(o)
    keep = (kind[pp.edge] != 2) & ~(window_grades(pg, F, pp) > RAMP_PCT)
    o2 = route_pairs(*split_graph(pg, pp, keep), P, I, J)[0]
    out["ped_ramp8@unit"] = share(o2)
    out["_outcomes"]["ped_stairs"] = np.bincount(o, minlength=5).tolist()
    out["_outcomes"]["ped_ramp8"] = np.bincount(o2, minlength=5).tolist()
    return out


def pedestrian_graph(g, ped=None):
    """Convention build of the vehicular chains plus the pedestrian lines: (graph, kind per edge: 0 vehicular,
    1 footway / path, 2 steps; an edge shared with a street is a street)."""
    ch = chains(g)
    lines = [c for c, _, _ in ch]
    kinds = [0] * len(ch)
    lv, st = [l_ for _, l_, _ in ch], [s_ for _, _, s_ in ch]
    if ped is not None:
        pl, steps = ped
        lines += [np.asarray(x, float)[:, :2] for x in pl]
        kinds += [2 if s else 1 for s in steps]
        lv += [0] * len(pl)
        st += [0] * len(pl)
    if all(a == 0 and b == 0 for a, b in zip(lv, st)):
        b = build_graph(lines, classes=kinds)
    else:
        b = build_graph(lines, classes=kinds, levels=lv, structs=st)
    return b, np.asarray(b.classes(), int)


# ---------------------------------------------------------------- scoring (D3a)
@dataclass(frozen=True)
class Metric:
    transform: str            # "log": log(x + c); "logit": logit((x + c) / (1 + 2c)); "id": x   (D3a families)
    unit: float               # D3b measurement unit, in the metric's units ...
    per: str | None = None    # ... divided by the median of this cell quantity over the reference towns
    street: bool = True       # a street-shape metric: >= 1 km of real street in the cell (definedness, reference)


_LOG = lambda u, per=None, s=True: Metric("log", u, per, s)
_LGT = lambda u, per=None, s=True: Metric("logit", u, per, s)
_ID = lambda u, s=True: Metric("id", u, None, s)
_ACCESS = _LGT(0.001, s=False)
METRICS = {                                                # units: the D3b table
    "density": _LOG(0.03, "land_km2", False),              # 0.03 km / median class land
    "coverage": _LGT(0.001, "land_km2", False),            # 0.001 km2 / median class land
    "share_gt15": _LGT(0.01, "street_km"), "lens_share": _LGT(0.01, "street_km"),            # 10 m / median street
    "lens_share_2ap": _LGT(0.01, "street_km"), "lcc_share": _LGT(0.01, "street_km"),
    "share_3way": _LGT(1.0, "junctions"), "share_4way": _LGT(1.0, "junctions"),              # 1 / median junctions
    "close15": _LGT(1.0, "junctions_focal"),
    "deadend_per_km": _LOG(1.0, "street_km"), "hairpin_rate": _LOG(1.0, "street_km"),       # 1 / median street km
    "hairpin_rate_conv": _LOG(1.0, "street_km"), "hairpin_rate_cls": _LOG(1.0, "street_km"),
    "loops_per_km2": _LOG(1.0, "land_km2"),                # 1 / median land km2
    "g2_tight_share": _LGT(1.0, "hairpins"), "g2_tight_share_conv": _LGT(1.0, "hairpins_conv"),   # 1 / median count
    "run_p50": _LOG(10.0), "run_p90": _LOG(10.0),          # 10 m
    "contour_index": _ID(0.01), "phi_med": _ID(0.01),
    "grade_p90": _LOG(0.5), "grade_change_p95": _LOG(0.5), "grade_change_p95_conv": _LOG(0.5),   # 0.5 pp
    "face_area_med": _LOG(64.0), "face_area_p90": _LOG(64.0),                                     # 64 m2
    "face_area_p10": _LOG(64.0), "small_face_share": _LGT(1.0, "faces"),                         # D3c mesh
    "acute_share": _LGT(1.0, "junctions"), "jj_len_med": _LOG(10.0),
    "block_contour_cos": _ID(0.03), "tortuosity": _ID(0.003),
    **{f"m6_{t:g}": _ACCESS for t in M6_STEEP}, **{f"g1_{t:g}": _ACCESS for t in G1_CAPS},
    "ped_stairs": _ACCESS, "ped_ramp8": _ACCESS,
}
CELLS = {m: KEYS for m in METRICS}
CELLS.update({"contour_index": KEYS[1:], "block_contour_cos": KEYS[2:]})
CELLS.update({m: ("gt15",) for m in ("hairpin_rate", "hairpin_rate_conv", "g2_tight_share", "g2_tight_share_conv")})
CELLS.update({m: ("unit",) for m in METRICS if m.startswith(("m6_", "g1_", "ped_"))})


def cells(*metrics) -> tuple:
    return tuple(f"{m}@{c}" for m in metrics for c in CELLS[m])


GROUPS = {
    "M1": cells("density", "coverage"), "M2": cells("contour_index", "grade_p90", "share_gt15"),
    "M3": cells("run_p50", "run_p90", "hairpin_rate"), "runs": cells("run_p50", "run_p90"),
    "hairpin": cells("hairpin_rate"),
    "M4": cells("deadend_per_km", "share_3way", "share_4way", "close15", "lcc_share", "loops_per_km2",
                "lens_share_2ap"),                                   # D3c: 2A/P primary
    "lens": cells("lens_share_2ap"), "lens_topo": cells("lens_share"),
    "lcc": cells("lcc_share"), "loops": cells("loops_per_km2"),
    "mesh": cells("face_area_p10", "small_face_share", "acute_share", "jj_len_med"),
    "M5": cells("face_area_med", "face_area_p90", "phi_med", "block_contour_cos"), "M7": cells("tortuosity"),
    "M6": cells("m6_15"), "M6_sec": cells("m6_12", "m6_10"), "G1": cells("g1_15"),
    "pedestrian": cells("ped_stairs", "ped_ramp8"),
}
# P2, the untargeted composite (D4 "Consequences for D3"; M3 moved to targeted). Change it here only.
P2_CELLS = cells("lens_share_2ap", "deadend_per_km", "lcc_share", "phi_med", "block_contour_cos", "tortuosity")
P2_CELLS_TOPO = cells("lens_share", "deadend_per_km", "lcc_share", "phi_med", "block_contour_cos", "tortuosity")
MESH_CELLS = GROUPS["mesh"]                                   # D3c: descriptive, their own frozen plan
# The suite composite of the validation gate (choice 15): every realism cell of M1-M5 and M7.
SUITE_CELLS = tuple(dict.fromkeys(sum((GROUPS[k] for k in ("M1", "M2", "M3", "M4", "M5", "M7")), ())))
ADVERSARIAL_TARGETS = {"switchback": ("M1", "hairpin"), "leastcostpath": ("lens",), "fall_line_grid": ("M2",),
                       "raster_zigzag": ("M7",), "contour_parallel": ("lcc", "loops")}
ADVERSARIAL_P2 = ("leastcostpath", "contour_parallel", "raster_zigzag")      # D3a: also above the real P2 p95
TARGET_SIGN = {"density": -1, "coverage": -1, "hairpin_rate": 1, "lens_share_2ap": 1, "lens_share": 1,   # D3c
               "contour_index": -1, "grade_p90": 1, "share_gt15": 1, "tortuosity": 1, "lcc_share": -1,
               "loops_per_km2": -1}
GATE2_MIN_SHARE = 0.9                                                         # D3a: fail on >= 90 % of towns


@dataclass(frozen=True)
class Z:
    z: float                  # signed, capped (NaN when the cell was scored by rule)
    absz: float
    flag: str = ""            # "", "gen_lt_1km", "gen_undefined"


@dataclass(frozen=True)
class Plan:
    """Frozen from all real dev-H towns: the scored keys (20-30 / > 30 replaced by 20-inf where pooled), c per key,
    and the declared keys left descriptive."""
    keys: tuple
    c: dict
    descriptive: tuple
    n_ref: dict
    unit: dict                # D3b measurement unit per key
    floor: dict               # D3b scale floor per key (frozen with c)


@dataclass(frozen=True)
class RefCell:
    c: float
    median: float
    scale: float
    n: int
    binding: str              # which term set the scale: "mad", "qrange" or "floor" (the declared deviation)


@dataclass(frozen=True)
class Reference:
    plan: Plan
    cells: dict               # scored key -> RefCell


def _street_km(row: dict, cell: str) -> float:
    return float(np.median(np.atleast_1d(row.get(f"street_km@{cell}", 0.0))))


def row_get(row: dict, key: str) -> float:
    v = row.get(key, np.nan)
    return float(v) if v is not None else np.nan


def _land_ok(row: dict, cell: str) -> bool:
    return cell == "unit" or row_get(row, f"land_km2@{cell}") >= CELL_LAND_KM2


def defined(real: dict, key: str) -> bool:
    """Cell rules on the unit's own real network: >= 0.05 km2 of class land, >= 1 km of real street (street
    metrics), a finite real value."""
    m, c = key.split("@")
    if not _land_ok(real, c) or (METRICS[m].street and _street_km(real, c) < STREET_KM_MIN):
        return False
    return bool(np.isfinite(row_get(real, key)))


def ref_ok(row: dict, key: str) -> bool:
    """A real town enters a cell's reference with >= 0.05 km2 of class land and, for street-shape metrics only,
    >= 1 km of street there (D3b)."""
    m, c = key.split("@")
    if not _land_ok(row, c) or (METRICS[m].street and _street_km(row, c) < STREET_KM_MIN):
        return False
    return bool(np.isfinite(row_get(row, key)))


def pooled_key(key: str) -> str:
    m, c = key.split("@")
    return f"{m}@{POOLED}" if c in KEYS[3:] else key


def expand(keys) -> set:
    """Declared keys plus their pooled 20-inf form (to restrict a composite whatever the plan pooled)."""
    return set(keys) | {pooled_key(k) for k in keys}


def t_of(key: str, x, c: float):
    tr = METRICS[key.split("@")[0]].transform
    x = np.asarray(x, float)
    if tr == "log":
        return np.log(x + c)
    if tr == "logit":
        p = (x + c) / (1 + 2 * c)
        return np.log(p) - np.log1p(-p)
    return x


def _floor(key: str, med_x: float, c: float, unit: float) -> float:
    """D3b: the span on the t scale of one measurement unit at the reference median, / sqrt(12) (the SD of a uniform
    quantisation error). The unit is shifted inside the metric's domain at a bound."""
    tr = METRICS[key.split("@")[0]].transform
    if tr == "id":
        return unit / math.sqrt(12)
    lo = max(med_x - unit / 2, 0.0)
    hi = lo + unit
    if tr == "logit" and hi > 1.0:
        hi, lo = 1.0, max(1.0 - unit, 0.0)
    return float(abs(t_of(key, hi, c) - t_of(key, lo, c))) / math.sqrt(12)


def _unit(key: str, rows) -> float:
    """D3b measurement unit of a key: a constant, or a constant / the median (over the reference towns, positive
    values) of the cell's denominator."""
    m, cell = key.split("@")
    spec = METRICS[m]
    if spec.per is None:
        return spec.unit
    d = np.array([row_get(r, f"{spec.per}@{cell}") for r in rows if ref_ok(r, key)])
    d = d[np.isfinite(d) & (d > 0)]
    return spec.unit / float(np.median(d)) if len(d) else spec.unit


def make_plan(rows, keys) -> Plan:
    """D3a: a cell is scored when >= 8 real towns have >= 1 km of street in it; otherwise 20-30 and > 30 are pooled
    into 20-inf (both, when either is short; choice 14); otherwise the cell is descriptive. c = 0.1 x the median of
    the positive reference values (log and logit metrics); a key with no positive value is descriptive."""
    rows = list(rows)
    nref = lambda k: sum(ref_ok(r, k) for r in rows)
    eff, desc = [], []
    for k in keys:
        m, cell = k.split("@")
        if cell in KEYS[3:]:
            pair = [f"{m}@{x}" for x in KEYS[3:]]
            if all(nref(p_) >= REF_MIN_N for p_ in pair):
                eff.append(k)
            elif nref(pooled_key(k)) >= REF_MIN_N:
                eff.append(pooled_key(k))
            else:
                desc.append(k)
        elif nref(k) >= REF_MIN_N:
            eff.append(k)
        else:
            desc.append(k)
    eff = list(dict.fromkeys(eff))
    c, keep, unit, floor = {}, [], {}, {}
    for k in eff:
        v = np.array([row_get(r, k) for r in rows if ref_ok(r, k)])
        if METRICS[k.split("@")[0]].transform == "id":
            c[k] = 0.0
        else:
            if not (v > 0).any():
                desc.append(k)
                continue
            c[k] = 0.1 * float(np.median(v[v > 0]))
        unit[k] = _unit(k, rows)
        floor[k] = _floor(k, float(np.median(v)), c[k], unit[k])
        keep.append(k)
    return Plan(tuple(keep), c, tuple(desc), {k: nref(k) for k in keep}, unit, floor)


def fit_reference(rows, keys, plan: Plan | None = None) -> Reference:
    """Per scored key: median of t(reference), scale = max(1.4826 MAD, (q90 - q10) / 2.563, floor) with the binding
    term logged; c and the floor come frozen in ``plan`` (from all real towns; built from ``rows`` when None)."""
    rows = list(rows)
    plan = make_plan(rows, keys) if plan is None else plan
    out = {}
    for k in plan.keys:
        v = np.array([row_get(r, k) for r in rows if ref_ok(r, k)])
        if len(v) < 2:
            continue
        t = t_of(k, v, plan.c[k])
        med = float(np.median(t))
        q10, q90 = np.quantile(t, [0.1, 0.9])
        terms = {"mad": 1.4826 * float(np.median(np.abs(t - med))), "qrange": float(q90 - q10) / 2.563,
                 "floor": plan.floor[k]}
        b = max(terms, key=terms.get)
        out[k] = RefCell(plan.c[k], med, terms[b], len(v), b)
    return Reference(plan, out)


def floor_bound(ref: Reference) -> list:
    """The cells whose scale is set by the floor (D3b: their number on real dev-H is reported)."""
    return [k for k, rc in ref.cells.items() if rc.binding == "floor"]


def score(x: dict, real: dict, ref: Reference, keys=None) -> dict:
    """Robust z of network ``x`` on a unit whose real network is ``real``, over the scored keys (restricted to the
    declared ``keys``)."""
    want = None if keys is None else expand(keys)
    out = {}
    for k in ref.plan.keys:
        if (want is not None and k not in want) or k not in ref.cells or not defined(real, k):
            continue
        m, cell = k.split("@")
        if METRICS[m].street and _street_km(x, cell) < STREET_KM_MIN:
            out[k] = Z(np.nan, Z_CAP, "gen_lt_1km")
            continue
        v = row_get(x, k)
        if not np.isfinite(v):
            out[k] = Z(np.nan, Z_CAP, "gen_undefined")
            continue
        rc = ref.cells[k]
        z = float(np.clip((t_of(k, v, rc.c) - rc.median) / rc.scale, -Z_CAP, Z_CAP))
        out[k] = Z(z, abs(z))
    return out


def composite(s: dict, keys=None) -> float:
    """Equal-weight mean |z| over the scored cells (restricted to the declared ``keys``)."""
    want = None if keys is None else expand(keys)
    v = [z.absz for k, z in s.items() if want is None or k in want]
    return float(np.mean(v)) if v else np.nan


def signed(s: dict, keys) -> float:
    """D3c: mean sign-oriented z (``TARGET_SIGN``) over the scored cells of ``keys``; cells scored by rule (NaN z)
    are left out; NaN when none remains."""
    want = expand(keys)
    v = [TARGET_SIGN[k.split("@")[0]] * z.z for k, z in s.items() if k in want and np.isfinite(z.z)]
    return float(np.mean(v)) if v else np.nan


def loo(rows: dict, keys, plan: Plan | None = None) -> dict:
    """Leave-one-out |z| of every real town (reference = the other towns; plan frozen from all)."""
    plan = make_plan(rows.values(), keys) if plan is None else plan
    return {t: score(rows[t], rows[t], fit_reference([rows[q] for q in rows if q != t], keys, plan)) for t in rows}


def reference_draws(rows: dict, keys, B: int = 1000, seed: int = 0, plan: Plan | None = None) -> list:
    """References refitted on towns resampled with replacement (the bootstrap over towns)."""
    plan = make_plan(rows.values(), keys) if plan is None else plan
    rng, names = np.random.default_rng(seed), list(rows)
    return [fit_reference([rows[names[i]] for i in rng.integers(0, len(names), len(names))], keys, plan)
            for _ in range(B)]


def structure_flags(x: dict, rows) -> dict:
    """D3a: per cell, the network's bridge + tunnel share against the real 95th percentile (flag when above)."""
    out = {}
    for cell in KEYS + (POOLED, "unit"):
        k = f"struct_share@{cell}"
        v = np.array([row_get(r, k) for r in rows])
        v, xv = v[np.isfinite(v)], row_get(x, k)
        if len(v) and np.isfinite(xv):
            p = float(np.percentile(v, 95))
            out[cell] = {"share": xv, "real_p95": p, "flag": bool(xv > p)}
    return out


def gate_real(rows: dict, keys=SUITE_CELLS, plan: Plan | None = None) -> dict:
    """Validation gate 1: pooled LOO coverage (|z| <= 2) >= (n - 1)/(n + 1) - 0.05, towns under 0.75 flagged, and
    the KS test of the LOO composite percentiles; each pair of towns scored against a reference excluding both
    (D3a). The KS p is APPROXIMATE (the percentiles are dependent)."""
    from scipy.stats import kstest
    plan = make_plan(rows.values(), keys) if plan is None else plan
    L = loo(rows, keys, plan)
    names, n = list(rows), len(rows)
    allz = np.array([z.absz for t in L for z in L[t].values()])
    per = {t: float(np.mean([z.absz <= Z_FIT for z in L[t].values()])) if L[t] else np.nan for t in L}
    A = {}
    for i, r in enumerate(names):
        for q in names[i + 1:]:
            ref = fit_reference([rows[s_] for s_ in names if s_ not in (r, q)], keys, plan)
            A[(r, q)] = composite(score(rows[r], rows[r], ref))
            A[(q, r)] = composite(score(rows[q], rows[q], ref))
    pct = {}
    for r in names:
        cmp = [(A[(r, q)], A[(q, r)]) for q in names if q != r and np.isfinite(A[(r, q)]) and np.isfinite(A[(q, r)])]
        if cmp:
            k = sum((b < a) + 0.5 * (b == a) for a, b in cmp)
            pct[r] = float((k + 0.5) / (len(cmp) + 1))
    cov = float(np.mean(allz <= Z_FIT)) if len(allz) else np.nan
    thr = (n - 1) / (n + 1) - 0.05
    full = fit_reference(rows.values(), keys, plan)
    ks = float(kstest(list(pct.values()), "uniform").pvalue) if len(pct) >= 2 else np.nan
    return {"n": n, "coverage": cov, "threshold": thr, "coverage_ok": bool(cov >= thr), "per_town": per,
            "flagged": [t for t, v in per.items() if v < 0.75], "composite": {t: composite(L[t]) for t in L},
            "percentiles": pct, "ks_p_approx": ks, "ks_ok": bool(ks > 0.05), "pass": bool(cov >= thr and ks > 0.05),
            "loo": L, "scale_binding": dict(Counter(rc.binding for rc in full.cells.values())),
            "floor_bound": len(floor_bound(full)), "floor_bound_cells": floor_bound(full)}


def gate_adversarial(adv: dict, real_unit: dict, rows: dict, case: str, keys=SUITE_CELLS, town: str | None = None,
                     plan: Plan | None = None, loo_out: dict | None = None) -> dict:
    """Validation gate 2 for one adversarial network on one unit (``case``: an ``ADVERSARIAL_TARGETS`` name). It
    fails (as it must) when: its suite composite (mean |z|) is above the 95th percentile of the real LOO composites;
    one of its targeted groups has a mean sign-oriented z (``signed``, D3c) above the 95th percentile of the real LOO
    sign-oriented group means; for ``ADVERSARIAL_P2`` its P2 composite (mean |z|) is above the real LOO P2 95th
    percentile; for the zigzag its sign-oriented M7 is above the largest real LOO value (D3a, D3c). Margins are
    reported. ``town``: the unit's own real town, left out of the reference."""
    plan = make_plan(rows.values(), keys) if plan is None else plan
    L = loo(rows, keys, plan) if loo_out is None else loo_out
    s = score(adv, real_unit, fit_reference([rows[t] for t in rows if t != town], keys, plan))
    p95 = lambda ks_: float(np.nanpercentile([composite(L[t], ks_) for t in L], 95))
    sp95 = lambda ks_: float(np.nanpercentile([signed(L[t], ks_) for t in L], 95))
    out = {"case": case, "composite": composite(s), "null_p95": p95(None), "groups": {}, "z": s}
    out["composite_fails"] = bool(out["composite"] > out["null_p95"])
    out["margins"] = {"composite": out["composite"] - out["null_p95"]}
    for gname in ADVERSARIAL_TARGETS[case]:
        gv, gp = signed(s, GROUPS[gname]), sp95(GROUPS[gname])
        out["groups"][gname] = {"mean_signed_z": gv, "mean_absz": composite(s, GROUPS[gname]), "null_p95": gp,
                                "fails": bool(np.isfinite(gv) and gv > gp)}
        out["margins"][gname] = gv - gp
    out["targeted_fails"] = any(v["fails"] for v in out["groups"].values())
    checks = [out["composite_fails"], out["targeted_fails"]]
    if case in ADVERSARIAL_P2:
        out["p2"], out["p2_null_p95"] = composite(s, P2_CELLS), p95(P2_CELLS)
        out["p2_fails"] = bool(out["p2"] > out["p2_null_p95"])
        out["margins"]["p2"] = out["p2"] - out["p2_null_p95"]
        checks.append(out["p2_fails"])
    if case == "raster_zigzag":
        out["m7"] = signed(s, GROUPS["M7"])
        out["m7_null_max"] = float(np.nanmax([signed(L[t], GROUPS["M7"]) for t in L]))
        out["m7_fails"] = bool(np.isfinite(out["m7"]) and out["m7"] > out["m7_null_max"])
        out["margins"]["m7"] = out["m7"] - out["m7_null_max"]
        checks.append(out["m7_fails"])
    out["fails"] = bool(all(checks))
    return out


def gate_adversarial_towns(adv_rows: dict, rows: dict, case: str, keys=SUITE_CELLS, plan: Plan | None = None,
                           loo_out: dict | None = None) -> dict:
    """D3a gate 2 for one case: the adversarial built on every dev-H town (``adv_rows`` town -> its metrics) must fail
    on at least 90 % of them. Nothing is tuned to make a case pass; a failure is recorded as a limitation."""
    plan = make_plan(rows.values(), keys) if plan is None else plan
    L = loo(rows, keys, plan) if loo_out is None else loo_out
    per = {t: gate_adversarial(adv_rows[t], rows[t], rows, case, keys, town=t, plan=plan, loo_out=L) for t in adv_rows}
    share = float(np.mean([r["fails"] for r in per.values()])) if per else np.nan
    names = sorted({m for r in per.values() for m in r["margins"]})
    margins = {m: {"min": float(np.nanmin(v)), "median": float(np.nanmedian(v))}
               for m in names if len(v := [r["margins"].get(m, np.nan) for r in per.values()])}
    return {"case": case, "towns": len(per), "share_failing": share, "pass": bool(share >= GATE2_MIN_SHARE),
            "margins": margins, "per_town": per}


def gate_site_control(adv: dict, real_site: dict, rows: dict, case: str, keys=SUITE_CELLS, plan: Plan | None = None,
                      loo_out: dict | None = None) -> dict:
    """D3c: a single-site adversarial (the leastcostpath case on Petra, n = 1, a declared deviation from D3a's 90 %
    rule) and the site's REAL network as control, both scored as ``gate_adversarial`` on the same field against the
    whole dev-H reference (the site is not a dev-H unit), definedness from the real site network. A check catches the
    adversarial only if the adversarial fails it and the real network does not; if both fail it the check is
    indeterminate. Reported with every cell's z for both and the per-group differences (adversarial - real)."""
    plan = make_plan(rows.values(), keys) if plan is None else plan
    L = loo(rows, keys, plan) if loo_out is None else loo_out
    a = gate_adversarial(adv, real_site, rows, case, keys, town=None, plan=plan, loo_out=L)
    r = gate_adversarial(real_site, real_site, rows, case, keys, town=None, plan=plan, loo_out=L)
    checks = ["composite_fails"] + [f"group:{g_}" for g_ in a["groups"]] + (["p2_fails"] if "p2_fails" in a else [])
    get = lambda o, c: o["groups"][c[6:]]["fails"] if c.startswith("group:") else o[c]
    verdict = {c: ("caught" if get(a, c) and not get(r, c) else "indeterminate" if get(a, c) and get(r, c)
                   else "not caught") for c in checks}
    groups = {g_: GROUPS[g_] for g_ in ("M1", "M2", "M3", "M4", "M5", "M7", "lens", "lcc", "loops")}
    topo = make_plan(rows.values(), GROUPS["lens_topo"])          # the sensitivity has its own plan (not in the suite)
    rtopo = fit_reference(rows.values(), GROUPS["lens_topo"], topo)
    zta, ztr = score(adv, real_site, rtopo), score(real_site, real_site, rtopo)
    ref = fit_reference(rows.values(), keys, plan)
    za, zr = score(adv, real_site, ref), score(real_site, real_site, ref)
    zt = lambda z_: {k: {"z": v.z, "absz": v.absz, "flag": v.flag} for k, v in z_.items()}
    p2g = {m_: cells(m_) for m_ in ("lens_share_2ap", "deadend_per_km", "lcc_share", "phi_med", "block_contour_cos",
                                    "tortuosity")}
    return {"case": case, "n_sites": 1, "deviation": "single site (n = 1), not D3a's >= 90 % of towns",
            "adversarial": {k: v for k, v in a.items() if k != "z"}, "real_control": {k: v for k, v in r.items() if k != "z"},
            "verdict": verdict,
            "caught_all": bool(all(v == "caught" for v in verdict.values())),
            "defined_cells": sorted(za), "z_adversarial": zt(za), "z_real": zt(zr),
            "group_absz_diff": {g_: composite(za, k_) - composite(zr, k_) for g_, k_ in groups.items()},
            "p2_components_absz": {m_: {"adversarial": composite(za, k_), "real": composite(zr, k_)} for m_, k_ in p2g.items()},
            "lens_topo_signed": {"adversarial": signed(zta, GROUPS["lens_topo"]), "real": signed(ztr, GROUPS["lens_topo"])}}


# ---------------------------------------------------------------- C2ST (secondary)
C2ST_FEATURES = ("density", "coverage", "contour_index", "grade_p90", "share_gt15", "run_p50", "run_p90",
                 "hairpin_rate_cls", "deadend_per_km", "share_3way", "share_4way", "close15", "lcc_share",
                 "loops_per_km2", "lens_share_2ap", "face_area_med", "face_area_p90", "phi_med", "block_contour_cos")
M1_FEATURES = ("density", "coverage")


def seed_for_window(window_id: str, seeds) -> int:
    """One generated seed per window, by hash (real and generated cells 1:1)."""
    return seeds[zlib.crc32(window_id.encode()) % len(seeds)]


def cell_rows(flat: dict, town: str, label: int, window: str = "") -> list:
    """The (window, slope class) cells of one unit_metrics row, for c2st_cells."""
    return [{"town": town, "window": window, "label": label, "cls": k, "land_km2": row_get(flat, f"land_km2@{k}"),
             "street_km": row_get(flat, f"street_km@{k}"), **{f: row_get(flat, f"{f}@{k}") for f in C2ST_FEATURES}}
            for k in KEYS]


def logistic_fit(X, y, C: float = 1.0):
    """L2 logistic regression, sklearn's objective: 0.5 |w|^2 + C sum log(1 + exp(-y' (Xw + b))), b unpenalised."""
    from scipy.optimize import minimize
    from scipy.special import expit
    X, ys = np.asarray(X, float), 2 * np.asarray(y, float) - 1
    d = X.shape[1]

    def f(wb):
        m = -ys * (X @ wb[:d] + wb[d])
        gz = -ys * expit(m)
        return 0.5 * wb[:d] @ wb[:d] + C * np.logaddexp(0, m).sum(), np.r_[wb[:d] + C * X.T @ gz, C * gz.sum()]
    r = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return r.x[:d], float(r.x[d])


def auc(y, s) -> float:
    from scipy.stats import rankdata
    y, s = np.asarray(y), np.asarray(s, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _oof(X, y, town, C):
    import warnings
    s = np.full(len(y), np.nan)
    for t in np.unique(town):
        te, tr = town == t, town != t
        if len(np.unique(y[tr])) < 2:
            s[te] = 0.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)            # a column with no value in this fold
            mu = np.nan_to_num(np.nanmean(np.where(np.isfinite(X[tr]), X[tr], np.nan), 0))
        Xtr = np.where(np.isfinite(X[tr]), X[tr], mu)
        sd = Xtr.std(0)
        sd[sd < 1e-12] = 1.0
        w, b = logistic_fit((Xtr - mu) / sd, y[tr], C)
        s[te] = ((np.where(np.isfinite(X[te]), X[te], mu) - mu) / sd) @ w + b
    return s


def c2st(X, y, town, n_boot: int = 1000, n_perm: int = 1999, seed: int = 0, C: float = 1.0) -> dict:
    """Leave-one-town-out C2ST: pooled out-of-fold AUC, town-bootstrap 95 % CI, within-town label-permutation p
    (>= 1,999 permutations, so that it can reject under Holm; D3a)."""
    X, y, town = np.asarray(X, float), np.asarray(y, int), np.asarray(town)
    rng = np.random.default_rng(seed)
    s = _oof(X, y, town, C)
    a = auc(y, s)
    names = np.unique(town)
    rows = {t: np.flatnonzero(town == t) for t in names}
    bs = [auc(y[ix], s[ix]) for ix in (np.concatenate([rows[t] for t in rng.choice(names, len(names))])
                                      for _ in range(n_boot))]
    perm = []
    for _ in range(n_perm):
        yp = y.copy()
        for t in names:
            yp[rows[t]] = rng.permutation(y[rows[t]])
        perm.append(auc(yp, _oof(X, yp, town, C)))
    perm = np.array(perm)
    return {"auc": a, "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))] if n_boot else None,
            "p": float((1 + (perm >= a).sum()) / (1 + n_perm)) if n_perm else np.nan,
            "cells": int(len(y)), "real": int((y == 0).sum()), "towns": int(len(names))}


def c2st_cells(cells_: list, n_boot: int = 1000, n_perm: int = 1999, seed: int = 0, min_real: int = 40) -> dict:
    """C2ST per slope class (20-30 and > 30 pooled if either has < ``min_real`` real cells), with all features and
    without M1. Cells need >= 0.02 km2 of land; cells without streets are kept with an indicator."""
    cells_ = [c for c in cells_ if c["land_km2"] >= C2ST_LAND_KM2]
    nreal = Counter(c["cls"] for c in cells_ if c["label"] == 0)
    groups = {k: (k,) for k in KEYS[:3]}
    if nreal[KEYS[3]] < min_real or nreal[KEYS[4]] < min_real:
        groups["20-inf"] = KEYS[3:]
    else:
        groups.update({KEYS[3]: (KEYS[3],), KEYS[4]: (KEYS[4],)})
    out = {"pooled_steep": "20-inf" in groups}
    for name, ks in groups.items():
        sub = [c for c in cells_ if c["cls"] in ks]
        if not sub:
            continue
        y = np.array([c["label"] for c in sub])
        town = np.array([c["town"] for c in sub])
        for variant, feats in (("all", C2ST_FEATURES), ("no_M1", tuple(f for f in C2ST_FEATURES if f not in M1_FEATURES))):
            X = np.array([[c[f] for f in feats] + [float(not c["street_km"] > 0)] for c in sub], float)
            X = X[:, np.isfinite(X).any(0)]
            out[f"{name}|{variant}"] = c2st(X, y, town, n_boot, n_perm, seed)
    return out


# ---------------------------------------------------------------- adversarial networks (validation gate 2)
def _frame(F: HillField):
    """Land centroid, uphill unit vector (mean gradient over land), contour unit vector, land radius."""
    C = F.centres(F.land)
    centre = C.mean(0)
    g = np.array([np.nanmean(F.gx[F.land]), np.nanmean(F.gy[F.land])])
    up = g / np.hypot(*g) if np.hypot(*g) > 1e-9 else np.array([0.0, 1.0])
    return centre, up, np.array([-up[1], up[0]]), float(np.hypot(*(C - centre).T).max()) + F.res


def _clip(lines, F: HillField, min_len: float = 20.0) -> list:
    """Pieces of the lines inside the unit (densified every 2 m)."""
    out = []
    for l_ in lines:
        l_ = np.asarray(l_, float)
        seg = np.hypot(*np.diff(l_, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        if s[-1] <= 0:
            continue
        t = np.r_[np.arange(0, s[-1], SAMPLE_M), s[-1]]
        P = np.c_[np.interp(t, s, l_[:, 0]), np.interp(t, s, l_[:, 1])]
        ins = F.inside(P)
        st = np.flatnonzero(np.diff(np.r_[0, ins.astype(int), 0]) == 1)
        en = np.flatnonzero(np.diff(np.r_[0, ins.astype(int), 0]) == -1)
        for a, b in zip(st, en):
            if b - a >= 2 and t[b - 1] - t[a] >= min_len:
                out.append(np.asarray(shapely.LineString(P[a:b]).simplify(0.01).coords))
    return out


def adv_switchback(F: HillField, gap: float = 30.0, leg: float = 150.0, n_legs: int | None = None) -> RoadGraph:
    """One road climbing the unit's fall line in contour legs joined by hairpins of radius gap / 2 (its longest piece
    in the unit)."""
    c, up, ct, R = _frame(F)
    n_legs = int(min(400, max(2, 2 * R // gap))) if n_legs is None else n_legs
    r, pts = gap / 2, []
    base = c - up * (n_legs - 1) * gap / 2
    phi = np.linspace(0, np.pi, 13)[1:-1]
    for i in range(n_legs):
        d = ct if i % 2 == 0 else -ct
        y0 = base + up * i * gap
        a, b = y0 - d * leg / 2, y0 + d * leg / 2
        pts += [a, b]
        if i < n_legs - 1:
            pts += list(b + up * r + r * (np.cos(phi)[:, None] * -up + np.sin(phi)[:, None] * d))
    pieces_ = _clip([np.array(pts)], F)
    if not pieces_:
        return lines_graph([])
    return lines_graph([max(pieces_, key=lambda p: shapely.LineString(p).length)], node=False)


def adv_contour_parallel(F: HillField, spacing: float = 60.0) -> RoadGraph:
    """Streets along the contours of ``S`` (every ``spacing`` m at the median land slope), never joined."""
    from skimage.measure import find_contours
    E = F.S.elev.astype(float)
    E = np.where(np.isfinite(E), E, np.nanmean(E))
    dz = spacing * float(np.nanmedian(F.slope[F.land])) / 100
    lo, hi = float(E[F.land].min()), float(E[F.land].max())
    lines = []
    for lev in np.arange(lo + dz / 2, hi, dz):
        for cc in find_contours(E, lev):
            lines.append(np.c_[F.S.x0 + (cc[:, 1] + 0.5) * F.res, F.S.y1 - (cc[:, 0] + 0.5) * F.res])
    return lines_graph(_clip(lines, F, 30.0), node=False)


def adv_fall_line_grid(F: HillField, spacing: float = 80.0) -> RoadGraph:
    """A straight grid with one axis on the mean fall line."""
    c, up, ct, R = _frame(F)
    o = np.arange(-R, R + 1e-9, spacing)
    lines = [np.array([c + ct * k - up * R, c + ct * k + up * R]) for k in o]
    lines += [np.array([c + up * k - ct * R, c + up * k + ct * R]) for k in o]
    return lines_graph(_clip(lines, F), node=True)


def adv_raster_zigzag(g, seed: int = 0, legs=(4.0, 9.0)) -> RoadGraph:
    """Every segment of ``g`` as an axis-aligned staircase whose major-axis legs are U(legs) m long."""
    rng = np.random.default_rng(seed)
    out = []
    for poly, _, _ in chains(g):
        pts = [poly[0]]
        for a, b in zip(poly[:-1], poly[1:]):
            d = b - a
            ax = int(np.argmax(np.abs(d)))
            if abs(d[ax]) < 1e-9:
                continue
            t, p = 0.0, a.copy()
            while t < 1.0 - 1e-9:
                dt = min(1.0 - t, rng.uniform(*legs) / abs(d[ax]))
                step = d * dt
                p = p + np.where(np.arange(2) == ax, step, 0.0)
                pts.append(p.copy())
                p = p + np.where(np.arange(2) == ax, 0.0, step)
                pts.append(p.copy())
                t += dt
            pts[-1] = b.copy()
        out.append(np.array(pts))
    return lines_graph(out, node=False)


LCP_PETRA_S1 = ROOT / "outputs" / "sites" / "petra" / "cache" / "ext_leastcostpath_s1.pkl"


def load_lcp_petra_s1(crop: bool = True) -> RoadGraph:
    """The leastcostpath adversarial: the actual M18 Petra seed-1 network (D3 sign-off). Stored by
    scripts/m18_external.py in the site benchmark cache (``LCP_PETRA_S1``, key "graph"; site-local metres of the 4 m
    site domain; the raw R output is data/processed/sites/petra/external/ext_leastcostpath_s1/paths.csv).
    ``crop``: cut to the fixed city limits with site_benchmark.crop, as every site score. Loads; generates nothing.
    Its HillField must be built in the same site frame (site_benchmark.Site("petra"))."""
    import pickle
    g = pickle.loads(LCP_PETRA_S1.read_bytes())["graph"]
    if crop:
        sb = _script("site_benchmark")
        site = sb.Site("petra")
        g = sb.crop(g, np.load(site.data / "city_mask.npz")["mask"], site)
    return g
