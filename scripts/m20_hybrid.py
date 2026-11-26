"""Milestone 20, D4: the grade hybrid, H1 connectors + H2 steep runs, as post-processing of a sampler state
(docs/M20_LOG.md D4 with its sign-off amendments and the D4a and D4c rulings; the constants there are frozen and
copied below unchanged).

    router    the A5 cost (m19_proposal_diag.route: 4 m grid, 16 neighbours, length x (1 + (g / 0.08)^2)) plus a turn
              cost of 1 m per radian (D4a), with the grade cap at 15 % (declared short ramps allowed), existing
              vehicular edges x 0.2 (the reuse set: no stairs, not the run being replaced, not an H1 dead end's own
              chain), cells within 10 m of an existing edge but not on it x 10, crossings of existing streets only at
              nodes or at >= 60 deg; only the new pieces are committed; two passes (D4c): reuse x 0.2, then reuse
              x 1 only if the first route breaks detour <= 4 on its full length;
    geometry  Douglas-Peucker 8 m (a simplification that would put a piece over the cap is rejected), Chaikin to a
              centreline radius >= 12 m (binding, D4a), re-check (grade, water, crossings); on failure one fewer Chaikin
              iteration (radius still >= 12 m), then one re-route with the apex pushed 12 m outward, else the route is
              rejected ('smoothing_fail');
    ramps     a declared short ramp is <= 60 m and <= 25 % (<= 40 m and <= 20 % at a hairpin apex); ramps less than
              30 m apart are one run (D4a);
    H1        A5's dead-end eligibility, destinations, route-length limit and order (m19_arms.posthoc), committed along
              the smoothed polyline under the sampler's rules;
    H2        steep runs (stroke stretches whose 30 m-window grade exceeds 15 %) on the vehicular network: pieces above
              30 % become stairs regardless; short ramps (<= 60 m, <= 25 %) stay vehicular; other runs longest first
              (hash ties), re-checked after each change: (1) stairs if the ends stay joined in the G1 drivable subgraph
              with detour <= 4, (2) a switchback by the router (detour <= 4) and the run becomes stairs, (3) kept;
    output    a class-tagged graph (vehicular / stepped_ramp / stairs), snapshots after each connector or switchback
              with the gross new vehicular length (P1b's matched-length reading) and per-town logs.

The sampler is injected: ``evaluate(s, u, d) -> (outcome, ok, reason)`` resolves and scores one ground step u -> u + d
on a GenState (``sampler_evaluator`` wraps an M14 sampler's evaluate_candidate exactly as A5 calls it; tests pass a
stub), and every commit goes through ``GenState.commit``. The terrain objects are the caller's: ``T`` (a
heightfield.Terrain, optionally with ``water_kind``) is the rule / router / metric terrain (D4: FABDEM at 32 m) and
``grad_at`` the landform gradient for A5's dead-end eligibility; the model input (GLO-30 or FABDEM) is the caller's run.

    python scripts/m20_hybrid.py smoke [--unit natural] [--backbone TNg_s0] [--seed 1] [--ramp-m 60] [--out FILE]
        replay the stored M19 A0 run (must be bit-identical, as m20_blind_repair.job), apply H1 + H2, print the logs

Implementation choices (left open by D4; the committee's rulings of D4a are applied and cited):
 1. Router grid: the terrain's own pixels when it is on a 4 m grid (A5's grid exactly), else bilinear samples at 4 m
    cell centres (water by nearest pixel). Box = the endpoints' bounding box padded by A5's 60 m, inside the domain.
 2. Short-ramp allowance in the router (D4a b): a route with every 4 m-grid step <= 15 % is searched first; if none
    exists, a route with steps <= 25 % is searched and kept only if each stretch of consecutive steps above 15 % is
    <= 60 m, stretches less than 30 m apart counting as one (D4a 1).
 3. Turn cost (D4a a, now a D4 constant): the search state is (cell, incoming direction) and a heading change of
    theta costs 1 m x theta (radians). Without it the 4 m grid's equal-cost paths zig-zag at random with 9 m legs on
    any slope steeper than the cap (tested on a 20 % plane), which Douglas-Peucker may not remove (the cap rule) and
    Chaikin cannot round to 12 m.
 4. "On an edge" = a cell centre within 3 m (0.75 cell) of the segment; "within 10 m" = centre within 10 m. A step's
    multiplier is the largest of its two cells' and its midpoint's (0.2 on a reuse edge, 10 near any existing edge but
    not on a reuse edge, 1 elsewhere; the midpoint stops an 8.9 m knight step from jumping the 10 m band onto a
    street); so cells on stairs, on the run being replaced and on bridges or tunnels (never reused) cost x 10 too.
 5. Crossings: a grid step (and a committed chord) that properly crosses an existing edge must do so within MERGE_M
    (8 m, the sampler's crossing-to-node snap) of a junction or dead end of the network, or at >= 60 deg. Interior
    vertices of a street (degree 2, every ~25 m on generated streets) are not "nodes" here, or nearly every shallow
    crossing would pass. Steps moving along an edge (both cells on it) are not crossings. Stairs count as edges.
 6. New pieces: the route is cut where it runs along a reuse edge (two consecutive on-edge cells); each off-network
    stretch is shaped and committed separately, from the projection of its last on-edge cell to the projection of the
    next one. A stretch that leaves a street mid-edge starts from a node inserted there by GenState.split_edge (the
    split the sampler's T-junction rule uses), or from an existing node within MERGE_M.
 7. Grade checks (Douglas-Peucker chords and smoothed routes) use hilltown.window_grades on the polyline itself (pieces
    <= 10 m, a 30 m window centred on each piece, shifted inside near the ends): every stretch above 15 % must be a
    declared short ramp (D4a 1): <= 60 m and <= 25 %, stretches less than 30 m apart (at <= 15 %) merged into one with
    the gap, and a stretch at a hairpin apex <= 40 m and <= 20 %. "At a hairpin apex" = overlapping (within 1 m) the
    turning window of a G2 hairpin (hairpin_windows: hilltown.hairpins' rule and constants, recomputed for the
    window's extent, since its reported apex is the window's first vertex on an evenly sampled arc). Water (any wet
    sample every 2 m) and the crossing rule are checked with it.
 8. Smoothing (D4a e, 2): Chaikin (1/4-3/4 corner cutting, end points fixed) with k = 2..4 iterations; the first k
    whose committed polyline (choice 9) has no G2 hairpin under 12 m (hilltown.hairpins, the metric's own radius) is
    used. The radius is binding: if its re-check fails, k - 1 iterations (>= 1) are accepted only with every G2
    hairpin still >= 12 m; otherwise, and directly when no k reaches 12 m, the route is re-run once through the apex
    pushed 12 m away from the inside of its turn (the Douglas-Peucker vertex nearest the tightest hairpin's apex, or of
    largest turn after a grade failure); if that fails too it is rejected and logged 'smoothing_fail'. Bends under
    120 deg are not constrained; the smallest circumradius of committed vertices outside the hairpins (half the
    hairpin window of path around each apex) is logged descriptively, 'bend_r_min_m' and 'bends_under_12m'.
 9. Commit polyline: from the smoothed curve, greedy chords of 10-25 m whose skipped vertices lie within 1 m of the
    chord (>= 10 m keeps every step above MERGE_M; 25 m is A5's step; a last chord under 10 m is merged into the one
    before only if that moves the dropped vertex <= 2 m). Each chord is one evaluate + GenState.commit; a
    step cut short at a crossing continues from the crossing node to the same point; a remaining distance under
    MERGE_M moves on to the next point. A stretch is complete when a step reaches the network (merge, T-junction,
    crossing) and the rest of it lies within MERGE_M of the network it was planned on, or within MERGE_M of its end.
    All checks of choice 8 run on this committed polyline.
10. H1 keeps A5's commit semantics: an infeasible step ends the connector, which stays partial; a rejected route
    commits nothing. Destinations and eligibility use the network as it is (A5's destination()). D4a 3 (no loop on a
    stick): the dead end's own chain (from it through degree-2 nodes) is out of the reuse set and the x 10 band (it
    still counts for crossings), and the route's first step is never reuse, so every connector starts at the dead
    end ('start_reused' stays 0).
11. H2 switchbacks are all-or-nothing: the placement, the route and the commit run on a copy of the state, adopted
    only if every new stretch reaches its end ('commit_failure' otherwise, the run is kept).
12. H2 runs are found on the native vehicular graph (the GenState, node ids preserved): hilltown.pieces and
    hilltown.window_grades on its junction-to-junction chains, then maximal consecutive pieces above 15 % along the
    strokes of network_outcomes.strokes (through junctions), unsigned. Bridges and tunnels have no grade (never steep,
    always drivable), as hilltown choice 1. A run's length is the sum of its pieces; its grade, the max window grade.
13. Pieces above 30 % (the same window grade) are forced to stairs as maximal stretches, longest first, before the runs
    and again whenever they appear. "Cut off" = vehicular km in components split off by a forced placement, except
    the largest piece of each split component.
14. Stair placement: each end moves outward along the stroke to the nearest junction (degree >= 3 in the whole
    network) within 30 m, else the edge is split at the end; stairs never shrink. Then every vehicular chain shorter
    than 30 m that dangles from the new stairs (vehicular degree 1 at the stairs) is absorbed, repeatedly, and a stairs
    end whose chain was absorbed moves to the chain's far node (the ends rules (1) and (2) use).
15. G1 rule (1): on the state after the placement, the drivable subgraph is the vehicular pieces with 30 m-window grade
    <= 15 % (NaN kept, as hilltown G1) plus the pieces of short ramps; its shortest length between the placement's two
    end nodes must be <= 4 x their straight distance (an end with no drivable edge is not joined).
16. Rule (2) routes between the same two end nodes on that state (the run already pedestrian, so outside the reuse
    set) in a box padded by max(60 m, 2 x their straight distance), clipped to the domain (D4a 4; 'pad_max_m'; the
    router's time is logged as 'router_s'); detour = route length (new + reused) / straight distance <= 4. No route,
    or a longer one, is 'failed for room'.
17. Runs less than 30 m apart along a stroke are one run, the gentle gap included (D4a 1); a run holding a G2 hairpin
    apex of the vehicular graph is a short ramp only if <= 40 m and <= 20 %. A kept run is remembered by its piece
    midpoints; a later run with at least half its midpoints within 2 m of them is
    the same run. Runs are re-detected from scratch after every change (D4: re-checked after each change).
18. Typing: pedestrian pieces are typed by their 30 m-window grade along the pedestrian chains: > 25 % stairs, else
    stepped ramp (absorbed stubs and junction extensions under 15 % included); edges are cut where the type changes.
    Landings are left to the 3D builder (the graph has no landing nodes).
19. Snapshots: (gross new vehicular m, typed graph json) at the start, after every ``snap_every``-th connector or
    switchback in the declared order (H1 connectors by route length, then H2 switchbacks in run order), and at the end
    (after the last stairs). Gross = the length committed by H1 and the switchbacks; stairs are never subtracted.
20. The hybrid works on a copy of the state whose node arrays get spare capacity for its own splits, and max_nodes
    grows by one per own split, so the sampler's budget of new nodes for commits is what A5 had.
21. A route lying entirely on existing streets commits nothing (H1: logged 'no_new_pieces'; H2: the run becomes
    stairs as a switchback with no new length, 'switchback_no_new_pieces').
22. Two-pass router (D4c), H1 and H2 alike: pass 1 has reuse x 0.2; pass 2 (reuse x 1; the same net, band, crossing
    rules, pad, turn cost and D4a 3 start) runs only if pass 1's full route length (new plus reused) breaks detour
    <= 4, and must meet it itself; otherwise the run is kept / the connector rejected ('room:detour'). Detour <= 4 thus
    binds H1 too, on top of A5's limit (3 x straight + 60 m), which still applies to whichever pass is used; a pass-1
    route within detour 4 but beyond A5's limit gets no pass 2 ('room:limit'). The pushed re-route of choice 8 stays
    in the pass that found the route, and H1's ordering pass uses the same two passes. Logged per stage: 'pass1' /
    'pass2' (committed connectors or switchbacks by pass), 'pass2_tried', 'pass2_share'; 'pass2_m' = final vehicular
    edges laid by pass 2 (midpoint within 0.5 m of a committed pass-2 segment) and 'pass2_lens_share' = their share on
    lens faces (hilltown._faces, the M4 rule, on the final native vehicular graph; descriptive, never filtered).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from m19_arms import destination
from m19_proposal_diag import G0, NB, PAD_M, ROUTE_PAD, ROUTE_X, SLOPE_MIN
from urbanmorph.graphs.network_outcomes import strokes
from urbanmorph.graphs.road_graph import MERGE_M, RoadGraph
from urbanmorph.metrics import hilltown as H
from urbanmorph.terrain.road_feasibility import water_kind_at

# ---------------------------------------------------------------- D4 constants (frozen; docs/M20_LOG.md D4 and D4a)
CAP_PCT = 15.0                      # drivable grade (M6 / G1)
FORCE_PCT = 30.0                    # stairs regardless (M18 F12)
RAMP_M, RAMP_PCT = 60.0, 25.0       # short ramp (60 m: reviewer's judgement; sensitivities 40 and 100 m)
DETOUR = 4.0                        # detour (M6)
REUSE_X, NEAR_X, NEAR_M = 0.2, 10.0, 10.0   # reuse discount, proximity penalty within 10 m (Emilien-style)
CROSS_DEG = 60.0                    # crossing angle
DP_M, RADIUS_M = 8.0, 12.0          # Douglas-Peucker, radius (G2)
SNAP_M, STUB_M = 30.0, 30.0         # snap and stub
PUSH_M = 12.0                       # hairpin vertex pushed outward (sign-off amendment)
TURN_X = 1.0                        # turn cost, m per radian of heading change (D4a a, Galin-2010-style)
RAMP_GAP_M = 30.0                   # back-to-back ramps need >= 30 m at <= 15 % between them, else one run (D4a 1)
APEX_M, APEX_PCT = 40.0, 20.0       # a hairpin apex is a short ramp only if <= 40 m and <= 20 % (D4a 1)
# ---------------------------------------------------------------- implementation constants (choices 1-9)
GRID_M, ON_M = 4.0, 3.0
CHAIKIN_MIN, CHAIKIN_MAX = 2, 4
COMMIT_MIN, COMMIT_MAX, COMMIT_TOL = 10.0, 25.0, 1.0
STEP_MAX = GRID_M * math.hypot(1, 2)          # longest grid step
CLS = 3                                        # A5's connector class
VEH, PED = 0, 1                                # internal edge kinds
KIND_NAMES = ("vehicular", "stepped_ramp", "stairs")
NET_KINDS = ("crossing_split", "crossing_endpoint", "merge_node", "t_junction")


@dataclass(frozen=True)
class Params:
    """The hybrid's constants; ``ramp_m`` is the declared sensitivity switch (40 / 60 / 100 m)."""
    cap: float = CAP_PCT
    force: float = FORCE_PCT
    ramp_m: float = RAMP_M
    ramp_pct: float = RAMP_PCT
    detour: float = DETOUR
    reuse_x: float = REUSE_X
    near_x: float = NEAR_X
    near_m: float = NEAR_M
    cross_deg: float = CROSS_DEG
    dp_m: float = DP_M
    radius_m: float = RADIUS_M
    snap_m: float = SNAP_M
    stub_m: float = STUB_M
    push_m: float = PUSH_M
    pad_m: float = PAD_M
    turn_x: float = TURN_X
    ramp_gap_m: float = RAMP_GAP_M
    apex_m: float = APEX_M
    apex_pct: float = APEX_PCT


# ---------------------------------------------------------------- the network as the router sees it
@dataclass
class Net:
    """Existing edges: segments (E, 2, 2), reuse flags (E,), end node ids (E, 2) of the GenState, the coordinates
    of the network's chain-end nodes (junctions and dead ends: degree != 2), where crossings may be, and band flags
    (E,): the edge sets the x 10 band (D4a 3 hides the H1 dead end's own chain from the band and the reuse set)."""
    seg: np.ndarray
    reuse: np.ndarray
    ends: np.ndarray
    nodes: np.ndarray
    band: np.ndarray

    @classmethod
    def of(cls, s, kinds: dict, exclude=(), hidden=()):
        e = s.live_edges()
        e = np.unique(np.sort(e, 1), axis=0) if len(e) else e.reshape(0, 2)
        ex = {(min(a, b), max(a, b)) for a, b in exclude}
        hid = {(min(a, b), max(a, b)) for a, b in hidden}
        band = np.array([(int(a), int(b)) not in hid for a, b in e], bool)
        reuse = np.array([kinds.get((int(a), int(b)), VEH) == VEH and (int(a), int(b)) not in ex
                          and (int(a), int(b)) not in s.raised for a, b in e], bool) & band
        deg = np.bincount(e.ravel(), minlength=s.n) if len(e) else np.zeros(s.n, int)
        return cls(s.nodes[e].reshape(-1, 2, 2) if len(e) else np.zeros((0, 2, 2)), reuse, e,
                   s.nodes[:s.n][(deg >= 1) & (deg != 2)].copy(), band)

    @classmethod
    def empty(cls):
        return cls(np.zeros((0, 2, 2)), np.zeros(0, bool), np.zeros((0, 2), int), np.zeros((0, 2)), np.zeros(0, bool))

    def at_node(self, X):
        if not len(self.nodes) or not len(X):
            return np.zeros(len(X), bool)
        if getattr(self, "_tree", None) is None:
            self._tree = cKDTree(self.nodes)
        return self._tree.query(X, distance_upper_bound=MERGE_M)[0] < MERGE_M

    def near_box(self, lo, hi, m):
        if not len(self.seg):
            return np.zeros(0, int)
        a, b = self.seg[:, 0], self.seg[:, 1]
        return np.flatnonzero((np.maximum(a, b) >= lo - m).all(1) & (np.minimum(a, b) <= hi + m).all(1))


def merged_stretches(steep, ln, gap: float):
    """Index ranges [x0, x1) of the steep stretches of a piece sequence (``steep`` mask, lengths ``ln``); two
    back-to-back stretches less than ``gap`` m apart are one stretch, the gentle gap included (D4a 1)."""
    m = np.r_[False, np.asarray(steep, bool), False].astype(int)
    out = []
    for x0, x1 in zip(np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)):
        if out and float(np.sum(ln[out[-1][1]:x0])) < gap - 1e-9:
            out[-1] = (out[-1][0], int(x1))
        else:
            out.append((int(x0), int(x1)))
    return out


def _seg_dist(P, A, B):
    """Distances of points P (n, 2) to segments A -> B (n, 2) (pairwise rows)."""
    d = B - A
    t = np.clip(((P - A) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    return np.hypot(*(A + t[:, None] * d - P).T)


def _crossings(p, q, A, B):
    """Proper crossings of segments p -> q with A -> B (pairwise rows): (mask, t along p->q, point, |cos angle|)."""
    r, s_ = q - p, B - A
    den = r[:, 0] * s_[:, 1] - r[:, 1] * s_[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((A[:, 0] - p[:, 0]) * s_[:, 1] - (A[:, 1] - p[:, 1]) * s_[:, 0]) / den
        u = ((A[:, 0] - p[:, 0]) * r[:, 1] - (A[:, 1] - p[:, 1]) * r[:, 0]) / den
    m = (np.abs(den) > 1e-9) & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)
    X = p + np.where(m, t, 0.0)[:, None] * r
    cos = np.abs((r * s_).sum(1)) / np.maximum(np.hypot(*r.T) * np.hypot(*s_.T), 1e-12)
    return m, t, X, cos


def _cross_bad(X, cos, net: Net, P: Params):
    """A crossing is allowed at a node (within MERGE_M of a junction or dead end) or at >= 60 deg (choice 5)."""
    return (cos > math.cos(math.radians(P.cross_deg)) + 1e-12) & ~net.at_node(X)


# ---------------------------------------------------------------- the router
def _grid(T, lo, hi, size):
    """Router grid over the box: (Z, ok, xs, ys) (choice 1)."""
    if abs(float(T.res) - GRID_M) < 1e-9:                              # A5's grid (m19_proposal_diag.route)
        r0, c0 = T.rc(np.array([lo[0]]), np.array([hi[1]]))
        r1, c1 = T.rc(np.array([hi[0]]), np.array([lo[1]]))
        r0, c0, r1, c1 = int(np.floor(r0[0])), int(np.floor(c0[0])), int(np.ceil(r1[0])), int(np.ceil(c1[0]))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, T.elev.shape[0] - 1), min(c1, T.elev.shape[1] - 1)
        Z = T.elev[r0:r1 + 1, c0:c1 + 1].astype(float)
        ok = np.isfinite(Z)
        wk = getattr(T, "water_kind", None)
        if wk is not None:
            ok &= np.asarray(wk)[r0:r1 + 1, c0:c1 + 1] == 0
        xs = T.x0 + (np.arange(c0, c1 + 1) + 0.5) * T.res
        ys = T.y1 - (np.arange(r0, r1 + 1) + 0.5) * T.res
    else:
        xs = np.arange(math.floor(lo[0] / GRID_M), math.ceil(hi[0] / GRID_M)) * GRID_M + GRID_M / 2
        ys = (np.arange(math.ceil(hi[1] / GRID_M), math.floor(lo[1] / GRID_M), -1) * GRID_M - GRID_M / 2)
        X, Y = np.meshgrid(xs, ys)
        Z = T.sample(X.ravel(), Y.ravel()).reshape(X.shape)
        ok = np.isfinite(Z)
        if getattr(T, "water_kind", None) is not None:
            ok &= water_kind_at(T, X.ravel(), Y.ravel()).reshape(X.shape) == 0
    ok &= ((xs >= 0) & (xs <= size))[None, :] & ((ys >= 0) & (ys <= size))[:, None]
    return Z, ok, xs, ys


def route(T, a, b, size, net: Net, P: Params = Params(), pad: float | None = None):
    """The fabric-aware least-cost route a -> b (module docstring, choices 1-5), or None. ``pad``: the box margin
    (default A5's 60 m; H2 uses max(60 m, 2 x straight), D4a 4), always clipped to the domain.

    Returns {"poly" (n, 2) cell centres with the exact end points, "length", "on" (n,) index of the reuse segment the
    vertex lies on or -1, "grades" (n - 1,), "ramp": True if found by the short-ramp pass}."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pad = P.pad_m if pad is None else float(pad)
    lo = np.maximum(np.minimum(a, b) - pad, 0.0)
    hi = np.minimum(np.maximum(a, b) + pad, size)
    Z, ok, xs, ys = _grid(T, lo, hi, size)
    Hh, W = Z.shape
    if Hh * W < 2:
        return None
    C = np.c_[np.tile(xs, Hh), np.repeat(ys, W)]                       # cell centres, row-major
    n = Hh * W
    # ---- network fields: distances of cells to nearby segments (pairs within near_m)
    sel = net.near_box(lo, hi, P.near_m + STEP_MAX)                    # every edge counts for crossings
    d_any = np.full(n, np.inf)
    d_re = np.full(n, np.inf)
    on_id = np.full(n, -1)
    K = len(NB)
    dirs = np.array([(dc, -dr) for dr, dc in NB], float) * GRID_M
    m_any = np.full((K, n), np.inf)                                    # the same for each step's midpoint, so a
    m_re = np.full((K, n), np.inf)                                     # knight step cannot jump the 10 m band
    pc_, pe_, pd_ = np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    if len(sel):
        tree = shapely.STRtree(shapely.linestrings(net.seg[sel]))
        ci, sj = tree.query(shapely.points(C), predicate="dwithin", distance=P.near_m + STEP_MAX)
        e = sel[sj]
        r_, bd = net.reuse[e], net.band[e]                             # band edges set the x 10 band (D4a 3)
        for k in range(K):
            dm = _seg_dist(C[ci] + dirs[k] / 2, net.seg[e, 0], net.seg[e, 1])
            np.minimum.at(m_any[k], ci[bd], dm[bd])
            np.minimum.at(m_re[k], ci[r_], dm[r_])
        dd = _seg_dist(C[ci], net.seg[e, 0], net.seg[e, 1])
        keep = dd <= max(P.near_m, STEP_MAX)
        pc_, pe_, pd_ = ci[keep], e[keep], dd[keep]
        bd = net.band[pe_]
        np.minimum.at(d_any, pc_[bd], pd_[bd])
        r_ = net.reuse[pe_]
        np.minimum.at(d_re, pc_[r_], pd_[r_])
        on = r_ & (pd_ <= ON_M)                                        # nearest reuse segment of each on-cell
        o = np.lexsort((pd_[on], pc_[on]))
        c_on, e_on = pc_[on][o], pe_[on][o]
        first = np.r_[True, c_on[1:] != c_on[:-1]] if len(c_on) else np.zeros(0, bool)
        on_id[c_on[first]] = e_on[first]
    mult = lambda d_r, d_a: np.where(d_r <= ON_M, P.reuse_x, np.where(d_a <= P.near_m, P.near_x, 1.0))
    mc, mm = mult(d_re, d_any), mult(m_re, m_any)
    idx = np.arange(n).reshape(Hh, W)
    un = dirs / np.hypot(*dirs.T)[:, None]
    ang = np.arccos(np.clip(un @ un.T, -1, 1))
    # ---- forbidden crossings (choice 5), per (cell, direction)
    bad = np.zeros((K, n), bool)
    near_pairs = pd_ <= STEP_MAX + 1e-9
    if near_pairs.any():
        pc2, pe2, pd2 = pc_[near_pairs], pe_[near_pairs], pd_[near_pairs]
        on_e = {}
        for c_, e_, d_ in zip(pc2, pe2, pd2):
            if d_ <= ON_M:
                on_e.setdefault(int(e_), set()).add(int(c_))
        rr, cc = np.divmod(pc2, W)
        for k, (dr, dc) in enumerate(NB):
            r2, c2 = rr + dr, cc + dc
            inb = (r2 >= 0) & (r2 < Hh) & (c2 >= 0) & (c2 < W)
            if not inb.any():
                continue
            p = C[pc2[inb]]
            q = p + dirs[k]
            A_, B_ = net.seg[pe2[inb], 0], net.seg[pe2[inb], 1]
            m, _, X, cos = _crossings(p, q, A_, B_)
            if not m.any():
                continue
            badk = m & _cross_bad(X, cos, net, P)
            if not badk.any():
                continue
            src_c, dst_c, e_c = pc2[inb][badk], (r2 * W + c2)[inb][badk], pe2[inb][badk]
            along = np.array([int(x) in on_e.get(int(e_), ()) and int(y) in on_e.get(int(e_), ())
                              for x, y, e_ in zip(src_c, dst_c, e_c)], bool)
            bad[k, src_c[~along]] = True

    def search(cap):
        I, J, Cs, Ks = [], [], [], []
        for k, (dr, dc) in enumerate(NB):
            ra, rb = max(0, -dr), Hh - max(0, dr)
            ca, cb = max(0, -dc), W - max(0, dc)
            if ra >= rb or ca >= cb:
                continue
            A = (slice(ra, rb), slice(ca, cb))
            B = (slice(ra + dr, rb + dr), slice(ca + dc, cb + dc))
            L = GRID_M * math.hypot(dr, dc)
            g = np.abs(Z[B] - Z[A]) / L
            m = ok[A] & ok[B] & (g <= cap / 100 + 1e-12) & ~bad[k][idx[A]]
            src, dst = idx[A][m], idx[B][m]
            cost = L * (1 + (g[m] / G0) ** 2) * np.maximum(np.maximum(mc[src], mc[dst]), mm[k][src])
            I.append(src)
            J.append(dst)
            Cs.append(cost)
            Ks.append(np.full(len(src), k))
        if not I:
            return None
        I, J, Cs, Ks = map(np.concatenate, (I, J, Cs, Ks))
        rc = lambda p: (int(np.clip(round((ys[0] - p[1]) / GRID_M), 0, Hh - 1)),
                        int(np.clip(round((p[0] - xs[0]) / GRID_M), 0, W - 1)))
        sa, sb = idx[rc(a)], idx[rc(b)]
        if not ok.ravel()[sa] or not ok.ravel()[sb]:
            return None
        # state = cell * K + incoming direction; source node S = n * K
        S = n * K
        kin = np.arange(K)
        src = (I[:, None] * K + kin[None, :]).ravel()
        dst = np.repeat(J * K + Ks, K)
        w = (Cs[:, None] + P.turn_x * ang[kin[None, :], Ks[:, None]]).ravel()
        st = I == sa
        src = np.r_[src, np.full(int(st.sum()), S)]
        dst = np.r_[dst, J[st] * K + Ks[st]]
        w = np.r_[w, Cs[st]]
        G = coo_matrix((w, (src, dst)), shape=(S + 1, S + 1)).tocsr()
        dist, pred = dijkstra(G, indices=S, return_predecessors=True)
        tgt = np.arange(sb * K, sb * K + K)
        if sa == sb:
            return [sa]
        k_best = int(np.argmin(dist[tgt]))
        if not np.isfinite(dist[tgt[k_best]]):
            return None
        path, x = [], int(tgt[k_best])
        while x != S:
            path.append(x // K)
            x = int(pred[x])
            if x < 0:
                return None
        path.append(sa)
        return path[::-1]

    for cap, ramp in ((P.cap, False), (P.ramp_pct, True)):
        path = search(cap)
        if path is None:
            continue
        rr, cc = np.divmod(np.array(path), W)
        Pp = np.c_[xs[cc], ys[rr]]
        z = Z[rr, cc]
        Pp[0], Pp[-1] = a, b
        seg = np.hypot(*np.diff(Pp, axis=0).T)
        cell_len = np.hypot(*np.diff(np.c_[xs[cc], ys[rr]], axis=0).T)
        gr = np.abs(np.diff(z)) / np.maximum(cell_len, 1e-9) * 100
        if ramp:                                                      # choice 2: stretches above the cap <= ramp_m,
            if any(cell_len[x0:x1].sum() > P.ramp_m                    # back-to-back ones merged (D4a 1)
                   for x0, x1 in merged_stretches(gr > P.cap, cell_len, P.ramp_gap_m)):
                return None
        return {"poly": Pp, "length": float(seg.sum()), "on": on_id[np.array(path)], "grades": gr, "ramp": ramp}
    return None


# ---------------------------------------------------------------- route geometry (choices 7-9)
class Geo:
    """Grade, water and crossing checks of candidate polylines on the rule terrain (one per town or tile)."""

    def __init__(self, T, P: Params = Params()):
        self.T, self.P = T, P
        self.F = H.HillField(T, np.ones(T.elev.shape, bool))

    def window_grades(self, C):
        """30 m-window grade (%) of the <= 10 m pieces of polyline C, in order, and their lengths (hilltown)."""
        wg, pc = self._pieces(C)
        return wg, pc.length

    def _pieces(self, C):
        g = _poly_graph(C)
        pc = H.pieces(g, self.F)
        return H.window_grades(g, self.F, pc), pc

    def grade_ok(self, C) -> bool:
        """Every stretch above the cap (back-to-back ones less than 30 m apart merged) is a short ramp: <= ramp_m and
        <= ramp_pct, or <= 40 m and <= 20 % if it holds a G2 hairpin apex (D4a 1); no missing data."""
        wg, pc = self._pieces(C)
        if not np.isfinite(wg).all():
            return False
        P = self.P
        apex = hairpin_windows(_poly_graph(C))
        for x0, x1 in merged_stretches(wg > P.cap, pc.length, P.ramp_gap_m):
            lim_m, lim_g = P.ramp_m, P.ramp_pct
            if _touches(apex, pc.p0[x0:x1], pc.p1[x0:x1]):
                lim_m, lim_g = min(lim_m, P.apex_m), min(lim_g, P.apex_pct)
            if pc.length[x0:x1].sum() > lim_m + 1e-9 or wg[x0:x1].max() > lim_g + 1e-9:
                return False
        return True

    def dry(self, C) -> bool:
        seg = np.hypot(*np.diff(C, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        t = np.linspace(0, s[-1], max(2, int(math.ceil(s[-1] / 2.0)) + 1))
        x, y = np.interp(t, s, C[:, 0]), np.interp(t, s, C[:, 1])
        if not np.isfinite(self.T.sample(x, y)).all():
            return False
        return getattr(self.T, "water_kind", None) is None or not (water_kind_at(self.T, x, y) > 0).any()

    def crossings_ok(self, C, net: Net) -> bool:
        """No chord crosses an existing edge except at a node or at >= 60 deg (choice 5); contacts at the polyline's
        own end points are joins, not crossings."""
        if not len(net.seg) or len(C) < 2:
            return True
        lo, hi = C.min(0), C.max(0)
        sel = net.near_box(lo, hi, 1.0)
        if not len(sel):
            return True
        k, e = np.divmod(np.arange((len(C) - 1) * len(sel)), len(sel))
        p, q = C[k], C[k + 1]
        A_, B_ = net.seg[sel[e], 0], net.seg[sel[e], 1]
        m, t, X, cos = _crossings(p, q, A_, B_)
        if not m.any():
            return True
        ends = np.minimum(np.hypot(*(X - C[0]).T), np.hypot(*(X - C[-1]).T)) < 1.0
        return not (m & ~ends & _cross_bad(X, cos, net, self.P)).any()

    def chord_ok(self, A, B, net: Net) -> bool:
        C = np.array([A, B], float)
        return self.grade_ok(C) and self.dry(C) and self.crossings_ok(C, net)

    def checks(self, C, net: Net) -> bool:
        return self.grade_ok(C) and self.dry(C) and self.crossings_ok(C, net)


def _poly_graph(C):
    C = np.asarray(C, float)
    return RoadGraph(C, np.c_[np.arange(len(C) - 1), np.arange(1, len(C))], np.zeros(len(C), bool))


def hairpin_windows(g) -> list:
    """The turning windows of the G2 hairpins of graph g, as vertex arrays: hilltown.hairpins' rule (the minimal
    window turning more than HAIRPIN_DEG within HAIRPIN_M of path, greedy, along junction-to-junction chains) with the
    metric's constants, recomputed here because hilltown.hairpins returns only the apex vertex (on an evenly sampled
    arc, the window's first vertex)."""
    out = []
    if len(g.edges) == 0:
        return out
    lim = math.radians(H.HAIRPIN_DEG)
    view = RoadGraph(np.asarray(g.nodes, float), np.asarray(g.edges), np.zeros(len(g.nodes), bool))
    for P in H._M18.ground_polylines(view):
        ln = np.hypot(*np.diff(P, axis=0).T)
        P = np.vstack([P[:1], P[1:][ln > 1e-6]])
        d = np.diff(P, axis=0)
        ln = np.hypot(*d.T)
        if len(P) < 3:
            continue
        h = np.arctan2(d[:, 1], d[:, 0])
        turn = (np.diff(h) + np.pi) % (2 * np.pi) - np.pi
        s = np.cumsum(ln)[:-1]
        i = 0
        while i < len(turn):
            acc, j, found = 0.0, i, -1
            while j < len(turn) and s[j] - s[i] <= H.HAIRPIN_M:
                acc += turn[j]
                if abs(acc) > lim:
                    found = j
                    break
                j += 1
            if found < 0:
                i += 1
                continue
            out.append(P[i + 1:found + 2].copy())
            i = found + 1
    return out


def _touches(windows, A, B, tol: float = 1.0) -> bool:
    """Does a stretch (pieces A -> B) overlap a hairpin's turning window (a window vertex within ``tol`` of a piece,
    or a piece midpoint within ``tol`` of the window's polyline)?"""
    if not len(windows) or not len(A):
        return False
    M = (A + B) / 2
    for W_ in windows:
        if any((_seg_dist(np.repeat(x[None], len(A), 0), A, B) <= tol).any() for x in W_):
            return True
        if len(W_) >= 2 and any((_seg_dist(np.repeat(m[None], len(W_) - 1, 0), W_[:-1], W_[1:]) <= tol).any()
                                for m in M):
            return True
    return False


def douglas_peucker(Pts, keep_chord, tol: float = DP_M):
    """Douglas-Peucker at ``tol``; a span is replaced by its chord only if ``keep_chord(A, B)`` (D4: a simplification
    that would push a piece over the cap is rejected, and the span is split at its farthest vertex instead)."""
    Pts = np.asarray(Pts, float)
    n = len(Pts)
    keep = np.zeros(n, bool)
    keep[[0, n - 1]] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        m = Pts[i + 1:j]
        d = _seg_dist(m, np.repeat(Pts[i][None], len(m), 0), np.repeat(Pts[j][None], len(m), 0))
        k = i + 1 + int(np.argmax(d))
        if d.max() <= tol and keep_chord(Pts[i], Pts[j]):
            continue
        keep[k] = True
        stack += [(i, k), (k, j)]
    return Pts[keep]


def chaikin(Pts, k: int):
    """k iterations of Chaikin corner cutting (1/4, 3/4), end points fixed."""
    Q = np.asarray(Pts, float)
    for _ in range(k):
        if len(Q) < 3:
            break
        a, b = Q[:-1], Q[1:]
        L, R = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
        mid = np.empty((2 * (len(Q) - 1) - 2, 2))
        mid[0::2], mid[1::2] = R[:-1], L[1:]
        Q = np.vstack([Q[:1], mid, Q[-1:]])
    return Q


def densify(S, step: float):
    """The same polyline with extra vertices on its segments, at most ``step`` m apart (original vertices kept)."""
    S = np.asarray(S, float)
    out = [S[:1]]
    for a, b in zip(S[:-1], S[1:]):
        m = max(1, int(math.ceil(np.hypot(*(b - a)) / step)))
        out.append(a + np.linspace(0, 1, m + 1)[1:, None] * (b - a))
    return np.vstack(out)


def commit_points(S, lo: float = COMMIT_MIN, hi: float = COMMIT_MAX, tol: float = COMMIT_TOL):
    """Greedy chords along the smoothed curve S (choice 9): the farthest vertex within ``hi`` m of path whose skipped
    vertices lie within ``tol`` of the chord, at least ``lo`` m away unless it is the end; a last chord under ``lo``
    is merged into the previous one."""
    S = densify(S, 2.0)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(S, axis=0).T))]
    out, i, n = [0], 0, len(S)
    while i < n - 1:
        best = None
        j = i + 1
        while j < n and s[j] - s[i] <= hi + 1e-9:
            m = S[i + 1:j]
            dev = (_seg_dist(m, np.repeat(S[i][None], len(m), 0), np.repeat(S[j][None], len(m), 0)).max()
                   if len(m) else 0.0)
            ch = float(np.hypot(*(S[j] - S[i])))
            if dev <= tol and (ch >= lo or j == n - 1):
                best = j
            elif dev > tol and best is not None:
                break
            j += 1
        if best is None:                                              # tight curve: first vertex >= lo away
            j = i + 1
            while j < n - 1 and np.hypot(*(S[j] - S[i])) < lo:
                j += 1
            best = j
        out.append(best)
        i = best
    if len(out) > 2 and np.hypot(*(S[out[-1]] - S[out[-2]])) < lo:                 # merge a short last chord unless
        a_, b_, m_ = S[out[-3]], S[out[-1]], S[out[-2]]                            # that would cut a corner
        if _seg_dist(m_[None], a_[None], b_[None])[0] <= 2 * tol:
            del out[-2]
    return S[out]


def tight_hairpins(C, r_min: float = RADIUS_M) -> list:
    """G2 hairpins (hilltown.hairpins: > 120 deg within the metric's window) of polyline C under r_min, tightest
    first: [(apex, turn deg, radius)]."""
    return sorted([h for h in H.hairpins(_poly_graph(C)) if h[2] < r_min - 1e-9], key=lambda h: h[2])


def radius_ok(C, r_min: float = RADIUS_M) -> bool:
    """No G2 hairpin of polyline C has a centreline radius under r_min (choice 8; binding, D4a 2)."""
    return not tight_hairpins(C, r_min)


def other_bend_radius(C) -> float:
    """Smallest circumradius at a vertex of C outside the G2 hairpins (vertices within half the hairpin window of
    path from an apex excluded); descriptive only (D4a e). inf if C has no bend."""
    C = np.asarray(C, float)
    if len(C) < 3:
        return float("inf")
    a, b, c = C[:-2], C[1:-1], C[2:]
    la, lb, lc = np.hypot(*(b - a).T), np.hypot(*(c - b).T), np.hypot(*(c - a).T)
    area2 = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(area2 > 1e-9, la * lb * lc / np.maximum(2 * area2, 1e-12), np.inf)
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(C, axis=0).T))]
    keep = np.ones(len(R), bool)
    for h in H.hairpins(_poly_graph(C)):
        k = int(np.argmin(np.hypot(*(C - h[0]).T)))
        keep &= np.abs(s[1:-1] - s[k]) > H.HAIRPIN_M / 2
    return float(R[keep].min()) if keep.any() else float("inf")


def hairpin_vertex(D, apex=None):
    """(index, outward unit vector) of the interior vertex of polyline D to push: the one nearest ``apex`` (the
    tightest G2 hairpin) when given, else the one of largest turn; None if there is no turning vertex."""
    if len(D) < 3:
        return None
    u = np.diff(D, axis=0)
    u = u / np.maximum(np.hypot(*u.T), 1e-9)[:, None]
    turn = np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1, 1))
    j = int(np.argmax(turn)) if apex is None else int(np.argmin(np.hypot(*(D[1:-1] - np.asarray(apex)).T)))
    w = u[j + 1] - u[j]
    if np.hypot(*w) < 1e-9:
        return None
    return j + 1, -w / np.hypot(*w)


def shape(geo: Geo, Pts, net: Net):
    """Douglas-Peucker, Chaikin to radius, re-check, one fewer iteration (choice 8). Returns (committed polyline or
    None, info, the Douglas-Peucker polyline)."""
    D = douglas_peucker(Pts, lambda A, B: geo.chord_ok(A, B, net), geo.P.dp_m)
    Cs = {}
    kstar = None
    for k in range(CHAIKIN_MIN, CHAIKIN_MAX + 1):
        Cs[k] = commit_points(chaikin(D, k))
        if radius_ok(Cs[k], geo.P.radius_m):
            kstar = k
            break
    if kstar is None:                                                 # the radius cannot be reached by smoothing
        return None, {"fail": "radius", "apex": tight_hairpins(Cs[CHAIKIN_MAX], geo.P.radius_m)[0][0]}, D
    if geo.checks(Cs[kstar], net):
        return Cs[kstar], {"chaikin": kstar}, D
    C = commit_points(chaikin(D, kstar - 1))                          # one fewer iteration (>= 1), never a tight
    if geo.checks(C, net) and radius_ok(C, geo.P.radius_m):            # radius (D4a 2)
        return C, {"chaikin": kstar - 1, "fewer": True}, D
    return None, {"fail": "recheck"}, D


# ---------------------------------------------------------------- the working state: GenState + edge kinds
def _key(a, b):
    return (int(a), int(b)) if a < b else (int(b), int(a))


class State:
    """A GenState with edge kinds (absent = vehicular, PED = stairs / stepped ramp) and the hybrid's own edits."""

    def __init__(self, s, kinds=None):
        self.s, self.kinds = s, ({} if kinds is None else kinds)

    @classmethod
    def start(cls, s, spare: int = 4096):
        """A copy with spare node capacity for the hybrid's own splits (choice 20)."""
        s = s.copy()
        s.__dict__["_m14_extra_open"] = ()
        s.nodes = np.vstack([s.nodes, np.zeros((spare, 2))])
        s.border = np.r_[s.border, np.zeros(spare, bool)]
        return cls(s)

    def copy(self):
        return State(self.s.copy(), dict(self.kinds))

    def edges(self):
        """Live edges as sorted unique pairs (E, 2), their kinds, classes and structures."""
        s = self.s
        ec = s.live_edges(with_class=True)
        if not len(ec):
            z = np.zeros(0, int)
            return np.zeros((0, 2), int), z, z, z
        ec[:, :2] = np.sort(ec[:, :2], 1)
        _, first = np.unique(ec[:, :2], axis=0, return_index=True)
        ec = ec[np.sort(first)]
        e = ec[:, :2]
        kind = np.array([self.kinds.get((int(a), int(b)), VEH) for a, b in e], int)
        st = np.array([s.raised.get((int(a), int(b)), (0, 0))[1] for a, b in e], int)
        return e, kind, ec[:, 2], st

    def view(self, mask_fn=None):
        """RoadGraph over the GenState's node ids (no remap) of the edges whose kind passes ``mask_fn``."""
        e, kind, cls, st = self.edges()
        m = np.ones(len(e), bool) if mask_fn is None else mask_fn(kind)
        n = self.s.n
        g = RoadGraph(self.s.nodes[:n].copy(), e[m], np.zeros(n, bool), edge_class=cls[m], edge_struct=st[m])
        return g, kind[m]

    def vadj(self):
        adj = [set() for _ in range(self.s.n)]
        for a, b in self.s.live_edges():
            if self.kinds.get(_key(a, b), VEH) == VEH:
                adj[a].add(int(b))
                adj[b].add(int(a))
        return adj

    def split(self, a, b, p):
        """Insert a node at p on edge (a, b) (GenState.split_edge; an end within 0.5 m is reused); kinds follow."""
        s = self.s
        pa, pb = s.nodes[a], s.nodes[b]
        if np.hypot(*(p - pa)) < 0.5:
            return int(a)
        if np.hypot(*(p - pb)) < 0.5:
            return int(b)
        if _key(a, b) in s.raised or s.n >= len(s.nodes):             # never split a bridge / tunnel
            return int(a) if np.hypot(*(p - pa)) <= np.hypot(*(p - pb)) else int(b)
        k = self.kinds.pop(_key(a, b), None)
        v = s.split_edge(int(a), int(b), np.asarray(p, float))
        s.max_nodes += 1                                              # the sampler's new-node budget is unchanged
        if k is not None:
            self.kinds[_key(v, a)] = self.kinds[_key(v, b)] = k
        return v

    def node_at(self, p, tol: float = 1.0):
        """An existing node within MERGE_M of p, or a node split into the live edge within ``tol`` of p, or None."""
        s = self.s
        live = np.array([len(x) > 0 for x in s.adj], bool)
        if live.any():
            d = np.hypot(*(s.nodes[:s.n] - p).T)
            d[~live] = np.inf
            v = int(np.argmin(d))
            if d[v] < MERGE_M:
                return v
        e = s.live_edges()
        if not len(e):
            return None
        dd = _seg_dist(np.repeat(np.asarray(p, float)[None], len(e), 0), s.nodes[e[:, 0]], s.nodes[e[:, 1]])
        k = int(np.argmin(dd))
        if dd[k] > tol:
            return None
        a, b = s.nodes[e[k, 0]], s.nodes[e[k, 1]]
        t = float(np.clip(((p - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1))
        return self.split(e[k, 0], e[k, 1], a + t * (b - a))

    def commit_step(self, tip, o, cls: int = CLS):
        """GenState.commit; a street split by the step (T-junction, crossing) keeps its kind on both halves."""
        k = None
        if o["kind"] in ("crossing_split", "t_junction"):
            k = self.kinds.pop(_key(*o["edge"]), None)
        v = self.s.commit(tip, o, cls)
        if k is not None and v >= 0:
            a, b = o["edge"]
            self.kinds[_key(v, a)] = self.kinds[_key(v, b)] = k
        return v

    def commit_stretch(self, tip, C, evaluate, log: Counter, net: Net | None = None, seglog: list | None = None):
        """Commit polyline C from node ``tip`` step by step under the sampler's rules (choice 9). A step the rules
        join to the network (merge, T-junction, crossing) completes the stretch when the rest of C runs within MERGE_M
        of the network the route was planned on (the rest is reuse; a further step would run along that street), and
        so does reaching the network within MERGE_M of the end. Returns (complete, added length m, last node);
        ``seglog`` collects the committed segments (start, end)."""
        s = self.s
        added, last_kind = 0.0, None

        def rest_on_net(qi):
            if net is None or not len(net.seg):
                return False
            R_ = densify(C[qi:], 2.0)
            d = np.full(len(R_), np.inf)
            for A_, B_ in net.seg[net.near_box(R_.min(0), R_.max(0), MERGE_M)]:
                d = np.minimum(d, _seg_dist(R_, np.repeat(A_[None], len(R_), 0), np.repeat(B_[None], len(R_), 0)))
            return bool((d < MERGE_M).all())
        for qi in range(1, len(C)):
            q, last = C[qi], qi == len(C) - 1
            for _ in range(16):
                d = q - s.nodes[tip]
                dist = float(np.hypot(*d))
                if dist < 1.0 or (dist < MERGE_M and not last):
                    break
                if dist < MERGE_M and last_kind in NET_KINDS:         # already on the network within MERGE_M of
                    return True, added, tip                           # the end: a further step would be dropped
                o, ok, reason = evaluate(s, tip, d)
                if not ok:
                    log["step_infeasible"] += 1
                    log[f"infeasible:{reason or o['kind']}"] += 1
                    return False, added, tip
                L = float(np.hypot(*(o["q"] - s.nodes[tip])))
                p0 = s.nodes[tip].copy()
                v = self.commit_step(tip, o)
                if v < 0:
                    return False, added, tip
                if seglog is not None:
                    seglog.append((p0, np.asarray(o["q"], float).copy()))
                added += L
                tip, last_kind = v, o["kind"]
                if o["kind"] in NET_KINDS and rest_on_net(qi if o["kind"] in ("crossing_split", "crossing_endpoint")
                                                          else qi + 1 if not last else qi):
                    return True, added, tip                           # the rest of the stretch is existing street
                if o["kind"] in ("crossing_split", "crossing_endpoint"):
                    continue                                          # resume toward q from the crossing node
                break
        if last_kind is None:                                         # no step was needed: start = end
            return bool(np.hypot(*(s.nodes[tip] - C[-1])) < MERGE_M), added, tip
        return last_kind in NET_KINDS, added, tip


def sampler_evaluator(sm, i: int, cls: int = CLS):
    """evaluate(s, u, d) from an M14 sampler, exactly as m19_arms.posthoc calls it."""
    def ev(s, u, d):
        o, _, lw, reason = sm.evaluate_candidate(i, s, u, np.asarray(d, float), cls, 0)
        return o, bool(np.isfinite(lw) and o["kind"] != "dropped"), reason
    return ev


def stub_evaluator(s, u, d):
    """The geometric insertion rules only (GenState.resolve_step), for tests."""
    o = s.resolve_step(u, np.asarray(d, float))
    return o, o["kind"] != "dropped", "" if o["kind"] != "dropped" else "unrepresentable"


# ---------------------------------------------------------------- steep runs on the vehicular network (choice 12)
@dataclass
class Run:
    path: list            # stroke node path (GenState ids)
    S: np.ndarray         # arc position of each path node
    s0: float
    s1: float
    pieces: np.ndarray    # piece ids (of the analysis) in stroke order
    length: float
    wgmax: float
    mids: np.ndarray      # piece midpoints
    apex: bool = False    # holds a G2 hairpin apex (D4a 1)

    @property
    def key(self) -> int:
        a = tuple(np.round(self.mids[0], 1))
        b = tuple(np.round(self.mids[-1], 1))
        return int(hashlib.sha256(repr(sorted([a, b])).encode()).hexdigest()[:16], 16)


class Analysis:
    """Pieces, 30 m-window grades, steep runs (> cap; back-to-back runs less than 30 m apart are one run, D4a 1)
    and forced stretches (> force) of the vehicular network."""

    def __init__(self, st: State, F, P: Params):
        self.P = P
        self.G, _ = st.view(lambda k: k == VEH)
        G = self.G
        self.pc = H.pieces(G, F)
        self.wg = H.window_grades(G, F, self.pc)
        self.runs, self.forced = [], []
        if not len(G.edges):
            return
        apex = hairpin_windows(G)
        eid = {(int(a), int(b)): i for i, (a, b) in enumerate(G.edges)}
        off, pc = self.pc.off, self.pc
        mids = (pc.p0 + pc.p1) / 2
        for path in strokes(G)[0]:
            idx, arc = [], [0.0]
            for u, v in zip(path[:-1], path[1:]):
                e = eid[_key(u, v)]
                r = np.arange(off[e], off[e + 1])
                idx.append(r if G.edges[e, 0] == u else r[::-1])
                arc.append(arc[-1] + float(pc.length[r].sum()))
            if not idx:
                continue
            S = np.array(arc)
            idx = np.concatenate(idx)
            pos = np.r_[0, np.cumsum(pc.length[idx])]
            w = np.nan_to_num(self.wg[idx], nan=0.0)
            for thr, out, gap in ((P.cap, self.runs, P.ramp_gap_m), (P.force, self.forced, 0.0)):
                for x0, x1 in merged_stretches(w > thr, pc.length[idx], gap):
                    ix = idx[x0:x1]
                    out.append(Run(list(path), S, float(pos[x0]), float(pos[x1]), ix, float(pos[x1] - pos[x0]),
                                   float(w[x0:x1].max()), mids[ix], _touches(apex, pc.p0[ix], pc.p1[ix])))

    def short_ramp(self, r: Run) -> bool:
        """<= 60 m and <= 25 %; <= 40 m and <= 20 % if the run holds a hairpin apex (D4a 1)."""
        lim_m, lim_g = self.P.ramp_m, self.P.ramp_pct
        if r.apex:
            lim_m, lim_g = min(lim_m, self.P.apex_m), min(lim_g, self.P.apex_pct)
        return r.length <= lim_m + 1e-9 and r.wgmax <= lim_g + 1e-9

    def drivable(self):
        """The G1 drivable piece mask: window grade <= cap (NaN kept) or a piece of a short ramp (choice 15)."""
        keep = ~(self.wg > self.P.cap)
        for r in self.runs:
            if self.short_ramp(r):
                keep[r.pieces] = True
        return keep


def own_chain(s, u) -> list:
    """The edges of dead end u's own chain: from u through degree-2 nodes to the first node of another degree."""
    out, prev, cur = [], -1, int(u)
    for _ in range(s.n):
        nxt = [w for w in s.adj[cur] if w != prev]
        if not nxt or (cur != u and len(s.adj[cur]) != 2):
            break
        out.append(_key(cur, nxt[0]))
        prev, cur = cur, int(nxt[0])
        if len(s.adj[cur]) != 2:
            break
    return out


def _order(runs):
    return sorted(runs, key=lambda r: (-round(r.length, 6), r.key))


# ---------------------------------------------------------------- the hybrid
class Hybrid:
    """H1 + H2 on a copy of a sampler state (module docstring). ``T``: rule terrain; ``grad_at(x, y)``: landform
    gradient for A5's eligibility; ``evaluate``: the sampler's step evaluator."""

    def __init__(self, s, T, grad_at, evaluate, P: Params = Params(), snap_every: int = 1):
        self.st = State.start(s)
        self.T, self.grad_at, self.evaluate, self.P = T, grad_at, evaluate, P
        self.geo = Geo(T, P)
        self.size = float(s.size)
        self.gross = 0.0
        self.snap_every, self.n_snap = snap_every, 0
        self.log = {"H1": Counter(), "H2": Counter()}
        self.kept_mids = []
        self._net = None
        self._pass, self._segs = 1, []
        self.pass2_segs = {"H1": [], "H2": []}
        self.snapshots = [{"stage": "start", "gross_m": 0.0, "graph": self.typed_json()}]

    # -- snapshots (choice 19)
    def snapshot(self, stage, force=False):
        self.n_snap += 0 if force else 1
        if force or self.n_snap % self.snap_every == 0:
            self.snapshots.append({"stage": stage, "gross_m": self.gross, "graph": self.typed_json()})

    # -- routing with geometry and its fallbacks (choices 6, 8)
    def stretches(self, R):
        """Off-network stretches of route R: [(polyline, starts at the route start)] (choice 6)."""
        P_, on = R["poly"], R["on"]
        n = len(P_)
        reuse = (on[:-1] >= 0) & (on[1:] >= 0)
        net = self._net
        out = []
        k = 0
        while k < n - 1:
            if reuse[k]:
                k += 1
                continue
            k1 = k
            while k1 < n - 1 and not reuse[k1]:
                k1 += 1
            pts = P_[k:k1 + 1].copy()
            for j, vi in ((0, k), (-1, k1)):
                if 0 < vi < n - 1 and on[vi] >= 0:
                    A, B = net.seg[on[vi], 0], net.seg[on[vi], 1]
                    t = float(np.clip(((P_[vi] - A) @ (B - A)) / max((B - A) @ (B - A), 1e-12), 0, 1))
                    pts[j] = A + t * (B - A)
            if np.hypot(*(pts[-1] - pts[0])) >= MERGE_M or k == 0 or k1 == n - 1:
                out.append((pts, k == 0))
            k = k1
        return out

    def _shape_all(self, R):
        out = []
        for pts, at_start in self.stretches(R):
            C, info, D = shape(self.geo, pts, self._net)
            if C is None:
                return None, D, info
            out.append((C, at_start, info))
        return out, None, {}

    def _route(self, a, b, log: Counter, pad=None, start_new=False, P: Params | None = None):
        t0 = time.time()
        R = route(self.T, a, b, self.size, self._net, self.P if P is None else P, pad)
        log["router_calls"] += 1
        log["router_s"] += time.time() - t0
        if R is not None and start_new:                               # D4a 3: the route starts at the node itself
            R["on"] = R["on"].copy()
            R["on"][0] = -1
        return R

    def two_pass(self, a, b, log: Counter, detour_max: float, accept=None, pad=None, start_new=False):
        """D4c: pass 1 with reuse x 0.2; pass 2 with reuse x 1 (band, crossings, pad, tie-breaks and the D4a 3 rules
        unchanged: same net, same ``start_new``) only if pass 1 breaks detour <= 4, measured on the full route length
        (new pieces plus reused streets). ``accept``: a further length rule (H1: A5's limit). Returns (route, pass,
        params) or (None, reason, None) with reason "no_route", "detour" (both passes) or "limit"."""
        R = self._route(a, b, log, pad, start_new)
        if R is None:
            return None, "no_route", None
        if R["length"] > detour_max + 1e-9:
            P2 = replace(self.P, reuse_x=1.0)
            log["pass2_tried"] += 1
            R = self._route(a, b, log, pad, start_new, P2)
            if R is None or R["length"] > detour_max + 1e-9:
                return None, "detour", None
            return (R, 2, P2) if accept is None or accept(R["length"]) else (None, "limit", None)
        if accept is not None and not accept(R["length"]):
            return None, "limit", None
        return R, 1, self.P

    def plan(self, st: State, a, b, detour_max: float, log: Counter, accept=None, pad=None, hidden=(),
             start_new=False):
        """Route a -> b on state ``st`` (two passes, D4c), shape every new stretch, fall back as D4 / D4a say.
        Returns (shaped stretches, None) or (None, reason in {"room", "smoothing"}); ``self._pass`` = the pass used.
        The reuse set is the vehicular edges of ``st`` (stairs and a run being replaced are already typed pedestrian
        there) minus ``hidden``, which is also out of the x 10 band; ``pad``: the box margin; ``start_new``: the first
        new stretch starts at a. The pushed re-route stays in the pass that found the route (D4c adds nothing else)."""
        self._net = Net.of(st.s, st.kinds, hidden=hidden)
        R, pas, Pp = self.two_pass(a, b, log, detour_max, accept, pad, start_new)
        if R is None:                                                 # sub-reason logged for diagnosis
            log[f"room:{pas}"] += 1
            return None, "room"
        self._pass = pas
        ok_len = lambda L: L <= detour_max + 1e-9 and (accept is None or accept(L))
        log["route_ramp"] += int(R["ramp"])
        sh, D, info = self._shape_all(R)
        if sh is None:                                                # one re-route, the apex pushed 12 m outward
            hv = hairpin_vertex(D, info.get("apex"))
            if hv is not None:
                V = D[hv[0]] + self.P.push_m * hv[1]
                R1 = self._route(a, V, log, pad, start_new, Pp)
                R2 = self._route(V, b, log, pad, P=Pp) if R1 is not None else None
                if R2 is not None and ok_len(R1["length"] + R2["length"]):
                    R = {"poly": np.vstack([R1["poly"], R2["poly"][1:]]), "on": np.r_[R1["on"], R2["on"][1:]],
                         "length": R1["length"] + R2["length"], "ramp": R1["ramp"] or R2["ramp"]}
                    sh, _, info = self._shape_all(R)
                    log["pushed_reroute"] += 1
            if sh is None:                                            # rejected (D4a 2: smoothing_fail)
                log["smoothing_fail"] += 1
                log[f"smoothing_fail:{info.get('fail', '?')}"] += 1
                return None, "smoothing"
        for _, _, inf in sh:
            log["fewer_chaikin"] += int(bool(inf.get("fewer")))
        return sh, None

    def log_pass(self, stage: str):
        """D4c: count the committed connector / switchback by the pass that found it; keep pass-2 segments."""
        self.log[stage][f"pass{self._pass}"] += 1
        if self._pass == 2:
            self.pass2_segs[stage] += self._segs

    def pass2_lens(self):
        """D4c: per stage, the length of the final vehicular edges laid by pass 2 and its lens-face share
        (hilltown._faces, the M4 lens rule, on the final vehicular graph)."""
        g, k = self.typed()
        gv = RoadGraph(g.nodes, g.edges[k == 0], np.zeros(g.n, bool))
        fc = H._faces(gv, self.geo.F) if len(gv.edges) else None
        lens = fc["on_lens"] if fc is not None else np.zeros(len(gv.edges), bool)
        for stage, segs in self.pass2_segs.items():
            log = self.log[stage]
            n1, n2 = log.get("pass1", 0), log.get("pass2", 0)
            if n1 + n2:
                log["pass2_share"] = n2 / (n1 + n2)
            if not segs or not len(gv.edges):
                continue
            mid = (gv.nodes[gv.edges[:, 0]] + gv.nodes[gv.edges[:, 1]]) / 2
            L = np.hypot(*(gv.nodes[gv.edges[:, 0]] - gv.nodes[gv.edges[:, 1]]).T)
            on = np.zeros(len(mid), bool)
            for a, b in segs:
                on |= _seg_dist(mid, np.repeat(a[None], len(mid), 0), np.repeat(b[None], len(mid), 0)) <= 0.5
            log["pass2_m"] = float(L[on].sum())
            log["pass2_lens_m"] = float(L[on & lens].sum())
            if log["pass2_m"] > 0:
                log["pass2_lens_share"] = log["pass2_lens_m"] / log["pass2_m"]

    def log_bends(self, sh, log: Counter):
        """Descriptive: the smallest radius of the committed bends outside G2 hairpins (D4a e)."""
        for C, _, _ in sh:
            r = other_bend_radius(C)
            log["bend_r_min_m"] = min(log.get("bend_r_min_m", float("inf")), r)
            log["bends_under_12m"] += int(r < self.P.radius_m)

    def commit_plan(self, st: State, start_node, sh, log: Counter):
        """Commit the shaped stretches on ``st``. Returns (all complete, added m); ``self._first`` = the node the
        first new stretch starts from, ``self._segs`` the committed segments."""
        added, ok = 0.0, True
        self._first, self._segs = None, []
        for C, at_start, _ in sh:
            tip = start_node if at_start else st.node_at(C[0])
            if tip is None:
                log["stretch_start_missing"] += 1
                return False, added
            self._first = tip if self._first is None else self._first
            done, L, _ = st.commit_stretch(tip, C, self.evaluate, log, self._net, self._segs)
            added += L
            log["new_stretches"] += 1
            if not done:
                return False, added
        return ok, added

    # -- H1 connectors (A5's eligibility, destinations, limit and order; m19_arms.posthoc)
    def h1(self):
        log, s = self.log["H1"], self.st.s
        cand = []
        for u in range(s.n):
            if len(s.adj[u]) != 1 or s.border[u]:
                continue
            gx, gy = self.grad_at(*s.nodes[u])
            if float(np.hypot(gx[0], gy[0])) < SLOPE_MIN:
                continue
            t = destination(s, u)
            if t is None:
                continue
            self._net = Net.of(s, self.st.kinds, hidden=own_chain(s, u))      # D4a 3
            R, _, _ = self.two_pass(s.nodes[u], t["xy"], log, self.P.detour * t["straight"],    # D4c, both passes
                                    lambda L: L <= ROUTE_X * t["straight"] + ROUTE_PAD, start_new=True)
            if R is not None:
                cand.append((R["length"], u))
        for _, u in sorted(cand):
            s = self.st.s
            if len(s.adj[u]) != 1:
                continue
            t = destination(s, u)
            if t is None:
                continue
            sh, why = self.plan(self.st, s.nodes[u].copy(), t["xy"], self.P.detour * t["straight"], log,
                                accept=lambda L: L <= ROUTE_X * t["straight"] + ROUTE_PAD, hidden=own_chain(s, u),
                                start_new=True)
            if sh is None:
                log[f"rejected_{why}"] += 1
                continue
            if not sh:                                                # the route is all existing street
                log["no_new_pieces"] += 1
                continue
            ok, added = self.commit_plan(self.st, u, sh, log)
            log["start_reused"] += int(self._first is not None and self._first != u)     # 0 by D4a 3
            self.log_bends(sh, log)
            self.log_pass("H1")
            self.gross += added
            log["connectors_complete" if ok else "connectors_partial"] += 1
            log["added_m"] += added
            self.snapshot("H1")

    # -- H2 steep runs
    def placement(self, st: State, r: Run):
        """Stairs on run r (choice 14) on state ``st``: (end node A, end node B, metres of stubs absorbed); an end
        whose dangling stub was absorbed moves to the stub's far node."""
        s = st.s
        e, _, _, _ = st.edges()
        deg = np.bincount(e.ravel(), minlength=s.n) if len(e) else np.zeros(s.n, int)
        S, path = r.S, r.path
        junc = [j for j in range(len(path)) if deg[path[j]] >= 3]
        s0, s1 = r.s0, r.s1
        lo = [S[j] for j in junc if s0 - self.P.snap_m - 1e-6 <= S[j] <= s0 + 1e-6]
        hi = [S[j] for j in junc if s1 - 1e-6 <= S[j] <= s1 + self.P.snap_m + 1e-6]
        s0, s1 = (max(lo) if lo else s0), (min(hi) if hi else s1)
        segs = []
        for j in range(len(path) - 1):
            Lj = S[j + 1] - S[j]
            if S[j + 1] <= s0 + 1e-6 or S[j] >= s1 - 1e-6 or Lj <= 0:
                continue
            segs.append((path[j], path[j + 1], max(0.0, (s0 - S[j]) / Lj), min(1.0, (s1 - S[j]) / Lj)))
        touched, ends = set(), []
        for u, v, f0, f1 in segs:
            pu, pv = s.nodes[u].copy(), s.nodes[v].copy()
            a, b = u, v
            if f1 < 1 - 1e-9:
                b = st.split(a, b, pu + f1 * (pv - pu))
            if f0 > 1e-9:
                a = st.split(a, b, pu + f0 * (pv - pu))
            if a != b:
                st.kinds[_key(a, b)] = PED
            touched |= {a, b}
            ends.append((a, b))
        if not ends:
            return None
        A, B = ends[0][0], ends[-1][1]
        absorbed, moved = self.absorb(st, touched)
        for _ in range(len(moved) + 1):
            A, B = moved.get(A, A), moved.get(B, B)
        return A, B, absorbed

    def absorb(self, st: State, touched: set):
        """Absorb vehicular chains < stub_m dangling from the stairs (vehicular degree 1 there). Returns (m absorbed,
        {stairs node: far node of its absorbed chain})."""
        s = st.s
        total, todo, moved = 0.0, set(touched), {}
        for _ in range(64):
            vadj = st.vadj()
            hit = None
            for x in sorted(todo):
                if len(vadj[x]) != 1:
                    continue
                chain, prev, cur, L = [], x, next(iter(vadj[x])), 0.0
                while True:
                    chain.append((prev, cur))
                    L += float(np.hypot(*(s.nodes[cur] - s.nodes[prev])))
                    if len(vadj[cur]) != 2 or L >= self.P.stub_m or cur == x:
                        break
                    prev, cur = cur, next(w for w in vadj[cur] if w != prev)
                if L < self.P.stub_m:
                    hit = (chain, cur, L)
                    break
            if hit is None:
                break
            chain, end, L = hit
            for a, b in chain:
                st.kinds[_key(a, b)] = PED
            todo |= {end}
            moved[chain[0][0]] = end
            total += L
        return total, moved

    def joined(self, st: State, A: int, B: int) -> bool:
        """G1 rule (1): A and B joined in the drivable subgraph with detour <= 4 (choice 15)."""
        straight = float(np.hypot(*(st.s.nodes[A] - st.s.nodes[B])))
        if A == B or straight < 1e-6:
            return True
        an = Analysis(st, self.geo.F, self.P)
        nodes, edges = H.split_graph(an.G, an.pc, an.drivable())
        if not len(edges):
            return False
        L = np.hypot(*(nodes[edges[:, 0]] - nodes[edges[:, 1]]).T)
        n = len(nodes)
        G = coo_matrix((np.r_[L, L] + 1e-12, (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(n, n)).tocsr()
        d = dijkstra(G, indices=A, limit=self.P.detour * straight + 1e-6)
        return bool(d[B] <= self.P.detour * straight + 1e-6)

    def _cutoff(self, before: State, after: State):
        """(pieces, m) of vehicular street split off by a forced placement, the largest piece of each split
        component excepted (choice 13)."""
        def comps(st):
            G, _ = st.view(lambda k: k == VEH)
            if not len(G.edges):
                return G, np.zeros(0, int), np.zeros(0)
            A = coo_matrix((np.ones(len(G.edges)), (G.edges[:, 0], G.edges[:, 1])), shape=(G.n, G.n))
            lab = connected_components(A, directed=False)[1]
            L = np.hypot(*(G.nodes[G.edges[:, 0]] - G.nodes[G.edges[:, 1]]).T)
            return G, lab, L
        G0_, lab0, _ = comps(before)
        G1_, lab1, L1 = comps(after)
        if not len(G1_.edges):
            return 0, 0.0
        old = {}
        n0 = len(lab0)
        for (a, b), c1, l in zip(G1_.edges, lab1[G1_.edges[:, 0]], L1):
            c0 = lab0[a] if a < n0 else lab0[b] if b < n0 else -1     # new nodes are splits of old edges
            old.setdefault(c0, Counter())[c1] += l
        pieces, metres = 0, 0.0
        for c0, parts in old.items():
            if len(parts) > 1:
                big = max(parts, key=parts.get)
                pieces += len(parts) - 1
                metres += sum(v for k, v in parts.items() if k != big)
        return pieces, metres

    def _same_kept(self, r: Run) -> bool:
        if not self.kept_mids:
            return False
        d, _ = cKDTree(np.vstack(self.kept_mids)).query(r.mids)
        return bool((d <= 2.0).mean() >= 0.5)

    def h2(self):
        log, P = self.log["H2"], self.P
        an = Analysis(self.st, self.geo.F, P)
        log["runs_initial"] = sum(not an.short_ramp(r) for r in an.runs)
        log["short_ramps_initial"] = sum(an.short_ramp(r) for r in an.runs)
        log["forced_initial"] = len(an.forced)
        cap = 50 + 10 * (len(an.runs) + len(an.forced))
        for it in range(cap):
            if it:
                an = Analysis(self.st, self.geo.F, P)
            forced = [r for r in an.forced if not self._same_kept(r)]
            if forced:                                                # D4: a piece above 30 % becomes stairs regardless
                r = _order(forced)[0]
                st2 = self.st.copy()
                pl = self.placement(st2, r)
                if pl is None:
                    log["placement_failed"] += 1
                    self.kept_mids.append(r.mids)
                    continue
                pcs, m = self._cutoff(self.st, st2)
                log["forced_stairs"] += 1
                log["forced_stairs_m"] += r.length
                log["forced_cutoff_pieces"] += pcs
                log["forced_cutoff_m"] += m
                log["stubs_absorbed_m"] += pl[2]
                self.st = st2
                continue
            cands = [r for r in an.runs if not an.short_ramp(r) and not self._same_kept(r)]
            if not cands:
                break
            if it == cap - 1:
                log["iteration_cap"] += 1
            r = _order(cands)[0]
            st2 = self.st.copy()
            pl = self.placement(st2, r)
            if pl is None:
                log["placement_failed"] += 1
                self.kept_mids.append(r.mids)
                continue
            A, B, absorbed = pl
            if self.joined(st2, A, B):                                # (1) stairs
                log["runs_stairs"] += 1
                log["runs_stairs_m"] += r.length
                log["stubs_absorbed_m"] += absorbed
                self.st = st2
                continue
            straight = float(np.hypot(*(st2.s.nodes[A] - st2.s.nodes[B])))   # route with the run already stairs
            pad = max(P.pad_m, 2 * straight)                                 # D4a 4 (clipped to the domain)
            log["pad_max_m"] = max(log.get("pad_max_m", 0.0), pad)
            sh, why = self.plan(st2, st2.s.nodes[A].copy(), st2.s.nodes[B].copy(), P.detour * straight, log, pad=pad)
            if sh is not None:
                log["switchback_no_new_pieces"] += int(not sh)
                ok, added = self.commit_plan(st2, A, sh, log)
                if ok:                                                # (2) switchback
                    self.log_bends(sh, log)
                    self.log_pass("H2")
                    log["runs_switchback"] += 1
                    log["runs_switchback_m"] += r.length
                    log["switchback_added_m"] += added
                    log["stubs_absorbed_m"] += absorbed
                    self.st = st2
                    self.gross += added
                    self.snapshot("H2")
                    continue
                why = "commit"
            log["runs_kept"] += 1                                     # (3) kept
            log["kept_m"] += r.length
            log[{"room": "kept_failed_for_room", "smoothing": "kept_smoothing_fail",
                 "commit": "kept_commit_failure"}[why]] += 1
            self.kept_mids.append(r.mids)
        an = Analysis(self.st, self.geo.F, P)
        log["short_ramps_final"] = sum(an.short_ramp(r) for r in an.runs)
        log["short_ramps_final_m"] = sum(r.length for r in an.runs if an.short_ramp(r))
        log["drivable_claim"] = int(log["runs_kept"] == 0)

    # -- output (choice 18)
    def typed(self):
        """(RoadGraph remapped as GenState.graph, kind per edge: 0 vehicular, 1 stepped ramp, 2 stairs)."""
        st = self.st
        e, kind, cls, sts = st.edges()
        nodes = [st.s.nodes[:st.s.n]]
        E, K, Cl, St = [], [], [], []
        veh = kind == VEH
        E += [x for x in e[veh]]
        K += [0] * int(veh.sum())
        Cl += cls[veh].tolist()
        St += sts[veh].tolist()
        if (~veh).any():
            Gp = RoadGraph(st.s.nodes[:st.s.n].copy(), e[~veh], np.zeros(st.s.n, bool), edge_class=cls[~veh],
                           edge_struct=sts[~veh])
            pc = H.pieces(Gp, self.geo.F)
            wg = H.window_grades(Gp, self.geo.F, pc)
            typ = np.where(np.nan_to_num(wg, nan=0.0) > self.P.ramp_pct, 2, 1)
            n_new = st.s.n
            off = pc.off
            for j, (a, b) in enumerate(Gp.edges):
                tj = typ[off[j]:off[j + 1]]
                cut = np.flatnonzero(np.diff(tj)) + 1
                bounds = np.r_[0, cut, len(tj)]
                ids = [a]
                pa, pb = Gp.nodes[a], Gp.nodes[b]
                for c in cut:
                    nodes.append((pa + c / len(tj) * (pb - pa))[None])
                    ids.append(n_new)
                    n_new += 1
                ids.append(b)
                for x in range(len(bounds) - 1):
                    E.append(np.array([ids[x], ids[x + 1]]))
                    K.append(int(tj[bounds[x]]))
                    Cl.append(int(Gp.edge_class[j]))
                    St.append(int(Gp.edge_struct[j]))
        N = np.vstack(nodes)
        E = np.array(E, int).reshape(-1, 2)
        E = np.sort(E, 1)
        used = np.unique(E.ravel()) if len(E) else np.zeros(0, int)
        remap = -np.ones(len(N), int)
        remap[used] = np.arange(len(used))
        g = RoadGraph(N[used], remap[E] if len(E) else E, np.zeros(len(used), bool), edge_class=np.array(Cl, int),
                      edge_struct=np.array(St, int))
        return g, np.array(K, int)

    def typed_json(self):
        from urbanmorph.experiments.graph_io import graph_to_json
        g, k = self.typed()
        return {**graph_to_json(g), "kind": k.tolist()}

    def km(self):
        g, k = self.typed()
        L = np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T) if len(g.edges) else np.zeros(0)
        return {n_: float(L[k == i].sum() / 1000) for i, n_ in enumerate(KIND_NAMES)}


def run_hybrid(s, T, grad_at, evaluate, P: Params = Params(), snap_every: int = 1, h1: bool = True, h2: bool = True):
    """H1 then H2 on a copy of GenState ``s``. Returns {"graph", "kind", "snapshots", "log", "km", "gross_m"}."""
    hb = Hybrid(s, T, grad_at, evaluate, P, snap_every)
    t0 = time.time()
    if h1:
        hb.h1()
    t1 = time.time()
    if h2:
        hb.h2()
    hb.snapshot("end", force=True)
    hb.pass2_lens()
    g, k = hb.typed()
    log = {st: dict(c) for st, c in hb.log.items()}
    log["seconds"] = {"H1": round(t1 - t0, 2), "H2": round(time.time() - t1, 2)}
    return {"graph": g, "kind": k, "snapshots": hb.snapshots, "log": log, "km": hb.km(), "gross_m": hb.gross,
            "state": hb.st}


# ---------------------------------------------------------------- smoke run on a stored M19 A0 run
SAMPLER_KEYS = ("cfg", "tfs", "pots", "caps", "cds_topo", "stats")


def replay_a0(unit: str, bb: str, seed: int):
    """Replay M19 A0 exactly as m20_blind_repair.job builds it; abort unless every graph equals the stored A0 graph.
    Returns (ids, chosen final states, sampler)."""
    import torch
    from m18_joint_round import cross_over, joint_params
    from m19_arms import Arm, run_file, unit_tiles
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    from urbanmorph.experiments.graph_io import graph_to_json
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import NetSampler, net_preset
    from urbanmorph.graphs.tc_potentials import Potentials
    tile_mode = unit in ("gen2", "natural")
    torch.set_num_threads(2 if tile_mode else 1)
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    m = method("NA-SMC")
    model, names, norm, mcfg = load_backbone(bb, "cpu")
    over = dict(joint_params()) if tile_mode else dict(cross_over(unit)[0])
    cfg = net_preset(m.preset, **{**m.overrides, **over})
    if tile_mode:
        T_ = unit_tiles(unit)
        ds = [t["d"] for t in T_]

        class S_(Arm, NetSampler):
            pass
        try:
            crop = model_inputs(mcfg, [t["sid"] for t in T_], ds)
        except FileNotFoundError:
            crop = None
        sm = S_(model, ds, names, norm, pots, cfg, "cpu", crop)
    else:
        import site_benchmark as sb
        from m16_kaist_all import AnchoredGlobalNet
        site = sb.Site(unit)
        big = site.domain()

        class S_(Arm, AnchoredGlobalNet):
            pass
        sm = S_(model, {**big, "site": f"{site.cfg['id']}_NA-SMC"}, site.L, names, norm, pots, cfg)
        sm.anchor = None
    sm._arm_init("A0", seed)
    sm.voc_log = None
    gs = sm.run(seed)
    stored = pickle.loads(zlib.decompress(run_file(unit, "A0", bb, seed).read_bytes()))
    if [graph_to_json(g) for g in gs] != stored["graphs"]:
        raise RuntimeError(f"A0 replay differs from the stored run: {(unit, bb, seed)}")
    chosen = []
    for i, g in enumerate(gs):
        S = sm.states[i]
        p = next((k for k, x in enumerate(S) if np.array_equal(x.graph(with_class=True).nodes, g.nodes)
                  and np.array_equal(x.graph(with_class=True).edges, g.edges)), 0)
        chosen.append(S[p])
    return stored["ids"], chosen, sm


def sampler_proxy(parts: dict):
    """A TCSampler carrying only what evaluate_candidate reads (cached smoke runs)."""
    from urbanmorph.graphs.tc_sampler import TCSampler
    px = TCSampler.__new__(TCSampler)
    px.__dict__.update(parts)
    return px


def smoke(unit, bb, seed, P: Params, out=None, cache=None, only=None):
    t0 = time.time()
    if cache is not None and Path(cache).exists():
        c = pickle.loads(zlib.decompress(Path(cache).read_bytes()))
        ids, states, sm, identical = c["ids"], c["states"], sampler_proxy(c["sampler"]), c["identical"]
        print(f"loaded cached A0 replay ({cache})", flush=True)
    else:
        ids, states, sm = replay_a0(unit, bb, seed)
        identical = True
        print(f"A0 replay identical to the stored run: {len(ids)} graphs, {time.time() - t0:.0f} s", flush=True)
        if cache is not None:
            Path(cache).write_bytes(zlib.compress(pickle.dumps(
                {"ids": ids, "states": states, "identical": identical,
                 "sampler": {k: getattr(sm, k) for k in SAMPLER_KEYS}}), 6))
    rows, tot = [], {"H1": Counter(), "H2": Counter()}
    for i, uid in enumerate(ids):
        if only is not None and i not in only:
            continue
        tf = sm.tfs[i]
        res = run_hybrid(states[i], tf.t, tf.grad_at, sampler_evaluator(sm, i), P, snap_every=1)
        g = res["snapshots"]
        row = {"uid": uid, "km": {k: round(v, 3) for k, v in res["km"].items()}, "gross_m": round(res["gross_m"], 1),
               "snapshots": len(g), "monotone": all(a["gross_m"] <= b["gross_m"] + 1e-9 for a, b in zip(g[:-1], g[1:])),
               "log": {k: ({a: (round(b, 1) if isinstance(b, float) else b) for a, b in v.items()}
                           if isinstance(v, dict) else v) for k, v in res["log"].items()}}
        rows.append(row)
        for st in ("H1", "H2"):
            for k, v in res["log"][st].items():
                if k.endswith("_share"):
                    continue
                if k.endswith("_min_m"):
                    tot[st][k] = min(tot[st].get(k, float("inf")), v)
                elif k.endswith("_max_m"):
                    tot[st][k] = max(tot[st].get(k, 0.0), v)
                else:
                    tot[st][k] += v
        print(json.dumps(row), flush=True)
    for st in ("H1", "H2"):                                           # D4c shares over the unit set
        n12 = tot[st]["pass1"] + tot[st]["pass2"]
        if n12:
            tot[st]["pass2_share"] = tot[st]["pass2"] / n12
        if tot[st]["pass2_m"]:
            tot[st]["pass2_lens_share"] = tot[st]["pass2_lens_m"] / tot[st]["pass2_m"]
    summary = {"unit": unit, "backbone": bb, "seed": seed, "ramp_m": P.ramp_m, "a0_identical": identical,
               "tiles": len(rows), "totals": {k: {a: (round(b, 1) if isinstance(b, float) else b) for a, b in v.items()}
                                              for k, v in tot.items()},
               "km": {k: round(sum(r["km"][k] for r in rows), 3) for k in KIND_NAMES},
               "gross_km": round(sum(r["gross_m"] for r in rows) / 1000, 3), "seconds": round(time.time() - t0)}
    print(json.dumps(summary, indent=1), flush=True)
    if out:
        Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return summary, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke"])
    ap.add_argument("--unit", default="natural")
    ap.add_argument("--backbone", default="TNg_s0")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ramp-m", type=float, default=RAMP_M, help="short-ramp length (D4 sensitivities: 40, 100)")
    ap.add_argument("--out", default=None, help="write the logs here (JSON)")
    ap.add_argument("--cache", default=None, help="cache of the verified A0 replay (pickle) for repeated smoke runs")
    ap.add_argument("--only", default=None, help="comma-separated tile indices")
    a = ap.parse_args()
    only = None if a.only is None else {int(x) for x in a.only.split(",")}
    smoke(a.unit, a.backbone, a.seed, Params(ramp_m=a.ramp_m), a.out, a.cache, only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
