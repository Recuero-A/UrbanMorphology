"""Terrain-aware sampling (Milestone 12): the ``terrain`` policy of ``generate.sample_graphs``.

Modes (docs/M12_EVALUATION_PROTOCOL.md §2, §5):
* ``none``: every sampled step is built. Its structure is the model's own sample from the structure head (bridge /
  tunnel / ground): a LEARNED decision with no veto. A ground step through water stays an unbridged crossing.
* ``R``: every step is checked with ``terrain.road_feasibility.evaluate``. A step that is not buildable is dropped.
  A buildable river crossing or valley span is built as a bridge. Each bridge is counted as model-placed (the
  structure head also sampled "bridge") or rule-placed (the rule overrode a "ground" sample). A model "bridge" on a
  ground-feasible step is kept only where the step crosses an existing street (an overpass); otherwise it is
  downgraded and counted.
* ``HRV`` (EXPLORATORY, added after the pre-registered results; not part of H4): HR plus, for a step that is too
  steep, an extension across a dry valley to the far rim (``road_feasibility.far_rim``) built as a viaduct, tried
  BEFORE the other replanning candidates (otherwise the contour turn always wins and no valley is ever spanned).
* ``HR``: as R, but a non-buildable proposal is replanned before it is dropped: up to 4 redraws from the same learned
  step distribution, half the step, a turn to the contour nearest the proposal, and (for a step ending in water) an
  extension to the far bank within the class span. At most 8 evaluations per step.
Every outcome is counted in ``stats`` so that learned decisions and rules can be told apart.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from urbanmorph.graphs.generate import GenState, clip_to_tile
from urbanmorph.graphs.levels import BRIDGE
from urbanmorph.graphs.road_graph import MERGE_M
from urbanmorph.graphs.sequence import DISP_SCALE, rotate
from urbanmorph.graphs.terrain_sequence import channel_list, terrain_crops, tile_canvas, tile_layers
from urbanmorph.terrain.heightfield import Terrain
from urbanmorph.terrain.road_feasibility import Thresholds, evaluate, far_bank, far_rim, water_kind_at

N_REDRAW = 4


def terrain_of(d: dict) -> Terrain:
    t = Terrain(np.where(d["valid"], d["elev"], np.nan).astype(np.float32), -256.0, 768.0, 4.0)
    t.water_kind = d["water"].astype(np.int8)
    return t


def apply_bridge(s: GenState, u: int, d: np.ndarray, cls: int) -> bool:
    """A street step built as a bridge: it passes over streets and water (no crossing split, no T-junction) and lands
    on an existing ground node within MERGE_M or on a new node."""
    a = s.nodes[u]
    q, _ = clip_to_tile(a, a + d)
    if np.hypot(*(q - a)) < MERGE_M:
        return False
    dist = np.hypot(*(s.nodes[:s.n] - q).T)
    dist[u] = np.inf
    if s.raised_nodes:
        dist[list(s.raised_nodes)] = np.inf
    v = int(dist.argmin()) if s.n > 1 else -1
    if v < 0 or dist[v] >= MERGE_M:
        v = s.add_node(q, parent=u)
        s.node_rule[v] = "bridge_end"
    else:
        s.reach(v, u)
    s.connect(u, v, cls)
    s.raised[(min(u, v), max(u, v))] = (1, BRIDGE)
    s.rules["bridge"] = s.rules.get("bridge", 0) + 1
    return True


def crosses_street(s: GenState, a, q) -> bool:
    e = s.live_edges()
    if not len(e):
        return False
    A, B = s.nodes[e[:, 0]], s.nodes[e[:, 1]]
    r, w = q - a, B - A
    den = r[0] * w[:, 1] - r[1] * w[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = ((A[:, 0] - a[0]) * w[:, 1] - (A[:, 1] - a[1]) * w[:, 0]) / den
        uu = ((A[:, 0] - a[0]) * r[1] - (A[:, 1] - a[1]) * r[0]) / den
    return bool(((np.abs(den) > 1e-9) & (tt > 0.05) & (tt < 0.95) & (uu > 0.02) & (uu < 0.98)).any())


class TerrainPolicy:
    def __init__(self, tiles: list[dict], names, norm: dict, mode: str = "none", th: Thresholds | None = None,
                 crop_tiles: list[dict] | None = None):
        """``crop_tiles``: what the MODEL sees (e.g. low-passed terrain); rules and metrics always use ``tiles``."""
        assert mode in ("none", "R", "HR", "HRV")
        self.names, self.norm, self.mode, self.th = list(names), norm, mode, th or Thresholds()
        self.layers = [tile_layers(d, 0, self.names, norm) if self.names else {} for d in (crop_tiles or tiles)]
        self.canv = [tile_canvas(l, self.names, norm) if self.names else np.zeros((0, 128, 128), np.float16)
                     for l in self.layers]
        self.terr = [terrain_of(d) for d in tiles]
        self.n_ch = len(channel_list(self.names))
        self.stats = [Counter() for _ in tiles]

    # -- inputs
    def start_extra(self, i):
        return self.canv[i]

    def crop_extra(self, i, pos, hd):
        if not self.names:
            return np.zeros((0, 48, 48), np.float16)
        return terrain_crops(self.layers[i], self.names, np.asarray(pos)[None], np.array([hd]), self.norm)[0]

    def start_ok(self, i, p) -> bool:
        if self.mode == "none":
            return True
        t = self.terr[i]
        z = t.sample(p[:1], p[1:2])
        return bool(np.isfinite(z).all() and water_kind_at(t, p[:1], p[1:2])[0] == 0)

    # -- steps
    def apply(self, i, s: GenState, u, d, cls, spred, mix, hd, gen):
        st = self.stats[i]
        st["proposals"] += 1
        st[f"model_struct_{spred}"] += 1
        if self.mode == "none":
            if spred == BRIDGE and apply_bridge(s, u, d, cls):
                st["bridge_model"] += 1
            else:
                s.apply_step(u, d, cls)
            return
        a = s.nodes[u]
        f = self._check(i, a, d, cls)
        if not f.buildable and self.mode in ("HR", "HRV"):
            cands = []                                               # (label, step): tried in this order
            if self.mode == "HRV" and f.kind == "excessive_grade":   # exploratory: span a dry valley FIRST
                fr = far_rim(self.terr[i], a, d / max(np.hypot(*d), 1e-9), cls, self.th)
                if fr is not None:
                    cands.append(("far_rim", fr - a))
            cands += [("redraw", self._redraw(mix, hd, gen)) for _ in range(N_REDRAW)]
            cands += [("halved", 0.5 * d), ("contour", self._contour(i, a, d))]
            if f.kind == "ends_in_water":
                fb = far_bank(self.terr[i], a, d / max(np.hypot(*d), 1e-9), self.th.bridge_span_max[cls])
                if fb is not None:
                    cands.append(("far_bank", fb - a))
            for label, dd in cands:
                if dd is None:
                    continue
                st["replan_evals"] += 1
                f2 = self._check(i, a, dd, cls)
                if f2.buildable:
                    st[label] += 1
                    d, f = dd, f2
                    break
        if not f.buildable:
            st[f"dropped_{f.kind}"] += 1
            return
        q, _ = clip_to_tile(a, a + d)
        if f.structure == 1:
            kind = "viaduct" if f.kind == "possible_viaduct" else "bridge"
            if apply_bridge(s, u, d, cls):
                st[f"{kind}_{'model' if spred == BRIDGE else 'rule'}"] += 1
            return
        if spred == BRIDGE:
            if crosses_street(s, a, q) and apply_bridge(s, u, d, cls):
                st["overpass_model"] += 1
                return
            st["model_bridge_downgraded"] += 1
        s.apply_step(u, d, cls)
        st["ground"] += 1

    def _check(self, i, a, d, cls):
        q, _ = clip_to_tile(a, a + d)
        return evaluate(self.terr[i], a, q, cls, self.th)

    def _redraw(self, mix, hd, gen):
        from urbanmorph.models.graph_gen import mixture_sample
        lp, mu, ls, rho = (torch.as_tensor(x)[None] for x in mix)
        dn = mixture_sample(lp, mu, ls, rho, gen)[0].numpy()
        return rotate(dn * DISP_SCALE, hd)

    def _contour(self, i, a, d):
        t = self.terr[i]
        e = 4.0
        gx = (t.sample(np.array([a[0] + e]), np.array([a[1]])) - t.sample(np.array([a[0] - e]), np.array([a[1]])))[0] / (2 * e)
        gy = (t.sample(np.array([a[0]]), np.array([a[1] + e])) - t.sample(np.array([a[0]]), np.array([a[1] - e])))[0] / (2 * e)
        if not np.isfinite([gx, gy]).all() or np.hypot(gx, gy) < 1e-6:
            return None
        c = np.array([-gy, gx]) / np.hypot(gx, gy)
        if c @ d < 0:
            c = -c
        return c * np.hypot(*d)

    def summary(self) -> dict:
        tot = Counter()
        for c in self.stats:
            tot.update(c)
        return dict(tot)
