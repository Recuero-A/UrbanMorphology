"""Milestone 16: style-aware proposals and topology rules for the K-candidate samplers (docs/M16_METHOD.md).

Hypothesis: the learned turtle sometimes runs out of candidates that satisfy both the terrain and the street-pattern
constraints. Targeted proposals add candidates where the pattern wants a street; the K learned candidates always stay
in the pool, every candidate is resolved and scored by the SAME insertion rules and terrain / network potentials.

Modes (SamplerConfig.prop_mode; cumulative with the M15 heading reranking, which stays on):
* ``""``          no proposals (every M13-M15 sampler bit for bit);
* ``style``       style-aware proposals: up to ``prop_m`` extra candidates per decision, along the local pattern
                  directions (grid / organic cross field, radial spokes and ring arcs, cul-de-sac branches off collectors),
                  length = the median length of the decision's learned candidates;
* ``style_topo``  + topology rules: grid lattice (parallel streets at ``grid_spacing_m``, straight continuation, turns only
                  where a lattice line crosses, endpoints placed on lattice intersections, closure to a street ahead),
                  radial rings at multiples of ``ring_spacing_m`` with spokes held on their ray, organic infill closures at
                  irregular angles, and cul-de-sac terminal rules (local branches end at depth ``term_depth`` and never
                  close a loop: hard rules that apply to EVERY candidate, learned ones included).

Weighting of the pool (``prop_weight``):
* ``is``         balance-heuristic importance weights: candidates come from Q = (K p + M q) / (K + M), p the model's
                 displacement x class x structure density, q the style proposal (isotropic Gaussians of ``prop_sigma_m``
                 about the targets, class drawn from the model, ground only). Each candidate's log weight gains
                 log p - log Q, so selection and the stop normaliser still target p x (potentials): proposals reduce
                 variance only and cannot move mass the model gives no density.
* ``heuristic``  the feasible style candidates share selection mass ``prop_alpha`` (the pattern strength) and the
                 feasible learned candidates 1 - alpha, each group weighted within itself by its potentials; when one
                 group has no feasible member the other takes all the mass (no proposal can then stop an arm that the
                 learned pool would continue). Optional plausibility term ``prop_beta`` x min(0, log p_d(x) - median_k
                 log p_d(learned)). This CHANGES the target and the stop probability (Z = (1 - alpha) mean_learned w +
                 alpha mean_style w): heuristic selection with no inference guarantee.

Trigger (``prop_trigger``): ``always``; ``deficit:<k>`` fires when fewer than k feasible learned candidates lie within
15 degrees of a pattern direction; ``random:<p>`` fires with probability p (the matched-rate control).
Proposals are not used inside look-ahead roll-outs (``prop_in_rollout``), so roll-out cost is unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from urbanmorph.graphs.road_graph import MERGE_M
from urbanmorph.graphs.sequence import DISP_SCALE, rotate

LOCAL = (2, 3)                     # road classes: 0 major, 1 mid, 2 local, 3 minor
CONTINUE = ("grid_straight", "grid_turn", "radial_spoke", "radial_ring", "organic_straight", "cds_collector",
            "cds_branch")
                                   # topology rules: moves that continue the pattern (lower the arm's stop odds)
OCCUPIED_DEG = 30.0
NODE_TOL_M = 4.0


def patterns(cfg) -> tuple:
    """Pattern names of the proposals: ``prop_pattern`` ("radial+culdesac", ...) or derived from the M15 style."""
    if cfg.prop_pattern:
        return tuple(cfg.prop_pattern.split("+"))
    if cfg.style in ("grid", "organic", "radial"):
        return (cfg.style,)
    if cfg.style.startswith("mix:"):
        return tuple(p.split("=")[0] for p in cfg.style[4:].split("+"))
    return ("culdesac",) if cfg.style_loop or cfg.stop_bias else ()


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _unit(a: float) -> np.ndarray:
    return np.array([math.cos(a), math.sin(a)])


def occupied(s, u: int) -> list[float]:
    p = s.nodes[u]
    return [math.atan2(*(s.nodes[w] - p)[::-1]) for w in s.adj[u]]


def free(a: float, occ: list[float]) -> bool:
    return all(abs(_wrap(a - b)) > math.radians(OCCUPIED_DEG) for b in occ)


def ray_hit(s, a: np.ndarray, direction: float, reach: float, skip: int) -> float | None:
    """Distance along a ray from a to the first existing street (edges incident to ``skip`` excluded)."""
    e = s.live_edges()
    if not len(e):
        return None
    e = e[(e[:, 0] != skip) & (e[:, 1] != skip)]
    if not len(e):
        return None
    r = _unit(direction) * reach
    A, B = s.nodes[e[:, 0]], s.nodes[e[:, 1]]
    sv = B - A
    den = r[0] * sv[:, 1] - r[1] * sv[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((A[:, 0] - a[0]) * sv[:, 1] - (A[:, 1] - a[1]) * sv[:, 0]) / den
        w = ((A[:, 0] - a[0]) * r[1] - (A[:, 1] - a[1]) * r[0]) / den
    ok = (np.abs(den) > 1e-9) & (t > 0) & (t <= 1) & (w >= 0) & (w <= 1)
    return float(t[ok].min() * reach) if ok.any() else None


def edge_class(s, a: int, b: int) -> int:
    for x in s.edges:
        if x[0] >= 0 and {x[0], x[1]} == {a, b}:
            return int(x[2])
    return -1


def local_depth(s, u: int) -> tuple[int, bool]:
    """(local / minor edges between u and the nearest collector, whether a collector (major / mid edge) was reached).
    A branch grown from the start without any collector is not rooted: the terminal rules leave it alone."""
    d, v, seen = 0, u, set()
    while v >= 0 and s.parent[v] >= 0 and v not in seen:
        seen.add(v)
        c = edge_class(s, s.parent[v], v)
        if c not in LOCAL:
            return d, True
        d += 1
        v = s.parent[v]
    return d, False


def near_branch_root(s, u: int, gap: float) -> bool:
    """Is there a branch root (a node joining a collector edge and a local edge) within ``gap`` m of u?"""
    loc, col = set(), set()
    for a, b, c in s.edges:
        if a < 0:
            continue
        (loc if c in LOCAL else col).update((a, b))
    roots = [v for v in loc & col if v != u]
    return bool(roots) and float(np.min(np.hypot(*(s.nodes[roots] - s.nodes[u]).T))) < gap


def _even(t: float, L0: float) -> float:
    """A step that reaches distance t in a whole number of near-L0 steps."""
    return t / max(1, round(t / L0))


# ---------------------------------------------------------------------------------------------------- targets
def grid_targets(s, u, hd, theta, cfg, L0, topo, rng, occ, organic=False):
    """(displacement, label) for a cross field of direction theta at u."""
    a = s.nodes[u]
    axes = [theta + k * math.pi / 2 for k in range(4)]
    straight = min(axes, key=lambda x: abs(_wrap(x - hd)))
    out = []
    if organic:
        straight += math.radians(rng.normal(0, 8.0))
        side = 1 if rng.uniform() < 0.5 else -1                               # asymmetric branching
        turns = [straight + side * math.radians(rng.uniform(60, 120))]
        if rng.uniform() < 0.3:
            turns.append(straight - side * math.radians(rng.uniform(60, 120)))
        lengths = lambda: L0 * float(np.exp(rng.normal(0, 0.3)))
    else:
        turns = [straight + math.pi / 2, straight - math.pi / 2]
        lengths = lambda: L0
    if not topo:
        for ang, lab in [(straight, "straight")] + [(t, "turn") for t in turns]:
            if free(ang, occ):
                out.append((lengths() * _unit(ang), lab))
        return out
    if organic:
        for ang, lab in [(straight, "straight")] + [(t, "turn") for t in turns]:
            if not free(ang, occ):
                continue
            h = ray_hit(s, a, ang, 2.5 * L0, u)
            if h is not None and h > MERGE_M:                                    # infill closure: T-junction ahead
                out.append(((h + 1.0) * _unit(ang), "closure"))
            out.append((lengths() * _unit(ang), lab))
        return out
    sp = cfg.grid_spacing_m
    o = np.asarray(cfg_centre(cfg))
    e1, e2 = _unit(theta), _unit(theta + math.pi / 2)
    c1, c2 = float((a - o) @ e1), float((a - o) @ e2)
    near = lambda c: abs(c - sp * round(c / sp))
    for ang, lab in [(straight, "straight")] + [(t, "turn") for t in turns]:
        if not free(ang, occ):
            continue
        along_e1 = abs(math.cos(ang - theta)) > 0.7
        lateral, coord = (c2, c1) if along_e1 else (c1, c2)
        if lab == "turn" and near(lateral) > NODE_TOL_M:
            continue                                          # turn only where a lattice line passes
        dvec = _unit(ang)
        sgn = 1.0 if float(dvec @ (e1 if along_e1 else e2)) > 0 else -1.0
        nxt = sp * (math.floor(sgn * coord / sp + 1e-6) + 1)                  # next lattice line ahead
        t = nxt - sgn * coord
        step = _even(t, L0) if t <= 4 * L0 else L0
        snap = (sp * round(lateral / sp) - lateral) if near(lateral) < 0.25 * sp else 0.0
        side = e2 if along_e1 else e1                                         # hold the street on its lattice line
        out.append((step * dvec + snap * side, lab))
        h = ray_hit(s, a, ang, 1.5 * sp, u)
        if h is not None and h > step + MERGE_M:
            out.append(((h + 1.0) * dvec, "closure"))
    return out


def cfg_centre(cfg) -> tuple:
    cx = 256.0 if not np.isfinite(cfg.style_cx) else cfg.style_cx
    cy = 256.0 if not np.isfinite(cfg.style_cy) else cfg.style_cy
    return cx, cy


def radial_targets(s, u, hd, cfg, L0, topo, occ):
    a = s.nodes[u]
    c = np.asarray(cfg_centre(cfg))
    rv = a - c
    r = float(np.hypot(*rv))
    out = []
    if r < MERGE_M:                                                     # at the centre: spokes on fixed bearings
        n = cfg.n_spokes
        for k in range(n):
            ang = math.radians(cfg.style_theta0) + 2 * math.pi * k / n
            if free(ang, occ):
                out.append(((cfg.ring_spacing_m if topo else L0) * _unit(ang) + (c - a), "spoke"))
        return out
    phi = math.atan2(rv[1], rv[0])
    for ang in (phi, phi + math.pi):                                      # spokes out and in, held on the ray
        if not free(ang, occ):
            continue
        sgn = 1.0 if ang == phi else -1.0
        if topo:
            sp = cfg.ring_spacing_m
            nxt = sp * (math.floor(r / sp + 1e-6) + 1) if sgn > 0 else sp * (math.ceil(r / sp - 1e-6) - 1)
            step = _even(abs(nxt - r), L0) if 0 < abs(nxt - r) <= 4 * L0 else L0
        else:
            step = L0
        r2 = max(r + sgn * step, 0.0)
        out.append((c + r2 * _unit(phi) - a, "spoke"))
    on_ring = abs(r - cfg.ring_spacing_m * round(r / cfg.ring_spacing_m)) < NODE_TOL_M and r >= 0.5 * cfg.ring_spacing_m
    if topo and not on_ring:
        return out
    rr = cfg.ring_spacing_m * round(r / cfg.ring_spacing_m) if topo else r
    for sgn in (1.0, -1.0):                                               # ring arcs (chords on the circle)
        ang = phi + sgn * math.pi / 2
        if not free(ang, occ):
            continue
        dphi = sgn * L0 / max(rr, 1.0)
        out.append((c + rr * _unit(phi + dphi) - a, "ring"))
    return out


def culdesac_targets(s, u, hd, cfg, L0, topo, rng, occ, in_cls):
    """A collector backbone (class mid) that continues, with local terminal branches (class local) off it; on a local
    branch, a short continuation while its depth allows."""
    out = []
    if in_cls in LOCAL:
        depth, rooted = local_depth(s, u)
        if (not rooted or depth < cfg.term_depth) and free(hd, occ):
            out.append((cfg.branch_len_m * _unit(hd + math.radians(rng.normal(0, 10))), "branch_cont", 2))
        return out
    if not (topo and near_branch_root(s, u, cfg.branch_gap_m)):      # topology: branches spaced along the collector
        for side in (1, -1):
            ang = hd + side * math.radians(90 + rng.normal(0, 12))
            if free(ang, occ):
                out.append((cfg.branch_len_m * _unit(ang), "branch", 2))
    if free(hd, occ):
        out.append((L0 * _unit(hd), "collector", 1))
    return out


# ---------------------------------------------------------------------------------------------------- the hook
def aligned(sf, a, q, tol_deg=15.0) -> bool:
    d = q - a
    if d @ d < 1.0 or not sf.weights:
        return False
    th = math.atan2(d[1], d[0])
    m = (a + q) / 2
    return any(abs(_wrap(4 * (th - sf.direction(k, m)))) / 4 <= math.radians(tol_deg) for k in sf.weights)


def fires(cfg, rng, sf, s, u, learned) -> bool:
    tr = cfg.prop_trigger
    if tr == "always":
        return True
    if tr.startswith("random:"):
        return bool(rng.uniform() < float(tr.split(":")[1]))
    if tr.startswith("deficit:"):
        k = int(tr.split(":")[1])
        ok = [c for c in learned if np.isfinite(c["lw"]) and c["o"]["kind"] != "dropped"
              and aligned(sf, s.nodes[u], c["o"]["q"])]
        return len(ok) < k
    raise ValueError(tr)


def targets(cfg, sf, s, u, hd, L0, rng):
    """(world displacement, source label, class or None) of this decision's style candidates (at most prop_m)."""
    topo = cfg.prop_mode == "style_topo"
    occ = occupied(s, u)
    in_cls = edge_class(s, s.parent[u], u) if s.parent[u] >= 0 else -1
    out = []
    a = s.nodes[u]
    for p in patterns(cfg):
        if p == "grid":
            out += [(d, f"grid_{l}", None) for d, l in grid_targets(s, u, hd, sf.theta0, cfg, L0, topo, rng, occ)]
        elif p == "organic":
            out += [(d, f"organic_{l}", None) for d, l in
                    grid_targets(s, u, hd, sf.direction("organic", a), cfg, L0, topo, rng, occ, organic=True)]
        elif p == "radial":
            out += [(d, f"radial_{l}", None) for d, l in radial_targets(s, u, hd, cfg, L0, topo, occ)]
        elif p == "culdesac":
            out += [(d, f"cds_{l}", c) for d, l, c in culdesac_targets(s, u, hd, cfg, L0, topo, rng, occ, in_cls)]
    return out[:cfg.prop_m]


def logpdf_d(prop, b: int, dd_world: np.ndarray, hd: float) -> np.ndarray:
    """Model log density of world displacements (n, 2) at row b, in the model's normalised displacement units."""
    from urbanmorph.graphs.tc_sampler import mixture_logpdf
    lp, mu, ls, rho = prop["mix"]
    x = np.stack([rotate(v, -hd) / DISP_SCALE for v in dd_world])[None].astype(np.float32)
    return mixture_logpdf(torch.from_numpy(x), lp[b:b + 1], mu[b:b + 1], ls[b:b + 1], rho[b:b + 1])[0].numpy()


def logq_d(dd_world: np.ndarray, centres: np.ndarray, sigma_m: float, hd: float) -> np.ndarray:
    """Style proposal log density (normalised units): equal-weight isotropic Gaussians about the target displacements."""
    s = sigma_m / DISP_SCALE
    x = np.stack([rotate(v, -hd) / DISP_SCALE for v in dd_world])
    c = np.stack([rotate(v, -hd) / DISP_SCALE for v in centres])
    d2 = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    comp = -d2 / (2 * s * s) - math.log(2 * math.pi * s * s)
    m = comp.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(comp - m).mean(1)))
