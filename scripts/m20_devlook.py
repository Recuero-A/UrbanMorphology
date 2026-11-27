"""Milestone 20, D4: the ONE development look of the grade hybrid (docs/M20_LOG.md D4 "Development looks", "Logged per
town", the P1b snapshot rule; D4a; D4c; D3c's lens switch). P2 stays SEALED: its cells are computed here by script
and written only to gzip files whose sha256 sits in a manifest; nothing printed or written in the open carries a P2
key (``open_split`` / ``scrub``).

Stages (resumable: every (town, seed) result is cached and skipped when present; torch 1 thread per worker):
    generate   A0 = the frozen J on TNg_s0, NA-SMC, whole-town (m19_arms.job's site branch: Arm + AnchoredGlobalNet on
               the town's GLO-30 4 m domain, sm.anchor = None, _arm_init("A0", seed)), seeds 1-24 per settlement
               -> <out>/<town>/a0/sNN.json.gz (graph, logs) + sNN.state.pkl.gz (the chosen particle, for the hybrid);
               the seed-free sampler parts the step evaluator reads (m20_hybrid.SAMPLER_KEYS) once per town
    verify     replay A0 for the given seeds in fresh processes; the graph must equal the stored one bit for bit
    hybrid     H1 only and H1 + H2 (no H3: D4b) on each stored A0 state, T = the town's FABDEM 32 m terrain
               (m20_towns.terrain_of), the router on the 4 m grid (m20_hybrid choice 1), the sampler's own step rules
               -> <out>/<town>/hybrid/sNN.json.gz (class-tagged graphs, D4 / D4c logs) + sNN.snaps.pkl.gz (P1b);
               the D4c lens share of pass-2 pieces goes to <out>/sealed/<town>/sNN.hybrid_log.json.gz
    measure    hilltown.unit_metrics on the unit (the settlement polygon, or a city's core, D4d) for A0, H1, H1 + H2
               (cropped to the city limits as site_benchmark.crop), definedness from the real network of real.pkl
               -> <out>/<town>/open/sNN.json (M1, M2, M6, G1 and the logs) and <out>/sealed/<town>/sNN.json.gz (the rest)
    report     per town and arm: median and IQR over the runs of the open metrics, the per-unit run logs, the pilot
               for hold-H's N (D4d ruling 7); the sealed bundle and its manifest; timing and
               the projected CPU-hours (per stage scaled by L^k: k fitted when several L were measured, else 2)
               -> <out>/report_open.json, <out>/sealed/MANIFEST.json (+ sealed_bundle.json.gz)
    maps       one PNG per town: A0 vs H1 + H2 for one seed, streets by class over FABDEM contours -> <out>/maps/
    all        generate, hybrid, measure, report, maps

    python scripts/m20_devlook.py STAGE --split scratch|dev [--seeds 1-24] [--workers N<=6] [--ids A,B] [--delta none]
                                        [--out DIR] [--towns-n 32] [--mix 2048:7,4096:13,8192:12]

Splits. ``scratch``: the one unpinned scratch town of m20_towns --scratch (data/processed/m20/scratch_unpinned/),
outputs under scratch_unpinned/devlook/ unless --out. ``dev``: the hash-frozen dev-H unit list
(configs/m20_pool/dev_units.json: reference + descriptive) and the pinned town builds; outputs under
outputs/m20/devlook/. No hold-H or study-H list is ever read: the guard is the dev unit list itself.

Declared choices, with the committee's D4d rulings (docs/M20_LOG.md D4d ruling 6, 7):
 1. A0 = J with delta = 0 (APPROVED): M19's A0 on a site is J plus the OTHER site's D14 delta (cross_over); a dev-H town
    has no other site and the pooled delta was never fitted. Only ``--delta none`` runs on dev-H, labelled
    "A0 (J, δ = 0)", never "M19's A0"; ``--delta pooled|petra|machu_picchu`` is for scratch checks only. The delta
    tag names the output directory. ``a0_sampler`` is m19_arms.job's site branch (+ the pass-through ``RoundLog``):
    built on the M19 Petra site with --delta machu_picchu it reproduces the stored M19 A0 (petra, TNg_s0, seed 1) bit
    for bit, under the D4d round cap too. Per run: rounds used of the D4d cap (tc_global.round_cap), windows started,
    a run that hits the cap flagged (never dropped).
 2. ``grad_at`` (A5's dead-end eligibility) = the FABDEM 32 m gradient, nearest pixel (APPROVED; A5 and A5b use it too).
 3. The hybrid runs on the whole A0 state, and every log is attributed to the UNIT (CHANGED): an H1 connector by its
    dead end (its ordering-pass routing too), an H2 run by any piece midpoint inside the unit polygon (a city's core);
    the per-unit counts are primary and the domain-wide ones secondary. P1b's added km is counted inside the unit
    (``gross_unit_m`` on every snapshot). "Drivable" is claimed per unit only: no run kept in the unit and no steep
    run or forced stretch left in it.
 4. Snapshots (P1b, after every connector or switchback, snap_every = 1): stored as exact edge deltas of the typed
    graph json (``Compact``, a subclass of m20_hybrid.Hybrid that records but never decides; the full graphs would
    need GBs per whole-town run). ``snapshot_graphs`` rebuilds each one (node order canonical).
 5. Measured networks: the arm's class-tagged graph cropped by site_benchmark.crop (kind carried through the crop);
    vehicular edges are the network, stairs and stepped ramps its pedestrian layer (steps flag = stairs only); A0 has
    no pedestrian layer (hilltown: vehicular only).
 6. Open = M1 (density, coverage, with land_km2 / street_km), M2, M6 (m6_15 / 12 / 10, the D3a by-class shares, the
    pair count and outcomes), G1 (g1_15 / 12), the run logs and the maps (CHANGED: D4's list). Everything else is
    sealed: P2 and P2_TOPO, pedestrian reach, the mesh diagnostics, M3-M5 and M7, and the D4c lens share of pass-2
    pieces.
 7. The pilot for hold-H's N (CHANGED, ``pilot_n``) on m6_15, contrast H1 + H2 - A0: v = sigma2_b + sigma2_w / R
    (sigma2_b the between-settlement variance of the settlement-mean differences, sigma2_w the pooled run-to-run
    variance, R = 24), v+ its one-sided 80 % upper confidence limit (chi-square on the settlement count),
    N = (1.96 + 1.2816)^2 v+ / 0.008^2; the between-block variance (1-degree blocks of dev_units.json) and N from block
    means as the sensitivity; the larger N, and N >= 40. The contrast is a proxy for P1a (no H3 yet); P1b is
    unpiloted (no A5b arm).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import resource
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import m20_hybrid as hy
from urbanmorph.metrics import hilltown as H

BB, METHOD = "TNg_s0", "NA-SMC"
SEEDS = tuple(range(1, 25))                       # D4: 24 runs per settlement (the hold-H count)
MAX_WORKERS = 6
SCRATCH = ROOT / "data" / "processed" / "m20" / "scratch_unpinned"
OUT_DEV = ROOT / "outputs" / "m20" / "devlook"
UNITS = ROOT / "configs" / "m20_pool" / "dev_units.json"
JOINT_DIR = ROOT / "outputs" / "m18" / "joint_round"
POOLED_DELTA = ROOT / "configs" / "m18_terrain_potentials_TNg_whole.json"
DELTAS = ("none", "pooled", "petra", "machu_picchu")
ARMS = ("A0", "H1", "H12")
A0_LABEL = "A0 (J, δ = 0)"                         # D4d ruling 6, choice 1: not M19's A0 (no cross-fitted delta)
ARM_NAMES = {"A0": A0_LABEL, "H1": "H1 only", "H12": "H1 + H2"}
OPEN_OUTCOMES = ("m6_", "g1_")                     # access outcomes opened with M6 / G1 (pedestrian reach is sealed)
PILOT_R, PILOT_DELTA, PILOT_Z, PILOT_N_MIN = 24, 0.008, 1.96 + 1.2816, 40      # D4d ruling 7
TOWN_PARTS = tuple(k for k in hy.SAMPLER_KEYS if k != "stats")      # seed-free; "stats" is stored per run
SIZES = (2048, 4096, 8192)

# ---------------------------------------------------------------- the open / sealed split (P2 sealed, D4)
P2_METRICS = tuple(sorted({k.split("@")[0] for k in H.P2_CELLS + H.P2_CELLS_TOPO}))
SEALED_ALSO = tuple(sorted({k.split("@")[0] for g in ("pedestrian", "mesh") for k in H.GROUPS[g]}))   # D4d choice 6
SCRUB_TOKENS = P2_METRICS + SEALED_ALSO + ("lens", "deadend", "lcc", "phi", "block_contour", "tortuos")
OPEN_METRICS = frozenset({k.split("@")[0] for g in ("M1", "M2", "M6", "M6_sec", "G1") for k in H.GROUPS[g]}
                         | {"land_km2", "street_km", "pairs"} | {f"m6_{t:g}_steepcls" for t in H.M6_STEEP})
assert not OPEN_METRICS & set(P2_METRICS) and not any(t in m for m in OPEN_METRICS for t in SCRUB_TOKENS)
MAIN_KEYS = ("density@{c}", "coverage@{c}", "share_gt15@{c}", "grade_p90@{c}")
UNIT_KEYS = ("m6_15@unit", "m6_12@unit", "m6_10@unit", "g1_15@unit", "g1_12@unit")


def is_open(key: str) -> bool:
    return key.split("@")[0] in OPEN_METRICS and not any(t in key for t in SCRUB_TOKENS)


def open_split(m: dict) -> tuple:
    """(open, sealed) of a flat ``metric@cell`` dict: open = the declared open metrics only (choice 6)."""
    op, se = {}, {}
    for k, v in m.items():
        (op if is_open(k) else se)[k] = v
    return op, se


def scrub(x):
    """Drop every dict key naming a P2 / lens quantity, recursively (logs and reports before they are opened)."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if not any(t in str(k) for t in SCRUB_TOKENS)}
    if isinstance(x, (list, tuple)):
        return [scrub(v) for v in x]
    return x


def sealed_part(x):
    """The complement of ``scrub`` on a log dict (the keys it drops), recursively."""
    out = {}
    for k, v in x.items():
        if any(t in str(k) for t in SCRUB_TOKENS):
            out[k] = v
        elif isinstance(v, dict):
            s = sealed_part(v)
            if s:
                out[k] = s
    return out


# ---------------------------------------------------------------- small io helpers
def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_json(obj) -> str:
    return sha_bytes(json.dumps(obj, sort_keys=True).encode())


def _tmp(f: Path) -> Path:
    return f.with_name(f.name + ".part")


def put_bytes(f: Path, b: bytes):
    f.parent.mkdir(parents=True, exist_ok=True)
    _tmp(f).write_bytes(b)
    _tmp(f).replace(f)


def put_json(f: Path, obj):
    put_bytes(f, json.dumps(obj, indent=1, default=_default).encode())


def put_jgz(f: Path, obj):
    put_bytes(f, gzip.compress(json.dumps(obj, default=_default).encode(), compresslevel=6, mtime=0))


def get_jgz(f: Path):
    return json.loads(gzip.decompress(f.read_bytes()))


def put_pkl(f: Path, obj):
    put_bytes(f, zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), 6))


def get_pkl(f: Path):
    return pickle.loads(zlib.decompress(f.read_bytes()))


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def _num(d: dict) -> dict:
    return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in d.items()
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)}


def maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 1024


# ---------------------------------------------------------------- towns, paths, layout
def paths(split: str):
    from m20_towns import Paths
    return Paths(scratch=(split == "scratch"))


def town_ids(split: str, ids=()) -> list:
    """The towns of a split: scratch = the scratch site configs; dev = the frozen dev-H unit list (never a hold-H or
    study-H list). ``ids`` must be a subset."""
    if split == "scratch":
        own = sorted(f.stem for f in (SCRATCH / "sites").glob("*.json"))
    else:
        if not UNITS.exists():
            raise SystemExit(f"{UNITS} does not exist: the dev-H unit list is frozen first "
                             "(scripts/m20_gate.py townlist)")
        U = json.loads(UNITS.read_text())
        own = list(U["reference"]) + list(U["descriptive"])
    if ids:
        bad = [i for i in ids if i not in own]
        if bad:
            raise SystemExit(f"refused: {bad} not in the {split} towns")
        return [i for i in own if i in set(ids)]
    return own


class Layout:
    """Where one look writes (``base`` = the split's root or --out; one directory per A0 density setting)."""

    def __init__(self, split: str, delta: str = "none", base: Path | None = None):
        self.split, self.delta = split, delta
        root = Path(base) if base is not None else (SCRATCH / "devlook" if split == "scratch" else OUT_DEV)
        self.root = root / f"J_delta-{delta}"

    def town(self, sid):
        return self.root / sid

    def parts(self, sid):
        return self.town(sid) / "sampler_parts.pkl.gz", self.town(sid) / "sampler_parts.json"

    def a0(self, sid, seed):
        d = self.town(sid) / "a0"
        return d / f"s{seed:02d}.json.gz", d / f"s{seed:02d}.state.pkl.gz"

    def verify(self, sid, seed):
        return self.town(sid) / "verify" / f"s{seed:02d}.json"

    def hybrid(self, sid, seed):
        d = self.town(sid) / "hybrid"
        return d / f"s{seed:02d}.json.gz", d / f"s{seed:02d}.snaps.pkl.gz"

    def open_(self, sid, seed):
        return self.town(sid) / "open" / f"s{seed:02d}.json"

    def sealed(self, sid, seed):
        return self.root / "sealed" / sid / f"s{seed:02d}.json.gz"

    def sealed_log(self, sid, seed):
        return self.root / "sealed" / sid / f"s{seed:02d}.hybrid_log.json.gz"

    def real_open(self, sid):
        return self.town(sid) / "open" / "real.json"

    def real_sealed(self, sid):
        return self.root / "sealed" / sid / "real.json.gz"

    def manifest(self):
        return self.root / "sealed" / "MANIFEST.json"

    def bundle(self):
        return self.root / "sealed" / "sealed_bundle.json.gz"

    def report(self):
        return self.root / "report_open.json"

    def map(self, sid, seed):
        return self.root / "maps" / f"{sid}_s{seed:02d}.png"

    def timing(self):
        return self.root / "timing.jsonl"


def log_stage(lay: Layout, stage: str, wall: float, jobs: list, workers: int):
    """Append one stage record (wall time, summed job seconds) to timing.jsonl."""
    lay.root.mkdir(parents=True, exist_ok=True)
    rec = {"stage": stage, "split": lay.split, "wall_s": round(wall, 1), "workers": workers, "jobs": len(jobs),
           "job_s": round(sum(j.get("seconds", 0.0) for j in jobs), 1),
           "job_cpu_s": round(sum(j.get("cpu_s", 0.0) for j in jobs), 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(lay.timing(), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{stage}] {len(jobs)} jobs, wall {wall:.0f} s, job time {rec['job_s']:.0f} s", flush=True)


# ---------------------------------------------------------------- A0 (M19's whole-town A0, choice 1 for delta)
def a0_params(delta: str) -> dict:
    from m18_joint_round import joint_params
    over = dict(joint_params())
    if delta == "none":
        return over
    if delta == "pooled":
        if not POOLED_DELTA.exists():
            raise SystemExit(f"{POOLED_DELTA.relative_to(ROOT)} does not exist (m18_joint_round.py whole_pooled)")
        tm = json.loads(POOLED_DELTA.read_text())["params"]["tm_len"]
    else:
        tm = json.loads((JOINT_DIR / f"whole_fit_{delta}.json").read_text())["selected"]["tm_len"]
    over["tm_len"] = tuple(tm)                            # as m18_joint_round.cross_over
    return over


class RoundLog:
    """D4d logging, outside the sampler: counts the iterations of TCSampler.run's round loop that did work. An
    iteration calls do_starts (when a particle needs a start) and / or pops closed by one contexts call; A0 has no
    look-ahead (horizon 0), whose roll-outs would also call contexts. Pass-through: nothing the sampler does changes."""
    _rounds, _phase, _t0, PROGRESS = 0, "end", None, 1000

    def _tick(self):
        self._rounds += 1
        if self._t0 is None:
            self._t0 = time.time()
        if self._rounds % self.PROGRESS == 0:                # progress on stderr only
            print(f"  [A0] round {self._rounds}, {time.time() - self._t0:.0f} s", file=sys.stderr, flush=True)

    def do_starts(self, need, gen, rng):
        self._tick()
        self._phase = "starts"
        return super().do_starts(need, gen, rng)

    def pop(self, s, gen, conf):
        if self._phase == "end":
            self._tick()
        self._phase = "pops"
        return super().pop(s, gen, conf)

    def contexts(self, items):
        self._phase = "end"
        return super().contexts(items)


def a0_sampler(sid: str, P, delta: str, seed: int):
    """The M19 A0 sampler of m19_arms.job (site branch), on the town's TownSite domain, initialised for ``seed``; with
    the D4d round cap (tc_global.round_cap) and the pass-through RoundLog."""
    import torch
    from m16_kaist_all import AnchoredGlobalNet
    from m19_arms import Arm
    from m20_towns import TownSite
    from urbanmorph.experiments.backbones import load_backbone
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import net_preset
    from urbanmorph.graphs.tc_potentials import Potentials
    torch.set_num_threads(1)                              # as the stored whole-town runs
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    m = method(METHOD)
    model, names, norm, _ = load_backbone(BB, "cpu")
    over = a0_params(delta)
    cfg = net_preset(m.preset, **{**m.overrides, **over})
    site = TownSite(sid, P)
    big = site.domain()

    class S_(RoundLog, Arm, AnchoredGlobalNet):
        pass
    sm = S_(model, {**big, "site": f"{site.cfg['id']}_{METHOD}"}, site.L, names, norm, pots, cfg)
    sm.anchor = None
    sm._arm_init("A0", seed)
    sm.voc_log = None
    return sm, site, over


def parts_fingerprint(sm) -> str:
    """sha256 of what the step evaluator reads and that does not depend on the seed (choice: tfs arrays + reprs)."""
    h = hashlib.sha256()
    for tf in sm.tfs:
        for a in (tf.t.elev, getattr(tf.t, "water_kind", None), tf.gx, tf.gy, tf.hw):
            if a is not None:
                h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((sm.cfg, sm.caps, sm.cds_topo)).encode())
    h.update((ROOT / "configs" / "m13_potentials" / "TN.json").read_bytes())
    return h.hexdigest()


def a0_job(spec) -> dict:
    """One A0 run (or, with ``replay``, its bit-identity replay)."""
    from m20_towns import TownSite  # noqa: F401  (import check before the long run)
    from urbanmorph.graphs.tc_global import round_cap
    import site_benchmark as sb
    from urbanmorph.experiments.graph_io import graph_to_json
    split, sid, seed, delta, base, replay = spec
    lay = Layout(split, delta, base)
    P = paths(split)
    t0, c0 = time.time(), time.process_time()
    sm, site, over = a0_sampler(sid, P, delta, seed)
    t1 = time.time()
    gs = sm.run(seed)
    t_run = time.time() - t1
    gj = graph_to_json(gs[0])
    sha = sha_json(gj)
    if replay:
        f, _ = lay.a0(sid, seed)
        out = {"town": sid, "seed": seed, "replay_sha256": sha, "seconds": time.time() - t0,
               "cpu_s": time.process_time() - c0, "run_s": t_run}
        if f.exists():
            stored = get_jgz(f)
            out.update(stored_sha256=stored["graph_sha256"], identical=bool(stored["graph"] == gj))
        put_json(lay.verify(sid, seed), out)
        return out
    S = sm.states[0]
    p = next((k for k, x in enumerate(S) if np.array_equal(x.graph(with_class=True).nodes, gs[0].nodes)
              and np.array_equal(x.graph(with_class=True).edges, gs[0].edges)), None)
    if p is None:
        raise RuntimeError(f"{sid} s{seed}: no particle holds the returned A0 graph")
    st = S[p]
    fp = parts_fingerprint(sm)
    fpk, fpj = lay.parts(sid)
    if fpj.exists():
        if json.loads(fpj.read_text())["fingerprint"] != fp:
            raise RuntimeError(f"{sid}: the sampler parts differ between seeds")
    else:
        put_pkl(fpk, {k: getattr(sm, k) for k in TOWN_PARTS})
        put_json(fpj, {"fingerprint": fp, "keys": list(TOWN_PARTS)})
    mask = np.load(P.towns / sid / "city_mask.npz")["mask"]
    km = lambda g: float(np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T).sum() / 1000) if len(g.edges) else 0.0
    max_rounds = round_cap(len(sm.windows))                      # D4d: what GlobalTCSampler.run used
    rec = {"town": sid, "split": split, "seed": seed, "delta": delta, "backbone": BB, "method": METHOD,
           "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in over.items()}, "L": site.L,
           "site_key": f"{site.cfg['id']}_{METHOD}", "graph": gj, "graph_sha256": sha, "chosen_particle": p,
           "particles": len(S), "parts_fingerprint": fp,
           "seconds": time.time() - t0, "run_s": t_run, "setup_s": t1 - t0, "cpu_s": time.process_time() - c0,
           "forward_passes": int(sm.n_fwd), "stats": [_num(dict(x)) for x in sm.stats], "mech": dict(sm.mech),
           "max_rounds": max_rounds, "rounds_used": int(sm._rounds), "round_cap_hit": bool(not all(x.done for x in S)),
           "windows": len(sm.windows), "windows_started": int((st.win_ncomp > 0).sum()),
           "windows_started_ids": np.flatnonzero(st.win_ncomp > 0).tolist(),
           "windows_ended": int(st.win_done.sum()), "starts": int(st.ncomp),
           "node_budget_hit": bool(st.n >= st.max_nodes), "nodes": int(st.n), "edges": len(gj["edges"]),
           "km_domain": km(gs[0]), "km_limits": km(sb.crop(gs[0], mask, site)), "maxrss_mb": maxrss_mb(),
           "torch_threads": 1}
    fj, fs = lay.a0(sid, seed)
    put_pkl(fs, {"state": st, "stats": sm.stats, "graph_sha256": sha})
    put_jgz(fj, rec)
    return {k: rec[k] for k in ("town", "seed", "seconds", "cpu_s", "run_s", "forward_passes", "km_domain",
                                "km_limits", "rounds_used", "max_rounds", "round_cap_hit", "windows_started", "windows",
                                "node_budget_hit", "maxrss_mb")}


# ---------------------------------------------------------------- the hybrid with compact snapshots (choice 4)
def edge_keys(gj: dict) -> Counter:
    """Multiset of typed edges by their (rounded, as graph_to_json) end coordinates and attributes."""
    N = gj["nodes"]
    kind = gj.get("kind", [0] * len(gj["edges"]))
    out = Counter()
    for (a, b), c, s, k in zip(gj["edges"], gj["cls"], gj["st"], kind):
        pa, pb = tuple(N[a]), tuple(N[b])
        out[(*min(pa, pb), *max(pa, pb), int(c), int(s), int(k))] += 1
    return out


def keys_graph(keys: Counter) -> dict:
    """A typed graph json from an edge multiset (nodes in sorted coordinate order)."""
    E = sorted(keys.elements())
    pts = sorted({(e[0], e[1]) for e in E} | {(e[2], e[3]) for e in E})
    ix = {p: i for i, p in enumerate(pts)}
    return {"nodes": [list(p) for p in pts], "edges": [[ix[(e[0], e[1])], ix[(e[2], e[3])]] for e in E],
            "cls": [e[4] for e in E], "st": [e[5] for e in E], "kind": [e[6] for e in E]}


NON_ADDITIVE = ("_min_m", "_max_m", "_share")
STAGE_TOTALS = frozenset({"runs_initial", "short_ramps_initial", "forced_initial", "short_ramps_final",
                          "short_ramps_final_m", "drivable_claim", "iteration_cap"})     # whole-network, not per event


def _additive(k, v) -> bool:
    return (isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            and not str(k).endswith(NON_ADDITIVE) and k not in STAGE_TOTALS)


class Compact(hy.Hybrid):
    """m20_hybrid.Hybrid with two side records and no change to its decisions:
    * snapshots after the first are edge deltas (add / rem multisets) of the typed graph json: the same snapshot
      points, stages and gross metres (m20_hybrid choice 19), a fraction of the memory; each also carries
      ``gross_unit_m``, the committed length inside the unit (P1b's added km counted inside the unit, D4d ruling 6);
    * every H1 connector (at its dead end; its ordering-pass routing too) and every H2 run (any piece midpoint) is
      attributed to the unit or not: ``events`` = [{stage, kind, unit, delta of the additive log counters}] (D4d
      ruling 6, choice 3); with a unit holding the whole domain the per-unit log equals the domain log."""

    def __init__(self, *a, unit=None, **k):
        super().__init__(*a, **k)
        self._prev = edge_keys(self.snapshots[0]["graph"])
        self.unit = unit
        if unit is not None:
            import shapely
            shapely.prepare(unit)
        self.gross_unit, self.events, self._mark, self._in_plan = 0.0, [], None, False
        self.snapshots[0]["gross_unit_m"] = 0.0

    # -- unit attribution
    def _inside(self, P) -> bool:
        import shapely
        P = np.asarray(P, float).reshape(-1, 2)
        return True if self.unit is None else bool(shapely.contains_xy(self.unit, P[:, 0], P[:, 1]).any())

    def _unit_len(self, segs) -> float:
        import shapely
        if not segs:
            return 0.0
        L = shapely.linestrings(np.stack([np.stack([np.asarray(a, float), np.asarray(b, float)]) for a, b in segs]))
        return float(shapely.length(L).sum() if self.unit is None else shapely.length(shapely.intersection(L, self.unit)).sum())

    def _flush(self):
        if self._mark is None:
            return
        stage, kind, inside, before = self._mark
        now = {k: v for k, v in self.log[stage].items() if _additive(k, v)}
        delta = {k: v - before.get(k, 0) for k, v in now.items() if v != before.get(k, 0)}
        self.events.append({"stage": stage, "kind": kind, "unit": inside, "delta": delta})
        self._mark = None

    def _open(self, stage, kind, inside):
        self._flush()
        self._mark = (stage, kind, bool(inside), {k: v for k, v in self.log[stage].items() if _additive(k, v)})

    def two_pass(self, a, b, log, *args, **kw):
        if log is self.log["H1"] and not self._in_plan:               # H1's ordering pass, per candidate dead end
            self._open("H1", "order", self._inside(a))
        return super().two_pass(a, b, log, *args, **kw)

    def plan(self, st, a, b, detour_max, log, *args, **kw):
        if log is self.log["H1"]:                                     # one H1 connector, from its dead end a
            self._open("H1", "decision", self._inside(a))
        self._in_plan = True
        try:
            return super().plan(st, a, b, detour_max, log, *args, **kw)
        finally:
            self._in_plan = False

    def placement(self, st, r):                                       # one H2 run (forced or not)
        self._open("H2", "decision", self._inside(r.mids))
        return super().placement(st, r)

    def h1(self):
        super().h1()
        self._flush()

    def h2(self):
        super().h2()
        self._flush()

    # -- snapshots
    def snapshot(self, stage, force=False):
        if not force:                                                 # one connector or switchback was committed
            self.gross_unit += self._unit_len(self._segs)
        self.n_snap += 0 if force else 1
        if force or self.n_snap % self.snap_every == 0:
            cur = edge_keys(self.typed_json())
            self.snapshots.append({"stage": stage, "gross_m": self.gross, "gross_unit_m": self.gross_unit,
                                   "add": sorted((cur - self._prev).elements()),
                                   "rem": sorted((self._prev - cur).elements())})
            self._prev = cur

    def unit_log(self) -> dict:
        """The per-unit log (primary, D4d): the summed deltas of the events inside the unit, per stage."""
        res = {}
        for stg in ("H1", "H2"):
            c = Counter()
            for e in self.events:
                if e["unit"] and e["stage"] == stg:
                    c.update(e["delta"])
            d = {k: float(v) for k, v in c.items()}
            n12 = d.get("pass1", 0) + d.get("pass2", 0)
            if n12:
                d["pass2_share"] = d.get("pass2", 0) / n12
            d["events"] = sum(1 for e in self.events if e["unit"] and e["stage"] == stg and e["kind"] == "decision")
            res[stg] = d
        return res


def run_compact(s, T, grad_at, evaluate, P: hy.Params = hy.Params(), snap_every: int = 1, h1: bool = True,
                h2: bool = True, unit=None) -> dict:
    """m20_hybrid.run_hybrid, step for step, with ``Compact`` (its output equals run_hybrid's but for the snapshot
    encoding and the unit records; tests/test_m20_devlook.py checks this). ``unit``: the unit polygon (domain m)."""
    hb = Compact(s, T, grad_at, evaluate, P, snap_every, unit=unit)
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
            "state": hb.st, "log_unit": hb.unit_log(), "gross_unit_m": hb.gross_unit, "events": hb.events}


def snapshot_graphs(snaps: list):
    """Yield (stage, gross m, typed graph json) for every stored snapshot (the first one as it was stored)."""
    cur = None
    for sn in snaps:
        if "graph" in sn:
            cur = edge_keys(sn["graph"])
            yield sn["stage"], sn["gross_m"], sn["graph"]
            continue
        cur = cur - Counter(map(tuple, sn["rem"])) + Counter(map(tuple, sn["add"]))
        yield sn["stage"], sn["gross_m"], keys_graph(cur)


def fab_grad(T):
    """grad_at on the FABDEM 32 m grid (choice 2): nearest pixel of Terrain.gradient, NaN as 0 (as TerrainField)."""
    gx, gy = T.gradient()
    gx, gy = np.nan_to_num(gx), np.nan_to_num(gy)

    def f(x, y):
        r, c = T.rc(np.atleast_1d(x), np.atleast_1d(y))
        r = np.clip(np.round(r).astype(int), 0, gx.shape[0] - 1)
        c = np.clip(np.round(c).astype(int), 0, gx.shape[1] - 1)
        return gx[r, c], gy[r, c]
    return f


def steep_left(state, T, unit=None, P: hy.Params = hy.Params()) -> dict:
    """The unit's own reading (D4d ruling 6, choice 3): steep runs (not short ramps) and forced stretches left on the
    vehicular network with a piece midpoint inside the unit polygon (domain-wide counts are secondary)."""
    import shapely
    an = hy.Analysis(state, hy.Geo(T, P).F, P)
    runs = [r for r in an.runs if not an.short_ramp(r)]
    ins = (lambda r: True) if unit is None else (lambda r: bool(shapely.contains_xy(unit, r.mids[:, 0], r.mids[:, 1]).any()))
    inside = [r for r in runs if ins(r)]
    forced = [r for r in an.forced if ins(r)]
    return {"steep_runs_domain": len(runs), "steep_km_domain": sum(r.length for r in runs) / 1000,
            "steep_runs_unit": len(inside), "steep_km_unit": sum(r.length for r in inside) / 1000,
            "forced_left_unit": len(forced), "drivable_unit": int(not inside and not forced)}


def typed_json_of(g, kind) -> dict:
    from urbanmorph.experiments.graph_io import graph_to_json
    return {**graph_to_json(g), "kind": np.asarray(kind, int).tolist()}


def hybrid_job(spec) -> dict:
    from m20_towns import load_town, terrain_of
    from urbanmorph.experiments.graph_io import graph_to_json
    split, sid, seed, delta, base = spec
    lay = Layout(split, delta, base)
    P = paths(split)
    t0, c0 = time.time(), time.process_time()
    fj, fs = lay.a0(sid, seed)
    rec, stt = get_jgz(fj), get_pkl(fs)
    s = stt["state"]
    if graph_to_json(s.graph(with_class=True)) != rec["graph"] or stt["graph_sha256"] != rec["graph_sha256"]:
        raise RuntimeError(f"{sid} s{seed}: the stored state is not the A0 graph")    # m20_hybrid.replay_a0's rule
    parts = get_pkl(lay.parts(sid)[0])
    sm = hy.sampler_proxy({**parts, "stats": stt["stats"]})
    real = load_town(sid, P)
    T = terrain_of(real["terrain32"])
    unit = real["polygon"]                                    # the unit polygon (a city's core, D4d), domain metres
    grad_at = fab_grad(T)
    out = {"town": sid, "seed": seed, "delta": delta, "a0_sha256": rec["graph_sha256"], "arms": {},
           "unit_rule": real.get("unit_rule", "settlement polygon"), "a0_steep_left": steep_left(hy.State(s), T, unit)}
    snaps, sealed_logs = {}, {}
    for arm, h2 in (("H1", False), ("H12", True)):
        ta, ca = time.time(), time.process_time()
        res = run_compact(s, T, grad_at, hy.sampler_evaluator(sm, 0), hy.Params(), snap_every=1, h2=h2, unit=unit)
        after = sha_json(graph_to_json(s.graph(with_class=True)))
        if after != rec["graph_sha256"]:
            raise RuntimeError(f"{sid} s{seed}: the A0 state changed under {arm}")      # pairing (D4)
        sealed_logs[arm] = sealed_part(res["log"])                # the D4c lens share of pass-2 pieces (choice 6)
        a = {"graph": typed_json_of(res["graph"], res["kind"]), "km": res["km"], "gross_m": res["gross_m"],
             "gross_unit_m": res["gross_unit_m"], "log_unit": scrub(res["log_unit"]), "log": scrub(res["log"]),
             "events": scrub(res["events"]), "input_sha256": after, "snapshots": len(res["snapshots"]),
             "seconds": time.time() - ta, "cpu_s": time.process_time() - ca}
        if h2:                                                # "drivable" per unit only (D4d): nothing kept or left
            a["steep_left"] = steep_left(res["state"], T, unit)
            a["drivable_unit"] = int(a["steep_left"]["drivable_unit"] and not res["log_unit"]["H2"].get("runs_kept", 0))
        out["arms"][arm] = a
        snaps[arm] = res["snapshots"]
    out.update(seconds=time.time() - t0, cpu_s=time.process_time() - c0, maxrss_mb=maxrss_mb())
    fh, fsn = lay.hybrid(sid, seed)
    put_pkl(fsn, snaps)
    put_jgz(lay.sealed_log(sid, seed), {"town": sid, "seed": seed, "logs": sealed_logs})
    put_jgz(fh, out)
    return {"town": sid, "seed": seed, "seconds": out["seconds"], "cpu_s": out["cpu_s"],
            **{f"{k}_s": round(v["seconds"]) for k, v in out["arms"].items()}}


# ---------------------------------------------------------------- measure (choice 5)
def split_kinds(gj: dict, site, mask):
    """(vehicular RoadGraph, pedestrian layer or None, km by kind) of a typed graph json cropped as site_benchmark."""
    import site_benchmark as sb
    from urbanmorph.graphs.road_graph import RoadGraph
    nodes = np.array(gj["nodes"], float).reshape(-1, 2)
    E = np.array(gj["edges"], int).reshape(-1, 2)
    kind = np.array(gj.get("kind", [0] * len(E)), int)
    cls, st = np.array(gj["cls"], int), np.array(gj["st"], int)
    g = RoadGraph(nodes, E, np.zeros(len(nodes), bool), edge_class=cls + 100 * kind, edge_struct=st)
    c = sb.crop(g, mask, site)
    ck, cc, cs = np.asarray(c.classes()) // 100, np.asarray(c.classes()) % 100, np.asarray(c.structs())
    L = np.hypot(*(c.nodes[c.edges[:, 0]] - c.nodes[c.edges[:, 1]]).T) if len(c.edges) else np.zeros(0)
    km = {n_: float(L[ck == i].sum() / 1000) for i, n_ in enumerate(hy.KIND_NAMES)}
    v = ck == 0
    used = np.unique(c.edges[v].ravel()) if v.any() else np.zeros(0, int)
    remap = -np.ones(len(c.nodes), int)
    remap[used] = np.arange(len(used))
    veh = RoadGraph(c.nodes[used], remap[c.edges[v]].reshape(-1, 2), np.zeros(len(used), bool),
                    edge_class=cc[v], edge_struct=cs[v])
    pe = ~v
    ped = None
    if pe.any():
        ped = ([c.nodes[e] for e in c.edges[pe]], [bool(k == 2) for k in ck[pe]])
    return veh, ped, km


def real_row(sid: str, lay: Layout, P) -> dict:
    """The real town's unit metrics (as m20_gate.town_job): sealed in full, open part with definedness; cached."""
    from m20_towns import hill_field, load_town
    if lay.real_sealed(sid).exists():
        return get_jgz(lay.real_sealed(sid))["metrics"]
    real = load_town(sid, P)
    F = hill_field(real)
    t0 = time.time()
    m = H.unit_metrics(real["vehicular"]["unit_native"], F,
                       ped=(real["pedestrian"]["unit_lines"], real["pedestrian"]["unit_steps"]), access=True)
    store_real(lay, sid, m, time.time() - t0)
    return {k: v for k, v in m.items() if k != "_outcomes"}


def store_real(lay: Layout, sid: str, m: dict, seconds: float):
    """The real row: sealed in full; its open keys, outcomes and definedness in the open."""
    m = dict(m)
    outc = m.pop("_outcomes", {})
    op, _ = open_split(m)
    put_jgz(lay.real_sealed(sid), {"town": sid, "metrics": m, "outcomes": outc})
    put_json(lay.real_open(sid), scrub({"town": sid, "seconds": seconds, "metrics": op,
                                        "outcomes": {k: v for k, v in outc.items() if k.startswith(OPEN_OUTCOMES)},
                                        "defined": {k: defined(m, k) for k in op}}))


def defined(real: dict, key: str):
    """hilltown.defined on the real town for scored metrics; None (descriptive) for counts and by-class shares."""
    return bool(H.defined(real, key)) if key.split("@")[0] in H.METRICS else None


def window_grid(L: float) -> list:
    """GlobalTCSampler's start windows (x0, y0), in its order (tc_global: row-major from the SW corner, 512 m)."""
    n = int(L // 512)
    return [(c * 512.0, r * 512.0) for r in range(n) for c in range(n)]


def a0_coverage(started, L: float, F, g_a0, near_m: float = 60.0) -> dict:
    """D4e ruling 6 logs (no output changes): the ids of the windows started, the share of unit land (F.land cell
    centres) inside them, and the share of the unit's M6 pairs (hilltown.od_pairs) with an endpoint more than 60 m
    from the A0 network as M6 measures it (the cropped vehicular A0 graph)."""
    import shapely
    started = sorted(int(i) for i in started)
    P = F.centres(F.land)
    n = int(L // 512)
    if len(P):
        c = np.clip((P[:, 0] // 512).astype(int), 0, n - 1)
        r = np.clip((P[:, 1] // 512).astype(int), 0, n - 1)
        land_share = float(np.isin(r * n + c, started).mean())
    else:
        land_share = float("nan")
    Q, I, J = H.od_pairs(F)
    far_share = float("nan")
    if len(I):
        if len(g_a0.edges):
            tree = shapely.STRtree(shapely.linestrings(g_a0.nodes[g_a0.edges]))
            _, d = tree.query_nearest(shapely.points(Q), return_distance=True, all_matches=False)
            far = d > near_m
        else:
            far = np.ones(len(Q), bool)
        far_share = float((far[I] | far[J]).mean())
    return {"windows_started_ids": started, "windows": n * n, "unit_land_share_in_started_windows": land_share,
            "pairs": int(len(I)), "pairs_endpoint_far_share": far_share, "near_m": near_m}


def measure_job(spec) -> dict:
    from m20_towns import TownSite, hill_field, load_town
    split, sid, seed, delta, base = spec
    lay = Layout(split, delta, base)
    P = paths(split)
    t0, c0 = time.time(), time.process_time()
    site = TownSite(sid, P)
    mask = np.load(P.towns / sid / "city_mask.npz")["mask"]
    real = load_town(sid, P)
    F = hill_field(real)
    rrow = real_row(sid, lay, P)
    a0 = get_jgz(lay.a0(sid, seed)[0])
    hb = get_jgz(lay.hybrid(sid, seed)[0])
    if hb["a0_sha256"] != a0["graph_sha256"]:
        raise RuntimeError(f"{sid} s{seed}: the hybrid was not run on this A0")
    graphs = {"A0": a0["graph"], "H1": hb["arms"]["H1"]["graph"], "H12": hb["arms"]["H12"]["graph"]}
    measured = {}
    for arm, gj in graphs.items():
        ta = time.time()
        veh, ped, km = split_kinds(gj, site, mask)
        measured[arm] = (H.unit_metrics(veh, F, ped=ped, access=True), km, time.time() - ta)
        if arm == "A0":                                         # D4e ruling 6: coverage logs of the capped A0
            st = get_pkl(lay.a0(sid, seed)[1])["state"]
            cov = a0_coverage(np.flatnonzero(st.win_ncomp > 0), site.L, F, veh)
    op_rec, se_rec = assemble(sid, seed, delta, a0, hb, measured, rrow)
    op_rec["a0_coverage"] = cov
    op_rec["seconds"], op_rec["cpu_s"] = time.time() - t0, time.process_time() - c0
    put_jgz(lay.sealed(sid, seed), se_rec)
    put_json(lay.open_(sid, seed), op_rec)
    return {"town": sid, "seed": seed, "seconds": op_rec["seconds"], "cpu_s": op_rec["cpu_s"]}


def assemble(sid, seed, delta, a0: dict, hb: dict, measured: dict, rrow: dict) -> tuple:
    """(open record, sealed record) of one run: ``measured`` = {arm: (unit_metrics dict, km by kind, seconds)}; the
    open record carries the open keys only (M1, M2, M6, G1; D4d) and the run logs, and is scrubbed; the sealed one
    carries everything else (P2, pedestrian reach, the mesh diagnostics, the pass-2 lens logs, ...)."""
    op_all, se_all, se_out = {}, {}, {}
    for arm, (m, km, sec) in measured.items():
        m = dict(m)
        outc = m.pop("_outcomes", {})
        op, se = open_split(m)
        op_all[arm] = {"metrics": op, "outcomes": {k: v for k, v in outc.items() if k.startswith(OPEN_OUTCOMES)},
                       "km_limits": km, "seconds": sec}
        se_all[arm] = se
        se_out[arm] = {k: v for k, v in outc.items() if not k.startswith(OPEN_OUTCOMES)}
    H_ = ("H1", "H12")
    logs = {arm: hb["arms"][arm]["log"] for arm in H_}
    op_rec = {"town": sid, "seed": seed, "delta": delta, "a0_label": A0_LABEL, "a0_sha256": a0["graph_sha256"],
              "unit_rule": hb.get("unit_rule", "settlement polygon"), "arms": op_all,
              "defined": {k: defined(rrow, k) for k in op_all["A0"]["metrics"]},
              "logs_unit": {arm: hb["arms"][arm].get("log_unit", {}) for arm in H_},          # primary (D4d)
              "logs_domain": logs,                                                             # secondary
              "steep_left": {"A0": hb["a0_steep_left"], "H12": hb["arms"]["H12"]["steep_left"]},
              "drivable_unit": int(hb["arms"]["H12"].get("drivable_unit", 0)),
              "km": {arm: hb["arms"][arm]["km"] for arm in H_},
              "gross_m": {arm: hb["arms"][arm]["gross_m"] for arm in H_},
              "gross_unit_m": {arm: hb["arms"][arm].get("gross_unit_m", float("nan")) for arm in H_},
              "a0": {k: a0.get(k) for k in ("seconds", "run_s", "forward_passes", "km_domain", "km_limits", "round_cap_hit",
                                            "rounds_used", "max_rounds", "windows", "windows_started", "node_budget_hit",
                                            "nodes", "edges", "L", "chosen_particle", "maxrss_mb")},
              "hybrid_s": {arm: hb["arms"][arm]["seconds"] for arm in H_}}
    se_rec = {"town": sid, "seed": seed, "arms": se_all, "outcomes": se_out, "logs": sealed_part(logs)}
    return scrub(op_rec), se_rec


# ---------------------------------------------------------------- report (open summary, sealed bundle, timing)
def q(v, p):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.percentile(v, p)) if len(v) else float("nan")


def summary_of(vals: list) -> dict:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return {"median": q(v, 50), "q25": q(v, 25), "q75": q(v, 75), "n": len(v), "n_nan": len(vals) - len(v)}


def var1(v) -> float:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.var(v, ddof=1)) if len(v) >= 2 else float("nan")


LOG_KEYS = {"H1": ("events", "connectors_complete", "connectors_partial", "rejected_room", "rejected_smoothing",
                   "smoothing_fail", "no_new_pieces", "pass1", "pass2", "pass2_share", "added_m", "router_calls",
                   "router_s"),
            "H2": ("events", "forced_stairs", "forced_stairs_m", "forced_cutoff_pieces", "forced_cutoff_m", "runs_stairs",
                   "runs_stairs_m", "runs_switchback", "runs_switchback_m", "switchback_added_m", "runs_kept", "kept_m",
                   "kept_failed_for_room", "kept_smoothing_fail", "kept_commit_failure", "smoothing_fail",
                   "placement_failed", "pass1", "pass2", "pass2_share", "stubs_absorbed_m", "router_s")}
DOMAIN_ONLY = ("runs_initial", "short_ramps_initial", "forced_initial", "short_ramps_final", "iteration_cap", "pad_max_m")


def _log_summary(recs, key: str, arm: str, stg: str, names) -> dict:
    out = {}
    for k in names:
        vals = [r[key][arm].get(stg, {}).get(k, 0) for r in recs]
        out[f"{arm}.{stg}.{k}"] = {**summary_of(vals), "sum": float(np.nansum(vals))}
    return out


def town_report(lay: Layout, sid: str, seeds) -> dict | None:
    recs = [json.loads(lay.open_(sid, s).read_text()) for s in seeds if lay.open_(sid, s).exists()]
    if not recs:
        return None
    rr = json.loads(lay.real_open(sid).read_text())
    keys = sorted(recs[0]["arms"]["A0"]["metrics"])
    dfd = recs[0]["defined"]
    out = {"runs": [r["seed"] for r in recs], "n_runs": len(recs), "unit_rule": recs[0].get("unit_rule"),
           "real": rr["metrics"], "defined": dfd, "undefined_on_real": sorted(k for k in keys if dfd.get(k) is False),
           "arms": {}}
    for arm in ARMS:
        ms = {k: summary_of([r["arms"][arm]["metrics"].get(k) for r in recs]) for k in keys if dfd.get(k) is not False}
        kms = {k: summary_of([r["arms"][arm]["km_limits"][k] for r in recs]) for k in hy.KIND_NAMES}
        out["arms"][arm] = {"metrics": ms, "km_limits": kms}
    m6 = {arm: [r["arms"][arm]["metrics"].get("m6_15@unit") for r in recs] for arm in ARMS}
    diff = lambda a: [x - y if x is not None and y is not None else None for x, y in zip(m6[a], m6["A0"])]
    d12 = [x for x in diff("H12") if x is not None and np.isfinite(x)]
    out["m6_15_pilot"] = {**{f"var_{a}": var1(m6[a]) for a in ARMS},
                          "var_H12_minus_A0": var1(diff("H12")), "var_H1_minus_A0": var1(diff("H1")),
                          "mean_H12_minus_A0": float(np.mean(d12)) if d12 else float("nan"),
                          "diffs_H12_minus_A0": d12, "n": len(recs)}
    # D4d ruling 6, choice 3: per-unit logs primary, domain-wide secondary; "drivable" per unit only
    unit, dom = {}, {}
    for arm, stg in (("H1", ("H1",)), ("H12", ("H1", "H2"))):
        for st in stg:
            unit.update(_log_summary(recs, "logs_unit", arm, st, LOG_KEYS[st]))
            dom.update(_log_summary(recs, "logs_domain", arm, st, [k for k in LOG_KEYS[st] if k != "events"]
                                    + (list(DOMAIN_ONLY) if st == "H2" else [])))
    for k in ("steep_runs_unit", "steep_km_unit", "forced_left_unit"):
        for arm in ("A0", "H12"):
            unit[f"{arm}.left.{k}"] = summary_of([r["steep_left"][arm][k] for r in recs])
    for k in ("steep_runs_domain", "steep_km_domain"):
        for arm in ("A0", "H12"):
            dom[f"{arm}.left.{k}"] = summary_of([r["steep_left"][arm][k] for r in recs])
    for arm in ("H1", "H12"):
        unit[f"{arm}.gross_unit_km"] = summary_of([r["gross_unit_m"][arm] / 1000 for r in recs])   # P1b, in the unit
        dom[f"{arm}.gross_km"] = summary_of([r["gross_m"][arm] / 1000 for r in recs])
        for k in hy.KIND_NAMES:
            dom[f"{arm}.km_domain.{k}"] = summary_of([r["km"][arm][k] for r in recs])
    out["logs_unit"], out["logs_domain"] = unit, dom
    out["drivable_unit"] = {"runs_claimed": sum(int(r["drivable_unit"]) for r in recs), "n_runs": len(recs)}
    out["a0"] = {k: summary_of([float(r["a0"][k]) for r in recs if r["a0"].get(k) is not None])
                 for k in ("seconds", "run_s", "forward_passes", "km_domain", "km_limits", "maxrss_mb", "rounds_used",
                           "windows_started")}
    out["a0_coverage"] = {k: summary_of([r.get("a0_coverage", {}).get(k) for r in recs])
                          for k in ("unit_land_share_in_started_windows", "pairs_endpoint_far_share")}
    out["a0"]["round_cap_hit"] = sum(int(bool(r["a0"]["round_cap_hit"])) for r in recs)
    out["a0"]["node_budget_hit"] = sum(int(bool(r["a0"]["node_budget_hit"])) for r in recs)
    for k in ("L", "max_rounds", "windows"):
        out["a0"][k] = recs[0]["a0"].get(k)
    return out


def pilot_n(diffs: dict, blocks: dict | None = None, R: int = PILOT_R, delta: float = PILOT_DELTA) -> dict:
    """D4d ruling 7 on the m6_15 paired differences H1 + H2 - A0 ({settlement: [per-run d]}):
    v = sigma2_b + sigma2_w / R (sigma2_b: the variance of the settlement-mean differences; sigma2_w: the pooled
    run-to-run variance), v+ = its one-sided 80 % upper confidence limit (chi-square, settlement count - 1 df),
    N = (1.96 + 1.2816)^2 v+ / delta^2; the same from 1-degree block means (sensitivity); the larger N, and N >= 40."""
    from scipy.stats import chi2
    D = {t: np.asarray([x for x in d if x is not None and np.isfinite(x)], float) for t, d in diffs.items()}
    D = {t: d for t, d in D.items() if len(d)}
    df_w = sum(len(d) - 1 for d in D.values())
    s2w = float(sum(((d - d.mean()) ** 2).sum() for d in D.values()) / df_w) if df_w > 0 else float("nan")
    means = {t: float(d.mean()) for t, d in D.items()}

    def n_of(vals):
        k = len(vals)
        if k < 2 or not np.isfinite(s2w):
            return {"n": k, "var": float("nan"), "v": float("nan"), "v_ucl80": float("nan"), "N": float("nan")}
        s2b = float(np.var(list(vals), ddof=1))
        v = s2b + s2w / R
        vp = (k - 1) * v / float(chi2.ppf(0.20, k - 1))
        return {"n": k, "var": s2b, "v": v, "v_ucl80": vp, "N": PILOT_Z ** 2 * vp / delta ** 2}
    st = n_of(means.values())
    bk = {}
    for t, m in means.items():
        bk.setdefault((blocks or {}).get(t) or f"?{t}", []).append(m)
    bl = n_of([float(np.mean(v)) for v in bk.values()])
    Ns = [x for x in (st["N"], bl["N"]) if np.isfinite(x)]
    return {"metric": "m6_15@unit", "contrast": "H1 + H2 - A0 (paired on A0's sample path)", "R": R, "delta": delta,
            "z_sum": PILOT_Z, "sigma2_w": s2w, "settlements": {**st, "sigma2_b": st["var"]},
            "blocks": {**bl, "sigma2_block": bl["var"], "ids": {b: len(v) for b, v in bk.items()}},
            "N": int(max(math.ceil(max(Ns)), PILOT_N_MIN)) if Ns else None, "N_min": PILOT_N_MIN,
            "statement": "The look's contrast H1 + H2 - A0 is a proxy for P1a: H3 is designed only after the look (D4b). "
                         "The look has no A5b arm, so P1b goes unpiloted.",
            "rule": "D4d ruling 7: v = sigma2_b + sigma2_w / R, v+ = one-sided 80 % UCL (chi-square), "
                    "N = (1.96 + 1.2816)^2 v+ / 0.008^2, also from block means; the larger N, N >= 40"}


def seal_bundle(lay: Layout, towns, seeds) -> dict:
    """One gzip JSON of every sealed record, its sha256 and each record's sha256 in the manifest (no key names)."""
    files = [lay.real_sealed(t) for t in towns] + [f for t in towns for s in seeds
                                                    for f in (lay.sealed(t, s), lay.sealed_log(t, s))]
    files = [f for f in files if f.exists()]
    bundle = {str(f.relative_to(lay.root)): get_jgz(f) for f in files}
    put_jgz(lay.bundle(), bundle)
    man = {"sealed": True, "rule": "D4: P2 cells (hilltown.P2_CELLS, P2_CELLS_TOPO) and every other non-open quantity "
                                   "of the dev-H look, unread until the hybrid is frozen",
           "bundle": str(lay.bundle().relative_to(ROOT)) if lay.bundle().is_relative_to(ROOT) else str(lay.bundle()),
           "bundle_sha256": sha_bytes(lay.bundle().read_bytes()), "records": len(files),
           "files": {str(f.relative_to(lay.root)): sha_bytes(f.read_bytes()) for f in files},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    put_json(lay.manifest(), man)
    return man


def projection(lay: Layout, towns, seeds, n_towns: int, runs: int = len(SEEDS), mix: dict | None = None) -> dict:
    """CPU-hours per run from the measured stages (1 thread each), scaled to L = 2048 / 4096 / 8192 m by L^k per stage:
    k fitted (log-log least squares) when the measured runs span several L, else k = 2 (area)."""
    per = {"generate": [], "hybrid": [], "measure": []}
    for t in towns:
        for s in seeds:
            f = lay.a0(t, s)[0]
            if not f.exists():
                continue
            r = get_jgz(f)
            L = r.get("L")
            per["generate"].append((L, r.get("cpu_s")))
            if lay.hybrid(t, s)[0].exists():
                per["hybrid"].append((L, get_jgz(lay.hybrid(t, s)[0]).get("cpu_s")))
            if lay.open_(t, s).exists():
                per["measure"].append((L, json.loads(lay.open_(t, s).read_text()).get("cpu_s")))
    per = {k: [(L, x) for L, x in v if L is not None and x is not None and x > 0] for k, v in per.items()}
    if not all(per.values()):
        return {}
    L0 = float(np.median([L for L, _ in per["generate"]]))
    fit = {}
    for k, v in per.items():
        Ls, xs = np.log([L for L, _ in v]), np.log([x for _, x in v])
        kk = float(np.polyfit(Ls, xs, 1)[0]) if len(set(Ls)) >= 2 else 2.0
        fit[k] = (kk, lambda L, v=v, kk=kk: float(np.exp(np.mean([np.log(x) + kk * (math.log(L) - math.log(L_))
                                                                    for L_, x in v]))))
    out = {"measured_L": sorted({L for L, _ in per["generate"]}), "per_run_cpu_s": {k: float(np.mean([x for _, x in v]))
                                                                                     for k, v in per.items()},
           "k": {k: f[0] for k, f in fit.items()}, "runs_per_town": runs, "towns": n_towns, "by_L": {}}
    for L in SIZES:
        tot = sum(f[1](L) for f in fit.values())
        out["by_L"][str(L)] = {"per_run_cpu_h": tot / 3600, "total_cpu_h": tot * runs * n_towns / 3600,
                               "by_stage_cpu_h": {k: f[1](L) * runs * n_towns / 3600 for k, f in fit.items()},
                               "wall_h_at_6_workers": tot * runs * n_towns / 3600 / MAX_WORKERS}
    if mix:                                               # towns per L (--mix), e.g. cities 8192, towns 4096
        tot = sum(out["by_L"][str(L)]["per_run_cpu_h"] * n * runs for L, n in mix.items())
        out["mix"] = {"towns_by_L": {str(L): n for L, n in mix.items()}, "total_cpu_h": tot,
                      "wall_h_at_6_workers": tot / MAX_WORKERS}
    return out


def fmt(x, d=3):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def render(rep: dict) -> str:
    """The printed look: open quantities only (the report is scrubbed before this)."""
    lines = [f"D4 development look ({rep['split']}; {A0_LABEL}, delta {rep['delta']}): {len(rep['towns'])} towns"]
    for sid, t in rep["towns"].items():
        lines.append(f"\n== {sid}: {t['n_runs']} runs (seeds {t['runs']}), L {t['a0']['L']:.0f} m ==")
        lines.append("  key                     real      A0 med [q25, q75]          H1 med [q25, q75]          "
                     "H1+H2 med [q25, q75]")
        keys = [k.format(c=c) for k in MAIN_KEYS for c in H.KEYS] + list(UNIT_KEYS)
        for k in keys:
            if k not in t["arms"]["A0"]["metrics"]:
                continue
            cells = []
            for arm in ARMS:
                s = t["arms"][arm]["metrics"][k]
                cells.append(f"{fmt(s['median'])} [{fmt(s['q25'])}, {fmt(s['q75'])}]".ljust(27))
            lines.append(f"  {k:22s} {fmt(t['real'].get(k)):>8s}  " + "".join(cells))
        if t["undefined_on_real"]:
            lines.append(f"  undefined on the real town (not summarised): {', '.join(t['undefined_on_real'])}")
        p = t["m6_15_pilot"]
        lines.append(f"  m6_15 run-to-run variance: A0 {fmt(p['var_A0'], 5)}, H1 {fmt(p['var_H1'], 5)}, "
                     f"H1+H2 {fmt(p['var_H12'], 5)}; paired H1+H2 - A0 {fmt(p['var_H12_minus_A0'], 5)} "
                     f"(mean {fmt(p['mean_H12_minus_A0'], 4)}), H1 - A0 {fmt(p['var_H1_minus_A0'], 5)}; n {p['n']}")
        u, dm = t["logs_unit"], t["logs_domain"]
        g = lambda k: u.get(k, {}).get("median", float("nan"))
        gd = lambda k: dm.get(k, {}).get("median", float("nan"))
        lines.append(f"  IN THE UNIT ({t.get('unit_rule')}; primary, medians per run):")
        lines.append(f"    H1: connectors complete {fmt(g('H12.H1.connectors_complete'), 0)}, partial "
                     f"{fmt(g('H12.H1.connectors_partial'), 0)}, rejected for room {fmt(g('H12.H1.rejected_room'), 0)}, "
                     f"smoothing_fail {fmt(g('H12.H1.smoothing_fail'), 0)}, pass-2 share {fmt(g('H12.H1.pass2_share'), 2)}")
        lines.append(f"    H2: steep runs in A0 {fmt(g('A0.left.steep_runs_unit'), 0)} -> stairs "
                     f"{fmt(g('H12.H2.runs_stairs'), 0)}, switchbacks {fmt(g('H12.H2.runs_switchback'), 0)}, kept "
                     f"{fmt(g('H12.H2.runs_kept'), 0)} (failed for room {fmt(g('H12.H2.kept_failed_for_room'), 0)}, "
                     f"smoothing_fail {fmt(g('H12.H2.kept_smoothing_fail'), 0)}), kept km "
                     f"{fmt(g('H12.H2.kept_m') / 1000, 2)}, forced stairs {fmt(g('H12.H2.forced_stairs'), 0)}, pass-2 "
                     f"share {fmt(g('H12.H2.pass2_share'), 2)}; steep runs left {fmt(g('H12.left.steep_runs_unit'), 0)}")
        lines.append(f"    added km in the unit (P1b's count): H1 {fmt(g('H1.gross_unit_km'), 2)}, H1+H2 "
                     f"{fmt(g('H12.gross_unit_km'), 2)}")
        d = t["drivable_unit"]
        lines.append(f"    drivable (per unit only; never where a run is kept or left): {d['runs_claimed']}/{d['n_runs']} runs")
        lines.append(f"  domain-wide (secondary): steep runs {fmt(gd('H12.H2.runs_initial'), 0)}, stairs "
                     f"{fmt(gd('H12.H2.runs_stairs'), 0)}, switchbacks {fmt(gd('H12.H2.runs_switchback'), 0)}, kept "
                     f"{fmt(gd('H12.H2.runs_kept'), 0)}, connectors {fmt(gd('H12.H1.connectors_complete'), 0)}, gross km "
                     f"{fmt(gd('H12.gross_km'), 2)}")
        a = t["a0"]
        lines.append(f"  A0: {fmt(a['run_s']['median'], 0)} s per run, {fmt(a['forward_passes']['median'], 0)} forward "
                     f"passes, {fmt(a['km_limits']['median'], 1)} km in the limits ({fmt(a['km_domain']['median'], 1)} "
                     f"in the domain), rounds {fmt(a['rounds_used']['median'], 0)} of {a['max_rounds']} (cap hit "
                     f"{a['round_cap_hit']}/{t['n_runs']}), windows started {fmt(a['windows_started']['median'], 0)} of "
                     f"{a['windows']}, node budget hit {a['node_budget_hit']}/{t['n_runs']}, peak RSS "
                     f"{fmt(a['maxrss_mb']['median'], 0)} MB")
        cv = t.get("a0_coverage", {})
        if cv:
            lines.append(f"  A0 coverage (D4e): unit land in started windows "
                         f"{fmt(cv['unit_land_share_in_started_windows']['median'], 3)}, M6 pairs with an endpoint > 60 m "
                         f"from A0 {fmt(cv['pairs_endpoint_far_share']['median'], 3)} (medians)")
    pl = rep.get("pilot") or {}
    if pl:
        st, bl = pl["settlements"], pl["blocks"]
        lines.append(f"\nPilot for hold-H's N (D4d ruling 7; {pl['metric']}, {pl['contrast']}): sigma2_w "
                     f"{fmt(pl['sigma2_w'], 6)}; settlements n {st['n']}, sigma2_b {fmt(st['sigma2_b'], 6)}, v "
                     f"{fmt(st['v'], 6)}, v+ {fmt(st['v_ucl80'], 6)}, N {fmt(st['N'], 1)}; blocks n {bl['n']}, "
                     f"between-block variance {fmt(bl['sigma2_block'], 6)}, N {fmt(bl['N'], 1)}; N used: "
                     f"{pl['N'] if pl['N'] is not None else 'undefined (fewer than 2 settlements)'} (>= {pl['N_min']})")
        lines.append(f"  {pl['statement']}")
    pj = rep.get("projection") or {}
    if pj:
        c, k = pj["per_run_cpu_s"], pj["k"]
        lines.append(f"\nProjection ({pj['towns']} towns x {pj['runs_per_town']} runs; measured at L {pj['measured_L']} m; "
                     f"per run CPU s: generate {c['generate']:.0f}, hybrid {c['hybrid']:.0f}, measure {c['measure']:.0f}; "
                     f"scaled by L^k, k = {k['generate']:.2f} / {k['hybrid']:.2f} / {k['measure']:.2f}):")
        for L, v in pj["by_L"].items():
            lines.append(f"  all towns at L {L} m: {v['per_run_cpu_h']:.2f} CPU-h per run, {v['total_cpu_h']:.0f} "
                         f"CPU-h total, {v['wall_h_at_6_workers']:.0f} h wall at {MAX_WORKERS} workers")
        if pj.get("mix"):
            lines.append(f"  mix {pj['mix']['towns_by_L']} (towns per L m): {pj['mix']['total_cpu_h']:.0f} CPU-h, "
                         f"{pj['mix']['wall_h_at_6_workers']:.0f} h wall at {MAX_WORKERS} workers")
    return "\n".join(lines)


def blocks_of(split: str, towns) -> dict:
    """1-degree blocks: the frozen dev unit list's "blocks" (dev); the site config's m20 block (scratch)."""
    if split == "dev":
        return dict(json.loads(UNITS.read_text()).get("blocks", {})) if UNITS.exists() else {}
    out = {}
    for t in towns:
        f = SCRATCH / "sites" / f"{t}.json"
        if f.exists():
            out[t] = json.loads(f.read_text()).get("m20", {}).get("block")
    return out


def stage_report(lay: Layout, towns, seeds, n_towns: int, mix: dict | None = None, blocks: dict | None = None) -> dict:
    rep = {"split": lay.split, "delta": lay.delta, "a0_label": A0_LABEL, "seeds": list(seeds), "towns": {}}
    for t in towns:
        r = town_report(lay, t, seeds)
        if r is not None:
            rep["towns"][t] = r
    rep["pilot"] = pilot_n({t: r["m6_15_pilot"]["diffs_H12_minus_A0"] for t, r in rep["towns"].items()},
                           blocks if blocks is not None else blocks_of(lay.split, towns))
    for t in towns:
        for s in seeds:
            v = lay.verify(t, s)
            if v.exists():
                x = json.loads(v.read_text())
                if "identical" not in x and lay.a0(t, s)[0].exists():
                    x["identical"] = x["replay_sha256"] == get_jgz(lay.a0(t, s)[0])["graph_sha256"]
                rep.setdefault("replay", {})[f"{t}:s{s}"] = x.get("identical")
    man = seal_bundle(lay, towns, seeds)
    rep["sealed"] = {"manifest": str(lay.manifest()), "bundle_sha256": man["bundle_sha256"], "records": man["records"]}
    rep["timing"] = [json.loads(x) for x in lay.timing().read_text().splitlines()] if lay.timing().exists() else []
    rep["projection"] = projection(lay, towns, seeds, n_towns, mix=mix)
    rep = scrub(rep)
    put_json(lay.report(), rep)
    text = render(rep)
    print(text, flush=True)
    if rep.get("replay"):
        print("A0 replay identical:", rep["replay"], flush=True)
    print(f"sealed: {man['records']} records, bundle sha256 {man['bundle_sha256'][:16]}... ({lay.manifest()})", flush=True)
    return rep


# ---------------------------------------------------------------- maps
def map_job(spec) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LightSource
    from matplotlib.lines import Line2D
    from m20_towns import load_town
    split, sid, seed, delta, base = spec
    lay = Layout(split, delta, base)
    P = paths(split)
    real = load_town(sid, P)
    t32 = real["terrain32"]
    Z = t32["elev"].astype(float)
    H_, W_ = Z.shape
    ext = (t32["x0"], t32["x0"] + W_ * t32["res"], t32["y1"] - H_ * t32["res"], t32["y1"])
    poly = real["polygon"]
    x0, y0, x1, y1 = poly.bounds
    pad = 300.0
    a0 = get_jgz(lay.a0(sid, seed)[0])["graph"]
    hb = get_jgz(lay.hybrid(sid, seed)[0])
    h12 = hb["arms"]["H12"]["graph"]
    # categorical slots 1-3 of the reference palette, validated all-pairs; class also coded by line width
    style = {0: ("#2a78d6", 0.55, "vehicular"), 1: ("#1baf7a", 1.6, "stepped ramp (15-25 %)"),
             2: ("#eb6834", 1.9, "stairs (> 25 %)")}
    Zf = np.where(np.isfinite(Z), Z, np.nanmean(Z))
    shade = LightSource(315, 40).hillshade(Zf, vert_exag=2.0, dx=t32["res"], dy=t32["res"])
    lo, hi = np.nanpercentile(Z, 1), np.nanpercentile(Z, 99)
    step = 10.0 if hi - lo < 400 else 20.0
    lev = np.arange(math.floor(lo / step) * step, hi + step, step)
    Xc = t32["x0"] + (np.arange(W_) + 0.5) * t32["res"]
    Yc = t32["y1"] - (np.arange(H_) + 0.5) * t32["res"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.6))
    for ax, (title, gj) in zip(axs, ((f"{A0_LABEL}, TNg", a0), (f"{A0_LABEL} + H1 + H2 (grade hybrid)", h12))):
        ax.imshow(shade, extent=ext, cmap="Greys_r", vmin=-0.2, vmax=1.4, interpolation="bilinear")
        ax.contour(Xc, Yc, Zf, levels=lev, colors="#6b6a64", linewidths=0.35, alpha=0.7)
        ax.contour(Xc, Yc, Zf, levels=[v for v in lev if v % (5 * step) == 0], colors="#52514e", linewidths=0.7)
        N = np.array(gj["nodes"], float).reshape(-1, 2)
        E = np.array(gj["edges"], int).reshape(-1, 2)
        K = np.array(gj.get("kind", [0] * len(E)), int)
        for k in (0, 1, 2):
            sel = K == k
            if sel.any():
                col, lw, _ = style[k]
                ax.add_collection(LineCollection(N[E[sel]], colors=col, linewidths=lw, capstyle="round", zorder=3 + k))
                if k:                                     # pedestrian pieces are short: a marker makes them findable
                    M = N[E[sel]].mean(1)
                    ax.scatter(M[:, 0], M[:, 1], s=22, c=col, edgecolors="white", linewidths=0.6, zorder=6 + k)
        full = real.get("polygon_full")
        rings = lambda g: [p.exterior.xy for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]
        if full is not None and not full.equals(poly):                # a city: the whole polygon, faint
            for px, py in rings(full):
                ax.plot(px, py, color="#52514e", lw=0.8, ls=":", zorder=6)
        for px, py in rings(poly):
            ax.plot(px, py, color="#0b0b0b", lw=1.1, ls="--", zorder=6)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.tick_params(labelsize=7, colors="#52514e")
        for sp in ax.spines.values():
            sp.set_color("#c3c2b7")
    lg = hb["arms"]["H12"].get("log_unit") or hb["arms"]["H12"]["log"]
    kept = lg["H2"].get("runs_kept", 0)
    handles = [Line2D([], [], color=c, lw=max(lw, 1.2) * 1.5, label=lab, marker="o" if k else None, markersize=5,
                      markeredgecolor="white") for k, (c, lw, lab) in style.items()]
    core = real.get("unit_rule") == "city core"
    handles.append(Line2D([], [], color="#0b0b0b", lw=1.1, ls="--", label="unit: city core" if core else "settlement polygon"))
    if core:
        handles.append(Line2D([], [], color="#52514e", lw=0.8, ls=":", label="whole GHSL polygon"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"{sid} ({'scratch, unpinned OSM' if split == 'scratch' else 'dev-H'}), seed {seed}: streets by class "
                 f"over FABDEM 32 m contours ({step:.0f} m; heavy every {5 * step:.0f} m). H2: "
                 f"{lg['H2'].get('runs_stairs', 0)} runs to stairs, {lg['H2'].get('runs_switchback', 0)} switchbacks, "
                 f"{kept} kept (in the unit: {hb.get('unit_rule', 'settlement polygon')})", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    f = lay.map(sid, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=150)
    plt.close(fig)
    return str(f)


# ---------------------------------------------------------------- stage drivers
def _pool(fn, specs, workers):
    from multiprocessing import Pool
    out = []
    if workers <= 1 or len(specs) <= 1:
        for sp in specs:
            out.append(fn(sp))
            print(json.dumps(scrub(out[-1]), default=_default), flush=True)
        return out
    with Pool(min(workers, len(specs)), maxtasksperchild=1) as p:
        for r in p.imap_unordered(fn, specs, chunksize=1):
            out.append(r)
            print(json.dumps(scrub(r), default=_default), flush=True)
    return out


def stage_generate(lay, towns, seeds, workers, replay=False):
    if replay:
        specs = [(lay.split, t, s, lay.delta, _base(lay), True) for t in towns for s in seeds
                 if not lay.verify(t, s).exists()]
    else:
        specs = [(lay.split, t, s, lay.delta, _base(lay), False) for t in towns for s in seeds
                 if not lay.a0(t, s)[0].exists()]
    t0 = time.time()
    jobs = _pool(a0_job, specs, workers)
    log_stage(lay, "verify" if replay else "generate", time.time() - t0, jobs, workers)
    if replay:
        bad = [j for j in jobs if j.get("identical") is False]
        if bad:
            raise SystemExit(f"A0 replay differs: {[(j['town'], j['seed']) for j in bad]}")
    return jobs


def _base(lay: Layout):
    return lay.root.parent


def stage_simple(lay, towns, seeds, workers, name):
    fn, done, need = {"hybrid": (hybrid_job, lambda t, s: lay.hybrid(t, s)[0], lambda t, s: lay.a0(t, s)[0]),
                      "measure": (measure_job, lambda t, s: lay.open_(t, s), lambda t, s: lay.hybrid(t, s)[0])}[name]
    specs = [(lay.split, t, s, lay.delta, _base(lay)) for t in towns for s in seeds
             if not done(t, s).exists() and need(t, s).exists()]
    if name == "measure":                                     # the real row once per town, before the pool
        for t in sorted({sp[1] for sp in specs}):
            real_row(t, lay, paths(lay.split))
    t0 = time.time()
    jobs = _pool(fn, specs, workers)
    log_stage(lay, name, time.time() - t0, jobs, workers)
    return jobs


def stage_maps(lay, towns, seeds):
    out = []
    for t in towns:
        s = next((s for s in seeds if lay.hybrid(t, s)[0].exists()), None)
        if s is not None:
            out.append(map_job((lay.split, t, s, lay.delta, _base(lay))))
            print("map:", out[-1], flush=True)
    return out


def parse_seeds(x: str) -> tuple:
    out = []
    for part in x.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return tuple(sorted(set(out)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["generate", "verify", "hybrid", "measure", "report", "maps", "all"])
    ap.add_argument("--split", choices=["dev", "scratch"], required=True)
    ap.add_argument("--seeds", default="1-24")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--ids", default="")
    ap.add_argument("--delta", choices=DELTAS, default="none", help="A0's whole-town delta (choice 1)")
    ap.add_argument("--out", default=None, help="output root (default: the split's devlook directory)")
    ap.add_argument("--towns-n", type=int, default=32, help="towns in the CPU-hour projection")
    ap.add_argument("--mix", default="", help="projection: towns per domain side, e.g. 2048:7,4096:13,8192:12")
    a = ap.parse_args(argv)
    if a.split == "dev" and a.delta != "none":
        raise SystemExit(f"D4d ruling 6: only {A0_LABEL} (--delta none) runs on dev-H")
    if not 1 <= a.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")
    towns = town_ids(a.split, [i for i in a.ids.split(",") if i])
    seeds = parse_seeds(a.seeds)
    lay = Layout(a.split, a.delta, a.out)
    print(f"{a.stage}: {a.split}, towns {towns}, seeds {seeds[0]}-{seeds[-1]} ({len(seeds)}), -> {lay.root}", flush=True)
    if a.stage in ("generate", "all"):
        stage_generate(lay, towns, seeds, a.workers)
    if a.stage == "verify":
        stage_generate(lay, towns, seeds, a.workers, replay=True)
    if a.stage in ("hybrid", "all"):
        stage_simple(lay, towns, seeds, a.workers, "hybrid")
    if a.stage in ("measure", "all"):
        stage_simple(lay, towns, seeds, a.workers, "measure")
    if a.stage in ("report", "all"):
        mix = {int(L): int(n) for L, n in (x.split(":") for x in a.mix.split(",") if x)}
        stage_report(lay, towns, seeds, sum(mix.values()) if mix else a.towns_n, mix or None)
    if a.stage in ("maps", "all"):
        t0 = time.time()
        fs = stage_maps(lay, towns, seeds)
        log_stage(lay, "maps", time.time() - t0, [{} for _ in fs], 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
