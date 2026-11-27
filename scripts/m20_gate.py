"""Milestone 20 D3 validation gate on the real dev-H towns (docs/M20_LOG.md D3, D3a, D3b), run once before the freeze.

    python scripts/m20_gate.py townlist                after the OSM gate, before any metric: the reference and the
                                                       descriptive ids, hash-frozen -> configs/m20_pool/dev_units.json
    python scripts/m20_gate.py metrics [--workers 6]   real and adversarial unit metrics of every built dev-H town
                                                       -> outputs/m20/gate/{real,adv/<case>}/<id>.json
    python scripts/m20_gate.py gate                    gate 1 (LOO coverage, KS) and gate 2 (every adversarial case)
                                                       -> outputs/m20/gate/report.json

Real units: the built towns of data/processed/m20/towns/; the network is the unit's native vehicular network, the
pedestrian layer its steps / footways / paths, the field FABDEM at 32 m (m20_towns.hill_field). The REFERENCE (plan,
gate 1, gate 2, LOO nulls) is the accepted dev-H towns and cities only; the dev-H villages ("evaluation only", D1) and
the two declared anchors (anchor_accepted.csv) are scored against it descriptively and decide nothing (D3c reading).
Adversarial cases (D3, D3a choice 17), built on every town's own field: switchback, contour_parallel, fall_line_grid,
raster_zigzag (of the town's real network, seed 0); targeted checks are sign-oriented (D3c). The leastcostpath case
(D3c) is a single-site check (n = 1, a declared deviation from D3a's 90 % rule): the actual M18 Petra seed-1 network
on the Petra field (``petra_field``), with Petra's real vehicular network as the control.
The only generated network read is that leastcostpath output (an M20 external; its Petra output was seen before the
freeze, D3c). No output of any M20 method is read, so nothing here can see P2 of a method.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

OUT = ROOT / "outputs" / "m20" / "gate"
WALK = ROOT / "data" / "processed" / "m20" / "walk"
CASES = ("switchback", "contour_parallel", "fall_line_grid", "raster_zigzag")


UNITS = ROOT / "configs" / "m20_pool" / "dev_units.json"


def stage_townlist() -> int:
    """D3c condition 1: freeze the unit lists (sha256 of the sorted ids) before any metric; n is that count. A town
    flagged in gate 1 (coverage < 0.75) is never removed: its mapping check is descriptive (declared now)."""
    import hashlib
    import pandas as pd
    if UNITS.exists():
        raise SystemExit(f"{UNITS.relative_to(ROOT)} exists: the lists are frozen")
    acc = pd.read_csv(WALK / "dev_accepted.csv")
    anc = pd.read_csv(WALK / "anchor_accepted.csv")
    ref = sorted(acc.id[acc.kind != "village"])
    desc = sorted(acc.id[acc.kind == "village"]) + sorted(anc.id)
    h = lambda ids: hashlib.sha256("\n".join(ids).encode()).hexdigest()
    blocks = dict(zip(acc.id, acc.block)) | dict(zip(anc.id, anc.block))
    out = {"reference": ref, "descriptive": desc, "reference_sha256": h(ref), "descriptive_sha256": h(desc),
           "n_reference": len(ref), "blocks": {i: blocks[i] for i in ref + desc},
           "rule": "reference = accepted dev-H towns and cities; descriptive = dev-H villages and the declared anchors "
                   "(D1, D3c reading); flagged towns are never removed"}
    UNITS.write_text(json.dumps(out, indent=1))
    print(f"frozen: {len(ref)} reference ({out['reference_sha256'][:12]}), {len(desc)} descriptive")
    return 0


def units(require_built: bool = False) -> tuple:
    """(reference ids, descriptive ids) of the frozen list (``stage_townlist``), restricted to the built towns; with
    ``require_built`` every frozen unit must be built (the gate is never run on a partial reference)."""
    from m20_towns import Paths
    if not UNITS.exists():
        raise SystemExit("run the townlist stage first (the unit lists are frozen before any metric)")
    U = json.loads(UNITS.read_text())
    P = Paths(scratch=False)
    built = lambda s: (P.towns / s / "real.pkl").exists()
    miss = [s for s in U["reference"] + U["descriptive"] if not built(s)]
    if require_built and miss:
        raise SystemExit(f"not built: {miss}")
    return [s for s in U["reference"] if built(s)], [s for s in U["descriptive"] if built(s)]


def _dump(f: Path, obj):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(f)


def town_job(job) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    from m20_towns import Paths, hill_field, load_town
    from urbanmorph.metrics import hilltown as H
    sid, adversarial = job
    real = load_town(sid, Paths(scratch=False))
    F = hill_field(real)
    g = real["vehicular"]["unit_native"]
    ped = (real["pedestrian"]["unit_lines"], real["pedestrian"]["unit_steps"])
    t0, done = time.time(), {}
    f = OUT / "real" / f"{sid}.json"
    if not f.exists():
        m = H.unit_metrics(g, F, ped=ped, access=True)
        _dump(f, {"id": sid, "seconds": time.time() - t0, "outcomes": m.pop("_outcomes", {}), "metrics": m})
    done["real"] = time.time() - t0
    build = {"switchback": lambda: H.adv_switchback(F), "contour_parallel": lambda: H.adv_contour_parallel(F),
             "fall_line_grid": lambda: H.adv_fall_line_grid(F), "raster_zigzag": lambda: H.adv_raster_zigzag(g, seed=0)}
    for case in CASES if adversarial else ():
        f = OUT / "adv" / case / f"{sid}.json"
        if f.exists():
            continue
        t0 = time.time()
        a = build[case]()
        m = H.unit_metrics(a, F, access=False)
        m.pop("_outcomes", None)
        _dump(f, {"id": sid, "case": case, "seconds": time.time() - t0, "street_km": float(m.get("street_km@unit", np.nan)),
                  "metrics": m})
        done[case] = time.time() - t0
    return {"id": sid, "seconds": done}


def petra_metrics() -> dict:
    """Unit metrics of Petra's real vehicular network and of the leastcostpath network, both cropped to the limits
    (site_benchmark.crop, as every site score), on the Petra field; cached."""
    import site_benchmark as sb
    from urbanmorph.metrics import hilltown as H
    f = OUT / "petra" / "metrics.json"
    if f.exists():
        return json.loads(f.read_text())
    F, site, mask, tiles = petra_field()
    out = {"tiles": tiles, "unit_km2": float(F.unit.sum() * F.res ** 2 / 1e6)}
    for name, g in (("real", sb.crop(site.real("vehicular"), mask, site)), ("leastcostpath", H.load_lcp_petra_s1())):
        m = H.unit_metrics(g, F, access=False)
        m.pop("_outcomes", None)
        out[name] = m
    _dump(f, out)
    return out


def stage_metrics(workers: int) -> int:
    ref, desc = units()
    print(f"{len(ref)} built dev-H towns and cities (reference), {len(desc)} villages and anchors (descriptive)", flush=True)
    petra_metrics()
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(town_job, [(s, True) for s in ref] + [(s, False) for s in desc]):
            print(r["id"], {k: round(v) for k, v in r["seconds"].items()}, flush=True)
    return 0


def petra_field():
    """The HillField of the M18 Petra site (D3c): FABDEM V1-2 averaged to 32 m on the site's 2048 m box (grid lines on
    the box corners, so through the site centre), unit = 32 m cells whose centre lies inside the fixed city limits,
    land = unit (the wadis are dry, as declared in M18; no water layer). Site-local metres, as the stored networks."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    import site_benchmark as sb
    from m20_towns import FRES, ll_bounds
    from m20_walk import fab_tile
    from urbanmorph.metrics.hilltown import HillField
    from urbanmorph.terrain.heightfield import Terrain
    site = sb.Site("petra")
    n = int(site.L // FRES)
    lo0, la0, lo1, la1 = ll_bounds(site.crs, site.x0, site.y0, site.x0 + site.L, site.y0 + site.L)
    tiles = sorted({fab_tile(la, lo) for la in range(int(np.floor(la0 - .01)), int(np.floor(la1 + .01)) + 1)
                    for lo in range(int(np.floor(lo0 - .01)), int(np.floor(lo1 + .01)) + 1)} - {None})
    srcs = [rasterio.open(t) for t in tiles]
    dem, dtr = merge(srcs, bounds=(lo0 - 0.01, la0 - 0.01, lo1 + 0.01, la1 + 0.01), nodata=np.nan, dtype="float32")
    nod = srcs[0].nodata
    for s in srcs:
        s.close()
    dem = dem[0]
    if nod is not None:
        dem[dem == nod] = np.nan
    Z = np.full((n, n), np.nan, np.float32)
    reproject(dem, Z, src_transform=dtr, src_crs="EPSG:4326", dst_transform=from_origin(site.x0, site.y0 + site.L, FRES, FRES),
              dst_crs=site.crs, resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    S = Terrain(Z, 0.0, float(site.L), FRES, native_res=30.0)
    c = (np.arange(n) + 0.5) * FRES
    X, Y = np.meshgrid(c, site.L - c)
    mask = np.load(site.data / "city_mask.npz")["mask"]
    r, k = site.rc(np.c_[X.ravel(), Y.ravel()])
    unit = mask[r, k].reshape(n, n)
    return HillField(S, unit & np.isfinite(Z), T=S, unit=unit, uid="petra"), site, mask, [str(t) for t in tiles]


def _load(d: Path, ids) -> dict:
    return {s: json.loads((d / f"{s}.json").read_text())["metrics"] for s in ids if (d / f"{s}.json").exists()}


def stage_gate() -> int:
    from urbanmorph.metrics import hilltown as H
    ids, _ = units(require_built=True)
    rows = _load(OUT / "real", ids)
    missing = sorted(set(ids) - set(rows))
    if missing:
        raise SystemExit(f"real metrics missing for {missing}: run the metrics stage first")
    plan = H.make_plan(rows.values(), H.SUITE_CELLS)
    g1 = H.gate_real(rows, H.SUITE_CELLS, plan)
    L = g1.pop("loo")
    blocks = json.loads(UNITS.read_text())["blocks"]
    g1b = H.gate_real(rows, H.SUITE_CELLS, plan, groups={t: blocks[t] for t in rows})     # D3c sensitivity
    g1b.pop("loo")
    mesh = H.make_plan(rows.values(), H.MESH_CELLS)              # D3c: descriptive, frozen with the suite
    topo = H.make_plan(rows.values(), H.GROUPS["lens_topo"])
    plan_rec = lambda p_: {"keys": list(p_.keys), "descriptive": list(p_.descriptive), "c": p_.c, "unit": p_.unit,
                           "floor": p_.floor, "n_ref": p_.n_ref}
    lens_n = {c: {"towns_lens_gt0": int(sum(r.get(f"lens_share@{c}", np.nan) > 0 for r in rows.values())),
                  "towns_2ap_gt0": int(sum(r.get(f"lens_share_2ap@{c}", np.nan) > 0 for r in rows.values()))}
              for c in H.KEYS + (H.POOLED,)}
    rep = {"towns": ids, "gate1": g1, "gate1_leave_block_out": g1b, "gate2": {}, "lens_towns_gt0": lens_n,
           "real_loo_p2": {t: H.composite(L[t], H.P2_CELLS) for t in L},
           "plans": {"suite": plan_rec(plan), "mesh": plan_rec(mesh), "lens_topo": plan_rec(topo)}}
    for case in CASES:
        adv = _load(OUT / "adv" / case, ids)
        r = H.gate_adversarial_towns(adv, rows, case, plan=plan, loo_out=L)
        for t in r["per_town"].values():
            t.pop("z", None)
        rep["gate2"][case] = r
    pm = petra_metrics()
    rep["leastcostpath_single_site"] = H.gate_site_control(pm["leastcostpath"], pm["real"], rows, "leastcostpath",
                                                           plan=plan, loo_out=L)
    full = H.fit_reference(rows.values(), H.SUITE_CELLS, plan)
    desc = _load(OUT / "real", units()[1])
    rep["descriptive_units"] = {}
    for sid, x in desc.items():
        z = H.score(x, x, full)
        c = H.composite(z)
        rep["descriptive_units"][sid] = {                       # descriptive only: no p-values (D3c)
            "composite": c, "share_absz_le2": float(np.mean([v.absz <= H.Z_FIT for v in z.values()])) if z else None,
            "cells_defined": len(z), "cells_declared": len(H.expand(plan.keys)),
            "z": {k: {"z": v.z, "absz": v.absz, "flag": v.flag} for k, v in z.items()}}
    _dump(OUT / "report.json", rep)
    print(f"gate 1: n {g1['n']}, coverage {g1['coverage']:.3f} (threshold {g1['threshold']:.3f}), "
          f"KS p (approx.) {g1['ks_p_approx']:.3f} -> {'PASS' if g1['pass'] else 'FAIL'}; "
          f"flagged {g1['flagged']}; floor-bound cells {g1['floor_bound']}; scale binding {g1['scale_binding']}")
    print(f"gate 1, leave-one-block-out (sensitivity): coverage {g1b['coverage']:.3f}, KS p (approx.) "
          f"{g1b['ks_p_approx']:.3f}, flagged {g1b['flagged']}")
    for case, r in rep["gate2"].items():
        print(f"gate 2 {case}: fails on {r['share_failing']:.2f} of {r['towns']} towns -> "
              f"{'PASS' if r['pass'] else 'FAIL'}; margins {json.dumps({k: round(v['median'], 2) for k, v in r['margins'].items()})}")
    lc = rep["leastcostpath_single_site"]
    print(f"leastcostpath, single site (n = 1, deviation): {lc['verdict']} -> "
          f"{'caught on every check' if lc['caught_all'] else 'NOT caught on every check (limitation)'}")
    print("towns with lens > 0 / 2A/P > 0 per cell:", json.dumps(lens_n))
    for sid, d in rep["descriptive_units"].items():
        print(f"descriptive {sid}: composite {d['composite']:.2f}, |z| <= 2 on {d['share_absz_le2']} of "
              f"{d['cells_defined']} defined cells ({d['cells_declared']} declared)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["townlist", "metrics", "gate"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.stage == "townlist":
        return stage_townlist()
    return stage_metrics(a.workers) if a.stage == "metrics" else stage_gate()


if __name__ == "__main__":
    sys.exit(main())
