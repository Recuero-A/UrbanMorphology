"""Milestone 17, step 4: fresh geographic holdout (docs/M17_PROTOCOL.md §6-7; frozen before these runs).

For every arm (M*, NA-fixed, NA-SMC, M* + organic, M* + radial): original, F1 reuse (frozen radius), the spacing
ablation and F0 post-hoc consolidation of the original graphs, paired by window and seed against the original.
Primary: REBUILT close15 excess over the SAME window's real public streets; reductions and overcorrection. Companions:
nn p10, consol10 (raw and rebuilt). Preservation: the frozen margins. Uncertainty: bootstrap over metros (geographic
blocks), then windows within metro, then seeds. Mechanism: edit classes of the remaining close pairs.

    python scripts/m17_holdout_report.py   ->  outputs/m17/holdout.{md,json}, outputs/m17/claims.md
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from m17_diagnose import attribute, rebuild
from m17_fixdev_report import MARGINS
from urbanmorph.experiments.graph_io import graph_from_json
from urbanmorph.experiments.jsonio import atomic_json
from urbanmorph.experiments.runner import load_run
from urbanmorph.experiments.sites import site_set
from urbanmorph.graphs.junction_reuse import consolidate_graph
from urbanmorph.graphs.junctions import junction_row
from urbanmorph.graphs.m14_metrics import tile_metrics_m14
from urbanmorph.graphs.style_metrics import morphology_row
from urbanmorph.graphs.tc_potentials import TerrainField
from urbanmorph.graphs.terrain_generate import terrain_of

RUNS = ROOT / "outputs" / "m17" / "runs"
OUT = ROOT / "outputs" / "m17"
HOLD = ROOT / "data" / "processed" / "m17" / "holdout"


def collect(spec, real, terr, posthoc=None, attr=None):
    r = load_run(spec, RUNS)
    if r is None:
        return {}
    out = {}
    for sd, v in r["seeds"].items():
        for k, row in enumerate(v["rows"]):
            g = graph_from_json(v["graphs"][k])
            site = row["site"]
            rec = {"empty": row.get("empty"), "od_unreach": row.get("od_unreach_land"), "coverage_60": row.get("coverage_60"),
                   "street_km": row.get("street_km"), "share_gt15": row.get("share_gt15"),
                   "block_logA": np.log(row["block_area_med"]) if row.get("block_area_med") else np.nan,
                   "reuse_applied": row.get("log", {}).get("reuse_applied", 0), "cand_evals": row.get("log", {}).get("cand_evals", 0),
                   "seconds": v["seconds"] / len(v["rows"]), "diversity": np.nan}
            if posthoc:
                g = consolidate_graph(g, posthoc)
                t = terr[site]
                rng = np.random.default_rng([int(sd), 15])
                m = morphology_row(g, t.t, centre=None, rng=rng, grade_span=30.0)
                m14 = tile_metrics_m14(g, t, rng)
                rec.update(od_unreach=m14.get("od_unreach_land"), coverage_60=m14.get("coverage_60"), street_km=m["street_km"],
                           share_gt15=m.get("share_gt15"), empty=int(len(g.edges) == 0),
                           block_logA=np.log(m["block_area_med"]) if m.get("block_area_med") else np.nan)
            elif attr is not None:
                evs = (v.get("events") or [None] * len(v["rows"]))[k]
                if evs:
                    attr.update(attribute(g, evs)[0])
            raw, reb = junction_row(g), junction_row(rebuild(g))
            w = site[len("hold_"):]
            rec.update(metro=w.rsplit("_", 1)[0], raw_close15=raw["close15"], reb_close15=reb["close15"],
                       raw_p10=raw["nn_p10"], reb_p10=reb["nn_p10"], raw_consol10=raw["consol10_ratio"], n3=reb["n3"], n4=reb["n4"], n5p=reb["n5p"],
                       reb_consol10=reb["consol10_ratio"], density=reb["focal_density_km2"],
                       excess=reb["close15"] - real[w])
            out[(site, int(sd))] = rec
    return out


def boot3(d: dict, metro_of, rng, B=4000):
    """Mean over metros of (mean over windows of (mean over seeds)), bootstrap metros -> windows -> seeds."""
    tree = {}
    for (site, sd), v in d.items():
        if np.isfinite(v):
            tree.setdefault(metro_of[site], {}).setdefault(site, []).append(v)
    if not tree:
        return np.nan, np.nan, np.nan
    est = float(np.mean([np.mean([np.mean(s) for s in w.values()]) for w in tree.values()]))
    ms = list(tree)
    bs = []
    for _ in range(B):
        vals = []
        for mm in rng.choice(ms, len(ms)):
            ws = list(tree[mm])
            vals.append(np.mean([np.mean(rng.choice(tree[mm][w], len(tree[mm][w]))) for w in rng.choice(ws, len(ws))]))
        bs.append(np.mean(vals))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    rng = np.random.default_rng(17)
    man = json.loads((ROOT / "configs" / "m17_holdout_windows.json").read_text())["windows"]
    real = {w["window"]: junction_row(pickle.loads((HOLD / "streets" / f"{w['window']}.pkl").read_bytes()))["close15"] for w in man}
    real_p10 = {w["window"]: junction_row(pickle.loads((HOLD / "streets" / f"{w['window']}.pkl").read_bytes()))["nn_p10"] for w in man}
    metro_of = {f"hold_{w['window']}": w["metro"] for w in man}
    terr = {d["site"]: TerrainField(terrain_of(d)) for d in site_set("holdout_m17")}
    plan = json.loads((ROOT / "configs" / "m17_plans" / "holdout.json").read_text())["runs"]
    sel = json.loads((OUT / "fixdev.json").read_text())["selected"]
    r = float(sel["r"]) if sel else 12.0
    arms = [("M*", "off"), ("NA-fixed", "off"), ("NA-SMC", "off"), ("M*", "organic"), ("M*", "radial")]
    res, mech = [], {}
    real_mean = boot3({(f"hold_{w}", 0): v for w, v in real.items()}, metro_of, rng)
    for m, st in arms:
        arm = m + (f" + {st}" if st != "off" else "")
        sp = lambda pred: [s for s in plan if s["method"] == m and s["style"] == st and pred(s["overrides"])][0]
        A0, A1 = Counter(), Counter()
        base = collect(sp(lambda o: set(o) == {"event_log"}), real, terr, attr=A0)
        variants = [("original", base)]
        if sel:
            variants.append((f"F1 reuse r={r:g}", collect(sp(lambda o: "junction_reuse_m" in o), real, terr, attr=A1)))
        variants.append((f"F0 post-hoc tau={r:g}", collect(sp(lambda o: set(o) == {"event_log"}), real, terr, posthoc=r)))
        variants.append(("F0 post-hoc tau=9 (secondary)", collect(sp(lambda o: set(o) == {"event_log"}), real, terr, posthoc=9.0)))
        variants.append((f"spacing r={r:g} (ablation)", collect(sp(lambda o: "junction_min_m" in o), real, terr)))
        mech[arm] = {"original": dict(A0), "F1": dict(A1)}
        for lab, cur in variants:
            keys = [k for k in cur if k in base]
            rec = {"arm": arm, "variant": lab, "n": len(keys),
                   "excess": boot3({k: cur[k]["excess"] for k in keys}, metro_of, rng),
                   "delta_excess": boot3({k: cur[k]["excess"] - base[k]["excess"] for k in keys}, metro_of, rng),
                   "overcorrected_share": float(np.mean([cur[k]["excess"] < -0.05 for k in keys if np.isfinite(cur[k]["excess"])])),
                   "reb_close15": float(np.nanmean([cur[k]["reb_close15"] for k in keys])),
                   "raw_close15": float(np.nanmean([cur[k]["raw_close15"] for k in keys])),
                   "reb_p10": float(np.nanmean([cur[k]["reb_p10"] for k in keys])),
                   "raw_consol10": float(np.nanmean([cur[k]["raw_consol10"] for k in keys])),
                   "reb_consol10": float(np.nanmean([cur[k]["reb_consol10"] for k in keys])),
                   "density": float(np.nanmean([cur[k]["density"] for k in keys])),
                   "reuse_applied": float(np.mean([cur[k]["reuse_applied"] for k in keys])),
                   "cand_evals": float(np.mean([cur[k]["cand_evals"] for k in keys])),
                   "seconds": float(np.mean([cur[k]["seconds"] for k in keys]))}
            ok = True
            for key, (kind, mg) in MARGINS.items():
                b = np.nanmean([base[k][key] for k in keys])
                c = np.nanmean([cur[k][key] for k in keys])
                d = boot3({k: cur[k][key] - base[k][key] for k in keys}, metro_of, rng)
                rec[f"d_{key}"] = d
                rec[f"ok_{key}"] = bool(d[0] <= mg if kind == "abs" else ((d[0] / b if b else 0) >= mg if kind == "rel" else abs(d[0]) <= mg))
                ok &= rec[f"ok_{key}"]
            rec["margins_ok"] = bool(ok)
            res.append(rec)
            print(arm, lab, f"excess {rec['excess'][0]:+.3f} d {rec['delta_excess'][0]:+.3f} margins {ok}", flush=True)
    f = lambda x, n=3: "–" if x is None or not np.isfinite(x) else f"{x:+.{n}f}"
    ci = lambda t, n=3: f"{f(t[0], n)} [{f(t[1], n)}, {f(t[2], n)}]"
    L = ["# M17 step 4: fresh geographic holdout (CONFIRMATORY; 9 metros x 6 windows; seeds 101-103)", "",
         f"Metros: Gwangju, Ulsan, Jeonju, Cheongju (KR), Piura, Chiclayo, Huancayo (PE), Castellón, Figueres (ES); "
         f"GLO-30 DSM. Real close15 of these windows (mean of metros): {real_mean[0]:.3f}. Excess = REBUILT close15 − the "
         "same window's real close15. CIs: bootstrap metros → windows → seeds. Margins frozen in amendment 1.", "",
         "| Arm | Variant | excess [95% CI] | Δ excess vs original [95% CI] | windows overcorrected (< −0.05) | close15 raw / rebuilt | nn p10 rebuilt | consol10 raw / rebuilt | junctions / km² | Δ OD | Δ coverage | Δ street km | Δ > 15 % | Δ block log-area | margins | reuse / tile | evals / tile | s / tile |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in res:
        L.append(f"| {x['arm']} | {x['variant']} | {ci(x['excess'])} | {ci(x['delta_excess'])} | {x['overcorrected_share']:.0%} | "
                 f"{x['raw_close15']:.3f} / {x['reb_close15']:.3f} | {x['reb_p10']:.1f} | {x['raw_consol10']:.3f} / {x['reb_consol10']:.3f} | "
                 f"{x['density']:.0f} | {f(x['d_od_unreach'][0])} | {f(x['d_coverage_60'][0])} | {f(x['d_street_km'][0], 2)} | "
                 f"{f(x['d_share_gt15'][0])} | {f(x['d_block_logA'][0], 2)} | {'ok' if x['margins_ok'] else 'FAIL: ' + ', '.join(k[3:] for k in x if k.startswith('ok_') and not x[k])} | "
                 f"{x['reuse_applied']:.1f} | {x['cand_evals']:.0f} | {x['seconds']:.1f} |")
    L += ["", "## Mechanism: edit classes of the close pairs (RAW final graphs), original vs F1", ""]
    for arm, d in mech.items():
        for lab in ("original", "F1"):
            c = Counter(d[lab])
            n = sum(c.values())
            if n:
                L.append(f"* **{arm} — {lab}** ({n} pairs): " + ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common(5)))
    (OUT / "holdout.md").write_text("\n".join(L) + "\n")
    atomic_json(OUT / "holdout.json", {"rows": res, "mechanism": mech, "real_close15_mean": real_mean, "radius": r})
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
