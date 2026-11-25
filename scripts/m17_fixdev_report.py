"""Milestone 17, step 3 on development data: fixes vs the original, paired (docs/M17_PROTOCOL.md §5, amendment 1).

Original = the diagnosis replay runs (same windows, seeds, K); F1 reuse r in {6, 9, 12, 15}; spacing ablation r in
{9, 12, 15}; F2 closure-gain check (NA); F0 post-hoc topology-aware consolidation of the ORIGINAL graphs at tau in
{6, 9, 12, 15}. Outcomes per (window, seed): REBUILT close15 excess vs the same window's real streets (primary), raw
close15, nn p10, consol10, and the preservation measures with the frozen margins. Applies the F1 radius rule.

    python scripts/m17_fixdev_report.py  ->  outputs/m17/fixdev.{md,json}
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from m17_diagnose import rebuild
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
PACK = ROOT / "data" / "processed" / "m16" / "pack" / "streets"
ARMS = [("M*", "off"), ("NA-fixed", "off"), ("NA-SMC", "off"), ("M*", "organic"), ("M*", "radial")]
MARGINS = {"od_unreach": ("abs", 0.02), "coverage_60": ("rel", -0.05), "street_km": ("rel", -0.10),
           "empty": ("abs", 0.05), "share_gt15": ("abs", 0.02), "block_logA": ("absdiff", 0.25)}


def window_key(site):
    p = site.split("_")
    return p[1] + ("_" + p[2] if p[2] not in ("glo30", "mdt05") else "")


def rows_of(spec, real, terr, posthoc=None):
    r = load_run(spec, RUNS)
    if r is None:
        return None
    out = {}
    for sd, v in r["seeds"].items():
        for k, row in enumerate(v["rows"]):
            g = graph_from_json(v["graphs"][k])
            site = row["site"]
            rec = {"empty": row.get("empty"), "od_unreach": row.get("od_unreach_land"), "coverage_60": row.get("coverage_60"),
                   "street_km": row.get("street_km"), "share_gt15": row.get("share_gt15"),
                   "block_logA": np.log(row["block_area_med"]) if row.get("block_area_med") else np.nan,
                   "reuse_applied": row.get("log", {}).get("reuse_applied", 0), "seconds": v["seconds"] / len(v["rows"])}
            if posthoc:
                g = consolidate_graph(g, posthoc)
                t = terr[site]
                rng = np.random.default_rng([int(sd), 15])
                m = morphology_row(g, t.t, centre=(256.0, 256.0), rng=rng, grade_span=spec.get("grade_span", 10.0))
                m14 = tile_metrics_m14(g, t, rng)
                rec.update(od_unreach=m14.get("od_unreach_land"), coverage_60=m14.get("coverage_60"),
                           street_km=m["street_km"], share_gt15=m.get("share_gt15"),
                           block_logA=np.log(m["block_area_med"]) if m.get("block_area_med") else np.nan,
                           empty=int(len(g.edges) == 0))
            raw, reb = junction_row(g), junction_row(rebuild(g))
            rr = real[window_key(site)]
            rec.update(raw_close15=raw["close15"], reb_close15=reb["close15"], reb_p10=reb["nn_p10"],
                       raw_consol10=raw["consol10_ratio"], n3=reb["n3"], n4=reb["n4"], n5p=reb["n5p"], excess=reb["close15"] - rr, density=reb["focal_density_km2"])
            out[(site, int(sd))] = rec
    return out


def boot(d, rng, B=4000):
    sites = sorted({s for s, _ in d})
    by = {s: [v for (ss, _), v in d.items() if ss == s and np.isfinite(v)] for s in sites}
    by = {s: v for s, v in by.items() if v}
    if not by:
        return np.nan, np.nan, np.nan
    ks = list(by)
    est = float(np.mean([np.mean(by[s]) for s in ks]))
    bs = [np.mean([np.mean(rng.choice(by[s], len(by[s]))) for s in rng.choice(ks, len(ks))]) for _ in range(B)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    rng = np.random.default_rng(17)
    real = {f.stem: junction_row(pickle.loads(f.read_bytes()))["close15"] for f in PACK.glob("*.pkl")
            if "__" not in f.stem and "CAMPUS" not in f.stem}
    terr = {d["site"]: TerrainField(terrain_of(d)) for name in ("named_glo30", "named_mdt05") for d in site_set(name)}
    diag = json.loads((ROOT / "configs" / "m17_plans" / "diagnose.json").read_text())["runs"]
    fix = json.loads((ROOT / "configs" / "m17_plans" / "fixdev.json").read_text())["runs"]
    res, sel = [], {}
    for m, st in ARMS:
        def specs(plan, pred):
            return [s for s in plan if s["method"] == m and s["style"] == st and pred(s.get("overrides", {}))]
        orig_specs = specs(diag, lambda o: set(o) == {"event_log"})
        variants = [("original", orig_specs, None)]
        for r in (6.0, 9.0, 12.0, 15.0):
            variants.append((f"F1 reuse r={r:g}", specs(fix, lambda o, r=r: o.get("junction_reuse_m") == r), None))
        for r in (9.0, 12.0, 15.0):
            variants.append((f"spacing r={r:g} (ablation)", specs(fix, lambda o, r=r: o.get("junction_min_m") == r), None))
        if m.startswith("NA"):
            variants.append(("F2 closure gain 0.25 (check)", specs(fix, lambda o: "closure_gain" in o), None))
        for t in (6.0, 9.0, 12.0, 15.0):
            variants.append((f"F0 post-hoc tau={t:g}", orig_specs, t))
        base = {}
        for sp in orig_specs:
            base.update(rows_of(sp, real, terr) or {})
        for lab, sps, ph in variants:
            cur = {}
            for sp in sps:
                cur.update(rows_of(sp, real, terr, ph) or {})
            if not cur:
                continue
            keys = [k for k in cur if k in base]
            dd = lambda f: {k: f(cur[k]) - f(base[k]) for k in keys}
            ex0 = boot({k: base[k]["excess"] for k in keys}, rng)
            ex = boot({k: cur[k]["excess"] for k in keys}, rng)
            red = boot(dd(lambda x: x["excess"]), rng)
            rec = {"arm": f"{m}" + (f" + {st}" if st != "off" else ""), "variant": lab, "n": len(keys),
                   "excess_original": ex0[0], "excess": ex, "delta_excess": red,
                   "raw_close15": float(np.nanmean([cur[k]["raw_close15"] for k in keys])),
                   "reb_close15": float(np.nanmean([cur[k]["reb_close15"] for k in keys])),
                   "reb_p10": float(np.nanmean([cur[k]["reb_p10"] for k in keys])),
                   "raw_consol10": float(np.nanmean([cur[k]["raw_consol10"] for k in keys])),
                   "density": float(np.nanmean([cur[k]["density"] for k in keys])),
                   "reuse_applied": float(np.mean([cur[k]["reuse_applied"] for k in keys])),
                   "seconds": float(np.mean([cur[k]["seconds"] for k in keys]))}
            ok = True
            for key, (kind, mg) in MARGINS.items():
                b = np.nanmean([base[k][key] for k in keys])
                c = np.nanmean([cur[k][key] for k in keys])
                d = c - b
                rec[f"d_{key}"] = float(d)
                if kind == "abs":
                    ok &= d <= mg
                elif kind == "rel":
                    ok &= (d / b if b else 0.0) >= mg
                else:
                    ok &= abs(d) <= mg
            rec["margins_ok"] = bool(ok)
            res.append(rec)
            print(rec["arm"], lab, f"excess {ex[0]:+.3f} (orig {ex0[0]:+.3f}) d {red[0]:+.3f} [{red[1]:+.3f},{red[2]:+.3f}] margins {ok}", flush=True)
    df = pd.DataFrame(res)
    # F1 radius rule (amendment 1)
    rule = []
    for r in (6.0, 9.0, 12.0, 15.0):
        sub = df[df.variant == f"F1 reuse r={r:g}"]
        if len(sub) < len(ARMS):
            continue
        frac = float(np.mean([-row.delta_excess[0] / row.excess_original if row.excess_original > 0 else 0.0
                              for row in sub.itertuples()]))
        rule.append({"r": r, "mean_reduction_fraction": frac, "all_margins_ok": bool(sub.margins_ok.all())})
    pick = next((x for x in rule if x["mean_reduction_fraction"] >= 0.5 and x["all_margins_ok"]), None)
    if pick is None:
        okr = [x for x in rule if x["all_margins_ok"]]
        pick = max(okr, key=lambda x: (x["mean_reduction_fraction"], -x["r"])) if okr else None
    f = lambda x, n=3: "–" if x is None or not np.isfinite(x) else f"{x:+.{n}f}"
    L = ["# M17 step 3: fixes on development data (terrain pack windows, seeds 1-10; paired with the original)", "",
         "Excess = REBUILT close15 minus the same window's real close15 (primary). Δ = fix − original (two-level bootstrap "
         "over windows, then seeds). Margins (amendment 1): OD +0.02, coverage −5 % rel., street km −10 % rel., empty +0.05, "
         "> 15 % grade +0.02, |Δ block log-area| 0.25.", "",
         "| Arm | Variant | excess (original → fix) | Δ excess [95% CI] | close15 raw | nn p10 rebuilt | consol10 raw | junctions / km² | reuse applied / tile | Δ OD | Δ coverage | Δ street km | Δ > 15 % | margins | s / tile |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['arm']} | {r['variant']} | {r['excess_original']:+.3f} → {r['excess'][0]:+.3f} | {f(r['delta_excess'][0])} "
                 f"[{f(r['delta_excess'][1])}, {f(r['delta_excess'][2])}] | {r['raw_close15']:.3f} | {r['reb_p10']:.1f} | "
                 f"{r['raw_consol10']:.3f} | {r['density']:.0f} | {r['reuse_applied']:.1f} | {f(r['d_od_unreach'])} | "
                 f"{f(r['d_coverage_60'])} | {f(r['d_street_km'], 2)} | {f(r['d_share_gt15'])} | {'ok' if r['margins_ok'] else 'FAIL'} | {r['seconds']:.1f} |")
    L += ["", "## F1 radius rule (amendment 1)", "", "| r | mean reduction of the excess | all margins met |", "|---|---|---|"]
    for x in rule:
        L.append(f"| {x['r']:g} m | {x['mean_reduction_fraction']:.0%} | {'yes' if x['all_margins_ok'] else 'no'} |")
    L.append("")
    L.append(f"**Selected:** {'r = ' + format(pick['r'], 'g') + ' m' if pick else 'none (F1 fails on development data)'}")
    (OUT / "fixdev.md").write_text("\n".join(L) + "\n")
    atomic_json(OUT / "fixdev.json", {"rows": res, "rule": rule, "selected": pick})
    print("\n".join(L[-8:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
