"""Milestone 17 amendment 2: junction-cluster metrics per model and version, with the real streets.

    python scripts/m17_degree_report.py dev       # development: original, F1, F1 + F4 (lambda 0.5 / 1 / 2); lambda rule
    python scripts/m17_degree_report.py holdout   # holdout (EXPLORATORY reuse): original, F1, F0 tau 9, F4, F1 + F4

Per version: REBUILT close15 (window mean) and its excess over the same window's real streets, nn p10, consol10 (raw),
pooled 3- / 4- / 5+-way shares of focal junctions, junctions per km2, and the amendment-1 preservation margins
(paired with the original). The real streets of the same windows are scored identically.
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

from urbanmorph.experiments.jsonio import atomic_json
from urbanmorph.graphs.junctions import junction_row

OUT = ROOT / "outputs" / "m17"
ARMS = [("M*", "off"), ("NA-fixed", "off"), ("NA-SMC", "off"), ("M*", "organic"), ("M*", "radial")]
TRAIN_5P = json.loads((ROOT / "configs" / "m17_degree_prior.json").read_text())["pi"]


def summary(cur: dict, base: dict, margins) -> dict:
    keys = [k for k in cur if k in base]
    n3, n4, n5 = (sum(cur[k][x] for k in keys) for x in ("n3", "n4", "n5p"))
    tot = max(n3 + n4 + n5, 1)
    out = {"n": len(keys), "close15": float(np.nanmean([cur[k]["reb_close15"] for k in keys])),
           "excess": float(np.nanmean([cur[k]["excess"] for k in keys])),
           "p10": float(np.nanmean([cur[k]["reb_p10"] for k in keys])),
           "consol10": float(np.nanmean([cur[k]["raw_consol10"] for k in keys])),
           "share3": n3 / tot, "share4": n4 / tot, "share5p": n5 / tot,
           "density": float(np.nanmean([cur[k]["density"] for k in keys]))}
    ok, fails = True, []
    for key, (kind, mg) in margins.items():
        b = np.nanmean([base[k][key] for k in keys])
        c = np.nanmean([cur[k][key] for k in keys])
        d = c - b
        out[f"d_{key}"] = float(d)
        good = d <= mg if kind == "abs" else ((d / b if b else 0.0) >= mg if kind == "rel" else abs(d) <= mg)
        if not good:
            ok = False
            fails.append(key)
    out["margins_ok"], out["fails"] = ok, fails
    return out


def real_row(graphs) -> dict:
    rows = [junction_row(g) for g in graphs]
    n3, n4, n5 = (sum(r[x] for r in rows) for x in ("n3", "n4", "n5p"))
    tot = max(n3 + n4 + n5, 1)
    return {"close15": float(np.nanmean([r["close15"] for r in rows])), "excess": 0.0,
            "p10": float(np.nanmean([r["nn_p10"] for r in rows])),
            "consol10": float(np.nanmean([r["consol10_ratio"] for r in rows])),
            "share3": n3 / tot, "share4": n4 / tot, "share5p": n5 / tot,
            "density": float(np.nanmean([r["focal_density_km2"] for r in rows])), "margins_ok": True, "fails": []}


def table(res: list[dict], title: str, note: str) -> list[str]:
    f = lambda x, n=3: "–" if x is None or not np.isfinite(x) else f"{x:.{n}f}"
    L = [f"# {title}", "", note, "",
         "| Model | Version | close15 | excess vs real | nn p10 (m) | consol10 | 3-way | 4-way | **5+-way** | junctions / km² | Δ OD | Δ coverage | Δ street km | Δ block log-area | margins |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        d = lambda k, n=3: f"{r[k]:+.{n}f}" if k in r and np.isfinite(r[k]) else ""
        L.append(f"| {r['arm']} | {r['version']} | {f(r['close15'])} | {r['excess']:+.3f} | {f(r['p10'], 1)} | {f(r['consol10'])} | "
                 f"{r['share3']:.1%} | {r['share4']:.1%} | **{r['share5p']:.1%}** | {f(r['density'], 0)} | {d('d_od_unreach')} | "
                 f"{d('d_coverage_60')} | {d('d_street_km', 2)} | {d('d_block_logA', 2)} | "
                 f"{'ok' if r['margins_ok'] else 'FAIL: ' + ', '.join(r['fails'])} |")
    return L


def dev():
    from m17_fixdev_report import MARGINS, rows_of
    from urbanmorph.experiments.sites import site_set
    from urbanmorph.graphs.tc_potentials import TerrainField
    from urbanmorph.graphs.terrain_generate import terrain_of
    PACK = ROOT / "data" / "processed" / "m16" / "pack" / "streets"
    realg = {f.stem: pickle.loads(f.read_bytes()) for f in PACK.glob("*.pkl") if "__" not in f.stem and "CAMPUS" not in f.stem}
    real = {k: junction_row(g)["close15"] for k, g in realg.items()}
    terr = {d["site"]: TerrainField(terrain_of(d)) for name in ("named_glo30", "named_mdt05") for d in site_set(name)}
    diag = json.loads((ROOT / "configs" / "m17_plans" / "diagnose.json").read_text())["runs"]
    fix = json.loads((ROOT / "configs" / "m17_plans" / "fixdev.json").read_text())["runs"]
    deg = json.loads((ROOT / "configs" / "m17_plans" / "degdev.json").read_text())["runs"]
    res = []
    used = sorted({s for s in ("KAIST", "SEL_HBC", "MAD_LAV", "GUA_ESC")})
    res.append({"arm": "REAL streets (same windows)", "version": "reference", **real_row([realg[k] for k in used])})
    by_lam = {}
    for m, st in ARMS:
        arm = m + (f" + {st}" if st != "off" else "")
        sel = lambda plan, pred: [s for s in plan if s["method"] == m and s["style"] == st and pred(s.get("overrides", {}))]
        load = lambda specs: {k: v for sp in specs for k, v in (rows_of(sp, real, terr) or {}).items()}
        base = load(sel(diag, lambda o: set(o) == {"event_log"}))
        vers = [("original", base), ("F1 reuse 15 m", load(sel(fix, lambda o: o.get("junction_reuse_m") == 15.0)))]
        for lam_ in (0.5, 1.0, 2.0):
            vers.append((f"F1 + F4 lambda {lam_:g}", load(sel(deg, lambda o, l=lam_: o.get("degree_prior") == l))))
        f1 = None
        for lab, cur in vers:
            if not cur:
                continue
            row = {"arm": arm, "version": lab, **summary(cur, base, MARGINS)}
            res.append(row)
            if lab.startswith("F1 reuse"):
                f1 = row
            if lab.startswith("F1 + F4"):
                by_lam.setdefault(float(lab.split()[-1]), []).append((row, f1))
    rule = []
    target = 2 * (TRAIN_5P["5"] + TRAIN_5P["6"] + TRAIN_5P["7"] + TRAIN_5P["8"])
    for lam_, rows in sorted(by_lam.items()):
        s5 = float(np.mean([r["share5p"] for r, _ in rows]))
        ok = all(r["margins_ok"] for r, _ in rows) and all(r["excess"] - f["excess"] <= 0.02 for r, f in rows if f)
        rule.append({"lambda": lam_, "mean_share5p": s5, "conditions_ok": bool(ok and s5 <= target), "margins_ok": bool(ok)})
    pick = next((x for x in rule if x["conditions_ok"]), None) or min((x for x in rule if x["margins_ok"]),
                                                                     key=lambda x: x["mean_share5p"], default=None)
    L = table(res, "M17 amendment 2 on development data: junction clusters per model (terrain pack windows, seeds 1-10)",
              "REBUILT graphs, focal junctions (>= 40 m inside the window). Shares pooled over junctions. Δ = version − original.")
    L += ["", f"λ rule: mean 5+-way share ≤ 2 × training ({target:.1%}), all margins, excess ≤ F1 + 0.02.", "",
          "| λ | mean 5+-way share | all conditions |", "|---|---|---|"]
    L += [f"| {x['lambda']:g} | {x['mean_share5p']:.1%} | {'yes' if x['conditions_ok'] else 'no'} |" for x in rule]
    L += ["", f"**Selected λ = {pick['lambda']:g}**" if pick else "**No λ meets the margins.**"]
    (OUT / "degdev.md").write_text("\n".join(L) + "\n")
    atomic_json(OUT / "degdev.json", {"rows": res, "rule": rule, "selected_lambda": pick["lambda"] if pick else None})
    print("\n".join(L))


def holdout():
    from m17_fixdev_report import MARGINS
    from m17_holdout_report import collect
    from urbanmorph.experiments.sites import site_set
    from urbanmorph.graphs.tc_potentials import TerrainField
    from urbanmorph.graphs.terrain_generate import terrain_of
    HOLD = ROOT / "data" / "processed" / "m17" / "holdout"
    man = json.loads((ROOT / "configs" / "m17_holdout_windows.json").read_text())["windows"]
    realg = {w["window"]: pickle.loads((HOLD / "streets" / f"{w['window']}.pkl").read_bytes()) for w in man}
    real = {w: junction_row(g)["close15"] for w, g in realg.items()}
    terr = {d["site"]: TerrainField(terrain_of(d)) for d in site_set("holdout_m17")}
    plan = json.loads((ROOT / "configs" / "m17_plans" / "holdout.json").read_text())["runs"]
    plan2 = json.loads((ROOT / "configs" / "m17_plans" / "degholdout.json").read_text())["runs"]
    lam_ = json.loads((OUT / "degdev.json").read_text())["selected_lambda"] or 1.0      # exploratory (amendment 2)
    res = [{"arm": "REAL streets (54 holdout windows)", "version": "reference", **real_row(list(realg.values()))}]
    for m, st in ARMS:
        arm = m + (f" + {st}" if st != "off" else "")
        sp = lambda p, pred: [s for s in p if s["method"] == m and s["style"] == st and pred(s["overrides"])][0]
        base = collect(sp(plan, lambda o: set(o) == {"event_log"}), real, terr)
        vers = [("original", base), ("F1 reuse 15 m", collect(sp(plan, lambda o: "junction_reuse_m" in o), real, terr)),
                ("F0 post-hoc 9 m", collect(sp(plan, lambda o: set(o) == {"event_log"}), real, terr, posthoc=9.0)),
                (f"F4 degree prior λ {lam_:g}", collect(sp(plan2, lambda o: "junction_reuse_m" not in o), real, terr)),
                (f"F1 + F4 (λ {lam_:g})", collect(sp(plan2, lambda o: "junction_reuse_m" in o), real, terr))]
        for lab, cur in vers:
            res.append({"arm": arm, "version": lab, **summary(cur, base, MARGINS)})
            print(arm, lab, f"5+ {res[-1]['share5p']:.3f} excess {res[-1]['excess']:+.3f}", flush=True)
    L = table(res, "M17 amendment 2 on the holdout (EXPLORATORY reuse of the holdout): junction clusters per model",
              "9 fresh metros × 6 windows, seeds 101-103; REBUILT graphs, focal junctions; shares pooled. Δ = version − original.")
    (OUT / "degholdout.md").write_text("\n".join(L) + "\n")
    atomic_json(OUT / "degholdout.json", {"rows": res, "lambda": lam_})
    print("\n".join(L))


if __name__ == "__main__":
    {"dev": dev, "holdout": holdout}[sys.argv[1]]()
