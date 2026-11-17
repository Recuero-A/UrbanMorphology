"""Milestone 14 report: tables, hypothesis tests and figures from the stored runs (never generates).

    python scripts/m14_report.py       # -> outputs/m14/REPORT.md, outputs/m14/report.json, outputs/m14/figures/*

Every number is computed from outputs/m14/runs by scripts/m14_analyze.py functions. Missing runs are listed as such.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import m14_analyze as ma  # noqa: E402
from m14_plans import methods  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m14"
FIG = OUT / "figures"
FZ = json.loads((PROJECT_ROOT / "configs" / "m14_frozen.json").read_text())
LAM = FZ["lambda"]
TN = ("TN_s0", "TN_s1", "TN_s2")
SETS = {"gen2": "GEN2 (primary: Alcoi, Tarragona, Santa Barbara, Sacramento)", "devB": "DEV-B (secondary)",
        "gen": "GEN (secondary: Girona, Manresa, San Francisco)"}
CITY = {"ESP_ALY": "Alcoi", "ESP_TGN": "Tarragona", "USA_SBA": "Santa Barbara", "USA_SAC": "Sacramento",
        "ESP_GRO": "Girona", "ESP_MRS": "Manresa", "USA_SFO": "San Francisco"}
COLS = [("od_unreach_land", "OD unreach. (land)", "{:.3f}"), ("od_unreach_land_nonempty", "OD unreach. non-empty", "{:.3f}"),
        ("empty_share", "empty", "{:.3f}"), ("islands", "islands / tile", "{:.2f}"),
        ("dead_end_ratio_m13", "dead-end ratio", "{:.3f}"), ("closed_giant", "closed giant", "{:.3f}"),
        ("S10_cond", "S10", "{:.3f}"), ("S_T", "S_T", "{:.3f}"), ("density_ratio", "density / real", "{:.2f}"),
        ("premature_share", "premature", "{:.2f}"), ("river_crossings_unbridged", "unbridged / tile", "{:.2f}"),
        ("bridges", "bridges / tile", "{:.2f}"), ("coverage_60", "coverage 60 m", "{:.3f}"),
        ("od_detour", "detour", "{:.2f}"), ("energy_distance", "energy dist.", "{:.2f}"),
        ("jaccard_diversity", "Jaccard div.", "{:.3f}"), ("sec_per_tile_sample", "s / tile", "{:.2f}")]


def get(tiles, bb, label, seeds_any=True):
    for lab, p, ov, ex in methods(FZ):
        if lab == label:
            bb2 = "TN_s0" if p == "P" else bb
            try:
                return ma.find(tiles, bb2, p, ov, **ex)
            except FileNotFoundError:
                return None
    raise KeyError(label)


def complete(res, n_seeds=3):
    return res is not None and len(res["seeds"]) >= n_seeds


def fmt_cell(vals, f):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "—"
    return f.format(v.mean()) + (f" [{f.format(v.min())}, {f.format(v.max())}]" if len(v) > 1 else "")


def method_table(tiles, labels, bbs=TN, n_seeds=3):
    ref = ma.real(tiles)
    rows, summ = [], {}
    for lab in labels:
        ss = []
        for b in bbs:
            r = get(tiles, b, lab)
            if complete(r, n_seeds):
                ss.append(ma.summary(r, ref))
            if lab == "P":
                break
        summ[lab] = ss
    hdr = "| method | " + " | ".join(c[1] for c in COLS) + " |"
    md = [hdr, "|" + "---|" * (len(COLS) + 1)]
    for lab in labels:
        ss = summ[lab]
        if not ss:
            md.append(f"| {lab} | " + " | ".join("not run" for _ in COLS) + " |")
            continue
        md.append(f"| {lab} | " + " | ".join(fmt_cell([s.get(k) for s in ss], f) for k, _, f in COLS) + " |")
    s0 = next((summ[l][0] for l in labels if summ[l]), None)
    if s0:
        real = {k: s0.get(k + "_real") for k, _, _ in COLS}
        real.update(density_ratio=1.0)
        md.append("| real | " + " | ".join(f.format(real[k]) if real.get(k) is not None and np.isfinite(real[k])
                                           else "—" for k, _, f in COLS) + " |")
    return md, summ


def pairs(tiles, a, b, bbs=TN, n_seeds=3):
    out = []
    for bb in bbs:
        ra, rb = get(tiles, bb, a), get(tiles, bb, b)
        if complete(ra, n_seeds) and complete(rb, n_seeds):
            out.append((ra, rb))
    return out


def ctext(c, key="delta"):
    p = c.get("p_signflip", c.get("p_perm"))
    return f"Δ {c[key]:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}], p = {p:.4f}" + \
        (f" (min attainable {c['min_p']:.4f}, {c['n_blocks']} blocks)" if "min_p" in c else "")


def hypotheses(tiles, summ):
    """Pre-registered tests on one tile set. Returns (markdown lines, dict)."""
    md, res = [], {}
    ref = ma.real(tiles)
    P = lambda a, b: pairs(tiles, a, b)
    # H1
    pr = P("M*", "NA-fixed")
    if pr:
        c = ma.block_contrast(pr, "od_unreach_land")
        s10 = ma.set_contrast(pr, "S10_cond", ref)
        dens = np.mean([x["density_ratio"] for x in summ.get("NA-fixed", [])]) if summ.get("NA-fixed") else np.nan
        guard = s10["delta"] <= 0.05 and abs(dens - 1) <= 0.25
        res["H1"] = {"od_unreach_land": c, "S10_cond": s10, "density_ratio": dens, "guard": guard}
        verdict = ("**supported**" if guard else "connectivity bought with realism (guard failed)") if c["hi"] < 0 \
            else "not supported"
        md.append(f"| H1 | NA-fixed − M\\*: OD unreach. (land) | {ctext(c)} | S10: {ctext(s10)}; density/real {dens:.2f} | "
                  f"{verdict} |")
    # H3
    pr_af, pr_ar = P("NA-fixed", "NA-adaptive"), P("NA-random", "NA-adaptive")
    if pr_af:
        c = ma.block_contrast(pr_af, "od_unreach_land")
        s10 = ma.set_contrast(pr_af, "S10_cond", ref)
        sec = {l: np.mean([s["sec_per_tile_sample"] for s in summ.get(l, [])]) if summ.get(l) else np.nan
               for l in ("NA-fixed", "NA-adaptive")}
        a_ok = c["hi"] <= 0.010
        b_ok = s10["hi"] <= 0.02
        c_ok = sec["NA-adaptive"] <= 0.7 * sec["NA-fixed"]
        res["H3_vs_fixed"] = {"od_unreach_land": c, "S10_cond": s10, "sec": sec, "a": a_ok, "b": b_ok, "c": c_ok}
        md.append(f"| H3 (a, b, c) | NA-adaptive − NA-fixed | OD: {ctext(c)} (margin +0.010: {'met' if a_ok else 'not met'}) | "
                  f"S10: {ctext(s10)} (margin +0.02: {'met' if b_ok else 'not met'}); cost {sec['NA-adaptive']:.2f} vs "
                  f"{sec['NA-fixed']:.2f} s ({'≤ 0.7×' if c_ok else '> 0.7×'}) | {'met' if a_ok and b_ok and c_ok else 'not met'} |")
    if pr_ar:
        c = ma.block_contrast(pr_ar, "od_unreach_land")
        res["H3_vs_random"] = {"od_unreach_land": c}
        md.append(f"| H3 (vs random) | NA-adaptive − NA-random | {ctext(c)} | — | "
                  f"{'**supported**' if c['hi'] < 0 else 'not supported'} |")
    # H4
    if summ.get("NA-adaptive") and summ.get("A"):
        ra = np.mean([s["jaccard_diversity"] for s in summ["NA-adaptive"]]) / np.mean([s["jaccard_diversity"] for s in summ["A"]])
        res["H4"] = {"ratio": float(ra)}
        md.append(f"| H4 | Jaccard diversity NA-adaptive / A | ratio {ra:.3f} | — | {'**met** (≥ 0.9)' if ra >= 0.9 else 'not met'} |")
    return md, res


SEC_KEYS = ("od_unreach_land", "od_unreach_land_nonempty", "dead_end_ratio_m13", "islands", "S10_cond", "S_T",
            "energy_distance", "density_ratio")


def any_run(tiles, bb, pre, ov, n_seeds=2, **extra):
    try:
        r = ma.find(tiles, bb, pre, ov, **extra)
    except FileNotFoundError:
        return None
    if len(r["seeds"]) < n_seeds:
        return None
    # a 3-seed confirm run standing in for a 2-seed ablation is cut to the same seeds, so the rows stay comparable
    # (its Jaccard diversity is still the mean pairwise distance over all stored samples)
    keep = sorted(r["seeds"], key=int)[:n_seeds]
    return {**r, "seeds": {s: r["seeds"][s] for s in keep}}


def secondary(tiles, summ):
    """Secondary contrasts (no decisions): method - M* (and NA-fixed - M13-SMC), per-tile keys by block sign-flip."""
    md = [f"\n### Secondary contrasts on {tiles.upper()} (method − reference; block sign-flip p; CI = block bootstrap)\n",
          "| contrast | OD unreach. (land) | dead-end ratio | islands |", "|---|---|---|---|"]
    out = {}
    for a, b in (("M*", "NA-myopic"), ("M*", "NA-fixed"), ("M*", "NA-adaptive"), ("M*", "NA-SMC"), ("M*", "M*+repair"),
                 ("M13-SMC", "NA-SMC"), ("NA-SMC", "NA-fixed"), ("A", "M*")):
        pr = pairs(tiles, a, b)
        if not pr:
            md.append(f"| {b} − {a} | not run | | |")
            continue
        cs = {k: ma.block_contrast(pr, k) for k in ("od_unreach_land", "dead_end_ratio_m13", "islands")}
        out[f"{b} - {a}"] = cs
        md.append(f"| {b} − {a} | " + " | ".join(f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] p={c['p_signflip']:.3f}"
                                               for c in cs.values()) + " |")
    md += ["", "| contrast (set level) | S10 | S_T | energy distance |", "|---|---|---|---|"]
    ref = ma.real(tiles)
    for a, b in (("M*", "NA-fixed"), ("M*", "NA-SMC"), ("M*", "NA-myopic"), ("M*", "NA-adaptive"), ("M*", "M*+repair")):
        pr = pairs(tiles, a, b)
        if not pr:
            continue
        cs = {k: ma.set_contrast(pr, k, ref) for k in ("S10_cond", "S_T")}
        ed = [ma.summary(rb, ref)["energy_distance"] - ma.summary(ra, ref)["energy_distance"] for ra, rb in pr]
        out[f"{b} - {a} (set)"] = {**cs, "energy_distance_per_seed": ed}
        md.append(f"| {b} − {a} | " + " | ".join(f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] p={c['p_perm']:.3f}"
                                               for c in cs.values()) + f" | {np.mean(ed):+.3f} (seeds {', '.join(f'{x:+.2f}' for x in ed)}) |")
    return md, out


def per_city(tiles, labels=("A", "M*", "NA-myopic", "NA-adaptive", "NA-fixed", "NA-SMC")):
    ref = ma.real(tiles)
    cities = sorted({r["sid"][:7] for r in ref})
    md = [f"\n### Per city ({tiles.upper()}; TN, pooled over training and inference seeds)\n",
          "| city | tiles | metric | real | " + " | ".join(labels) + " |", "|---|---|---|---|" + "---|" * len(labels)]
    out = {}
    for c in cities:
        idx = [i for i, r in enumerate(ref) if r["sid"].startswith(c)]
        rr = [ref[i] for i in idx]
        vals = {}
        for l in labels:
            ss = []
            for b in TN:
                r = get(tiles, b, l)
                if complete(r):
                    g = ma.tile_means(r)
                    gg = [g[i] for i in idx]
                    ne = [x["rows"][i]["od_unreach_land"] for x in r["seeds"].values() for i in idx
                          if x["rows"][i]["street_km_per_km2"] > 0]
                    ss.append({"od_ne": np.nanmean(ne) if ne else np.nan,
                               "dead": np.nanmean([x["dead_end_ratio_m13"] for x in gg]),
                               "S10": ma.an.primary(gg, rr)["S10_cond"],
                               "dens": ma.an.primary(gg, rr)["density_ratio"]})
            vals[l] = {k: float(np.nanmean([x[k] for x in ss])) if ss else np.nan for k in ("od_ne", "dead", "S10", "dens")}
        realv = {"od_ne": np.nanmean([r["od_unreach_land"] for r in rr]), "dead": np.nanmean([r["dead_end_ratio_m13"] for r in rr]),
                 "S10": np.nan, "dens": 1.0}
        out[c] = {"n": len(idx), "real": realv, **vals}
        for k, lab in (("od_ne", "OD unreach. non-empty"), ("dead", "dead-end ratio"), ("S10", "S10"), ("dens", "density / real")):
            md.append(f"| {CITY.get(c, c)} | {len(idx)} | {lab} | " + ("—" if not np.isfinite(realv[k]) else f"{realv[k]:.3f}") + " | "
                      + " | ".join(f"{vals[l][k]:.3f}" if np.isfinite(vals[l][k]) else "—" for l in labels) + " |")
    return md, out


def strata_table(labels=("M*", "NA-myopic", "NA-adaptive", "NA-fixed", "NA-SMC")):
    """Pre-registered terrain strata, pooled over GEN2 + DEV-B + GEN (TN, all seeds)."""
    import m14_experiments as mx
    rows = []
    for tiles in ("gen2", "devB", "gen"):
        T = mx.tile_set(tiles)
        st = ma.tile_strata(T)
        ref = ma.real(tiles)
        runs = {l: [get(tiles, b, l) for b in TN] for l in labels}
        for k, t in enumerate(T):
            s = ref[k]["slope_mean_ctx"]
            cls = ["flat" if s < 0.05 else "moderate" if s < 0.10 else "steep" if s < 0.15 else "mountainous"]
            if st[t["sid"]]["river"]:
                cls.append("river")
            if st[t["sid"]]["coastal"]:
                cls.append("coastal")
            if st[t["sid"]]["water_any"] >= 0.01 and s >= 0.10:
                cls.append("mixed")
            row = {"strata": cls, "real_dead": ref[k]["dead_end_ratio_m13"], "real_dens": ref[k]["street_km_per_km2"],
                   "real_unbr": ref[k]["river_crossings_unbridged"]}
            for l in labels:
                vals = [x["rows"][k] for r in runs[l] if complete(r) for x in r["seeds"].values()]
                if vals:
                    ne = [v["od_unreach_land"] for v in vals if v["street_km_per_km2"] > 0]
                    row[l] = {"od_ne": np.nanmean(ne) if ne else np.nan, "dead": np.nanmean([v["dead_end_ratio_m13"] for v in vals]),
                              "dens": np.mean([v["street_km_per_km2"] for v in vals]),
                              "unbr": np.mean([v["river_crossings_unbridged"] for v in vals])}
            rows.append(row)
    md = ["\n## Terrain strata (GEN2 + DEV-B + GEN pooled; TN; descriptive; each stratum is defined from the real terrain)\n",
          "| stratum | tiles | metric | real | " + " | ".join(labels) + " |", "|---|---|---|---|" + "---|" * len(labels)]
    out = {}
    for sname in ("flat", "moderate", "steep", "mountainous", "river", "coastal", "mixed"):
        R = [r for r in rows if sname in r["strata"]]
        if not R:
            md.append(f"| {sname} | 0 | — | — |" + " — |" * len(labels))
            continue
        out[sname] = {"n": len(R)}
        for k, lab, realk in (("od_ne", "OD unreach. non-empty", None), ("dead", "dead-end ratio", "real_dead"),
                              ("dens", "density km/km²", "real_dens"), ("unbr", "unbridged / tile", "real_unbr")):
            rv = np.nanmean([r[realk] for r in R]) if realk else np.nan
            vs = {l: np.nanmean([r[l][k] for r in R if l in r]) for l in labels}
            out[sname][k] = {"real": rv, **vs}
            md.append(f"| {sname} | {len(R)} | {lab} | " + (f"{rv:.3f}" if np.isfinite(rv) else "—") + " | "
                      + " | ".join(f"{vs[l]:.3f}" for l in labels) + " |")
    return md, out


def h5_section():
    md = ["\n## H5: leakage controls on GEN2 (TNlp = low-passed DEM, TNse = street-erased DEM; inference seeds 1–2)\n",
          "| backbone | contrast | OD unreach. (land) | OD non-empty (per seed mean) | dead-end ratio | S10 |", "|---|---|---|---|---|---|"]
    out, ps = {}, {}
    fz = FZ
    from m14_plans import lam as L
    spec = {"M*": ("energy", {"K": 8, "lam_ratio": 0.5}), "NA-myopic": ("na_myopic", L()),
            "NA-adaptive": ("na_adaptive", {**L(), "trigger": fz["trigger"]}), "A": ("standard", {})}
    for fam in ("TNlp", "TNse"):
        for lab in ("NA-adaptive", "NA-myopic"):
            per = []
            for s in (0, 1, 2):
                ra = any_run("gen2", f"{fam}_s{s}", *spec["M*"], potentials=fam)
                rb = any_run("gen2", f"{fam}_s{s}", *spec[lab], potentials=fam)
                if ra is None or rb is None:
                    continue
                c = ma.block_contrast([(ra, rb)], "od_unreach_land")
                d = ma.block_contrast([(ra, rb)], "dead_end_ratio_m13")
                s10 = ma.set_contrast([(ra, rb)], "S10_cond", ma.real("gen2"))
                sa, sb = ma.summary(ra), ma.summary(rb)
                per.append({"seed": s, "od": c, "dead": d, "S10": s10,
                            "od_ne": (sa["od_unreach_land_nonempty"], sb["od_unreach_land_nonempty"])})
                md.append(f"| {fam}_s{s} | {lab} − M\\* | {ctext(c)} | {sa['od_unreach_land_nonempty']:.4f} → "
                          f"{sb['od_unreach_land_nonempty']:.4f} | {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | "
                          f"{s10['delta']:+.3f} [{s10['lo']:+.3f}, {s10['hi']:+.3f}] |")
            out[f"{fam} {lab}"] = per
            if lab == "NA-adaptive" and per:
                n_ok = sum(1 for x in per if x["od"]["hi"] < 0)
                ps[f"H5-{fam}"] = float(np.median([x["od"]["p_signflip"] for x in per]))
                md.append(f"| {fam} | H5 verdict ({lab}) | CI < 0 in {n_ok} of {len(per)} seeds | | | "
                          f"{'**supported**' if n_ok >= 2 and len(per) == 3 else 'not supported'} |")
    return md, out, ps


def ablation_section():
    from m14_plans import lam as L
    base = {**L(), "trigger": FZ["trigger"]}
    variants = [("frozen NA-adaptive", "na_adaptive", base),
                ("island only (λ_I = 1, heuristic)", "na_adaptive", {**base, "lam_dead": 0.0, "lam_island": 1.0}),
                ("dead end + island (λ_I = 1)", "na_adaptive", {**base, "lam_island": 1.0}),
                ("terrain potentials off", "na_adaptive", {**base, "soft": False}),
                ("H = 1", "na_adaptive", {**base, "horizon": 1}), ("H = 4", "na_adaptive", {**base, "horizon": 4}),
                ("K = 4", "na_adaptive", {**base, "K": 4}), ("K = 16", "na_adaptive", {**base, "K": 16}),
                ("trust region ε = 0.05", "na_myopic", {**L(), "kl_eps": 0.05}),
                ("trust region ε = 0.2", "na_myopic", {**L(), "kl_eps": 0.2}),
                ("λ × 0 (= M*)", "energy", {"K": 8, "lam_ratio": 0.5}), ("λ × 0.5 (myopic)", "na_myopic", L(0.5)),
                ("λ × 1 (myopic)", "na_myopic", L(1.0)), ("λ × 2 (myopic)", "na_myopic", L(2.0)),
                ("λ × 4 (myopic)", "na_myopic", L(4.0))]
    for f in ("island", "deg1", "water", "steep", "feas_low"):
        variants.append((f"trigger = {f}", "na_adaptive", {**L(), "trigger": f}))
    variants.append(("trigger = random (same rate)", "na_adaptive", {**L(), "trigger": f"random:{FZ['fire_share']:.4f}"}))
    variants.append(("roll-outs everywhere (NA-fixed)", "na_fixed", L()))
    keys = [("od_unreach_land_nonempty", "{:.4f}"), ("od_unreach_land", "{:.3f}"), ("dead_end_ratio_m13", "{:.3f}"),
            ("islands", "{:.2f}"), ("S10_cond", "{:.3f}"), ("S_T", "{:.3f}"), ("energy_distance", "{:.2f}"),
            ("density_ratio", "{:.2f}"), ("jaccard_diversity", "{:.3f}"), ("sec_per_tile_sample", "{:.2f}"),
            ("la_fire_share", "{:.2f}")]
    md = ["\n## Ablations (TN_s0, inference seeds 1–2; descriptive, never used to change the frozen configuration)\n"]
    out = {}
    for tiles in ("devA", "devB"):
        md += [f"\n### {tiles.upper()}\n", "| variant | " + " | ".join(k for k, _ in keys) + " |", "|---|" + "---|" * len(keys)]
        for lab, pre, ov in variants:
            r = any_run(tiles, "TN_s0", pre, ov)
            if r is None and tiles == "devB" and pre in ("energy", "na_myopic", "na_fixed", "na_adaptive"):
                try:                                      # confirm runs (3 seeds) stand in for the same configuration
                    r = ma.find(tiles, "TN_s0", pre, ov)
                except FileNotFoundError:
                    r = None
            if r is None:
                md.append(f"| {lab} | " + " | ".join("not run" for _ in keys) + " |")
                continue
            sm = ma.summary(r)
            out[f"{tiles} {lab}"] = {k: sm.get(k) for k, _ in keys}
            md.append(f"| {lab} | " + " | ".join(f.format(sm[k]) if sm.get(k) is not None and np.isfinite(sm[k]) else "—"
                                               for k, f in keys) + " |")
    return md, out


def shift_section():
    labels = ("A", "M*", "NA-myopic", "NA-adaptive")
    md = ["\n## Synthetic terrain shifts (TN_s0, seeds 1–2; robustness only, real streets are a mismatched reference)\n",
          "| set | method | OD unreach. (land) | OD non-empty | dead-end ratio | unbridged / tile | density / real | S10 | S_T |",
          "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for tiles in ("devB@x2", "devB@rough", "gen2@x2"):
        for l in labels:
            spec = [x for x in methods(FZ) if x[0] == l][0]
            r = any_run(tiles, "TN_s0", spec[1], spec[2], **spec[3])
            if r is None:
                md.append(f"| {tiles} | {l} | not run |||||||")
                continue
            s = ma.summary(r)
            out[f"{tiles} {l}"] = s
            md.append(f"| {tiles} | {l} | {s['od_unreach_land']:.3f} | {s['od_unreach_land_nonempty']:.4f} | "
                      f"{s['dead_end_ratio_m13']:.3f} | {s['river_crossings_unbridged']:.2f} | {s['density_ratio']:.2f} | "
                      f"{s['S10_cond']:.3f} | {s['S_T']:.3f} |")
    return md, out


def cost_section(summ_all):
    md = ["\n## Cost (GEN2, TN; seconds measured under parallel load, so approximate; forward passes and roll-out steps "
          "are load-independent)\n",
          "| method | s / tile | forward passes / tile | candidate evals / tile | roll-out steps / tile | roll-out share | km / s | mean ESS | peak RSS MB |",
          "|---|---|---|---|---|---|---|---|---|"]
    for l, ss in summ_all.items():
        if not ss:
            continue
        g = lambda k: np.nanmean([s.get(k, np.nan) if s.get(k) is not None else np.nan for s in ss])
        md.append(f"| {l} | {g('sec_per_tile_sample'):.2f} | {g('fwd_per_tile_sample'):.0f} | {g('cand_evals_per_tile_sample'):.0f} | "
                  f"{g('lookahead_steps_per_tile_sample'):.0f} | {g('la_fire_share'):.2f} | {g('km_per_second'):.2f} | "
                  f"{g('mean_ess'):.2f} | {g('peak_rss_mb'):.0f} |")
    return md


def diversity_section(summ_all):
    md = ["\n## Diversity (GEN2, TN; between the 3 inference samples of each tile)\n",
          "| method | Jaccard distance | orientation L1 | CV density | CV closed giant | CV dead-end ratio | S10 |", "|---|---|---|---|---|---|---|"]
    for l, ss in summ_all.items():
        if not ss:
            continue
        g = lambda f: np.nanmean([f(s) for s in ss])
        md.append(f"| {l} | {g(lambda s: s.get('jaccard_diversity', np.nan)):.3f} | {g(lambda s: s.get('orientation_diversity', np.nan)):.3f} | "
                  f"{g(lambda s: s['across_seed_cv']['street_km_per_km2']):.3f} | {g(lambda s: s['across_seed_cv']['closed_giant']):.3f} | "
                  f"{g(lambda s: s['across_seed_cv']['dead_end_ratio_m13']):.3f} | {g(lambda s: s['S10_cond']):.3f} |")
    return md


def density_section(tiles="gen2"):
    md = [f"\n## Density adjustment ({tiles.upper()}; per-tile difference regressed on the log-density difference; "
          "intercept = difference at equal density; block bootstrap)\n",
          "| contrast | metric | unadjusted Δ | density-adjusted Δ [95 % CI] | slope |", "|---|---|---|---|---|"]
    out = {}
    for a, b in (("M*", "NA-fixed"), ("M*", "NA-SMC"), ("M*", "NA-adaptive"), ("M*", "NA-myopic")):
        pr = pairs(tiles, a, b)
        if not pr:
            continue
        for k in ("od_unreach_land", "dead_end_ratio_m13"):
            u = ma.block_contrast(pr, k)
            d = ma.density_adjusted(pr, k)
            out[f"{b} - {a} {k}"] = {"unadjusted": u, "adjusted": d}
            md.append(f"| {b} − {a} | {k} | {u['delta']:+.4f} | {d['intercept']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | {d['slope']:+.3f} |")
    return md, out


def dm_find(tiles, bb, label, bias):
    for lab, p, ov, ex in methods(FZ):
        if lab == label:
            return any_run(tiles, bb, p, {**ov, "stop_bias": bias}, **ex)
    raise KeyError(label)


def dm_section(tiles="gen2"):
    """Density-matched runs: stop bias bisected on DEV-A (configs/m14_density.json), applied unchanged to GEN2."""
    dm = json.loads((PROJECT_ROOT / "configs" / "m14_density.json").read_text())["stop_bias"]
    ref = ma.real(tiles)
    md = [f"\n## Density-matched comparison ({tiles.upper()}; stop bias fitted on DEV-A to density ratio 1 ± 0.05, "
          "not re-fitted on GEN2; TN_s0–s2, inference seeds 1–2)\n",
          "| method | stop bias | " + " | ".join(c[1] for c in COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    runs, out = {}, {"summaries": {}, "contrasts": {}}
    for lab, bias in dm.items():
        rs = [dm_find(tiles, b, lab, bias) for b in TN]
        runs[lab] = [r for r in rs if r is not None]
        ss = [ma.summary(r, ref) for r in runs[lab]]
        out["summaries"][lab] = ss
        md.append(f"| {lab} (matched) | {bias:+.4f} | " + (" | ".join(fmt_cell([s.get(k) for s in ss], f) for k, _, f in COLS)
                                                          if ss else "not run") + " |")
    md += ["\n| contrast (matched) | OD unreach. (land) | dead-end ratio | islands | S10 |", "|---|---|---|---|---|"]
    for a, b in (("M*", "NA-SMC"), ("M*", "NA-myopic"), ("A", "NA-SMC")):
        pr = list(zip(runs.get(a, []), runs.get(b, [])))
        if len(pr) < len(TN):
            md.append(f"| {b} − {a} | incomplete | | | |")
            continue
        cs = {k: ma.block_contrast(pr, k) for k in ("od_unreach_land", "dead_end_ratio_m13", "islands")}
        cs["S10_cond"] = ma.set_contrast(pr, "S10_cond", ref)
        out["contrasts"][f"{b} - {a}"] = cs
        md.append(f"| {b} − {a} | " + " | ".join(ctext(cs[k]) for k in cs) + " |")
    return md, out


def h2_section():
    """Pre-registered H2 (protocol §1, §5.4): DEV-B dose-response of S10_cond over the lambda multiplier."""
    from m14_plans import lam as L
    floor = ma.noise_floor(ma.real("devB"))["median"]
    ms = (0.0, 0.5, 1.0, 2.0, 4.0)
    s10 = []
    for m in ms:
        pre, ov = ("energy", {"K": 8, "lam_ratio": 0.5}) if m == 0 else ("na_myopic", L(m))
        r = any_run("devB", "TN_s0", pre, ov)
        s10.append(ma.summary(r)["S10_cond"] if r is not None else np.nan)
    if not np.isfinite(s10).all():
        return ["\n## H2 (DEV-B dose-response)\n", "not run"], {}
    rise = s10[-1] - s10[0]
    steps = int(sum(b >= a for a, b in zip(s10, s10[1:])))
    ok = rise > floor and steps >= 3
    md = ["\n## H2: planning strength vs realism (DEV-B, TN_s0, seeds 1–2; m = 0 is M\\*)\n",
          "| m | " + " | ".join(f"{m:g}" for m in ms) + " |", "|---|" + "---|" * len(ms),
          "| S10_cond | " + " | ".join(f"{v:.3f}" for v in s10) + " |",
          f"\nS10(4) − S10(0) = {rise:+.3f} vs the DEV-B real-vs-real median {floor:.3f}; non-decreasing in {steps} of 4 "
          f"steps. H2 verdict: {'**supported**' if ok else 'not supported'}"
          + ("" if ok else (" (rise below the noise floor)" if rise <= floor else " (trend not monotone)")) + "."]
    return md, {"m": ms, "S10": s10, "rise": rise, "floor": floor, "steps_nondecreasing": steps, "supported": ok}


CONSTRAINED = ("M*", "M*+repair", "NA-myopic", "NA-adaptive", "NA-random", "NA-fixed", "NA-SMC", "M13-SMC")


def failure_section(sets, shifts):
    """Protocol §9 flags: premature share > A's + 0.10; density ratio outside [0.6, 1.6]; hard-constraint violation
    (unbridged river crossings > 0 for a method that enforces the hard constraints). Flagged results are kept."""
    md = ["\n## Failure flags (protocol §9; flagged results are reported, never dropped)\n",
          "| set | method | premature (A) | density / real | unbridged / tile | flags |", "|---|---|---|---|---|---|"]
    out = {}

    def row(tag, lab, prem, prem_a, dens, unbr):
        fl = []
        if np.isfinite(prem_a) and prem > prem_a + 0.10:
            fl.append("premature")
        if not 0.6 <= dens <= 1.6:
            fl.append("density")
        if lab in CONSTRAINED and unbr > 0:
            fl.append("hard constraint")
        out[f"{tag} {lab}"] = fl
        if fl:
            md.append(f"| {tag} | {lab} | {prem:.3f} ({prem_a:.3f}) | {dens:.2f} | {unbr:.3f} | {', '.join(fl)} |")

    for tiles, summ in sets.items():
        pa = np.mean([x["premature_share"] for x in summ["A"]]) if summ.get("A") else np.nan
        for lab, ss in summ.items():
            if ss:
                row(tiles, lab, np.mean([x["premature_share"] for x in ss]), pa,
                    np.mean([x["density_ratio"] for x in ss]), np.mean([x["river_crossings_unbridged"] for x in ss]))
    for key, s in shifts.items():
        tiles, lab = key.split(" ", 1)
        a = shifts.get(f"{tiles} A")
        row(tiles, lab, s["premature_share"], a["premature_share"] if a else np.nan, s["density_ratio"],
            s["river_crossings_unbridged"])
    if len(md) == 3:
        md.append("| — | — | — | — | — | none |")
    md.append("\nMethods without the hard constraints (A, P) are not flagged for unbridged crossings. "
              "Summaries are means over training seeds.")
    return md, out


def transfer_section():
    lamf = json.loads((PROJECT_ROOT / "configs" / "m14_lambda.json").read_text())
    md = ["\n## λ transfer check (train_cal, na_fixed, calibrated λ; no re-calibration)\n",
          "| backbone | dead ends / tile | islands / tile | target real |", "|---|---|---|---|"]
    from m14_plans import lam as L
    from urbanmorph.graphs.road_graph import RoadGraph
    from m14_calibrate import counts
    for b in ("TN_s1", "TN_s2"):
        r = any_run("train_cal", b, "na_fixed", L())
        if r is None:
            md.append(f"| {b} | not run | | |")
            continue
        c = []
        for x in r["seeds"].values():
            for G in x["graphs"]:
                nodes = np.array(G["nodes"], float).reshape(-1, 2)
                c.append(counts(RoadGraph(nodes, np.array(G["edges"], int).reshape(-1, 2), np.zeros(len(nodes), bool))))
        c = np.array(c, float).mean(0)
        md.append(f"| {b} | {c[0]:.2f} | {c[1]:.3f} | D {lamf['target_real']['D_per_tile']:.2f}, I {lamf['target_real']['I_per_tile']:.3f} |")
    md.append(f"| TN_s0 (calibration) | {lamf['final_gen']['D']:.2f} | {lamf['final_gen']['I']:.3f} | same |")
    return md


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    rep, md = {}, ["# Milestone 14 report (development data only; sealed test never read)\n",
                   "Generated by `scripts/m14_report.py` from stored runs. "
                   f"Frozen: λ_D = {LAM['lam_dead']}, λ_I = {LAM['lam_island']} (moment-matched on training tiles); trigger = "
                   f"`{FZ['trigger']}` (DEV-A firing share {FZ['fire_share']:.1%}, VoC captured {FZ['voc_captured']:.1%}; "
                   f"pilot validation vs random: {'passed' if FZ.get('validation_devA', {}).get('justified') else 'FAILED'}).\n"]
    labels = [l for l, _, _, _ in methods(FZ)]
    holm_ps = {}
    for tiles, name in SETS.items():
        t, summ = method_table(tiles, labels)
        md += [f"\n## {name}: all methods (TN, mean over training seeds [min, max]; inference seeds 1–3)\n"] + t
        fl = ma.noise_floor(ma.real(tiles))
        md.append(f"\nReal-vs-real S10_cond between random block halves of the real tiles: median {fl['median']:.3f}, "
                  f"p90 {fl['p90']:.3f} (half-size sets: an upper bound on the full-size noise).\n")
        refr = ma.real(tiles)
        rl = {k: float(np.nanmean([r.get(k, np.nan) for r in refr])) for k in ("od_unreach_land", "dead_end_ratio_m13", "islands")}
        md.append("Gap to real (|method − real|, mean over seeds): " + "; ".join(
            f"{l}: OD {abs(np.mean([x['od_unreach_land'] for x in summ[l]]) - rl['od_unreach_land']):.3f}, dead-end "
            f"{abs(np.mean([x['dead_end_ratio_m13'] for x in summ[l]]) - rl['dead_end_ratio_m13']):.3f}, islands "
            f"{abs(np.mean([x['islands'] for x in summ[l]]) - rl['islands']):.2f}" for l in labels if summ.get(l)) + "\n")
        h, hr = hypotheses(tiles, summ)
        md += [f"\n### Pre-registered tests on {tiles.upper()}" + (" (PRIMARY)" if tiles == "gen2" else " (secondary)") + "\n",
               "| ID | contrast | primary | secondary / guard | verdict |", "|---|---|---|---|---|"] + h
        sm, so = secondary(tiles, summ)
        md += sm
        pc, po = per_city(tiles) if tiles != "devB" else ([], {})
        md += pc
        rep[tiles] = {"summaries": {l: summ[l] for l in labels}, "noise_floor": fl, "tests": hr, "secondary": so, "cities": po}
        if tiles == "gen2":
            if "H1" in hr:
                holm_ps["H1"] = hr["H1"]["od_unreach_land"]["p_signflip"]
            if "H3_vs_random" in hr:
                holm_ps["H3-vs-random"] = hr["H3_vs_random"]["od_unreach_land"]["p_signflip"]
            md += cost_section(summ) + diversity_section(summ)
            dm, do = density_section("gen2")
            md += dm
            rep["density_adjusted"] = do
            dmm, dmo = dm_section("gen2")
            md += dmm
            rep["density_matched"] = dmo
    hm, ho, hps = h5_section()
    md += hm
    holm_ps.update(hps)
    rep["H5"] = ho
    if holm_ps:
        hres = ma.holm(holm_ps)
        md += ["\n## Holm–Bonferroni over the GEN2 primary tests (α = 0.05; H5 uses the median p over seeds)\n",
               "| test | p | significant after Holm |", "|---|---|---|"]
        md += [f"| {k} | {p:.4f} | {'yes' if hres[k] else 'no'} |" for k, p in sorted(holm_ps.items(), key=lambda x: x[1])]
        rep["holm"] = {"p": holm_ps, "significant": hres}
    st, so = strata_table()
    md += st
    rep["strata"] = so
    ab, ao = ablation_section()
    md += ab
    rep["ablations"] = ao
    sh, sho = shift_section()
    md += sh
    h2m, h2o = h2_section()
    md += h2m
    rep["H2"] = h2o
    fm, fo = failure_section({t: rep[t]["summaries"] for t in SETS if t in rep}, sho)
    md += fm
    rep["failure_flags"] = fo
    rep["shifts"] = {k: {kk: v.get(kk) for kk in ("od_unreach_land", "od_unreach_land_nonempty", "dead_end_ratio_m13",
                                                  "S10_cond", "S_T", "density_ratio")} for k, v in sho.items()}
    md += transfer_section()
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=lambda o: float(o) if np.isscalar(o) else str(o)))
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
