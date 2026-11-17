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
        res["H1"] = {"od_unreach_land": c, "S10_cond": s10}
        md.append(f"| H1 | NA-fixed − M\\*: OD unreach. (land) | {ctext(c)} | S10: {ctext(s10)} | "
                  f"{'**supported**' if c['hi'] < 0 and s10['delta'] <= 0.05 else 'not supported'} |")
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
        pr = P("A", "NA-adaptive")
        c = ma.block_contrast_div(pr) if hasattr(ma, "block_contrast_div") else None
        res["H4"] = {"ratio": float(ra)}
        md.append(f"| H4 | Jaccard diversity NA-adaptive / A | ratio {ra:.3f} | — | {'**met** (≥ 0.9)' if ra >= 0.9 else 'not met'} |")
    return md, res


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    rep, md = {}, ["# Milestone 14 report (development data only; sealed test never read)\n",
                   f"Frozen configuration: λ_D = {LAM['lam_dead']}, λ_I = {LAM['lam_island']}; trigger = `{FZ['trigger']}` "
                   f"(DEV-A firing share {FZ['fire_share']:.1%}, VoC captured {FZ['voc_captured']:.1%}).\n"]
    labels = [l for l, _, _, _ in methods(FZ)]
    for tiles, name in SETS.items():
        t, summ = method_table(tiles, labels)
        md += [f"\n## {name}: all methods (TN, mean over training seeds [min, max]; inference seeds 1–3)\n"] + t
        fl = ma.noise_floor(ma.real(tiles))
        md.append(f"\nReal-vs-real S10_cond between block halves: median {fl['median']:.3f}, p90 {fl['p90']:.3f}.\n")
        h, hr = hypotheses(tiles, summ)
        md += ["\n| ID | contrast | primary | secondary / guard | verdict |", "|---|---|---|---|---|"] + h
        rep[tiles] = {"summaries": {l: summ[l] for l in labels}, "noise_floor": fl, "tests": hr}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
