"""Milestone 15 report: protocol outcomes O1-O4, comparisons, the real-city pilot and figures, from STORED runs only
(docs/M15_PROTOCOL.md). Never generates.

    python scripts/m15_report.py        # outputs/m15/REPORT.md, outputs/m15/report.json, outputs/m15/figures/*.png

Every result is labelled PILOT. Intervals: percentile bootstrap (5000 draws) resampling seeds within sites
(synthetic) or windows within metros (real); synthetic sites are fixed conditions, not a population.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m15_run import RUNS, run_id
from m15_plans import eval_plan, real_plan
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.experiments.graph_io import graph_from_json
from urbanmorph.experiments.sites import CONDITIONS, grid_b, real_tiles, site_set
from urbanmorph.metrics.fidelity import prdc, real_vs_real

OUT = PROJECT_ROOT / "outputs" / "m15"
FIG = OUT / "figures"
B = 5000
SLOPED = ("incline8", "incline15", "ridge", "valley")
NONFLAT = ("incline8", "incline15", "ridge", "valley", "water")
FEATS = ("block_area_med", "block_area_p90", "share_3way", "share_4way", "circuity_100_200", "circuity_200_350")
LOGF = ("block_area_med", "block_area_p90")
RNG = np.random.default_rng(20261120)
SYN_START = "centre"          # protocol amendment 2; the as-run learned-start runs are summarised in an appendix


# ------------------------------------------------------------------------------------------------ loading
def load(spec):
    f = RUNS / f"{run_id(spec)}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return r if len(r["seeds"]) == len(spec["seeds"]) else None


def find(plan, **kw):
    """The stored run of the unique plan spec matching kw (absent keys must be absent / default)."""
    defaults = {"overrides": None, "grade_cap": None, "temperature": None, "shown": None, "bb": "TN_s0",
                "start": SYN_START if str(kw.get("sites", "")).startswith("syn") else None}
    want = {**defaults, **kw}
    hits = []
    for s in plan:
        cur = {"method": s["method"], "style": s.get("style", "off"), "sites": s["sites"], "overrides": s.get("overrides"),
               "grade_cap": s.get("grade_cap"), "temperature": s.get("temperature"), "shown": s.get("shown"), "bb": s["backbone"],
               "start": s.get("start")}
        if all(cur.get(k) == v for k, v in want.items()):
            hits.append(s)
    assert len(hits) <= 1, (kw, hits)
    return load(hits[0]) if hits else None


def arr(run, key, sites=None):
    """(n_seeds, n_sites) array; sites: list of site keys to keep (in that order)."""
    seeds = sorted(run["seeds"], key=int)
    names = [r["site"] for r in run["seeds"][seeds[0]]["rows"]]
    idx = [names.index(s) for s in sites] if sites else range(len(names))
    return np.array([[run["seeds"][s]["rows"][i].get(key, np.nan) for i in idx] for s in seeds], float)


def sites_of(cond):
    return [f"eval_{c}" for c in cond]


def boot_mean(D):
    """Mean of a (seeds, sites) paired-difference array; CI resampling seeds within each site."""
    D = np.asarray(D, float)
    pt = float(np.nanmean(D))
    n = D.shape[0]
    bs = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, (n, D.shape[1]))
        bs[b] = np.nanmean(np.take_along_axis(D, idx, 0))
    return pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def boot_ratio(Dt, Df):
    """Retention = mean(Dt) / mean(Df) with the same within-site seed resampling for both."""
    pt = float(np.nanmean(Dt) / np.nanmean(Df)) if np.nanmean(Df) else np.nan
    n = Dt.shape[0]
    bs = []
    for _ in range(B):
        it, i_f = RNG.integers(0, n, (n, Dt.shape[1])), RNG.integers(0, n, (n, Df.shape[1]))
        den = np.nanmean(np.take_along_axis(Df, i_f, 0))
        bs.append(np.nanmean(np.take_along_axis(Dt, it, 0)) / den if den else np.nan)
    return pt, float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def ci(t, f="{:+.3f}"):
    return f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]" if t and np.isfinite(t[0]) else "—"


# ------------------------------------------------------------------------------------------------ style descriptors
def descriptor(run, style, sites):
    """The pre-declared O1 descriptor of a style (higher = more of the requested style; cul-de-sac: two rows)."""
    if style == "grid":
        return {"R4": arr(run, "orth_order", sites)}
    if style == "radial":
        return {"radial+tangential": arr(run, "radial_share", sites) + arr(run, "tangential_share", sites)}
    if style == "culdesac":
        return {"loops/km2": arr(run, "loops_per_km2", sites), "dead ends/km": arr(run, "deadend_per_km", sites)}
    if style == "organic":
        return {"R4": arr(run, "orth_order", sites)}
    raise KeyError(style)


SMALLEST = {("grid", "R4"): 0.05, ("radial", "radial+tangential"): 0.05, ("culdesac", "loops/km2"): -5.0,
            ("culdesac", "dead ends/km"): 0.3, ("organic", "R4"): -0.05}


# ------------------------------------------------------------------------------------------------ synthetic report
def synthetic(md, rep):
    P = eval_plan()
    md += ["## A. Synthetic interventions (PILOT; evidence about model behaviour, not real-world generalisation)", ""]
    # A0 validity
    md += ["### A0. Validity, length and cost (all 6 evaluation sites x 10 seeds)", "",
           "| Variant | empty | street km | 60 m coverage | OD unreach. (land) | rejected cand. share | forced stops / sample | s / seed (6 sites) | Jaccard div. |",
           "|---|---|---|---|---|---|---|---|---|"]
    variants = [(f"{m} {st}", dict(method=m, style=st, sites="syn_eval")) for m in ("M*", "NA-fixed") for st in
                ("off", "grid", "organic", "radial", "culdesac")]
    variants += [(f"{m} off", dict(method=m, style="off", sites="syn_eval")) for m in ("A", "NA-myopic", "NA-SMC", "M13-SMC", "P")]
    variants += [(f"TRACER {st}", dict(method="TRACER", style=st, sites="syn_eval")) for st in ("grid", "organic", "radial")]
    variants += [(f"M* {st} morph-only", dict(method="M*", style=st, sites="syn_eval", overrides={"soft": False}))
                 for st in ("grid", "organic", "radial", "culdesac")]
    variants += [(f"{m} {st} cap p99", dict(method=m, style=st, sites="syn_eval", grade_cap="real_p99"))
                 for m in ("M*", "NA-fixed") for st in ("grid", "organic", "radial", "culdesac")]
    val = {}
    for lab, kw in variants:
        r = find(P, **kw)
        if r is None:
            md.append(f"| {lab} | missing | | | | | | | |")
            continue
        rows = [x for s in r["seeds"].values() for x in s["rows"]]
        rej = np.nansum([sum(v for k, v in x["log"].items() if k.startswith("hard_")) for x in rows])
        ev = np.nansum([x["log"].get("cand_evals", 0) for x in rows])
        fs = np.nanmean([x["log"].get("forced_stop", 0) for x in rows])
        sec = np.mean([s["seconds"] for s in r["seeds"].values()])
        div = np.nanmean([d["jaccard_distance"] for d in r["diversity"]])
        v = {"empty": np.mean([x["empty"] for x in rows]), "km": np.nanmean([x["street_km"] for x in rows]),
             "cov": np.nanmean([x["coverage_60"] for x in rows]), "od": np.nanmean([x["od_unreach_land"] for x in rows]),
             "rej": rej / ev if ev else np.nan, "fs": fs, "sec": sec, "div": div}
        val[lab] = v
        md.append(f"| {lab} | {v['empty']:.2f} | {v['km']:.2f} | {v['cov']:.3f} | {v['od']:.3f} | "
                  f"{v['rej']:.3f} | {v['fs']:.2f} | {v['sec']:.1f} | {v['div']:.3f} |")
    rep["validity"] = val
    md += ["", "Rejected-candidate share = candidates failing a hard rule / candidates evaluated (the sampler's own log); "
           "forced stop = a stop taken when every continuation had (near) zero weight; the tracer has no candidate log.", ""]

    # O1 + O2
    md += ["### O1 (style control on flat) and O2 (retention on terrain), paired style on - off", "",
           "| Sampler | Style | Descriptor | O1 flat: on - off [95% CI] | smallest effect | on terrain (5 non-flat): on - off | O2 retention [95% CI] | non-inferior (>= 0.5)? |",
           "|---|---|---|---|---|---|---|---|"]
    rep["O1_O2"] = {}
    for m in ("M*", "NA-fixed"):
        off = find(P, method=m, style="off", sites="syn_eval")
        for st in ("grid", "organic", "radial", "culdesac"):
            on = find(P, method=m, style=st, sites="syn_eval")
            if on is None or off is None:
                continue
            ref = find(P, method=m, style="grid", sites="syn_eval") if st == "organic" else off
            for name, Xon in descriptor(on, st, None).items():
                Xref = descriptor(ref, "grid" if st == "organic" else st, None)[name]
                flat_i, nf_i = [0], [1, 2, 3, 4, 5]
                Df, Dt = Xon[:, flat_i] - Xref[:, flat_i], Xon[:, nf_i] - Xref[:, nf_i]
                o1, o_t = boot_mean(Df), boot_mean(Dt)
                ret = boot_ratio(Dt, Df) if abs(o1[0]) > 1e-9 else (np.nan, np.nan, np.nan)
                smallest = SMALLEST[(st, name)]
                ok = "yes" if (np.isfinite(ret[1]) and ret[1] >= 0.5) else ("no" if np.isfinite(ret[0]) else "—")
                lab = f"{name} vs grid style" if st == "organic" else name
                md.append(f"| {m} | {st} | {lab} | {ci(o1)} | {smallest:+g} | {ci(o_t)} | {ci(ret, '{:.2f}')} | {ok} |")
                rep["O1_O2"][f"{m}|{st}|{name}"] = {"flat": o1, "terrain": o_t, "retention": ret}
    # tracer and controllability
    md += ["", "Controllability check (field alignment, the generator's own objective; 0.5 = uniform bearings), flat / non-flat mean:", "",
           "| Style | M* off | M* on | NA-fixed off | NA-fixed on | Tracer |", "|---|---|---|---|---|---|"]
    for st in ("grid", "organic", "radial"):
        cells = []
        for m, s_ in (("M*", "off"), ("M*", st), ("NA-fixed", "off"), ("NA-fixed", st), ("TRACER", st)):
            r = find(P, method=m, style=s_, sites="syn_eval")
            if r is None:
                cells.append("—")
                continue
            if s_ == "off":                      # alignment of style-off output against this style's field
                a = off_alignment(r, st)
            else:
                a = arr(r, "field_alignment")
            cells.append(f"{np.nanmean(a[:, 0]):.2f} / {np.nanmean(a[:, 1:]):.2f}")
        md.append(f"| {st} | " + " | ".join(cells) + " |")
    md.append("")

    # O3
    md += ["### O3. Do NA-fixed's connectivity gains survive style guidance? (paired NA-fixed - M*, same style, site, seed)", "",
           "| Style | OD unreach. (land) NA-fixed - M* [95% CI] | islands / tile | M* OD | NA-fixed OD | street km M* / NA-fixed |",
           "|---|---|---|---|---|---|"]
    rep["O3"] = {}
    pooled_on = []
    for st in ("off", "grid", "organic", "radial", "culdesac"):
        a, b = find(P, method="M*", style=st, sites="syn_eval"), find(P, method="NA-fixed", style=st, sites="syn_eval")
        if a is None or b is None:
            continue
        D = arr(b, "od_unreach_land") - arr(a, "od_unreach_land")
        Di = arr(b, "islands") - arr(a, "islands")
        if st != "off":
            pooled_on.append(D)
        t = boot_mean(D)
        rep["O3"][st] = t
        md.append(f"| {st} | {ci(t)} | {np.nanmean(Di):+.2f} | {np.nanmean(arr(a, 'od_unreach_land')):.3f} | "
                  f"{np.nanmean(arr(b, 'od_unreach_land')):.3f} | {np.nanmean(arr(a, 'street_km')):.2f} / {np.nanmean(arr(b, 'street_km')):.2f} |")
    if pooled_on:
        t = boot_mean(np.concatenate(pooled_on, axis=1))
        rep["O3"]["pooled_on"] = t
        verdict = "survives" if (t[0] <= -0.01 and t[2] < 0) else "does not survive (criterion: <= -0.01 and CI excluding 0)"
        md += [f"| **pooled, style on** | **{ci(t)}** | | | | |", "", f"O3 verdict: the gain **{verdict}**.", ""]

    # O4
    md += ["### O4. Terrain exposure on the four sloped conditions (true raster, 10 m sub-spans)", "",
           "| Variant | share > 15 % | share > 10 % | grade p90 | street km | coverage 60 m | flag |", "|---|---|---|---|---|---|---|"]
    rep["O4"] = {}
    ss = sites_of(SLOPED)

    def expo(r, sites=ss):
        return {k: arr(r, k, sites) for k in ("share_gt15", "share_gt10", "grade_p90", "street_km", "coverage_60")}
    exp_rows = [(f"{m} {st}", dict(method=m, style=st, sites="syn_eval")) for m in ("M*", "NA-fixed")
                for st in ("off", "grid", "organic", "radial", "culdesac")]
    exp_rows += [(f"TRACER {st}", dict(method="TRACER", style=st, sites="syn_eval")) for st in ("grid", "organic", "radial")]
    exp_rows += [(f"{m} {st} cap p99", dict(method=m, style=st, sites="syn_eval", grade_cap="real_p99"))
                 for m in ("M*", "NA-fixed") for st in ("grid",)]
    exp_rows += [(f"M* {st} morph-only", dict(method="M*", style=st, sites="syn_eval", overrides={"soft": False})) for st in ("grid",)]
    exp_rows += [(f"{m} {st} MISMATCH", dict(method=m, style=st, sites="syn_eval_nonflat", shown="syn_eval_mismatch"))
                 for m in ("M*", "NA-fixed", "TRACER") for st in (("off", "grid") if m != "TRACER" else ("grid",))]
    E = {}
    for lab, kw in exp_rows:
        r = find(P, **kw)
        if r is None:
            continue
        E[lab] = expo(r)
    base_km = {lab: np.nanmean(v["street_km"]) for lab, v in E.items()}
    for lab, v in E.items():
        ref = lab.split(" MISMATCH")[0] if "MISMATCH" in lab else None
        flag = ""
        if ref and ref in base_km and base_km[lab] < 0.8 * base_km[ref]:
            flag = "fewer roads than the matched variant"
        md.append(f"| {lab} | {np.nanmean(v['share_gt15']):.3f} | {np.nanmean(v['share_gt10']):.3f} | {np.nanmean(v['grade_p90']):.3f} | "
                  f"{np.nanmean(v['street_km']):.2f} | {np.nanmean(v['coverage_60']):.3f} | {flag} |")
    rep["O4_means"] = {lab: {k: float(np.nanmean(x)) for k, x in v.items()} for lab, v in E.items()}
    md += ["", "Paired contrasts on share > 15 % (sloped conditions; criterion (a): style on - off <= +0.02; (b) superiority vs tracer and vs mismatch):", "",
           "| Contrast | difference [95% CI] | street-km ratio | reading |", "|---|---|---|---|"]
    for m in ("M*", "NA-fixed"):
        for st in ("grid", "organic", "radial", "culdesac"):
            a, b = E.get(f"{m} {st}"), E.get(f"{m} off")
            if a is None or b is None:
                continue
            t = boot_mean(a["share_gt15"] - b["share_gt15"])
            kr = np.nanmean(a["street_km"]) / np.nanmean(b["street_km"])
            rd = "non-inferior" if t[2] <= 0.02 else "not shown non-inferior"
            md.append(f"| (a) {m} {st} - {m} off | {ci(t)} | {kr:.2f} | {rd} |")
            rep["O4"][f"a|{m}|{st}"] = t
        for st in ("grid", "organic", "radial"):
            a, b = E.get(f"{m} {st}"), E.get(f"TRACER {st}")
            if a is None or b is None:
                continue
            t = boot_mean(a["share_gt15"] - b["share_gt15"])
            kr = np.nanmean(a["street_km"]) / np.nanmean(b["street_km"])
            rd = ("lower exposure" if t[2] < 0 else "not lower") + ("; FEWER ROADS (< 0.8x)" if kr < 0.8 else "")
            md.append(f"| (b) {m} {st} - TRACER {st} | {ci(t)} | {kr:.2f} | {rd} |")
            rep["O4"][f"b_tracer|{m}|{st}"] = t
        for st in ("off", "grid"):
            a, b = E.get(f"{m} {st}"), E.get(f"{m} {st} MISMATCH")
            if a is None or b is None:
                continue
            nf_sloped = [list(NONFLAT).index(c) for c in SLOPED]
            D = a["share_gt15"] - b["share_gt15"][:, nf_sloped] if b["share_gt15"].shape[1] == 5 else a["share_gt15"] - b["share_gt15"]
            t = boot_mean(D)
            kr = np.nanmean(a["street_km"]) / np.nanmean(b["street_km"])
            md.append(f"| (b) {m} {st} - same with MISMATCHED terrain | {ci(t)} | {kr:.2f} | {'lower exposure' if t[2] < 0 else 'not lower'} |")
            rep["O4"][f"b_mismatch|{m}|{st}"] = t
    for m in ("M*", "NA-fixed"):                      # (c) below-cap comparison with the hard-mask control
        a, b = E.get(f"{m} grid"), E.get(f"{m} grid cap p99")
        if a is None or b is None:
            continue
        md.append(f"| (c) {m} grid vs {m} grid with ceiling: share 10-15 % | {np.nanmean(a['share_gt10'] - a['share_gt15']):.3f} vs "
                  f"{np.nanmean(b['share_gt10'] - b['share_gt15']):.3f} | {np.nanmean(a['street_km']) / np.nanmean(b['street_km']):.2f} | "
                  f"ceiling removes {np.nanmean(a['share_gt15']) - np.nanmean(b['share_gt15']):+.3f} of the > 15 % share |")
    md.append("")
    return E


def off_alignment(run, style):
    from urbanmorph.graphs.style import StyleField
    from urbanmorph.graphs.style_metrics import field_alignment
    from urbanmorph.graphs.tc_sampler import SamplerConfig
    from m15_run import style_overrides
    cfg = SamplerConfig(**style_overrides(style))
    seeds = sorted(run["seeds"], key=int)
    return np.array([[field_alignment(graph_from_json(G), StyleField.build(cfg, row["site"], int(s)))
                      for row, G in zip(run["seeds"][s]["rows"], run["seeds"][s]["graphs"])] for s in seeds])


def secondary(md, rep):
    P = eval_plan()
    md += ["### Temperature, separately (all else unchanged; style off and grid)", "",
           "| Sampler | Style | T | empty | street km | rejected cand. share | forced stops / sample | grid R4 (flat) | share > 15 % (sloped) | OD unreach. | Jaccard div. |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    rep["temperature"] = {}
    for m in ("M*", "NA-fixed"):
        for st in ("off", "grid"):
            for T in (None, 0.8, 1.0):
                r = find(P, method=m, style=st, sites="syn_eval", temperature=T)
                if r is None:
                    continue
                rows = [x for s in r["seeds"].values() for x in s["rows"]]
                rej = np.nansum([sum(v for k, v in x["log"].items() if k.startswith("hard_")) for x in rows])
                ev = np.nansum([x["log"].get("cand_evals", 0) for x in rows])
                v = dict(empty=np.mean([x["empty"] for x in rows]), km=np.nanmean([x["street_km"] for x in rows]),
                         rej=rej / ev if ev else np.nan, fs=np.nanmean([x["log"].get("forced_stop", 0) for x in rows]),
                         r4=np.nanmean(arr(r, "orth_order")[:, 0]), ex=np.nanmean(arr(r, "share_gt15", sites_of(SLOPED))),
                         od=np.nanmean(arr(r, "od_unreach_land")), div=np.nanmean([d["jaccard_distance"] for d in r["diversity"]]))
                rep["temperature"][f"{m}|{st}|{T or 0.5}"] = v
                md.append(f"| {m} | {st} | {T or 0.5} | {v['empty']:.2f} | {v['km']:.2f} | {v['rej']:.3f} | {v['fs']:.2f} | "
                          f"{v['r4']:.2f} | {v['ex']:.3f} | {v['od']:.3f} | {v['div']:.3f} |")
    md += ["", "### Grade-ceiling sensitivity (NA-fixed + grid)", "",
           "| Ceiling | street km | share > 10 % | share > 15 % | grade p90 | grid R4 on sloped | OD unreach. |", "|---|---|---|---|---|---|---|"]
    for cap in (None, "real_p99", "0.12", "0.08"):
        r = find(P, method="NA-fixed", style="grid", sites="syn_eval", grade_cap=cap)
        if r is None:
            continue
        ss = sites_of(SLOPED)
        md.append(f"| {cap or 'off'} | {np.nanmean(arr(r, 'street_km', ss)):.2f} | {np.nanmean(arr(r, 'share_gt10', ss)):.3f} | "
                  f"{np.nanmean(arr(r, 'share_gt15', ss)):.3f} | {np.nanmean(arr(r, 'grade_p90', ss)):.3f} | "
                  f"{np.nanmean(arr(r, 'orth_order', ss)):.2f} | {np.nanmean(arr(r, 'od_unreach_land')):.3f} |")
    md += ["", "### Ablation: un-normalised style factor (M* + grid)", "",
           "| Style factor | street km | forced stops / sample | grid R4 (flat) | R4 (non-flat) |", "|---|---|---|---|---|"]
    for lab, ov in (("normalised (frozen)", None), ("un-normalised", {"style_norm": False})):
        r = find(P, method="M*", style="grid", sites="syn_eval", overrides=ov)
        if r is None:
            continue
        rows = [x for s in r["seeds"].values() for x in s["rows"]]
        md.append(f"| {lab} | {np.nanmean([x['street_km'] for x in rows]):.2f} | {np.nanmean([x['log'].get('forced_stop', 0) for x in rows]):.2f} | "
                  f"{np.nanmean(arr(r, 'orth_order')[:, 0]):.2f} | {np.nanmean(arr(r, 'orth_order')[:, 1:]):.2f} |")
    md += ["", "### Second backbone (TN_s1): O1 grid and O3", "", "| Backbone | M* grid - off R4 (flat) | NA-fixed grid - off R4 (flat) | NA-fixed - M* OD (grid) |", "|---|---|---|---|"]
    for bb in ("TN_s0", "TN_s1"):
        g = {(m, st): find(P, method=m, style=st, sites="syn_eval", bb=bb) for m in ("M*", "NA-fixed") for st in ("off", "grid")}
        if any(v is None for v in g.values()):
            continue
        a = boot_mean(arr(g["M*", "grid"], "orth_order")[:, :1] - arr(g["M*", "off"], "orth_order")[:, :1])
        b = boot_mean(arr(g["NA-fixed", "grid"], "orth_order")[:, :1] - arr(g["NA-fixed", "off"], "orth_order")[:, :1])
        c = boot_mean(arr(g["NA-fixed", "grid"], "od_unreach_land") - arr(g["M*", "grid"], "od_unreach_land"))
        md.append(f"| {bb} | {ci(a)} | {ci(b)} | {ci(c)} |")
    md.append("")


# ------------------------------------------------------------------------------------------------ real pilot
def real_rows_cache():
    """Metrics of the REAL development windows (GLO-30, 30 m grade spans), cached."""
    f = OUT / "real_rows.json"
    if f.exists():
        return json.loads(f.read_text())
    from urbanmorph.graphs.m14_metrics import tile_metrics_m14
    from urbanmorph.graphs.style_metrics import morphology_row
    from urbanmorph.graphs.tc_potentials import TerrainField
    from urbanmorph.graphs.terrain_generate import terrain_of
    rows = []
    for t in real_tiles("glo30"):
        tf = TerrainField(terrain_of(t["d"]))
        rng = np.random.default_rng(0)
        rows.append({"site": t["sid"], "metro": t["metro"], "country": t["country"], "stratum": t["stratum"],
                     **morphology_row(t["graph"], tf.t, rng=rng, grade_span=30.0),
                     **{k: v for k, v in tile_metrics_m14(t["graph"], tf, rng).items() if not isinstance(v, (list, dict))}})
    f.write_text(json.dumps(rows, default=float))
    return rows


def feat(rows):
    X = np.array([[r.get(k, np.nan) for k in FEATS] for r in rows], float)
    for j, k in enumerate(FEATS):
        if k in LOGF:
            X[:, j] = np.log(np.where(X[:, j] > 0, X[:, j], np.nan))
    return X


def real(md, rep):
    P = real_plan()
    R = real_rows_cache()
    metro = np.array([r["metro"] for r in R])
    Xr = feat(R)
    md += ["## B. Real-city pilot (PILOT, development metros only; GLO-30 DSM elevation)", "",
           f"{len(R)} windows in {len(set(metro))} metros. Held-out metros are declared and untouched.", "",
           "### B1. Task A (unconditional, style off): fidelity on features outside every generator objective", "",
           "Features: " + ", ".join(FEATS) + " (block areas in log m2). k = 5. Real-vs-real = random metro-level halves (20 splits).", ""]
    rr = real_vs_real(Xr, metro)
    rep["real_vs_real"] = rr
    md += ["| Set | precision | recall | density | coverage | coverage / real-vs-real | street km | OD unreach. (land) | share > 15 % (30 m) | empty |",
           "|---|---|---|---|---|---|---|---|---|---|",
           f"| real vs real | {rr['precision']:.2f} | {rr['recall']:.2f} | {rr['density']:.2f} | {rr['coverage']:.2f} | 1.00 | "
           f"{np.nanmean([r['street_km'] for r in R]):.2f} | {np.nanmean([r['od_unreach_land'] for r in R]):.3f} | "
           f"{np.nanmean([r['share_gt15'] for r in R]):.3f} | 0.00 |"]
    rep["taskA"] = {}
    gens = {}
    for m in ("A", "M*", "NA-fixed", "NA-SMC", "P"):
        r = find(P, method=m, style="off", sites="real_dev")
        if r is None:
            md.append(f"| {m} | missing |")
            continue
        gens[m] = r
        rows = [x for s in r["seeds"].values() for x in s["rows"]]
        res = prdc(Xr, feat(rows))
        rep["taskA"][m] = res
        md.append(f"| {m} | {res['precision']:.2f} | {res['recall']:.2f} | {res['density']:.2f} | {res['coverage']:.2f} | "
                  f"{res['coverage'] / rr['coverage']:.2f} | {np.nanmean([x['street_km'] for x in rows]):.2f} | "
                  f"{np.nanmean([x['od_unreach_land'] for x in rows]):.3f} | {np.nanmean([x['share_gt15'] for x in rows]):.3f} | "
                  f"{np.mean([x['empty'] for x in rows]):.2f} |")
    md += ["", "Coverage-retention target (protocol §5): generated coverage >= 0.8 x real-vs-real coverage.", ""]
    # per-metro and leave-one-metro-out
    md += ["### B2. Per metro (Task A): street km and OD unreachability, real vs M* vs NA-fixed; leave-one-metro-out", "",
           "| Metro | windows | real km | M* km | NA-fixed km | real OD | M* OD | NA-fixed OD | NA-fixed - M* OD |", "|---|---|---|---|---|---|---|---|---|"]
    diffs = []
    for mt in dict.fromkeys(metro):
        idx = [i for i, r in enumerate(R) if r["metro"] == mt]
        sids = [R[i]["site"] for i in idx]
        if "M*" not in gens or "NA-fixed" not in gens:
            break
        a, b = arr(gens["M*"], "od_unreach_land", sids), arr(gens["NA-fixed"], "od_unreach_land", sids)
        d = float(np.nanmean(b - a))
        diffs.append(d)
        md.append(f"| {mt} | {len(idx)} | {np.nanmean([R[i]['street_km'] for i in idx]):.2f} | {np.nanmean(arr(gens['M*'], 'street_km', sids)):.2f} | "
                  f"{np.nanmean(arr(gens['NA-fixed'], 'street_km', sids)):.2f} | {np.nanmean([R[i]['od_unreach_land'] for i in idx]):.3f} | "
                  f"{np.nanmean(a):.3f} | {np.nanmean(b):.3f} | {d:+.3f} |")
    if diffs:
        loo = [np.mean([x for j, x in enumerate(diffs) if j != i]) for i in range(len(diffs))]
        sd = float(np.std(diffs, ddof=1))
        n_req = int(np.ceil(((1.96 + 0.84) * sd / 0.01) ** 2)) if sd > 0 else 0
        rep["metro_effect"] = {"per_metro": diffs, "loo_range": [min(loo), max(loo)], "sd": sd, "n_metros_for_0.01": n_req}
        md += ["", f"City-level NA-fixed - M* OD effect: mean {np.mean(diffs):+.3f}, metro SD {sd:.3f}, leave-one-metro-out range "
               f"[{min(loo):+.3f}, {max(loo):+.3f}].",
               f"Sample size (paired metro-level effects, two-sided alpha 0.05, power 0.8, smallest effect 0.01): "
               f"n = ((1.96 + 0.84) x {sd:.3f} / 0.01)^2 = **{n_req} metros**.", ""]
    # strata exposure
    md += ["### B3. Terrain exposure by GLO-30 slope stratum (share of ground streets > 15 %, 30 m sub-spans)", "",
           "| Stratum | windows | real | A | M* | NA-fixed | NA-SMC |", "|---|---|---|---|---|---|---|"]
    for stv in ("flat", "mixed", "hilly"):
        sids = [r["site"] for r in R if r["stratum"] == stv]
        if not sids:
            continue
        cells = [f"{np.nanmean([r['share_gt15'] for r in R if r['stratum'] == stv]):.3f}"]
        for m in ("A", "M*", "NA-fixed", "NA-SMC"):
            cells.append(f"{np.nanmean(arr(gens[m], 'share_gt15', sids)):.3f}" if m in gens else "—")
        md.append(f"| {stv} | {len(sids)} | " + " | ".join(cells) + " |")
    md += ["", "GLO-30 is a DSM: building and tree heights inflate grades in dense or wooded windows, for real and generated "
           "streets alike; 30 m sub-spans are the finest scale the source supports.", ""]
    # Task B
    md += ["### B4. Task B (declared grid, a SEPARATE task; R4 is conditioned and excluded as evidence)", "",
           "Windows with real R4 >= 0.5; theta0 declared from the real network. Per window, |generated - real| of the "
           "non-conditioned features, style off (Task A runs on the same windows) vs grid declared:", "",
           "| Sampler | feature | off: mean abs. error | grid: mean abs. error | difference [95% CI] |", "|---|---|---|---|---|"]
    gb = [t["sid"] for t in grid_b(real_tiles("glo30"))]
    Rb = {r["site"]: r for r in R}
    rep["taskB"] = {}
    for m in ("M*", "NA-fixed"):
        on = find(P, method=m, style="grid", sites="real_dev_gridB")
        off = gens.get(m)
        if on is None or off is None:
            continue
        for k in ("share_4way", "share_3way", "block_area_med", "circuity_200_350"):
            real_v = np.array([Rb[s][k] for s in gb], float)
            a_on, a_off = arr(on, k, gb), arr(off, k, gb)
            if k == "block_area_med":
                real_v, a_on, a_off = np.log(real_v), np.log(a_on), np.log(a_off)
            e_on, e_off = np.abs(a_on - real_v), np.abs(a_off - real_v)
            t = boot_mean(e_on - e_off)
            rep["taskB"][f"{m}|{k}"] = t
            md.append(f"| {m} | {k}{' (log)' if k == 'block_area_med' else ''} | {np.nanmean(e_off):.3f} | {np.nanmean(e_on):.3f} | {ci(t)} |")
        r4 = boot_mean(arr(on, "orth_order", gb) - arr(off, "orth_order", gb))
        md.append(f"| {m} | R4 (controllability only) | | | {ci(r4)} |")
    md.append("")
    # DEM sensitivity
    md += ["### B5. Elevation-source sensitivity (Spain, 60 windows): IGN MDT05 bare earth vs GLO-30 DSM", "",
           "| Sampler | DEM | street km | OD unreach. | share > 15 % | grade p90 |", "|---|---|---|---|---|---|"]
    sp = [r["site"] for r in R if r["country"] == "spain"]
    for m in ("M*", "NA-fixed"):
        g30, g5 = gens.get(m), find(P, method=m, style="off", sites="real_dev_spain_mdt05")
        for lab, r, s in (("GLO-30 (30 m spans)", g30, sp), ("MDT05 (10 m spans)", g5, None)):
            if r is None:
                continue
            md.append(f"| {m} | {lab} | {np.nanmean(arr(r, 'street_km', s)):.2f} | {np.nanmean(arr(r, 'od_unreach_land', s)):.3f} | "
                      f"{np.nanmean(arr(r, 'share_gt15', s)):.3f} | {np.nanmean(arr(r, 'grade_p90', s)):.3f} |")
    md += ["", "Grades differ between rows partly by construction (span and DSM vs DTM); the comparison shows how much the "
           "elevation source changes the GENERATED networks (street km, connectivity).", ""]
    return R, gens


# ------------------------------------------------------------------------------------------------ figures
def figures(E, rep):
    from m13_elevation_maps import contour_step
    from m13_three_tiles import draw
    FIG.mkdir(parents=True, exist_ok=True)
    P = eval_plan()
    eval_sites = site_set("syn_eval")
    # style-by-terrain matrix: NA-fixed + style, seed 101, every condition
    styles = ["off", "grid", "organic", "radial", "culdesac"]
    for m in ("NA-fixed", "M*"):
        runs = {st: find(P, method=m, style=st, sites="syn_eval") for st in styles}
        if any(r is None for r in runs.values()):
            continue
        f, ax = plt.subplots(len(styles), len(CONDITIONS), figsize=(2.3 * len(CONDITIONS), 2.45 * len(styles)), layout="constrained")
        for i, st in enumerate(styles):
            for j, d in enumerate(eval_sites):
                g = graph_from_json(runs[st]["seeds"]["101"]["graphs"][j])
                z = d["elev"][64:192, 64:192]
                lo, hi = float(z.min()), float(z.max()) + 1
                draw(ax[i, j], d, g, lo, hi, contour_step(lo, hi))
                row = runs[st]["seeds"]["101"]["rows"][j]
                ax[i, j].set_title(f"R4 {row['orth_order']:.2f} · r+t {row['radial_share'] + row['tangential_share']:.2f}\n"
                                   f"loops {row['loops_per_km2']:.0f} · km {row['street_km']:.1f}", fontsize=7)
                if j == 0:
                    ax[i, j].set_ylabel(st, fontsize=10)
                if i == 0:
                    ax[i, j].set_xlabel(CONDITIONS[j], fontsize=9)
                    ax[i, j].xaxis.set_label_position("top")
        f.suptitle(f"PILOT. {m} + style (rows) x synthetic terrain (columns), evaluation sites, seed 101 (first seed; no selection). "
                   "R4 = orthogonal order, r+t = radial + tangential share about the tile centre", fontsize=9)
        f.savefig(FIG / f"fig_style_terrain_matrix_{m.replace('*', 'star')}.png", dpi=110)
        plt.close(f)
    # trade-off: exposure (sloped) vs retention (grid descriptor), per variant
    pts = []
    for lab, kw in [("M* grid", dict(method="M*", style="grid")), ("NA-fixed grid", dict(method="NA-fixed", style="grid")),
                    ("TRACER grid", dict(method="TRACER", style="grid")), ("M* grid morph-only", dict(method="M*", style="grid", overrides={"soft": False})),
                    ("M* grid cap p99", dict(method="M*", style="grid", grade_cap="real_p99")),
                    ("NA-fixed grid cap p99", dict(method="NA-fixed", style="grid", grade_cap="real_p99")),
                    ("NA-fixed grid cap 0.12", dict(method="NA-fixed", style="grid", grade_cap="0.12")),
                    ("NA-fixed grid cap 0.08", dict(method="NA-fixed", style="grid", grade_cap="0.08"))]:
        r = find(P, sites="syn_eval", **kw)
        if r is None:
            continue
        R4 = arr(r, "orth_order")
        retention = np.nanmean(R4[:, 1:]) / np.nanmean(R4[:, :1])            # R4 on terrain relative to flat (same variant)
        pts.append((lab, float(np.nanmean(arr(r, "share_gt15", sites_of(SLOPED)))), float(retention),
                    float(np.nanmean(arr(r, "street_km", sites_of(SLOPED))))))
    if pts:
        f, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
        cols = plt.cm.tab10(np.arange(len(pts)) % 10)
        for (lab, x, y, km), c in zip(pts, cols):
            ax.scatter(x, y, s=40 + 25 * km, color=c, edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(f"{lab} ({km:.1f} km)", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("terrain exposure: share of ground street > 15 % grade, sloped conditions (lower = better)")
        ax.set_ylabel("morphology retention: grid R4 on terrain / on flat (higher = better)")
        ax.grid(alpha=0.3)
        ax.set_title("PILOT. Trade-off between terrain exposure and grid retention (marker size = street km on sloped sites)", fontsize=9)
        f.savefig(FIG / "fig_tradeoff_exposure_retention.png", dpi=120)
        plt.close(f)
        rep["tradeoff_points"] = pts


def real_figures(R, gens, rep):
    if not gens:
        return
    FIG.mkdir(parents=True, exist_ok=True)
    rr = rep["real_vs_real"]
    labs = [m for m in ("A", "M*", "NA-fixed", "NA-SMC", "P") if m in rep.get("taskA", {})]
    f, ax = plt.subplots(figsize=(6.5, 5.0), layout="constrained")
    ax.axvspan(*rr["precision_range"], color="#cccccc", alpha=0.35, lw=0)
    ax.axhspan(*rr["coverage_range"], color="#cccccc", alpha=0.35, lw=0)
    ax.scatter([rr["precision"]], [rr["coverage"]], marker="*", s=180, color="#444444", zorder=3)
    ax.annotate("real vs real (grey: range over splits)", (rr["precision"], rr["coverage"]), xytext=(6, -12), textcoords="offset points", fontsize=8)
    for m, c in zip(labs, plt.cm.tab10(np.arange(len(labs)))):
        v = rep["taskA"][m]
        ax.scatter(v["precision"], v["coverage"], s=60, color=c, edgecolor="white", zorder=3)
        ax.annotate(m, (v["precision"], v["coverage"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlim(0, 1.02), ax.set_ylim(0, 1.02)
    ax.set_xlabel("precision (generated windows inside the real manifold)")
    ax.set_ylabel("coverage (real windows reached by generated ones)")
    ax.set_title("PILOT. Real vs generated, Task A, 6 development metros (features outside every objective)", fontsize=9)
    ax.grid(alpha=0.3)
    f.savefig(FIG / "fig_real_precision_coverage.png", dpi=120)
    plt.close(f)


def failures(md):
    """Representative failures by a rule fixed in the protocol: for NA-fixed + each style on the evaluation sites, the
    (site, seed) with the LOWEST style descriptor on a non-flat condition (ties: lowest seed), drawn with style off."""
    from m13_elevation_maps import contour_step
    from m13_three_tiles import draw
    P = eval_plan()
    sites = site_set("syn_eval")
    picks = []
    for st, key in (("grid", "orth_order"), ("radial", None), ("culdesac", "loops_per_km2"), ("organic", "field_alignment")):
        r = find(P, method="NA-fixed", style=st, sites="syn_eval")
        if r is None:
            continue
        if st == "radial":
            X = arr(r, "radial_share") + arr(r, "tangential_share")
        elif st == "culdesac":
            X = -arr(r, "loops_per_km2")                 # most loops = worst cul-de-sac
        else:
            X = arr(r, key)
        X[:, 0] = np.inf                                  # non-flat only
        i, j = np.unravel_index(np.nanargmin(np.where(np.isfinite(X), X, np.inf)), X.shape)
        seed = sorted(r["seeds"], key=int)[i]
        picks.append((st, j, seed, r))
    if not picks:
        return
    f, ax = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.6), layout="constrained")
    ax = np.atleast_1d(ax)
    md += ["### Representative failures (rule: NA-fixed + style, the non-flat (site, seed) with the weakest style descriptor)", "",
           "| Style | site | seed | descriptor value | street km | OD unreach. |", "|---|---|---|---|---|---|"]
    for a, (st, j, seed, r) in zip(ax, picks):
        d = sites[j]
        g = graph_from_json(r["seeds"][seed]["graphs"][j])
        row = r["seeds"][seed]["rows"][j]
        z = d["elev"][64:192, 64:192]
        lo, hi = float(z.min()), float(z.max()) + 1
        draw(a, d, g, lo, hi, contour_step(lo, hi))
        dv = {"grid": row["orth_order"], "radial": row["radial_share"] + row["tangential_share"], "culdesac": row["loops_per_km2"],
              "organic": row["field_alignment"]}[st]
        a.set_title(f"{st} on {CONDITIONS[j]}, seed {seed}\ndescriptor {dv:.2f} · {row['street_km']:.1f} km", fontsize=8)
        md.append(f"| {st} | {CONDITIONS[j]} | {seed} | {dv:.3f} | {row['street_km']:.2f} | {row['od_unreach_land']:.3f} |")
    f.suptitle("PILOT. Representative failures of NA-fixed + style (documented rule; not cherry-picked)", fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    f.savefig(FIG / "fig_failures.png", dpi=110)
    plt.close(f)
    md.append("")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Milestone 15 report (PILOT)", "",
          "Generated by `scripts/m15_report.py` from stored runs (`outputs/m15/runs/`). Protocol: `docs/M15_PROTOCOL.md` "
          "(frozen before evaluation; amendments in §8). Metrics: `docs/M15_METRICS.md`. Every number is development data; "
          "no held-out metro or sealed test split was read.", ""]
    rep = {}
    E = synthetic(md, rep)
    secondary(md, rep)
    failures(md)
    R, gens = real(md, rep)
    figures(E, rep)
    real_figures(R, gens, rep)
    md += ["## Figures", "", "* `outputs/m15/figures/fig_style_terrain_matrix_NA-fixed.png`, `..._Mstar.png`: style x terrain matrix",
           "* `outputs/m15/figures/fig_tradeoff_exposure_retention.png`: terrain exposure vs grid retention",
           "* `outputs/m15/figures/fig_real_precision_coverage.png`: real vs generated precision / coverage with the real-vs-real benchmark",
           "* `outputs/m15/figures/fig_failures.png`: representative failures (documented rule)", ""]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
