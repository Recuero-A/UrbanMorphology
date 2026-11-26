"""Milestone 13 analysis: per-run summaries and paired block-bootstrap contrasts (docs/M13_EVALUATION_PROTOCOL.md §6).

    python scripts/m13_analyze.py --summaries            # -> outputs/m13/summaries.json (every run in outputs/m13/runs)

Library functions (used by m13_select.py and m13_report.py):
* ``summarize(res)``         every pre-registered metric of one run on its tile set;
* ``contrast(a, b, key)``    paired block bootstrap of key(b) - key(a) over geographic blocks (2,000 resamples).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from m10_street_diagnosis import KEYS  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.m13_metrics import JOINT_KEYS, energy_distance, feature_matrix, mmd_rbf, response_distance
from urbanmorph.graphs.terrain_metrics import S_T_KEYS

OUT = PROJECT_ROOT / "outputs" / "m13"
RUNS = OUT / "runs"
NORM = json.loads((PROJECT_ROOT / "configs" / "m13_feature_norm.json").read_text())
RC_KEYS = ("len_density", "grade_med", "align_mean")
N_BOOT = 2000


def w1_iqr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    from scipy.stats import wasserstein_distance
    iqr = np.subtract(*np.percentile(b, [75, 25]))
    return float(wasserstein_distance(a, b) / iqr) if iqr > 0 else np.nan


def score(gen, ref, keys, idx=None):
    vals = []
    for k in keys:
        a = np.array([r.get(k, np.nan) for r in gen], float)
        b = np.array([r.get(k, np.nan) for r in ref], float)
        if idx is not None:
            a, b = a[idx], b[idx]
        v = w1_iqr(a, b)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def real_rows(tiles_tag):
    return json.loads((OUT / "real" / f"{tiles_tag}.json").read_text())


def tile_means(res):
    """Per tile: the mean over inference seeds of every scalar metric; response curves averaged bin-wise."""
    seeds = sorted(res["seeds"], key=int)
    n = len(res["seeds"][seeds[0]]["rows"])
    out = []
    for k in range(n):
        rows = [res["seeds"][s]["rows"][k] for s in seeds]
        m = {"sid": rows[0]["sid"], "block": rows[0]["block"]}
        for key, v in rows[0].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                m[key] = float(np.nanmean([r.get(key, np.nan) for r in rows])) if any(np.isfinite(r.get(key, np.nan)) for r in rows) else np.nan
        m["response"] = {key: np.nanmean([r["response"][key] for r in rows], 0).tolist() for key in rows[0]["response"]}
        m["stats"] = {}
        for r in rows:
            for key, v in r.get("stats", {}).items():
                m["stats"][key] = m["stats"].get(key, 0) + v
        out.append(m)
    return out


def primary(gen, ref, idx=None):
    g = gen if idx is None else [gen[i] for i in idx]
    r = ref if idx is None else [ref[i] for i in idx]
    st = score(g, r, [f"m12_{k}" for k in S_T_KEYS])
    s10 = score(g, r, [f"s10_{k}" for k in KEYS])
    dens_g = np.array([x["m12_street_km_per_km2"] for x in g])
    dens_r = np.array([x["m12_street_km_per_km2"] for x in r])
    deficit = np.maximum(0, 1 - dens_g / np.maximum(dens_r, 1e-9))
    rc = {k: response_distance(g, r, k) for k in RC_KEYS}
    return {"S_T": st, "S10_cond": s10, "RC3": float(np.nanmean(list(rc.values()))), **{f"RC_{k}": v for k, v in rc.items()},
            "density": float(dens_g.mean()), "density_real": float(dens_r.mean()),
            "density_ratio": float(dens_g.mean() / max(dens_r.mean(), 1e-9)),
            "premature_share": float((deficit > 0.5).mean()), "termination_deficit": float(deficit.mean()),
            "giant_share": float(np.nanmean([x["giant_share"] for x in g])),
            "giant_share_real": float(np.nanmean([x["giant_share"] for x in r]))}


def summarize(res):
    tag = res["spec"]["tiles"]
    ref = real_rows(tag)
    gen = tile_means(res)
    assert [g["sid"] for g in gen] == [r["sid"] for r in ref]
    s = {"id": res["id"], "spec": res["spec"], "n_tiles": len(gen), "n_seeds": len(res["seeds"]), **primary(gen, ref)}
    for k in ("circuity_all", "pair_access", "unreachable_share", "dead_end_ratio_m13", "intersections_per_km2",
              "bridges", "river_crossings_unbridged", "river_crossings_bridged", "unsupported_structures", "viaducts",
              "earthwork_m3_per_km", "grade_p90_m13", "contour_align"):
        s[k] = float(np.nanmean([x.get(k, np.nan) for x in gen]))
        s[k + "_real"] = float(np.nanmean([x.get(k, np.nan) for x in ref]))
    s["earthwork_w1iqr"] = w1_iqr([x["earthwork_m3_per_km"] for x in gen], [x["earthwork_m3_per_km"] for x in ref])
    Xg, Xr = feature_matrix(gen, NORM), feature_matrix(ref, NORM)
    s["energy_distance"] = energy_distance(Xg, Xr)
    bw = NORM.get("bandwidth") or float(np.median(np.sqrt(((Xr[:, None] - Xr[None]) ** 2).sum(-1))[np.triu_indices(len(Xr), 1)]))
    s["mmd"] = mmd_rbf(Xg, Xr, bw)
    if res.get("diversity"):
        s["jaccard_diversity"] = float(np.nanmean([d["jaccard_distance"] for d in res["diversity"]]))
        s["orientation_diversity"] = float(np.nanmean([d["orientation_l1"] for d in res["diversity"]]))
    secs = [v["seconds"] for v in res["seeds"].values()]
    fwd = [v["forward_passes"] for v in res["seeds"].values()]
    tot = {}
    for x in gen:
        for k, v in x["stats"].items():
            tot[k] = tot.get(k, 0) + v
    ns = len(gen) * len(res["seeds"])
    s["sec_per_tile_sample"] = float(np.sum(secs) / ns)
    s["fwd_per_tile_sample"] = float(np.nansum(fwd) / ns) if np.isfinite(np.nansum(fwd)) else np.nan
    s["cand_evals_per_tile_sample"] = tot.get("cand_evals", 0) / ns
    hard = sum(v for k, v in tot.items() if k.startswith("hard_"))
    s["hard_rejection_rate"] = hard / max(tot.get("cand_evals", 0), 1)
    s["candidate_duplicate_rate"] = 1 - tot.get("cand_unique", 0) / max(tot.get("cand_evals", 0), 1)
    s["forced_stop_per_tile"] = tot.get("forced_stop", 0) / ns
    s["mean_ess"] = tot.get("ess_sum", 0) / max(tot.get("ess_n", 0), 1) if tot.get("ess_n") else np.nan
    s["km_per_second"] = float(s["density"] * 0.262144 * ns / max(np.sum(secs), 1e-9))
    s["peak_rss_mb"] = res.get("peak_rss_mb")
    return s


def contrast(res_a, res_b, key, rng=None, n_boot=N_BOOT):
    """Paired block bootstrap of key(b) - key(a): (delta, lo, hi, p two-sided)."""
    rng = rng or np.random.default_rng(2026)
    ref = real_rows(res_a["spec"]["tiles"])
    ga, gb = tile_means(res_a), tile_means(res_b)
    blocks = sorted({x["block"] for x in ref})
    bidx = {b: [i for i, x in enumerate(ref) if x["block"] == b] for b in blocks}
    f = lambda g, idx: primary(g, ref, idx)[key] if key in ("S_T", "S10_cond", "RC3", "premature_share", "density",
                                                                "giant_share", "termination_deficit", "density_ratio") \
        else float(np.nanmean([g[i].get(key, np.nan) for i in idx]))
    full = list(range(len(ref)))
    d0 = f(gb, full) - f(ga, full)
    ds = []
    for _ in range(n_boot):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        idx = [i for b in pick for i in bidx[b]]
        ds.append(f(gb, idx) - f(ga, idx))
    ds = np.array(ds)
    ds = ds[np.isfinite(ds)]
    p = 2 * min((ds > 0).mean(), (ds < 0).mean()) if len(ds) else np.nan
    return {"delta": float(d0), "lo": float(np.percentile(ds, 2.5)), "hi": float(np.percentile(ds, 97.5)), "p": float(p),
            "rel": float(d0 / f(ga, full)) if f(ga, full) else np.nan}


def holm(ps: dict, alpha=0.05):
    order = sorted(ps, key=lambda k: ps[k])
    out, m = {}, len(ps)
    for i, k in enumerate(order):
        out[k] = ps[k] * (m - i) <= alpha and all(out[j] for j in order[:i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", action="store_true")
    a = ap.parse_args()
    if a.summaries:
        out = {}
        for f in sorted(RUNS.glob("*.json")):
            res = json.loads(f.read_text())
            if res.get("seeds"):
                out[res["id"]] = summarize(res)
        (OUT / "summaries.json").write_text(json.dumps(out, indent=1, default=float))
        print(len(out), "runs summarised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
