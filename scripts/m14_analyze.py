"""Milestone 14 analysis library (docs/M14_EXPERIMENTAL_PROTOCOL.md §5-§7).

* ``find(tiles, backbone, preset, **match)``  the stored run of a spec (outputs/m14/runs);
* ``summary(res)``                          every outcome of one run on its tile set;
* ``block_contrast(pairs, key)``            per-tile metric, primary analysis: block sums of paired differences
                                            averaged over training seeds; exact sign-flip p and block-bootstrap CI;
* ``set_contrast(pairs, key)``              set-level metric (S10_cond, ...): pooled-over-seeds difference, joint
                                            block bootstrap CI and within-block label-randomisation p;
* ``noise_floor(tiles)``                    real-vs-real S10_cond between block halves;
* ``strata(rows)``                          pre-registered terrain strata per tile;
* ``density_adjusted(pairs, key)``          intercept of the per-tile difference regressed on the log-density difference.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import m13_analyze as an  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m14"
RUNS = OUT / "runs"
M14_KEYS = ("closed_giant", "islands", "island_share", "od_unreach", "od_unreach_land", "od_unreach_water",
            "od_detour", "coverage_60", "street_m_per_conn")
PER_TILE = M14_KEYS + ("dead_end_ratio_m13", "river_crossings_unbridged", "bridges", "street_km_per_km2",
                       "intersections_per_km2", "giant_share", "circuity_all", "unsupported_structures")
SET_LEVEL = ("S10_cond", "S_T", "RC3", "density_ratio", "premature_share")
N_BOOT = 2000


def real(tiles):
    return json.loads((OUT / "real" / f"{tiles}.json").read_text())


def find(tiles, backbone, pre, overrides=None, **extra):
    """The run whose spec matches exactly (overrides dict and any extra spec keys, e.g. repair=True)."""
    ov = overrides or {}
    hits = []
    for f in sorted(RUNS.glob(f"{tiles}__{backbone}__{pre}__*.json")):
        sp = json.loads(f.read_text())["spec"]
        if sp.get("overrides", {}) == ov and all(sp.get(k) == v for k, v in extra.items()) \
                and not any(k not in extra for k in ("repair", "voc") if sp.get(k)):
            hits.append(f)
    if not hits:
        raise FileNotFoundError((tiles, backbone, pre, ov, extra))
    return json.loads(hits[-1].read_text())


def tile_means(res):
    seeds = sorted(res["seeds"], key=int)
    n = len(res["seeds"][seeds[0]]["rows"])
    out = []
    for k in range(n):
        rows = [res["seeds"][s]["rows"][k] for s in seeds]
        m = {"sid": rows[0]["sid"], "block": rows[0]["block"]}
        for key, v in rows[0].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals = np.array([r.get(key, np.nan) for r in rows], float)
                m[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        m["response"] = {key: np.nanmean([r["response"][key] for r in rows], 0).tolist() for key in rows[0]["response"]}
        st = {}
        for r in rows:
            for key, v in r.get("stats", {}).items():
                st[key] = st.get(key, 0) + v
        m["stats"] = st
        out.append(m)
    return out


def summary(res, ref=None):
    ref = ref or real(res["spec"]["tiles"])
    gen = tile_means(res)
    assert [g["sid"] for g in gen] == [r["sid"] for r in ref]
    s = {"id": res["id"], "spec": res["spec"], **an.primary(gen, ref)}
    for k in PER_TILE:
        s[k] = float(np.nanmean([g.get(k, np.nan) for g in gen]))
        s[k + "_real"] = float(np.nanmean([r.get(k, np.nan) for r in ref]))
    # amendment 2 (secondary): same-land unreachability over tile-SAMPLES with a non-empty network
    ne = [r["od_unreach_land"] for x in res["seeds"].values() for r in x["rows"] if r["street_km_per_km2"] > 0]
    s["od_unreach_land_nonempty"] = float(np.mean(ne)) if ne else np.nan
    s["od_unreach_land_nonempty_real"] = s["od_unreach_land_real"]
    s["empty_share"] = float(np.mean([r["street_km_per_km2"] == 0 for x in res["seeds"].values() for r in x["rows"]]))
    from urbanmorph.graphs.m13_metrics import energy_distance, feature_matrix, mmd_rbf
    Xg, Xr = feature_matrix(gen, an.NORM), feature_matrix(ref, an.NORM)
    s["energy_distance"], s["mmd"] = energy_distance(Xg, Xr), mmd_rbf(Xg, Xr, an.NORM["bandwidth"])
    if res.get("diversity"):
        s["jaccard_diversity"] = float(np.nanmean([d["jaccard_distance"] for d in res["diversity"]]))
        s["orientation_diversity"] = float(np.nanmean([d["orientation_l1"] for d in res["diversity"]]))
    seeds = sorted(res["seeds"], key=int)
    cv = {}
    for key in ("street_km_per_km2", "closed_giant", "dead_end_ratio_m13"):
        V = np.array([[res["seeds"][x]["rows"][k][key] for x in seeds] for k in range(len(gen))], float)
        cv[key] = float(np.nanmean(np.nanstd(V, 1) / np.maximum(np.abs(np.nanmean(V, 1)), 1e-9)))
    s["across_seed_cv"] = cv
    ns = len(gen) * len(seeds)
    tot = {}
    for g in gen:
        for k, v in g["stats"].items():
            tot[k] = tot.get(k, 0) + v
    s["sec_per_tile_sample"] = float(sum(res["seeds"][x]["seconds"] for x in seeds) / ns)
    s["fwd_per_tile_sample"] = float(np.nansum([res["seeds"][x]["forward_passes"] for x in seeds]) / ns)
    s["cand_evals_per_tile_sample"] = tot.get("cand_evals", 0) / ns
    s["lookahead_steps_per_tile_sample"] = tot.get("lookahead_steps", 0) / ns
    fired, skipped = tot.get("la_fired", 0), tot.get("la_skipped", 0)
    s["la_fire_share"] = fired / (fired + skipped) if fired + skipped else np.nan
    s["mean_ess"] = tot.get("ess_sum", 0) / tot["ess_n"] if tot.get("ess_n") else np.nan
    s["maxT_last_arm_per_tile_sample"] = tot.get("maxT_last_arm", 0) / ns
    s["km_per_second"] = s["street_km_per_km2"] * 0.262144 / max(s["sec_per_tile_sample"], 1e-9)
    s["peak_rss_mb"] = res.get("peak_rss_mb")
    return s


# ------------------------------------------------------------------------------------------------ statistics
def _signflip_p(S, n_rand=20000, rng=None):
    S = np.asarray(S, float)
    obs = abs(S.sum())
    B = len(S)
    if B <= 16:
        signs = np.array(list(itertools.product((-1, 1), repeat=B)))
    else:
        signs = (rng or np.random.default_rng(0)).choice((-1, 1), size=(n_rand, B))
    stats = np.abs(signs @ S)
    return float(np.mean(stats >= obs - 1e-12))


def block_contrast(pairs, key, rng=None):
    """pairs = [(res_ref, res_method) per training seed] on the same tile set. Difference = method - ref."""
    rng = rng or np.random.default_rng(2026)
    per_seed, blocks = [], None
    for ra, rb in pairs:
        ga, gb = tile_means(ra), tile_means(rb)
        d = np.array([b.get(key, np.nan) - a.get(key, np.nan) for a, b in zip(ga, gb)], float)
        bl = [a["block"] for a in ga]
        blocks = sorted(set(bl))
        per_seed.append((d, bl))
    N = len(per_seed[0][0])
    Sb = np.zeros(len(blocks))
    seed_est = []
    for d, bl in per_seed:
        seed_est.append(float(np.nanmean(d)))
        for j, b in enumerate(blocks):
            Sb[j] += np.nansum([d[i] for i in range(N) if bl[i] == b]) / len(per_seed)
    nb = np.array([sum(1 for x in per_seed[0][1] if x == b) for b in blocks])
    est = float(Sb.sum() / N)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(len(blocks), size=len(blocks))
        boots.append(Sb[idx].sum() / nb[idx].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"key": key, "delta": est, "lo": float(lo), "hi": float(hi), "p_signflip": _signflip_p(Sb, rng=rng),
            "n_blocks": len(blocks), "per_seed": seed_est, "min_p": 2.0 ** (1 - len(blocks))}


def set_contrast(pairs, key, refrows, rng=None, n_perm=2000):
    """Set-level metric (an.primary key): difference pooled over seeds; joint block bootstrap; label randomisation."""
    rng = rng or np.random.default_rng(2026)
    G = [(tile_means(a), tile_means(b)) for a, b in pairs]
    blocks = sorted({r["block"] for r in refrows})
    bidx = {b: [i for i, r in enumerate(refrows) if r["block"] == b] for b in blocks}

    def f(idx, swap=None):
        out = []
        for ga, gb in G:
            if swap is not None:
                ga2, gb2 = list(ga), list(gb)
                for i in swap:
                    ga2[i], gb2[i] = gb[i], ga[i]
                ga, gb = ga2, gb2
            out.append(an.primary([gb[i] for i in idx], [refrows[i] for i in idx])[key]
                       - an.primary([ga[i] for i in idx], [refrows[i] for i in idx])[key])
        return float(np.mean(out)), out
    full = list(range(len(refrows)))
    est, per = f(full)
    boots = []
    for _ in range(N_BOOT // 2):
        pick = rng.choice(blocks, size=len(blocks))
        boots.append(f([i for b in pick for i in bidx[b]])[0])
    perm = []
    for _ in range(n_perm // 4):
        sw = [i for b in blocks if rng.uniform() < 0.5 for i in bidx[b]]
        perm.append(abs(f(full, sw)[0]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"key": key, "delta": est, "lo": float(lo), "hi": float(hi),
            "p_perm": float((1 + np.sum(np.array(perm) >= abs(est) - 1e-12)) / (1 + len(perm))), "per_seed": per}


def noise_floor(refrows, rng=None, n=2000, key="S10_cond"):
    """Distribution of the set-level metric between two block halves of the REAL tiles."""
    rng = rng or np.random.default_rng(7)
    blocks = sorted({r["block"] for r in refrows})
    vals = []
    for _ in range(n):
        perm = rng.permutation(blocks)
        h = set(perm[: len(blocks) // 2])
        a = [r for r in refrows if r["block"] in h]
        b = [r for r in refrows if r["block"] not in h]
        if len(a) >= 3 and len(b) >= 3:
            keys = [f"s10_{k}" for k in an.KEYS] if key == "S10_cond" else [f"m12_{k}" for k in an.S_T_KEYS]
            vals.append(an.score(a, b, keys))
    v = np.array(vals)
    return {"key": key, "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": len(v)}


def strata(r):
    s = r.get("slope_mean_ctx", np.nan)
    out = ["flat" if s < 0.05 else "moderate" if s < 0.10 else "steep" if s < 0.15 else "mountainous"]
    return out


def tile_strata(tiles):
    """Pre-registered strata from the terrain arrays (river share, sea, slope)."""
    res = {}
    for t in tiles:
        wk = t["d"]["water"][64:192, 64:192]
        res[t["sid"]] = {"river": bool((wk == 1).mean() >= 0.01), "coastal": bool((wk == 3).any()),
                         "water_any": float((wk > 0).mean())}
    return res


def density_adjusted(pairs, key, rng=None):
    """Per-tile difference of `key` regressed on the per-tile difference of log density; intercept + block bootstrap."""
    rng = rng or np.random.default_rng(3)
    X, Y, B = [], [], []
    for ra, rb in pairs:
        for a, b in zip(tile_means(ra), tile_means(rb)):
            da, db = a["street_km_per_km2"], b["street_km_per_km2"]
            if da > 0 and db > 0 and np.isfinite(a.get(key, np.nan)) and np.isfinite(b.get(key, np.nan)):
                X.append(np.log(db) - np.log(da))
                Y.append(b[key] - a[key])
                B.append(a["block"])
    X, Y, B = np.array(X), np.array(Y), np.array(B)
    fit = lambda m: np.polyfit(X[m], Y[m], 1) if m.sum() > 3 else (np.nan, np.nan)
    slope, icpt = fit(np.ones(len(X), bool))
    blocks = sorted(set(B))
    boots = []
    for _ in range(N_BOOT // 2):
        pick = rng.choice(blocks, size=len(blocks))
        m = np.concatenate([np.flatnonzero(B == b) for b in pick])
        if len(m) > 3:
            boots.append(np.polyfit(X[m], Y[m], 1)[1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"key": key, "intercept": float(icpt), "lo": float(lo), "hi": float(hi), "slope": float(slope), "n": len(X)}


def holm(ps: dict, alpha=0.05):
    return an.holm(ps, alpha)


# ------------------------------------------------------------------------------------------------ trigger analysis
def trigger_table(voc_rows, extra_q=(50, 75)):
    """VoC captured vs firing share for every candidate trigger (protocol §4)."""
    v = np.array([r["voc"] for r in voc_rows])
    F = {k: np.array([bool(r[k]) for r in voc_rows]) for k in ("island", "deg1", "water", "steep", "feas_low")}
    ws = np.array([r["w_spread"] for r in voc_rows])
    for q in extra_q:
        thr = float(np.percentile(ws, q))
        F[f"w_spread>{thr:.3f}"] = ws > thr
    single = dict(F)
    for (a, fa), (b, fb) in itertools.combinations(single.items(), 2):
        F[f"any:{a}+{b}"] = fa | fb
    tot = v.sum()
    rows = []
    for k, m in F.items():
        share = float(m.mean())
        cap = float(v[m].sum() / tot) if tot > 0 else np.nan
        rows.append({"trigger": k, "fire_share": share, "voc_captured": cap, "ratio": cap / max(share, 1e-9)})
    return sorted(rows, key=lambda r: (-r["voc_captured"], r["fire_share"])), float(tot), len(v)


def choose_trigger(rows, min_capture=0.70):
    ok = [r for r in rows if r["voc_captured"] >= min_capture and r["fire_share"] < 0.999]
    if ok:
        return min(ok, key=lambda r: (r["fire_share"], -r["voc_captured"])), False
    return max(rows, key=lambda r: r["ratio"]), True
