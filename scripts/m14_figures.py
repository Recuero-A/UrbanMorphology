"""Milestone 14 figures from stored runs (never generates).

    python scripts/m14_figures.py      # -> outputs/m14/figures/

* fig_gallery_<set>.png  real | A | M* | NA-myopic | NA-fixed | NA-adaptive on tiles chosen by TERRAIN rules fixed
                         before looking at results (flattest, steepest, largest river share, largest sea share), plus
                         one failure case chosen by result (largest NA-adaptive OD unreachability, labelled as such);
                         TN_s0, inference seed 1 (first sample, not cherry-picked);
* fig_tradeoff.png       connectivity (OD unreachable, land) vs realism (S10) per method and set; marker area ∝ cost;
* fig_dose.png           planning strength (λ multiplier) vs dead ends, OD unreachability, S10 (DEV-A / DEV-B ablation).
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
import m14_experiments as mx  # noqa: E402
from m13_elevation_maps import contour_step  # noqa: E402
from m13_scoreboard import graph  # noqa: E402
from m13_three_tiles import draw  # noqa: E402
from m14_report import FZ, TN, get, method_table  # noqa: E402
from m14_plans import lam  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

FIG = PROJECT_ROOT / "outputs" / "m14" / "figures"
SHOW = ["A", "M*", "NA-myopic", "NA-fixed", "NA-adaptive"]
COL = {"A": "#8a8a8a", "M*": "#b23a48", "M*+repair": "#e09f3e", "P": "#6c757d", "NA-myopic": "#90be6d",
       "NA-fixed": "#2a6f97", "NA-adaptive": "#014f86", "NA-random": "#89c2d9", "NA-SMC": "#6c9a8b", "M13-SMC": "#c08497"}


def pick_tiles(T, runs):
    ctx = []
    for k, t in enumerate(T):
        w = t["d"]["water"][64:192, 64:192]
        rows = runs["NA-adaptive"]["seeds"]["1"]["rows"][k] if runs.get("NA-adaptive") else {}
        ctx.append({"k": k, "slope": rows.get("slope_mean_ctx", np.nan), "river": float((w == 1).mean()),
                    "sea": float((w == 3).mean()), "fail": rows.get("od_unreach_land", 0) + 0.001 * rows.get("islands", 0)})
    out = [("flattest", min(ctx, key=lambda c: c["slope"])["k"]), ("steepest", max(ctx, key=lambda c: c["slope"])["k"]),
           ("largest river share", max(ctx, key=lambda c: c["river"])["k"]),
           ("largest sea share", max(ctx, key=lambda c: c["sea"])["k"])]
    used = {k for _, k in out}
    fail = max((c for c in ctx if c["k"] not in used), key=lambda c: c["fail"])
    out.append(("FAILURE CASE (largest NA-adaptive OD unreach.)", fail["k"]))
    return out


def gallery(tiles):
    T = mx.tile_set(tiles)
    runs = {l: get(tiles, "TN_s0", l) for l in SHOW}
    if any(r is None for r in runs.values()):
        print("gallery: missing runs for", tiles)
        return
    picks = pick_tiles(T, runs)
    fig, ax = plt.subplots(len(picks), 1 + len(SHOW), figsize=(3.0 * (1 + len(SHOW)), 3.25 * len(picks)),
                           layout="constrained")
    for r, (why, k) in enumerate(picks):
        t = T[k]
        d = t["d"]
        z, w = d["elev"][64:192, 64:192], d["water"][64:192, 64:192]
        land = z[w == 0] if (w == 0).any() else z
        lo, hi = float(np.floor(land.min())), float(np.ceil(land.max())) + 1
        panels = [("Real", t["graph"], None)]
        for l in SHOW:
            row = runs[l]["seeds"]["1"]["rows"][k]
            panels.append((l, graph(runs[l]["seeds"]["1"]["graphs"][k]), row))
        for c, (lab, g, row) in enumerate(panels):
            a = ax[r, c]
            draw(a, d, g, lo, hi, contour_step(lo, hi))
            sub = "" if row is None else f"\nOD {row['od_unreach_land']:.2f} · isl {row['islands']} · dead {row['dead_end_ratio_m13']:.2f}"
            a.set_title(f"{lab}{sub}", fontsize=8)
            if c == 0:
                a.set_ylabel(f"{t['sid']}\n{why}", fontsize=8)
    fig.suptitle(f"{tiles.upper()}: real vs generated (TN_s0, first sample). OD = same-land unreachable pairs, "
                 "isl = isolated components, dead = interior dead-end ratio. Terrain: green low → beige high; "
                 "water blue; bridges red", fontsize=10)
    fig.savefig(FIG / f"fig_gallery_{tiles}.png", dpi=110)


def tradeoff():
    labels = ["A", "M*", "M*+repair", "M13-SMC", "NA-myopic", "NA-random", "NA-adaptive", "NA-fixed", "NA-SMC"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    for ax, tiles in zip(axs, ("gen2", "devB", "gen")):
        _, summ = method_table(tiles, labels)
        for l in labels:
            ss = summ.get(l) or []
            if not ss:
                continue
            x = np.array([s["od_unreach_land"] for s in ss])
            y = np.array([s["S10_cond"] for s in ss])
            c = np.mean([s["sec_per_tile_sample"] for s in ss])
            ax.errorbar(x.mean(), y.mean(), xerr=[[x.mean() - x.min()], [x.max() - x.mean()]],
                        yerr=[[y.mean() - y.min()], [y.max() - y.mean()]], fmt="none", ecolor=COL[l], lw=0.8)
            ax.scatter(x.mean(), y.mean(), s=30 + 40 * c, color=COL[l], label=f"{l} ({c:.1f} s)", zorder=3,
                       edgecolor="white")
        ax.set_xlabel("OD pairs unreachable on the same land (lower = more connected)")
        ax.set_ylabel("S10 realism gap (lower = more realistic)")
        ax.set_title(tiles.upper())
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Connectivity–realism trade-off (TN, mean over 3 training seeds, bars = seed range; marker area ∝ seconds per tile)")
    fig.savefig(FIG / "fig_tradeoff.png", dpi=140)


def dose():
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for tiles, ls in (("devA", "--"), ("devB", "-")):
        xs, vals = [], {"dead_end_ratio_m13": [], "od_unreach_land": [], "S10_cond": []}
        for m in (0.0, 0.5, 1.0, 2.0, 4.0):
            try:
                r = ma.find(tiles, "TN_s0", "energy", {"K": 8, "lam_ratio": 0.5}) if m == 0 else \
                    ma.find(tiles, "TN_s0", "na_myopic", lam(m))
            except FileNotFoundError:
                continue
            s = ma.summary(r)
            xs.append(m)
            for k in vals:
                vals[k].append(s[k])
        if not xs:
            continue
        for a, (k, v) in zip(axs, vals.items()):
            a.plot(xs, v, "o" + ls, color="#2a6f97", label=tiles)
            if k == "dead_end_ratio_m13":
                a.axhline(s[k + "_real"], color="#b23a48", ls=ls, lw=0.8, label=f"real ({tiles})")
            a.set_xlabel("λ multiplier (0 = M*, 1 = calibrated)"), a.set_title(k)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Planning strength dose–response (NA-myopic, TN_s0, inference seeds 1–2)")
    fig.savefig(FIG / "fig_dose.png", dpi=140)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    for tiles in ("gen2", "devB", "gen"):
        gallery(tiles)
    tradeoff()
    dose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
