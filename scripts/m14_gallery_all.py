"""M14 side-by-side gallery of every method on GEN2 (one typical tile per city).

Tiles are chosen by a fixed rule, before looking at any generated sample: per city, the tile whose real street count is
the median of that city's tiles. Each panel is the first stored sample (inference seed 1, TN_s0) from the confirmatory runs;
nothing is re-generated.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import m14_experiments as mx  # noqa: E402
from m13_elevation_maps import contour_step  # noqa: E402
from m13_scoreboard import graph  # noqa: E402
from m13_three_tiles import draw  # noqa: E402
from m14_figures import FIG  # noqa: E402
from m14_report import get  # noqa: E402

SHOW = ["P", "A", "M*", "M*+repair", "M13-SMC", "NA-myopic", "NA-fixed", "NA-SMC"]
CITY = {"ALY": "Alcoi", "TGN": "Tarragona", "SAC": "Sacramento", "SBA": "Santa Barbara"}


def n_edges(g):
    return len(g.edges)


def pick(T):
    by_city = {}
    for k, t in enumerate(T):
        by_city.setdefault(t["sid"].split("_")[1], []).append(k)
    out = []
    for c, ks in by_city.items():
        ks = sorted(ks, key=lambda k: (n_edges(T[k]["graph"]), T[k]["sid"]))
        out.append((CITY.get(c, c), ks[len(ks) // 2]))
    return out


def main(seed="1") -> int:
    T = mx.tile_set("gen2")
    runs = {l: get("gen2", "TN_s0", l) for l in SHOW}
    missing = [l for l, r in runs.items() if r is None]
    if missing:
        print("missing runs:", missing)
        return 1
    picks = pick(T)
    fig, ax = plt.subplots(len(picks), 1 + len(SHOW), figsize=(2.7 * (1 + len(SHOW)), 2.95 * len(picks)),
                           layout="constrained")
    for r, (city, k) in enumerate(picks):
        t, d = T[k], T[k]["d"]
        z, w = d["elev"][64:192, 64:192], d["water"][64:192, 64:192]
        land = z[w == 0] if (w == 0).any() else z
        lo, hi = float(np.floor(land.min())), float(np.ceil(land.max())) + 1
        panels = [("Real", t["graph"], None)]
        for l in SHOW:
            s = runs[l]["seeds"][seed]
            panels.append((l, graph(s["graphs"][k]), s["rows"][k]))
        for c, (lab, g, row) in enumerate(panels):
            a = ax[r, c]
            draw(a, d, g, lo, hi, contour_step(lo, hi))
            sub = "" if row is None else f"\nOD {row['od_unreach_land']:.2f} · isl {row['islands']} · dead {row['dead_end_ratio_m13']:.2f}"
            a.set_title(f"{lab}{sub}", fontsize=8, fontweight="bold" if lab in ("Real", "NA-SMC") else "normal")
            if c == 0:
                a.set_ylabel(f"{city}\n{t['sid']}", fontsize=8)
    fig.suptitle("GEN2 (unseen cities): real vs every method. One median-density tile per city, TN_s0, inference seed "
                 f"{seed}. OD = same-land unreachable pairs, isl = isolated components, dead = dead-end ratio. "
                 "Terrain green low → beige high; water blue; bridges red", fontsize=10)
    out = FIG / f"fig_gallery_all_gen2_s{seed}.png"
    fig.savefig(out, dpi=110)
    print(out, [(c, T[k]["sid"]) for c, k in picks])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
