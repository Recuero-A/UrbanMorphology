"""Three real-terrain tiles from the unseen GEN cities: real streets vs M* vs A, with per-tile scores.

    python scripts/m13_three_tiles.py   # -> outputs/m13/three_tiles/{three_tiles.png, scores.json}

Exploratory illustration (development data; the held-out test splits are never read). Tiles are chosen by a fixed
TERRAIN rule before any generation: per GEN city, the tile with the largest water share x mean landform slope.
Backbone TN_s0; M* = the frozen selection (outputs/m13/selection.json), A = standard sampling; inference seeds 1-4,
the map shows seed 1 (first sample, not cherry-picked). Networks are drawn exactly as generated.

Per-tile gaps (NOT the protocol's S_T / S10, which compare distributions over many tiles): for each S_T or S10
metric, |generated - real on the same tile| / IQR of that metric over the real GEN tiles, averaged. 0 = the same as the
real streets of that tile. Reference (network gap only): the median gap between this tile's real streets and the
other real tiles of the same city, i.e. how different another real neighbourhood would score. No terrain-gap reference:
other tiles have other terrain, so that comparison would be unfair.
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
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource, LinearSegmentedColormap

import m13_experiments as ex  # noqa: E402
from m13_elevation_maps import HYPSO, contour_step  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.m13_metrics import tile_metrics_m13
from urbanmorph.graphs.tc_potentials import Potentials, TerrainField
from urbanmorph.graphs.tc_sampler import preset, sample_tc
from urbanmorph.graphs.terrain_generate import terrain_of

OUT = PROJECT_ROOT / "outputs" / "m13" / "three_tiles"
SEEDS = [1, 2, 3, 4]
CITY = {"ESP_GRO": "Girona", "ESP_MRS": "Manresa", "USA_SFO": "San Francisco"}
T_KEYS = ["m12_grade_p90", "m12_share_gt10", "m12_contour_index", "m12_steep_density_ratio", "m12_circuity"]
N_KEYS = ["s10_orientation_phi", "s10_orientation_entropy", "s10_junction_angle_dev_deg", "s10_intersections_per_km2",
          "s10_dead_end_ratio", "s10_mean_link_length_m", "s10_street_length_km_per_km2", "s10_loops_per_km2",
          "s10_share_deg4_of_junctions", "s10_components"]
SHOW = [("street_km_per_km2", "street density, km/km²", "{:.1f}"),
        ("intersections_per_km2", "intersections /km²", "{:.0f}"),
        ("giant_share", "largest connected piece", "{:.0%}"),
        ("dead_end_ratio", "dead-end ratio", "{:.2f}"),
        ("m12_grade_p90", "grade p90 (steep streets)", "{:.1%}"),
        ("m12_share_gt10", "length steeper than 10 %", "{:.0%}"),
        ("m12_contour_index", "contour index (0 along, 1 up)", "{:.2f}"),
        ("m12_circuity", "circuity", "{:.2f}"),
        ("bridges", "bridges", "{:.0f}"),
        ("m12_unbridged_water_m", "unbridged street over water, m", "{:.0f}")]


def pick(real):
    out = []
    for code in CITY:
        rows = [r for r in real if r["sid"].startswith(code)]
        out.append(max(rows, key=lambda r: r["water_share_ctx"] * r["slope_mean_ctx"])["sid"])
    return out


def iqr(real, keys):
    v = {k: np.array([r.get(k, np.nan) for r in real], float) for k in keys}
    return {k: float(np.subtract(*np.nanpercentile(x, [75, 25]))) or 1.0 for k, x in v.items()}


def gap(a, b, keys, q):
    d = [abs(a[k] - b[k]) / q[k] for k in keys if np.isfinite(a.get(k, np.nan)) and np.isfinite(b.get(k, np.nan))]
    return float(np.mean(d)) if d else np.nan


def draw(ax, d, g, lo, hi, step):
    z, w = d["elev"][64:192, 64:192], d["water"][64:192, 64:192]
    rgb = LightSource(315, 40).shade(z, cmap=HYPSO, vmin=lo, vmax=hi, blend_mode="soft", vert_exag=2.0, dx=4, dy=4)
    ext = (0, 512, 0, 512)
    ax.imshow(rgb, extent=ext, origin="upper")
    ax.imshow(np.ma.masked_where(w == 0, w), cmap=LinearSegmentedColormap.from_list("w", ["#5b9bd5", "#2e6aa8"]),
              vmin=0, vmax=4, extent=ext, origin="upper", alpha=0.9)
    yy, xx = np.mgrid[2:512:4, 2:512:4]
    ax.contour(xx, 512 - yy, z, levels=np.arange(np.ceil(lo / step) * step, hi + 1e-9, step),
               colors="#3b2f25", linewidths=0.35, alpha=0.55)
    if len(g.edges):
        seg, br = g.segments().reshape(-1, 2, 2), np.asarray(g.structs()) == 1
        for m, col, lw in ((~br, "#1a1a1a", 1.2), (br, "#c0392b", 2.0)):
            ax.add_collection(LineCollection(seg[m], colors="white", linewidths=lw + 1.4, capstyle="round"))
            ax.add_collection(LineCollection(seg[m], colors=col, linewidths=lw, capstyle="round"))
    ax.set_xlim(0, 512), ax.set_ylim(0, 512), ax.set_xticks([]), ax.set_yticks([])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = {t["sid"]: t for t in ex.tile_set("gen")}
    real = ex.real_rows(list(tiles.values()), "gen")
    real_by = {r["sid"]: r for r in real}
    sids = pick(real)
    qT, qN = iqr(real, T_KEYS), iqr(real, N_KEYS)
    sel = json.loads((ex.OUT / "selection.json").read_text())["M_star"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, names, norm, _ = ex.load_backbone("TN_s0", device)
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")
    ds = [tiles[s]["d"] for s in sids]
    tfs = [TerrainField(terrain_of(d)) for d in ds]
    methods = {"M*": preset(sel["preset"], **sel["overrides"]), "A": preset("standard")}
    gen, shown = {m: {s: [] for s in sids} for m in methods}, {}
    for m, cfg in methods.items():
        for seed in SEEDS:
            gs, _, _ = sample_tc(model, ds, names, norm, pots, cfg, seed=seed, device=device)
            rng = np.random.default_rng(1000 + seed)
            for s, g, tf, d in zip(sids, gs, tfs, ds):
                row = {**ex.m12_rows(g, d), **tile_metrics_m13(g, tf, rng)}
                row["terrain_gap"] = gap(row, real_by[s], T_KEYS, qT)
                row["network_gap"] = gap(row, real_by[s], N_KEYS, qN)
                gen[m][s].append(row)
                if seed == SEEDS[0]:
                    shown[(m, s)] = g
            print(m, "seed", seed, flush=True)
    import pickle
    (OUT / "graphs_seed1.pkl").write_bytes(pickle.dumps({f"{m}|{s}": g for (m, s), g in shown.items()}))
    res = {}
    for s in sids:
        others = [r for r in real if r["sid"][:7] == s[:7] and r["sid"] != s]
        ref = {"terrain_gap": float(np.nanmedian([gap(r, real_by[s], T_KEYS, qT) for r in others])),
               "network_gap": float(np.nanmedian([gap(r, real_by[s], N_KEYS, qN) for r in others]))}
        res[s] = {"city": CITY[s[:7]], "real": real_by[s], "other_real_tiles_same_city": ref,
                  **{m: {"seed1": gen[m][s][0], "mean_4_seeds": {k: float(np.nanmean([r[k] for r in gen[m][s]]))
                                                                 for k in [k for k, _, _ in SHOW] + ["terrain_gap", "network_gap"]}}
                     for m in methods}}
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))

    fig = plt.figure(figsize=(17, 5.2 * len(sids)), layout="constrained")
    sub = fig.subfigures(len(sids), 1)
    for sf, s, d in zip(sub, sids, ds):
        z = d["elev"][64:192, 64:192]
        land = z[d["water"][64:192, 64:192] == 0]
        lo, hi = float(np.floor(land.min())), float(np.ceil(land.max()))
        step = contour_step(lo, hi)
        ax = sf.subplots(1, 4, width_ratios=[1, 1, 1, 1.15])
        panels = [("Real streets", tiles[s]["graph"]), ("M* (energy, K=8, λ=0.5)", shown[("M*", s)]),
                  ("A (standard sampling)", shown[("A", s)])]
        for a, (t, g) in zip(ax[:3], panels):
            draw(a, d, g, lo, hi, step)
            a.set_title(t, fontsize=10)
        sm = plt.cm.ScalarMappable(cmap=HYPSO, norm=plt.Normalize(lo, hi))
        cb = sf.colorbar(sm, ax=ax[2], fraction=0.04, pad=0.01)
        cb.set_label(f"elevation (m), contours {step:g} m", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        r, gm, ga = res[s]["real"], res[s]["M*"]["seed1"], res[s]["A"]["seed1"]
        rows = [[lab, f.format(r.get(k, np.nan)), f.format(gm[k]), f.format(ga[k])] for k, lab, f in SHOW]
        ref = res[s]["other_real_tiles_same_city"]
        rows += [["terrain gap to real (lower better)", "—", f"{gm['terrain_gap']:.2f}",
                  f"{ga['terrain_gap']:.2f}"],
                 ["network gap to real (lower better)", f"{ref['network_gap']:.2f}*", f"{gm['network_gap']:.2f}",
                  f"{ga['network_gap']:.2f}"]]
        ax[3].set_axis_off()
        tb = ax[3].table(cellText=rows, colLabels=["seed 1", "Real", "M*", "A"], loc="center", cellLoc="center",
                         colWidths=[0.52, 0.16, 0.16, 0.16])
        tb.auto_set_font_size(False), tb.set_fontsize(7.5), tb.scale(1, 1.32)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor("#cccccc")
            if i == 0 or j == 0:
                c.set_text_props(weight="bold" if i == 0 else "normal", ha="left" if j == 0 else "center")
            if i > len(SHOW):
                c.set_facecolor("#f3efe6")
        ax[3].text(0.0, 0.02, "* network gap of the other real tiles of the same city (median): what a different real\n"
                   "  neighbourhood scores. Gaps are per-tile diagnostics, not the protocol's S_T / S10;\n"
                   "  a sparse network can show a small terrain gap simply by building little.",
                   transform=ax[3].transAxes, fontsize=6.5, color="#555555")
        sf.suptitle(f"{res[s]['city']}  ({s}):  water {r['water_share_ctx']:.0%} of tile, mean slope "
                    f"{r['slope_mean_ctx']:.0%}", fontsize=11, x=0.02, ha="left")
    fig.savefig(OUT / "three_tiles.png", dpi=120)
    print(json.dumps({s: {m: {k: round(v, 3) for k, v in res[s][m]["mean_4_seeds"].items()} for m in methods}
                      | {"ref": res[s]["other_real_tiles_same_city"]} for s in sids}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
