"""Milestone 18: how real street networks adapt to terrain (the reference every generator is judged against).

Real OSM street graphs on LiDAR bare-earth terrain (Spain IGN MDT05, USA 3DEP, resampled to 4 m): the M12 TRAIN and VAL
tiles that have a DEM, and the GEN2 tiles (Alcoi, Tarragona, Santa Barbara, Sacramento). The sealed TEST split is never
loaded. Every ground edge (bridges and tunnels left out) is cut into pieces of at most 10 m inside the 512 m tile:
* street grade = |dz| / length on the raw DEM (the road surface: LiDAR carries the street beds);
* terrain slope and fall line at the piece = gradient of the DEM low-passed with a Gaussian of sigma 16 m (the landform
  around the street, without its cut and fill);
* contour index = |cos| between the piece and the fall line (0 along the contours, 1 straight down; a random orientation
  averages 2/pi = 0.64).
Per terrain-slope class (0-5, 5-10, 10-20, 20-30, > 30 %), weighted by street length: the street grade quantiles, the
share of street length steeper than 10 / 15 / 20 %, the median ratio street grade / terrain slope, the mean contour
index, and street density relative to the land of that class (siting). Also per city.

    python scripts/m18_real_terrain_study.py  ->  outputs/m18/real_terrain/{fig_real_terrain.png, summary.json, README.md}
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
from scipy.ndimage import gaussian_filter

from urbanmorph.experiments.jsonio import atomic_json
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.terrain.heightfield import Terrain

OUT = ROOT / "outputs" / "m18" / "real_terrain"
TERRAIN = ROOT / "data" / "processed" / "m12" / "terrain"
TILE = 512.0
CLASSES = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 1000)]
CITY = {"ESP_BCN": "Barcelona", "ESP_MAD": "Madrid", "ESP_VLC": "Valencia", "USA_NYC": "New York", "USA_CHI": "Chicago",
        "USA_LAX": "Los Angeles", "ESP_ALY": "Alcoi", "ESP_TGN": "Tarragona", "USA_SBA": "Santa Barbara",
        "USA_SAC": "Sacramento"}


def tiles():
    """(sid, set, graph, d) for real tiles with a valid DEM: M12 train + val (never test) and GEN2."""
    d = pickle.loads((ROOT / "data" / "processed" / "m12" / "graphs3d.pkl").read_bytes())
    for split in ("train", "val"):
        for sid, _, g in d[split]:
            f = TERRAIN / f"{sid}.npz"
            if f.exists():
                z = dict(np.load(f))
                if z["valid"].all():
                    yield sid, split, g, z
    from m14_build_gen2 import gen2_tiles
    for t in gen2_tiles():
        yield t["sid"], "gen2", t["graph"], t["d"]


def pieces(g, d):
    """Street pieces inside the tile: length, street grade, terrain slope, contour index (ground edges only)."""
    E = np.asarray(g.edges)
    if len(E) == 0:
        return None
    P = np.asarray(g.nodes)[:, :2]
    st = np.asarray(g.structs()) if hasattr(g, "structs") else np.zeros(len(E), int)
    T = terrain_of(d)
    zs = gaussian_filter(np.where(d["valid"], d["elev"], np.nanmean(d["elev"])).astype(np.float64), 4.0)
    S = Terrain(zs.astype(np.float32), -256.0, 768.0, 4.0)
    gx, gy = S.gradient()
    A, B = P[E[:, 0]], P[E[:, 1]]
    L = np.hypot(*(B - A).T)
    n = np.maximum(1, np.ceil(L / 10.0)).astype(int)
    k = np.repeat(np.arange(len(E)), n)
    j = np.concatenate([np.arange(m) for m in n])
    p0 = A[k] + (j / n[k])[:, None] * (B[k] - A[k])
    p1 = A[k] + ((j + 1) / n[k])[:, None] * (B[k] - A[k])
    mid = (p0 + p1) / 2
    pl = L[k] / n[k]
    ok = (mid[:, 0] >= 0) & (mid[:, 0] <= TILE) & (mid[:, 1] >= 0) & (mid[:, 1] <= TILE) & (st[k] == 0) & (pl > 0.5)
    p0, p1, mid, pl = p0[ok], p1[ok], mid[ok], pl[ok]
    gr = np.abs(T.sample(p1[:, 0], p1[:, 1]) - T.sample(p0[:, 0], p0[:, 1])) / pl * 100.0
    r, c = S.rc(mid[:, 0], mid[:, 1])
    r = np.clip(np.round(r).astype(int), 0, zs.shape[0] - 1)
    c = np.clip(np.round(c).astype(int), 0, zs.shape[1] - 1)
    sx, sy = gx[r, c], gy[r, c]
    slope = np.hypot(sx, sy) * 100.0
    u = (p1 - p0) / pl[:, None]
    ci = np.abs(u[:, 0] * sx + u[:, 1] * sy) / np.maximum(np.hypot(sx, sy), 1e-9)
    m = np.isfinite(gr)
    return pl[m], gr[m], slope[m], ci[m]


def land_slope(d) -> np.ndarray:
    """Terrain slope (%) of every land cell inside the tile (sigma 16 m), for the siting denominator."""
    zs = gaussian_filter(np.where(d["valid"], d["elev"], np.nanmean(d["elev"])).astype(np.float64), 4.0)
    S = Terrain(zs.astype(np.float32), -256.0, 768.0, 4.0)
    gx, gy = S.gradient()
    m = S.window(0, 0, TILE, TILE) & (d["water"] == 0)
    return np.hypot(gx, gy)[m] * 100.0


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, c, x[o]))


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    rows, land, per_city, n_tiles = [], [], {}, {}
    for sid, split, g, d in tiles():
        pc = pieces(g, d)
        city = CITY.get(sid[:7], sid[:7])
        n_tiles[city] = n_tiles.get(city, 0) + 1
        ls = land_slope(d)
        land.append(ls)
        per_city.setdefault(city, {"pieces": [], "land": []})["land"].append(ls)
        if pc is not None:
            rows.append(pc)
            per_city[city]["pieces"].append(pc)
    print("tiles", sum(n_tiles.values()), n_tiles, flush=True)
    L, G, SL, CI = (np.concatenate([r[i] for r in rows]) for i in range(4))
    LAND = np.concatenate(land)
    cell_km2 = 16e-6

    def summarise(L, G, SL, CI, LAND):
        out = {}
        for lo, hi in CLASSES:
            m = (SL >= lo) & (SL < hi)
            lm = (LAND >= lo) & (LAND < hi)
            key = f"{lo}-{hi if hi < 1000 else 'inf'}"
            if m.sum() < 20:
                out[key] = None
                continue
            w = L[m]
            out[key] = {"street_km": float(w.sum() / 1000), "land_km2": float(lm.sum() * cell_km2),
                        "density_km_per_km2": float(w.sum() / 1000 / max(lm.sum() * cell_km2, 1e-9)),
                        "grade_p50": wq(G[m], w, 0.5), "grade_p90": wq(G[m], w, 0.9), "grade_p99": wq(G[m], w, 0.99),
                        "share_gt10": float(w[G[m] > 10].sum() / w.sum()), "share_gt15": float(w[G[m] > 15].sum() / w.sum()),
                        "share_gt20": float(w[G[m] > 20].sum() / w.sum()),
                        "ratio_grade_to_slope_med": wq(G[m] / np.maximum(SL[m], 1e-6), w, 0.5),
                        "contour_index_mean": float(np.average(CI[m], weights=w)),
                        "terrain_slope_med": wq(SL[m], w, 0.5)}
        out["all"] = {"street_km": float(L.sum() / 1000), "grade_p50": wq(G, L, 0.5), "grade_p90": wq(G, L, 0.9),
                      "grade_p99": wq(G, L, 0.99), "share_gt10": float(L[G > 10].sum() / L.sum()),
                      "share_gt15": float(L[G > 15].sum() / L.sum()), "share_gt20": float(L[G > 20].sum() / L.sum()),
                      "land_slope_p50": float(np.median(LAND)), "land_slope_p90": float(np.percentile(LAND, 90))}
        return out

    summ = {"tiles": n_tiles, "pooled": summarise(L, G, SL, CI, LAND), "cities": {}}
    for city, v in per_city.items():
        if not v["pieces"]:
            continue
        cl, cg, cs, cc = (np.concatenate([r[i] for r in v["pieces"]]) for i in range(4))
        summ["cities"][city] = summarise(cl, cg, cs, cc, np.concatenate(v["land"]))
    atomic_json(OUT / "summary.json", summ)

    # figure: street grade vs terrain slope (2-D density), contour index and density by terrain class, per-city tails
    ink, ink2, grid_c, accent, ref = "#1f2328", "#59636e", "#e6e8eb", "#2a78d6", "#d1242f"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.6))
    ax = axs[0, 0]
    h, xe, ye = np.histogram2d(np.clip(SL, 0, 60), np.clip(G, 0, 60), bins=60, range=[[0, 60], [0, 60]], weights=L)
    h = h / h.sum(axis=1, keepdims=True).clip(1e-9)                       # per terrain-slope column: where streets sit
    ax.pcolormesh(xe, ye, h.T, cmap="Blues", vmax=np.percentile(h[h > 0], 98))
    ax.plot([0, 60], [0, 60], color=ref, lw=1.2, ls="--")
    ax.text(41, 46, "street straight down\nthe slope (grade = slope)", color=ref, fontsize=8.5, ha="left")
    for t in (10, 15, 20):
        ax.axhline(t, color=ink2, lw=0.6, ls=":")
    xs = [(lo + min(hi, 60)) / 2 for lo, hi in CLASSES]
    p90 = [summ["pooled"][f"{lo}-{hi if hi < 1000 else 'inf'}"]["grade_p90"] if summ["pooled"][f"{lo}-{hi if hi < 1000 else 'inf'}"] else np.nan for lo, hi in CLASSES]
    p50 = [summ["pooled"][f"{lo}-{hi if hi < 1000 else 'inf'}"]["grade_p50"] if summ["pooled"][f"{lo}-{hi if hi < 1000 else 'inf'}"] else np.nan for lo, hi in CLASSES]
    ax.plot(xs, p50, "o-", color=ink, lw=1.6, ms=5, label="street grade median")
    ax.plot(xs, p90, "s--", color=ink, lw=1.2, ms=4.5, label="street grade p90")
    ax.set(xlim=(0, 60), ylim=(0, 60), xlabel="terrain slope around the street (%, σ 16 m)", ylabel="street grade (%)")
    ax.set_title("Real streets are much flatter than the land they cross", loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax = axs[0, 1]
    labels = [f"{lo}–{hi}" if hi < 1000 else f"> {lo}" for lo, hi in CLASSES]
    keys = [f"{lo}-{hi if hi < 1000 else 'inf'}" for lo, hi in CLASSES]
    ci = [summ["pooled"][k]["contour_index_mean"] if summ["pooled"][k] else np.nan for k in keys]
    ax.bar(range(len(keys)), ci, color=accent, width=0.6)
    ax.axhline(2 / np.pi, color=ref, lw=1.2, ls="--")
    ax.text(len(keys) - 0.6, 2 / np.pi + 0.015, "random orientation (0.64)", color=ref, ha="right", fontsize=8.5)
    for i, v in enumerate(ci):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=ink)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    ax.set(ylim=(0, 0.8), xlabel="terrain slope class (%)", ylabel="mean |cos| to the fall line (0 = along contours)")
    ax.set_title("On steeper land, streets turn to follow the contours", loc="left", fontweight="bold", fontsize=10.5)
    ax = axs[1, 0]
    dens = [summ["pooled"][k]["density_km_per_km2"] if summ["pooled"][k] else np.nan for k in keys]
    landkm = [summ["pooled"][k]["land_km2"] if summ["pooled"][k] else 0 for k in keys]
    ax.bar(range(len(keys)), dens, color=accent, width=0.6)
    for i, (v, a) in enumerate(zip(dens, landkm)):
        ax.text(i, v + 0.3, f"{v:.1f}\n({a:.0f} km² land)", ha="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels)
    ax.set(xlabel="terrain slope class (%)", ylabel="street km per km² of that land", ylim=(0, max(dens) * 1.3))
    ax.set_title("And they are built mostly on gentler land (siting)", loc="left", fontweight="bold", fontsize=10.5)
    ax = axs[1, 1]
    cities = sorted(summ["cities"], key=lambda c: summ["cities"][c]["all"]["share_gt10"])
    y = np.arange(len(cities))
    for t, mk, col in ((10, "o", ink), (15, "s", accent), (20, "^", ref)):
        ax.plot([summ["cities"][c]["all"][f"share_gt{t}"] * 100 for c in cities], y, mk, color=col, ms=6, label=f"> {t} %")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({n_tiles[c]} tiles)" for c in cities])
    ax.set(xlabel="share of street length steeper than the threshold (%)")
    ax.set_title("Steep streets are rare in every city, but not equally rare", loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for a in axs.ravel():
        a.grid(color=grid_c, lw=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(f"How real street networks adapt to terrain: {sum(n_tiles.values())} real tiles, {L.sum() / 1000:.0f} km of ground streets, "
                 "LiDAR terrain (Spain MDT05, USA 3DEP)", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "fig_real_terrain.png", dpi=130)
    plt.close(fig)
    md = ["# M18: how real street networks adapt to terrain", "", "Script `scripts/m18_real_terrain_study.py`; figure `fig_real_terrain.png`. "
          f"Tiles: {json.dumps(n_tiles)}. Test split never loaded.", "",
          "| Terrain slope (%) | street km | land km² | street km/km² | grade p50 / p90 / p99 (%) | > 10 / 15 / 20 % (share of length) | grade ÷ slope (median) | contour index (mean) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, lab in zip(keys, labels):
        s = summ["pooled"][k]
        if s:
            md.append(f"| {lab} | {s['street_km']:.0f} | {s['land_km2']:.1f} | {s['density_km_per_km2']:.1f} | {s['grade_p50']:.1f} / {s['grade_p90']:.1f} / {s['grade_p99']:.1f} | "
                      f"{s['share_gt10']:.3f} / {s['share_gt15']:.3f} / {s['share_gt20']:.3f} | {s['ratio_grade_to_slope_med']:.2f} | {s['contour_index_mean']:.2f} |")
    a = summ["pooled"]["all"]
    md += ["", f"All streets: grade p50 {a['grade_p50']:.1f} %, p90 {a['grade_p90']:.1f} %, p99 {a['grade_p99']:.1f} %; share > 10 / 15 / 20 %: "
           f"{a['share_gt10']:.3f} / {a['share_gt15']:.3f} / {a['share_gt20']:.3f}. Land slope p50 {a['land_slope_p50']:.1f} %, p90 {a['land_slope_p90']:.1f} %."]
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
