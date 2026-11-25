"""Milestone 18, leakage control (b) (docs/M18_LOG.md D7): terrain adaptation on natural terrain without street beds.

Urban LiDAR DEMs carry the imprint of the real streets, so a model's terrain response on city tiles can partly be
copying (outputs/m18/erasure). Here the same frozen generators run on NATURAL windows of the same LiDAR source (IGN MDT
5 m, the M12 training source for Spain) where almost no road exists, so there is nothing to copy. Their terrain
response is compared with the real-city reference of scripts/m18_real_terrain_study.py.

select   - Windows chosen by a rule written before any generation (outputs/m18/natural/windows.json):
           candidate centres on a 0.08 degree grid inside three Spanish regions with local OSM extracts (Cataluna,
           Madrid, Valencia; boxes below), visited in a seeded random order (seed 2026); a candidate is kept when
           (0) the window lies inside the region's OSM extract: at least 2 km of highway=* ways within 5 km of the
           centre (amendment, 2026-11-26, before any generation: the region boxes are wider than the extracts, so a
           window outside the extract showed 0 km of roads for lack of data, not for lack of roads),
           (1) all OSM ways tagged highway=* inside the 1,024 m window (tile + 256 m margin) total <= 0.5 km,
           (2) the IGN DTM covers the window fully, (3) the median terrain slope inside the 512 m tile (DEM low-passed
           with sigma 16 m) is between 5 and 35 %; the first 16 kept windows are used. Water layer: none (natural
           windows; OSM waterways are not rasterised here).
generate - The M14 samplers on the TN backbones (s0-s2, inference seeds 1-2): A, M*, M13-SMC, NA-myopic, NA-fixed,
           NA-SMC; P (procedural, seeds 1-2); and M* on N0 (terrain-blind), TNlp and TNse (s0-s2), the leakage-control
           backbones: on natural terrain their inputs carry no street beds, so TN, TNlp and TNse should respond alike if
           TN's terrain response is genuine.
score    - Street pieces of <= 10 m on ground edges, measured as in the real-city study: grade on the raw DEM, terrain
           slope and fall line on the DEM low-passed at sigma 16 m, contour index |cos| to the fall line; per terrain
           slope class: street density (siting), median grade / slope, mean contour index, shares above 10 / 15 / 20 %.

    python scripts/m18_natural_terrain.py select|generate|score  ->  outputs/m18/natural/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from scipy import ndimage

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m18" / "natural"
RAW = ROOT / "data" / "raw" / "m18_natural" / "windows"
PROC = ROOT / "data" / "processed" / "m18" / "natural"
OSM = ROOT / "data" / "raw" / "osm_extracts"
REGIONS = {"cataluna": ("europe_spain_cataluna-latest.osm.pbf", (40.55, 0.20, 42.85, 3.30)),
           "madrid": ("europe_spain_madrid-latest.osm.pbf", (39.90, -4.55, 41.15, -3.05)),
           "valencia": ("europe_spain_valencia-latest.osm.pbf", (37.85, -1.50, 40.80, 0.70))}
N_WINDOWS, SEED, STEP = 16, 2026, 0.08
N, RES, MARGIN = 256, 4.0, 256.0


def highways(region):
    """Highway ways of a region as a local GeoJSON-sequence file (osmium tags-filter + export, once)."""
    f, _ = REGIONS[region]
    hw = PROC / f"{region}_highways.geojsonseq"
    if not hw.exists():
        PROC.mkdir(parents=True, exist_ok=True)
        tmp = PROC / f"{region}_highways.osm.pbf"
        subprocess.run(["osmium", "tags-filter", str(OSM / f), "w/highway", "-o", str(tmp), "--overwrite"], check=True)
        subprocess.run(["osmium", "export", str(tmp), "-f", "geojsonseq", "-o", str(hw), "--overwrite",
                        "--geometry-types=linestring"], check=True)
    return hw


def load_lines(region):
    import shapely
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
    geoms = []
    for line in highways(region).read_text().splitlines():
        line = line.strip("\x1e").strip()
        if not line:
            continue
        g = json.loads(line)["geometry"]
        xy = np.array(g["coordinates"], float)
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        geoms.append(shapely.LineString(np.c_[x, y]))
    return shapely.STRtree(geoms), geoms


def stage_select():
    import shapely
    from m18_erasure_validation import natural_window
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True)
    cands = []
    for region, (_, (la0, lo0, la1, lo1)) in REGIONS.items():
        for la in np.arange(la0, la1, STEP):
            for lo in np.arange(lo0, lo1, STEP):
                cands.append((region, round(float(la), 4), round(float(lo), 4)))
    order = np.random.default_rng(SEED).permutation(len(cands))
    trees = {}
    kept, log = [], []
    for k in order:
        region, la, lo = cands[k]
        if region not in trees:
            trees[region] = load_lines(region)
        tree, geoms = trees[region]
        x, y = tr.transform(lo, la)
        win = shapely.box(x - 512, y - 512, x + 512, y + 512)
        km = sum(geoms[i].intersection(win).length for i in tree.query(win)) / 1000.0
        disc = shapely.Point(x, y).buffer(5000)
        cover = sum(geoms[i].intersection(disc).length for i in tree.query(disc)) / 1000.0
        rec = {"region": region, "lat": la, "lon": lo, "osm_highway_km_1024m": round(km, 3), "osm_km_within_5km": round(cover, 1)}
        if cover < 2.0:
            log.append({**rec, "verdict": "outside extract"})
            continue
        if km > 0.5:
            log.append({**rec, "verdict": "roads"})
            continue
        name = f"{region}_{la:.2f}_{lo:.2f}".replace("-", "m")
        try:
            z = natural_window(name, la, lo)
        except Exception as e:
            log.append({**rec, "verdict": f"dem: {type(e).__name__}"})
            continue
        if not np.isfinite(z).all() or (z < -100).any():
            log.append({**rec, "verdict": "dem gaps"})
            continue
        zs = ndimage.gaussian_filter(z.astype(np.float64), 4.0)
        gy, gx = np.gradient(zs, RES)
        med = float(np.median(np.hypot(gx, gy)[64:192, 64:192]) * 100)
        rec["median_slope_pct"] = round(med, 1)
        if not 5.0 <= med <= 35.0:
            log.append({**rec, "verdict": "slope"})
            continue
        PROC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PROC / f"{name}.npz", elev=z.astype(np.float32), valid=np.ones(z.shape, bool),
                            water=np.zeros(z.shape, np.int8))
        kept.append({**rec, "name": name})
        log.append({**rec, "verdict": "kept"})
        print("kept", name, rec, flush=True)
        if len(kept) == N_WINDOWS:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "windows.json", {"rule": __doc__.split("select   - ")[1].split("generate - ")[0].strip(),
                                       "regions": {k: v[1] for k, v in REGIONS.items()}, "seed": SEED,
                                       "windows": kept, "visited": log})
    print("kept", len(kept), "of", len(log), "visited", flush=True)
    return 0


def windows():
    w = json.loads((OUT / "windows.json").read_text())["windows"]
    return [{"sid": x["name"], "d": dict(np.load(PROC / f"{x['name']}.npz")), "meta": x} for x in w]


def stage_generate(device="cpu", seeds=(1, 2)):
    from m12_evaluate import TH, density_pool, real_graphs
    from urbanmorph.experiments.backbones import load_backbone, model_inputs
    from urbanmorph.experiments.graph_io import graph_to_json
    from urbanmorph.experiments.methods import method
    from urbanmorph.graphs.m14_sampler import NET_PRESETS, net_preset, sample_net
    from urbanmorph.graphs.procedural_terrain import sample_P
    from urbanmorph.graphs.tc_potentials import Potentials
    from urbanmorph.graphs.tc_sampler import preset, sample_tc
    from urbanmorph.graphs.terrain_generate import terrain_of
    W = windows()
    sids, ds = [w["sid"] for w in W], [w["d"] for w in W]
    f = OUT / "runs.json"
    res = json.loads(f.read_text()) if f.exists() else {"windows": sids, "runs": {}}
    jobs = [(lab, f"TN_s{b}") for lab in ("A", "M*", "M13-SMC", "NA-myopic", "NA-fixed", "NA-SMC") for b in range(3)]
    jobs += [("M*", f"{fam}_s{b}") for fam in ("N0", "TNlp", "TNse") for b in range(3)] + [("P", "-")]
    loaded = {}
    for lab, bb in jobs:
        for s in seeds:
            key = f"{lab}|{bb}|{s}"
            if key in res["runs"]:
                continue
            t0 = time.time()
            if lab == "P":
                rng = np.random.default_rng(s)
                pool = loaded.setdefault("pool", density_pool(real_graphs()))
                gs = [sample_P(terrain_of(d), rng, float(rng.choice(pool)), TH) for d in ds]
            else:
                if bb not in loaded:
                    loaded = {k: v for k, v in loaded.items() if k == "pool"}
                    loaded[bb] = load_backbone(bb, device)
                model, names, norm, mcfg = loaded[bb]
                fam = bb.split("_")[0]
                pots = Potentials.load(ROOT / "configs" / "m13_potentials" / f"{fam}.json")
                m = method(lab)
                cfg = net_preset(m.preset, **m.overrides) if m.preset in NET_PRESETS else preset(m.preset, **m.overrides)
                crop = model_inputs(mcfg, sids, ds) if fam == "TNlp" else None    # TNse: nothing to erase here
                if m.preset in NET_PRESETS:
                    gs, _, _ = sample_net(model, ds, names, norm, pots, cfg, seed=s, device=device, crop_tiles=crop, voc_log=None)
                else:
                    gs, _, _ = sample_tc(model, ds, names, norm, pots, cfg, seed=s, device=device, crop_tiles=crop)
            res["runs"][key] = {"graphs": [graph_to_json(g) for g in gs], "seconds": round(time.time() - t0, 1)}
            atomic_json(f, res)
            print(key, res["runs"][key]["seconds"], "s", flush=True)
    return 0


CLASSES = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 1000)]


def stage_score():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from m18_real_terrain_study import land_slope, pieces, wq
    from urbanmorph.experiments.graph_io import graph_from_json
    W = windows()
    runs = json.loads((OUT / "runs.json").read_text())["runs"]
    real = json.loads((ROOT / "outputs" / "m18" / "real_terrain" / "summary.json").read_text())["pooled"]
    land = [land_slope(w["d"]) for w in W]
    LAND = np.concatenate(land)
    keys = [f"{lo}-{hi if hi < 1000 else 'inf'}" for lo, hi in CLASSES]
    groups = {}
    for key, r in runs.items():
        lab, bb, s = key.split("|")
        fam = "TN" if bb.startswith("TN_") else ("P" if bb == "-" else bb.split("_")[0])
        groups.setdefault((lab, fam), []).append(r)
    summ = {"windows": len(W), "land_slope_p50": float(np.median(LAND)), "methods": {}}
    for (lab, fam), rs in groups.items():
        P = [pieces(graph_from_json(gj), w["d"]) for r in rs for gj, w in zip(r["graphs"], W)]
        P = [p for p in P if p is not None]
        if not P:
            continue
        L, G, SL, CI = (np.concatenate([p[i] for p in P]) for i in range(4))
        n_runs = len(rs)
        out = {"runs": n_runs, "street_km_per_km2": float(L.sum() / 1000 / (n_runs * len(W) * 0.262144)),
               "share_gt10": float(L[G > 10].sum() / L.sum()), "share_gt15": float(L[G > 15].sum() / L.sum()),
               "share_gt20": float(L[G > 20].sum() / L.sum()), "classes": {}}
        med = float(np.median(LAND))
        out["S_siting"] = float(L[SL < med].sum() / L.sum())
        steep = SL > 5
        out["C_contour"] = float(1 - np.average(CI[steep], weights=L[steep])) if steep.any() else np.nan
        out["G_gentle"] = 1 - out["share_gt15"]
        out["TAS"] = 100 * (out["G_gentle"] + out["C_contour"] + out["S_siting"]) / 3
        for (lo, hi), k in zip(CLASSES, keys):
            m = (SL >= lo) & (SL < hi)
            lm = (LAND >= lo) & (LAND < hi)
            if m.sum() < 20:
                out["classes"][k] = None
                continue
            w = L[m]
            out["classes"][k] = {"density": float(w.sum() / 1000 / max(n_runs * lm.sum() * 16e-6, 1e-9)),
                                 "ratio": wq(G[m] / np.maximum(SL[m], 1e-6), w, 0.5),
                                 "contour_index": float(np.average(CI[m], weights=w)),
                                 "share_gt15": float(w[G[m] > 15].sum() / w.sum())}
        summ["methods"][f"{lab} [{fam}]"] = out
    atomic_json(OUT / "summary.json", summ)
    ink, ink2, grid_c, accent, ref = "#1f2328", "#59636e", "#e6e8eb", "#2a78d6", "#d1242f"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4))
    xs = np.arange(len(keys))
    labels = ["0–5", "5–10", "10–20", "20–30", "> 30"]
    show = ["A [TN]", "M* [TN]", "NA-SMC [TN]", "NA-fixed [TN]", "M* [N0]", "M* [TNse]", "P [P]"]
    cols = {"A [TN]": "#9aa4ae", "M* [TN]": accent, "NA-SMC [TN]": "#1a5aa6", "NA-fixed [TN]": "#6aa6e8",
            "M* [N0]": "#e8a33c", "M* [TNse]": "#7b4fb0", "P [P]": "#4c9a5f"}
    for ax, (fld, ttl, refk) in zip(axs, (("contour_index", "Contour index |cos| (0 = along contours)", "contour_index_mean"),
                                         ("ratio", "Street grade ÷ terrain slope (median)", "ratio_grade_to_slope_med"),
                                         ("density", "Street km per km² of that land (siting)", "density_km_per_km2"))):
        rv = [real[k][refk] if real.get(k) else np.nan for k in keys]
        ax.plot(xs, rv, "-", color=ref, lw=3, label="real cities (LiDAR reference)", zorder=5)
        for nm in show:
            m = summ["methods"].get(nm)
            if not m:
                continue
            v = [m["classes"][k][fld] if m["classes"].get(k) else np.nan for k in keys]
            ax.plot(xs, v, "o-", color=cols[nm], lw=1.4, ms=4.5, label=nm)
        if fld == "contour_index":
            ax.axhline(2 / np.pi, color=ink2, ls=":", lw=1)
            ax.text(0, 2 / np.pi + 0.01, "random orientation", fontsize=8, color=ink2)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlabel("terrain slope class (%)")
        ax.set_title(ttl, loc="left", fontweight="bold", fontsize=10)
        ax.grid(color=grid_c, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Terrain response on {len(W)} natural LiDAR windows without street beds (no leakage possible), vs real cities",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig_natural.png", dpi=130)
    plt.close(fig)
    md = ["# M18: terrain response on natural terrain without street beds (leakage control b)", "",
          f"Windows: {len(W)} (rule in `windows.json`); script `scripts/m18_natural_terrain.py`; figure `fig_natural.png`.", "",
          "| Method [backbone] | runs | km/km² | share > 15 % | G | C (contour) | S (siting) | TAS |", "|---|---|---|---|---|---|---|---|"]
    for nm, m in sorted(summ["methods"].items(), key=lambda kv: -kv[1]["TAS"]):
        md.append(f"| {nm} | {m['runs']} | {m['street_km_per_km2']:.1f} | {m['share_gt15']:.3f} | {m['G_gentle']:.2f} | "
                  f"{m['C_contour']:.2f} | {m['S_siting']:.2f} | {m['TAS']:.1f} |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "generate", "score"])
    a = ap.parse_args()
    return {"select": stage_select, "generate": stage_generate, "score": stage_score}[a.stage]()


if __name__ == "__main__":
    sys.exit(main())
