"""Named terrain benchmark sites (EXPLORATORY; user request 2026-11-24): every street generator of the project on one
real terrain, cropped to fixed city limits (jungle or desert around), scored on realism and on adaptation to the terrain.
Protocol and the terrain adaptiveness score (declared before any generation): docs/SITE_BENCHMARKS.md.

Site config: configs/sites/<site>.json. Stages:
    data      OSM clip, GLO-30 terrain on the 4 m grid (+ 256 m margin), water, real roads and paths, city limits,
              a terrain map                                   ->  data/processed/sites/<site>/, outputs/sites/<site>/
    generate  every method x seed on the whole domain, no supplied roads (parallel, resumable cache)
                                                              ->  outputs/sites/<site>/cache/
    score     crop to the city limits; realism, terrain adaptiveness, network statistics
                                                              ->  outputs/sites/<site>/{per_seed,summary}.csv, summary.json
    figures   maps (overview sheet + one map per method and seed), realism vs adaptiveness, README table

    .venv/bin/python scripts/site_benchmark.py data      [--site machu_picchu]
    .venv/bin/python scripts/site_benchmark.py generate  [--workers 12] [--seeds 1 2 3] [--only KEY,KEY]
    .venv/bin/python scripts/site_benchmark.py score     [--workers 12]
    .venv/bin/python scripts/site_benchmark.py figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "themes" / "latin_caribbean" / "scripts"))

import numpy as np

W, RES, MARGIN = 512.0, 4.0, 256.0
SEEDS = (1, 2, 3)
# (label, version): the 15 methods of the KAIST comparisons (scripts/m16_kaist_all.ORDER), the M17 F1 + F4 versions of
# the seven improved methods (scripts/m17_kaist.METHODS) and the Chen-inspired field tracer of M15 (grid, organic)
ORIGINAL = ["M12-HR", "P", "A", "M*", "M* TNse", "M13-SMC", "NA-myopic", "NA-fixed", "NA-adaptive", "NA-random", "NA-SMC",
            "M* + grid", "M* + organic", "M* + radial", "M* + culdesac"]
IMPROVED = ["A", "M*", "M13-SMC", "NA-fixed", "NA-SMC", "M* + organic", "M* + radial"]
TRACER = ["Tracer · grid", "Tracer · organic"]
FIX = {"junction_reuse_m": 15.0, "degree_prior": 1.0}
MILESTONE = {"M12-HR": "M12", "P": "M12", "A": "M13", "M*": "M13", "M* TNse": "M13", "M13-SMC": "M13", "NA-myopic": "M14",
             "NA-fixed": "M14", "NA-adaptive": "M14", "NA-random": "M14", "NA-SMC": "M14", "M* + grid": "M16",
             "M* + organic": "M16", "M* + radial": "M16", "M* + culdesac": "M16", "Tracer · grid": "M15",
             "Tracer · organic": "M15"}
SLOW = ("M* + organic", "M* + radial", "NA-SMC", "NA-random", "NA-fixed", "NA-myopic", "NA-adaptive")


def jobs(seeds=SEEDS):
    out = [(lab, "original") for lab in ORIGINAL] + [(lab, "F1 + F4") for lab in IMPROVED] + [(lab, "original") for lab in TRACER]
    out = [(lab, ver, s) for lab, ver in out for s in seeds]
    return sorted(out, key=lambda j: (-(j[0] in SLOW[:2]) - (j[0] in SLOW), j[2]))     # longest first


def key_of(lab: str, ver: str = "original") -> str:
    k = lab.replace("*", "star").replace(" + ", "_").replace(" · ", "_").replace(" ", "_")
    return k + ("_F1F4" if ver == "F1 + F4" else "")


class Site:
    def __init__(self, name: str):
        self.name = name
        self.cfg = json.loads((ROOT / "configs" / "sites" / f"{name}.json").read_text())
        self.L = float(self.cfg["L"])
        self.crs = self.cfg["crs"]
        cx, cy = self.cfg["centre_utm"]
        self.x0, self.y0 = int(round(cx - self.L / 2)), int(round(cy - self.L / 2))
        self.n = int(self.L / RES + 2 * MARGIN / RES)
        self.data = ROOT / "data" / "processed" / "sites" / name
        self.out = ROOT / "outputs" / "sites" / name
        self.cache = self.out / "cache"

    # ---- grid helpers (row 0 = north, [-256, L + 256]^2 in domain metres)
    def xy_grid(self):
        xs = -MARGIN + (np.arange(self.n) + 0.5) * RES
        ys = self.L + MARGIN - (np.arange(self.n) + 0.5) * RES
        return np.meshgrid(xs, ys)

    def rc(self, P):
        P = np.asarray(P, float).reshape(-1, 2)
        r = ((self.L + MARGIN - P[:, 1]) / RES).astype(int)
        c = ((P[:, 0] + MARGIN) / RES).astype(int)
        return np.clip(r, 0, self.n - 1), np.clip(c, 0, self.n - 1)

    # ---- data
    def clip_osm(self) -> Path:
        from pyproj import Transformer
        out = ROOT / "data" / "raw" / "osm_extracts" / "cities" / f"{self.cfg['osm_clip']}.osm.pbf"
        if not out.exists():
            back = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform
            pad = MARGIN + 300.0
            lo, hi = back(self.x0 - pad, self.y0 - pad), back(self.x0 + self.L + pad, self.y0 + self.L + pad)
            bbox = f"{min(lo[0], hi[0])},{min(lo[1], hi[1])},{max(lo[0], hi[0])},{max(lo[1], hi[1])}"
            subprocess.run(["osmium", "extract", "-O", "-b", bbox, "-o", str(out), str(ROOT / self.cfg["osm_extract"])],
                           check=True, capture_output=True)
        return out

    def domain(self) -> dict:
        f = self.data / "terrain.npz"
        if f.exists():
            z = np.load(f)
            return {"elev": z["elev"], "valid": z["valid"], "water": z["water"]}
        from rasterio.transform import from_origin
        from m16_build_terrain_pack import window_terrain, write_tif
        from urbanmorph.terrain.water import city_features, rasterize_kinds, tile_window, water_layers
        self.data.mkdir(parents=True, exist_ok=True)
        self.clip_osm()
        z, tr, native = window_terrain(ROOT / self.cfg["dem"], self.crs, self.x0, self.y0, size=self.L)
        feats = city_features(self.cfg["osm_clip"], self.crs)
        layers = water_layers(feats, tile_window(self.x0, self.y0, MARGIN, size=self.L))
        water = rasterize_kinds(layers, from_origin(self.x0 - MARGIN, self.y0 + self.L + MARGIN, RES, RES), (self.n, self.n))
        if self.cfg.get("dry_wadis"):                      # ephemeral wadis (Petra): dry ground, not rivers
            water[:] = 0
        big = {"elev": np.nan_to_num(z).astype(np.float32), "valid": np.isfinite(z), "water": water}
        np.savez_compressed(f, **big)
        write_tif(self.data / "terrain_glo30_4m.tif", z, tr, self.crs)
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        clip = self.clip_osm()
        (self.data / "manifest.json").write_text(json.dumps({
            "site": self.cfg["name"], "crs": self.crs, "origin_utm_sw": [self.x0, self.y0], "L_m": self.L,
            "grid": f"{RES} m, {MARGIN} m margin, {self.n} x {self.n}, row 0 = north", "dem": self.cfg["dem"],
            "dem_sha256": sha(ROOT / self.cfg["dem"]), "dem_native_res_m": native, "dem_type": "GLO-30 DSM (surface model)",
            "osm_extract": self.cfg["osm_extract"], "osm_extract_mtime": time.ctime((ROOT / self.cfg["osm_extract"]).stat().st_mtime),
            "osm_clip_sha256": sha(clip), "water_cells": {k: int((water == k).sum()) for k in (1, 2, 3)},
            "attribution": ["Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, "
                            "provided under COPERNICUS by the European Union and ESA", "(c) OpenStreetMap contributors, ODbL 1.0"]},
            indent=1))
        return big

    def real(self, kind: str):
        """Real OSM ways of ``kind`` (vehicular / paths) as a graph in domain metres."""
        f = self.data / f"real_{kind}.pkl"
        if f.exists():
            return pickle.loads(f.read_bytes())
        import geopandas as gpd
        import shapely
        from urbanmorph.graphs.road_graph import RoadGraph, build_graph, highway_class, is_border
        seq = self.data / "osm_highways.geojsonseq"
        if not seq.exists():
            tmp = self.data / "_hw.osm.pbf"
            subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(self.clip_osm()), "w/highway"], check=True,
                           capture_output=True)
            subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "-o", str(seq), str(tmp)], check=True, capture_output=True)
            tmp.unlink()
        g = gpd.read_file(seq).set_crs(4326, allow_override=True).to_crs(self.crs)
        g = g[g.geom_type.isin(["LineString", "MultiLineString"]) & g.highway.isin(self.cfg["real"][kind])]
        win = shapely.box(self.x0, self.y0, self.x0 + self.L, self.y0 + self.L)
        lines, cls = [], []
        for geom, hw in zip(g.geometry.intersection(win), g.highway):
            for part in getattr(geom, "geoms", [geom]):
                if part.geom_type == "LineString" and part.length > 1:
                    lines.append(np.asarray(part.coords)[:, :2] - [self.x0, self.y0])
                    cls.append(highway_class(hw))
        gg = build_graph(lines, classes=cls)
        rg = RoadGraph(gg.nodes, gg.edges, is_border(gg.nodes, size=self.L), edge_class=gg.classes(), edge_struct=gg.structs())
        f.write_bytes(pickle.dumps(rg))
        return rg

    def city_mask(self, big) -> np.ndarray:
        from scipy import ndimage
        from urbanmorph.terrain.synthesis import gradient_noise
        c = self.cfg["city_limits"]
        X, Y = self.xy_grid()
        a = np.radians(c["angle_deg"])
        cx, cy = c["centre"]
        u = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
        v = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)
        r = np.hypot(u / c["semi_axes"][0], v / c["semi_axes"][1])
        for amp, scale, seed in c["noise"]:
            r = r + amp * gradient_noise(X / scale, Y / scale, seed)
        m = (r < 1) & (big["water"] == 0)
        m &= (X >= 0) & (X <= self.L) & (Y >= 0) & (Y <= self.L)
        lab, k = ndimage.label(m)
        return lab == 1 + int(np.argmax(ndimage.sum(m, lab, range(1, k + 1))))

    def bare(self, big):
        """Bare-earth proxy of the DSM: Gaussian low-pass, sigma 16 m (as scripts/m17_village_compare.py)."""
        from scipy import ndimage
        from urbanmorph.terrain.heightfield import Terrain
        return Terrain(ndimage.gaussian_filter(big["elev"].astype(float), 4.0).astype(np.float32), -MARGIN, self.L + MARGIN, RES)

    def slope(self, big):
        """|grad| of the bare-earth proxy (rise over run) on the 4 m grid."""
        from scipy import ndimage
        z = ndimage.gaussian_filter(big["elev"].astype(float), 4.0)
        gy, gx = np.gradient(z, RES)
        return np.hypot(gx, gy), gx, -gy                          # rows run north to south: y gradient flips


def stage_data(site: Site):
    big = site.domain()
    mask = site.city_mask(big)
    np.savez_compressed(site.data / "city_mask.npz", mask=mask)
    s, _, _ = site.slope(big)
    z = big["elev"]
    stats = {"city_km2": float(mask.sum() * RES ** 2 / 1e6), "domain_km2": site.L ** 2 / 1e6,
             "elev_city_min_max": [float(z[mask].min()), float(z[mask].max())],
             "elev_domain_min_max": [float(z.min()), float(z.max())],
             "slope_city_median": float(np.median(s[mask])),
             "slope_city_share_gt": {f"{t}%": float((s[mask] > t / 100).mean()) for t in (15, 30, 45, 60, 100)},
             "water_share_domain": float((big["water"] > 0).mean())}
    for kind in ("vehicular", "paths"):
        g = site.real(kind)
        Ln = np.linalg.norm(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]], axis=1)
        stats[f"real_{kind}_km_domain"] = float(Ln.sum() / 1000)
    site.out.mkdir(parents=True, exist_ok=True)
    (site.out / "site.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    terrain_map(site, big, mask)


def hillshade(site, big, mask, outside=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from scipy import ndimage
    z = ndimage.gaussian_filter(big["elev"].astype(float), 3.0)            # display only: hides the 30 m DEM cells
    rgb = LightSource(315, 40).shade(z, cmap=plt.cm.Greys_r, vert_exag=1.2, blend_mode="soft", vmin=z.min() - 150, vmax=z.max() + 50)
    if outside:                                                            # jungle / desert tint outside the limits
        rgb[~mask, :3] = rgb[~mask, :3] * 0.5 + np.array(site.cfg["outside"]["rgb"]) * 0.5
    rgb[big["water"] > 0, :3] = [0.55, 0.72, 0.92]
    return rgb


def contours(ax, site, big, step=50, lw=0.35):
    X, Y = site.xy_grid()
    from scipy import ndimage
    z = ndimage.gaussian_filter(big["elev"].astype(float), 2.0)
    lv = np.arange(np.floor(z.min() / step) * step, z.max() + step, step)
    cs = ax.contour(X, Y, z, levels=lv, colors="#3d3a33", linewidths=lw, alpha=0.45)
    major = [l for l in lv if l % (4 * step) == 0]
    ax.contour(X, Y, z, levels=major, colors="#3d3a33", linewidths=lw * 2.2, alpha=0.55)
    return cs


def terrain_map(site, big, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    s, _, _ = site.slope(big)
    L = site.L
    fig, axs = plt.subplots(1, 2, figsize=(20, 10.4))
    ax = axs[0]
    ax.imshow(hillshade(site, big, mask), extent=(-MARGIN, L + MARGIN, -MARGIN, L + MARGIN))
    cs = contours(ax, site, big)
    ax.clabel(cs, levels=[l for l in cs.levels if l % 200 == 0], fontsize=7, fmt="%d m")
    for kind, col, lw in (("paths", "#f4f1e8", 0.6), ("vehicular", "#c0392b", 1.6)):
        g = site.real(kind)
        ax.add_collection(LineCollection(g.nodes[g.edges], colors=col, linewidths=lw))
    ax.contour(*site.xy_grid(), mask.astype(float), levels=[0.5], colors="#f5c242", linewidths=1.6)
    ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, ec="#222", lw=1, ls="--"))
    ax.set_title(site.cfg["terrain_title"], loc="left", fontsize=10)
    ax = axs[1]
    sp = np.where(mask, s * 100, np.nan)
    im = ax.imshow(np.clip(s * 100, 0, 150), extent=(-MARGIN, L + MARGIN, -MARGIN, L + MARGIN), cmap="magma_r", vmin=0, vmax=150,
                   alpha=0.35)
    im = ax.imshow(np.clip(sp, 0, 150), extent=(-MARGIN, L + MARGIN, -MARGIN, L + MARGIN), cmap="magma_r", vmin=0, vmax=150)
    ax.contour(*site.xy_grid(), s * 100, levels=[float(np.median(s[mask]) * 100)], colors="#1f6fb2", linewidths=0.9)
    plt.colorbar(im, ax=ax, fraction=0.04, label="terrain slope (%) of the bare-earth proxy")
    ax.set_title(f"Slope inside the city limits (median {np.median(s[mask]) * 100:.0f} %, blue line = the median: "
                 "the 'steeper half' of the adaptiveness score)", loc="left", fontsize=10)
    for a in axs:
        a.set_xlim(-MARGIN, L + MARGIN)
        a.set_ylim(-MARGIN, L + MARGIN)
        a.set_xlabel("m")
    fig.tight_layout()
    fig.savefig(site.out / "site_terrain.png", dpi=90)
    plt.close(fig)


def merge(gs, offs, L):
    """Window graphs merged into domain coordinates without joining (as the KAIST P baseline)."""
    from urbanmorph.graphs.road_graph import RoadGraph, is_border
    nodes, edges, cls, st, k = [], [], [], [], 0
    for g, (dx, dy) in zip(gs, offs):
        if g is None or g.n == 0:
            continue
        nodes.append(g.nodes + [dx, dy])
        edges.append(g.edges + k)
        cls.append(g.classes())
        st.append(g.structs())
        k += g.n
    if not nodes:
        return RoadGraph(np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, bool))
    N = np.vstack(nodes)
    return RoadGraph(N, np.vstack(edges), is_border(N, size=L), edge_class=np.concatenate(cls), edge_struct=np.concatenate(st))


def style_overrides(name: str) -> dict:
    return dict(json.loads((ROOT / "configs" / "m15_style.json").read_text())["styles"][name])


def generate_one(job):
    """One (method, version, seed) on the whole domain; cached (resumable, parallel-safe)."""
    import torch
    site_name, lab, ver, seed = job
    site = Site(site_name)
    f = site.cache / f"{key_of(lab, ver)}_s{seed}.pkl"
    if f.exists():
        return job, "cached", 0.0
    torch.set_num_threads(1)
    big = site.domain()
    L = site.L
    n = int(L // W)
    offs = [(c * W, r * W) for r in range(n) for c in range(n)]
    t0, stats = time.time(), {}
    if lab == "M12-HR":                                     # the M12 loop with hard grade rules (scripts/m16_kaist_all.py)
        from m11_big_city import to_graph
        from m12_big_landscape import decode, grow
        from m12_evaluate import load_model
        from urbanmorph.experiments.references import TH
        from urbanmorph.terrain.heightfield import Terrain
        T = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -MARGIN, L + MARGIN, RES)
        T.water_kind = big["water"].astype(np.int8)
        T.water_surface = np.where(big["water"] > 0, big["elev"] - 1.0, np.nan).astype(np.float32)
        model, names, norm = load_model("TN_s0", "cpu")
        S, C, _ = grow(model, names, norm, "HR", TH, T, L, seed, 0.5)
        g = decode(to_graph(S, C, (0.0, 0.0)))
    elif lab == "P" or lab.startswith("Tracer"):            # per 512 m window, merged without joining
        from urbanmorph.graphs.tc_global import window_tiles
        from urbanmorph.graphs.terrain_generate import terrain_of
        tiles = window_tiles(big, L)
        if lab == "P":
            from urbanmorph.experiments.references import TH, median_density
            from urbanmorph.graphs.procedural_terrain import sample_P
            gs = [sample_P(terrain_of(t), np.random.default_rng([seed, j]), median_density(), TH) for j, t in enumerate(tiles)]
        else:                                               # M15 external baseline, as scripts/m15_run.py
            from urbanmorph.graphs.field_tracer import trace
            from urbanmorph.graphs.style import StyleField
            from urbanmorph.graphs.tc_potentials import Potentials
            from urbanmorph.graphs.tc_sampler import SamplerConfig
            cfg = SamplerConfig(**style_overrides(lab.split(" · ")[1]))
            pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
            gs = []
            for j, t in enumerate(tiles):
                d = {**t, "site": f"{site.cfg['id']}_w{j}"}
                gg, st = trace(d, StyleField.build(cfg, d["site"], seed), pots, np.random.default_rng([seed, j]))
                gs.append(gg)
                for k, v in dict(st).items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        stats[k] = stats.get(k, 0.0) + float(v)
        g = merge(gs, offs, L)
    else:                                                   # learned samplers, one global state, no supplied roads
        from m16_kaist_all import AnchoredGlobalNet
        from m16_kaist_campus import AnchoredGlobal
        from m16_plans import proposals
        from urbanmorph.experiments.backbones import load_backbone
        from urbanmorph.experiments.methods import method
        from urbanmorph.graphs.m14_sampler import net_preset
        from urbanmorph.graphs.tc_potentials import Potentials
        from urbanmorph.graphs.tc_sampler import preset
        bb = "TNse_s0" if lab == "M* TNse" else "TN_s0"
        base = "M*" if lab == "M* TNse" else lab.split(" + ")[0]
        m = method(base)
        model, names, norm, _ = load_backbone(bb, "cpu")
        pots = Potentials.load(ROOT / "configs" / "m13_potentials" / f"{bb.split('_')[0]}.json")
        ov = dict(m.overrides)
        if ver == "F1 + F4":
            ov.update(FIX)
        if " + " in lab:
            stl = lab.split(" + ")[1]
            fz = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]
            ov.update(style_overrides(stl))
            ov.update(proposals("style_topo", fz["M*"][stl]["style_topo"]["prop_alpha"]))
            cx, cy = site.cfg["city_limits"]["centre"]             # radial centre = the city centre
            ov.update(style_cx=float(cx), style_cy=float(cy))
        is_net = base.startswith("NA")
        cfg = net_preset(m.preset, **ov) if is_net else preset(m.preset, **ov)
        sm = (AnchoredGlobalNet if is_net else AnchoredGlobal)(model, {**big, "site": f"{site.cfg['id']}_{lab}"}, L, names,
                                                               norm, pots, cfg)
        sm.anchor = None
        g = sm.run(seed)[0]
        stats = {k: float(v) for k, v in dict(sm.stats[0]).items() if isinstance(v, (int, float, np.integer, np.floating))}
    site.cache.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps({"label": lab, "version": ver, "seed": seed, "graph": g, "seconds": time.time() - t0,
                                  "stats": stats}))
    tmp.rename(f)
    return job, "done", time.time() - t0


def stage_generate(site: Site, workers: int, seeds, only=None):
    from multiprocessing import Pool
    js = [(site.name, lab, ver, s) for lab, ver, s in jobs(seeds)]
    if only:
        keep = set(only.split(","))
        js = [j for j in js if key_of(j[1], j[2]) in keep]
    js = [j for j in js if not (site.cache / f"{key_of(j[1], j[2])}_s{j[3]}.pkl").exists()]
    print(f"{len(js)} jobs, {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers, maxtasksperchild=1) as p:
        for k, (job, how, sec) in enumerate(p.imap_unordered(generate_one, js, chunksize=1)):
            print(f"[{k + 1}/{len(js)}] {time.time() - t0:6.0f}s  {how} {job[1]} · {job[2]} · seed {job[3]}  ({sec:.0f}s)", flush=True)


# ---------------------------------------------------------------- score (definitions: docs/SITE_BENCHMARKS.md)
G = {}                                                      # per-worker state
SPARSE_KM_PER_KM2 = 5.0


def windows(site, mask, cover=0.7, overlap=0.3):
    """512 m windows >= ``cover`` inside the city limits, greedy, < ``overlap`` pairwise (as m17_village_compare)."""
    L = int(site.L)
    cov = lambda x0, y0: mask[int((L + MARGIN - (y0 + W)) / RES):int((L + MARGIN - y0) / RES),
                              int((x0 + MARGIN) / RES):int((x0 + W + MARGIN) / RES)].mean()
    cands = sorted(((cov(x, y), x, y) for x in range(0, int(L - W) + 1, 32) for y in range(0, int(L - W) + 1, 32)), reverse=True)
    best = []
    for c, x, y in cands:
        if c < cover:
            break
        if all(max(0, W - abs(x - bx)) * max(0, W - abs(y - by)) / W ** 2 < overlap for _, bx, by in best):
            best.append((float(c), x, y))
    return best


def init(site_name):
    from m16_realism import K, real_pool
    site = Site(site_name)
    big = site.domain()
    mask = np.load(site.data / "city_mask.npz")["mask"]
    s, gx, gy = site.slope(big)
    G.update(site=site, big=big, mask=mask, Tb=site.bare(big), slope=s, grad=(gx, gy), med_slope=float(np.median(s[mask])),
             wins=windows(site, mask))
    X, _ = real_pool()
    med = np.nanmedian(X, 0)
    Xi = np.where(np.isfinite(X), X, med)
    mu, sd = Xi.mean(0), np.where(Xi.std(0) > 1e-9, Xi.std(0), 1.0)
    Z = (Xi - mu) / sd
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    G["pool"] = (med, mu, sd, Z, float(np.mean(np.sort(D, 1)[:, :K].mean(1))), K)


def crop(g, mask, site):
    """kaist_village.crop without supplied main roads: edges with both ends inside, stubs < 10 m and isolated pieces
    < 120 m removed."""
    from urbanmorph.graphs.road_graph import RoadGraph, is_border
    if len(g.edges) == 0:
        return g
    r, c = site.rc(g.nodes)
    inside = mask[r, c]
    keep = inside[g.edges].all(1)
    L_e = np.linalg.norm(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]], axis=1)
    for _ in range(3):
        deg = np.bincount(g.edges[keep].ravel(), minlength=len(g.nodes))
        keep &= ~((L_e < 10) & ((deg[g.edges[:, 0]] == 1) | (deg[g.edges[:, 1]] == 1)))
    idx = np.flatnonzero(keep)
    lab = np.arange(len(g.nodes))

    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for e in idx:
        a, b = find(g.edges[e, 0]), find(g.edges[e, 1])
        if a != b:
            lab[a] = b
    comp = np.array([find(g.edges[e, 0]) for e in idx])
    tot = {}
    for e, k in zip(idx, comp):
        tot[k] = tot.get(k, 0.0) + L_e[e]
    keep[idx] = [tot[k] >= 120 for k in comp]
    E = g.edges[keep]
    used, inv = np.unique(E, return_inverse=True)
    N = g.nodes[used]
    return RoadGraph(N, inv.reshape(-1, 2), is_border(N, size=site.L), edge_class=g.classes()[keep],
                     edge_struct=g.structs()[keep])


def pieces(g, step=10.0):
    """Ground edges cut into <= step m pieces: grade on the bare proxy, length, midpoint, unit direction."""
    T = G["Tb"]
    st = np.asarray(g.structs())
    Gr, Wt, M, D = [], [], [], []
    for e in np.flatnonzero(st == 0):
        a, b = g.nodes[g.edges[e]]
        Ln = float(np.hypot(*(b - a)))
        if Ln < 1.0:
            continue
        cuts = np.r_[np.arange(0.0, Ln, step), Ln]
        cuts = cuts[np.r_[True, np.diff(cuts) > 0.5]]
        cuts[-1] = Ln
        P = a + np.outer(cuts / Ln, b - a)
        z = T.sample(P[:, 0], P[:, 1])
        d = np.diff(cuts)
        Gr.append(np.abs(np.diff(z)) / d)
        Wt.append(d)
        M.append((P[1:] + P[:-1]) / 2)
        D.append(np.repeat(((b - a) / Ln)[None], len(d), 0))
    if not Gr:
        z = np.zeros(0)
        return z, z, np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(Gr), np.concatenate(Wt), np.vstack(M), np.vstack(D)


def tas(g):
    """Terrain adaptiveness score and its three components (docs/SITE_BENCHMARKS.md)."""
    site = G["site"]
    Gr, Wt, M, D = pieces(g)
    out = {"TAS": np.nan, "G_gentle": np.nan, "C_contour": np.nan, "S_siting": np.nan, "contour_index": np.nan}
    if Wt.sum() <= 0:
        return out
    r, c = site.rc(M)
    s = G["slope"][r, c]
    gx, gy = G["grad"][0][r, c], G["grad"][1][r, c]
    out["G_gentle"] = float(1 - Wt[Gr > 0.15].sum() / Wt.sum())
    sel = s > 0.05
    if sel.any():
        ci = np.abs(D[sel, 0] * gx[sel] + D[sel, 1] * gy[sel]) / s[sel]
        out["contour_index"] = float(np.average(np.clip(ci, 0, 1), weights=Wt[sel]))
        out["C_contour"] = 1 - out["contour_index"]
    out["S_siting"] = float(Wt[s < G["med_slope"]].sum() / Wt.sum())
    if np.isfinite([out["G_gentle"], out["C_contour"], out["S_siting"]]).all():
        out["TAS"] = 100 * (out["G_gentle"] + out["C_contour"] + out["S_siting"]) / 3
    o = np.argsort(Gr)
    cw = np.cumsum(Wt[o]) / Wt.sum()
    out["grade_median_pct"] = float(100 * Gr[o][min(np.searchsorted(cw, 0.5), len(o) - 1)])
    for th in (12, 15, 20):
        out[f"over_{th}pct"] = float(Wt[Gr > th / 100].sum() / Wt.sum())
    return out


def city_stats(g):
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from urbanmorph.graphs.junctions import clean
    site, mask = G["site"], G["mask"]
    area = float(mask.sum() * RES ** 2 / 1e6)
    out = {"street_km_per_km2": 0.0, "coverage_60": 0.0, "junctions_per_km2": 0.0}
    if len(g.edges) == 0:
        return out
    N, E, _ = clean(g)
    Ln = np.linalg.norm(N[E[:, 0]] - N[E[:, 1]], axis=1)
    km = float(Ln.sum() / 1000)
    deg = np.bincount(E.ravel(), minlength=len(N))
    J = np.flatnonzero(deg >= 3)
    tot = max(len(J), 1)
    core = ndimage.binary_erosion(mask, iterations=8)
    dead = int(((deg == 1) & core[site.rc(N)]).sum())
    lab = np.arange(len(N))

    def find(i):
        while lab[i] != i:
            lab[i] = lab[lab[i]]
            i = lab[i]
        return i
    for a, b in E:
        ra, rb = find(a), find(b)
        if ra != rb:
            lab[ra] = rb
    comp = np.array([find(a) for a in E[:, 0]])
    lcc = float(max(np.bincount(np.unique(comp, return_inverse=True)[1], weights=Ln)) / max(Ln.sum(), 1e-9))
    P = np.vstack([N[E[:, 0]] + t * (N[E[:, 1]] - N[E[:, 0]]) for t in np.linspace(0, 1, 2 + int(Ln.max() / 2))])
    on = np.zeros_like(mask)
    on[site.rc(P)] = True
    near = ndimage.distance_transform_edt(~on) * RES <= 60.0
    out.update({"street_km_per_km2": km / area, "coverage_60": float(near[mask].mean()), "junctions_per_km2": len(J) / area,
                "share_3way": int((deg[J] == 3).sum()) / tot, "share_4way": int((deg[J] == 4).sum()) / tot,
                "share_5p": int((deg[J] >= 5).sum()) / tot,
                "close15": float((cKDTree(N[J]).query(N[J], 2)[0][:, 1] < 15.0).mean()) if len(J) >= 2 else np.nan,
                "deadends_per_km": dead / max(km, 1e-9), "largest_piece_share": lcc})
    return out


def water_unbridged_m(g):
    """Ground street length (whole domain) whose edge touches mapped water (9 samples per edge, as M12)."""
    if len(g.edges) == 0:
        return 0.0
    site, w = G["site"], G["big"]["water"]
    a, b = g.nodes[g.edges[:, 0]], g.nodes[g.edges[:, 1]]
    t = np.linspace(0, 1, 9)[:, None, None]
    P = (a[None] + t * (b - a)[None]).reshape(-1, 2)
    wet = (w[site.rc(P)] > 0).reshape(9, -1).any(0)
    ground = np.asarray(g.structs()) == 0
    return float(np.linalg.norm(b - a, axis=1)[wet & ground].sum())


def window_graph(g, x0, y0):
    import shapely
    from urbanmorph.graphs.road_graph import build_graph
    box = shapely.box(x0, y0, x0 + W, y0 + W)
    lines = []
    for a, b in g.edges:
        seg = shapely.LineString(g.nodes[[a, b]]).intersection(box)
        for part in getattr(seg, "geoms", [seg]):
            if part.geom_type == "LineString" and part.length > 0.5:
                lines.append(np.asarray(part.coords) - [x0, y0])
    return build_graph(lines) if lines else None


def typicality(g, rng):
    from m16_realism import vec
    from urbanmorph.graphs.network_outcomes import outcome_row
    from urbanmorph.graphs.style_metrics import morphology_row
    from urbanmorph.graphs.terrain_generate import terrain_of
    med, mu, sd, Z, real_typ, K = G["pool"]
    site, big = G["site"], G["big"]
    out = []
    for _, x0, y0 in G["wins"]:
        wg = window_graph(g, x0, y0)
        if wg is None or len(wg.edges) == 0:
            continue
        n0, c0 = int((site.L - y0 - W) / RES), int(x0 / RES)
        t = terrain_of({k: v[n0:n0 + 256, c0:c0 + 256] for k, v in big.items()})
        row = {**morphology_row(wg, t, centre=(256.0, 256.0), rng=rng, grade_span=30.0), **outcome_row(wg)}
        v = np.asarray(vec(row), float)
        v = np.where(np.isfinite(v), v, med)
        out.append(float(np.sort(np.sqrt(((((v - mu) / sd) - Z) ** 2).sum(1)))[:K].mean() / real_typ))
    return (float(np.mean(out)) if out else np.nan), len(out)


def score_one(job):
    lab, ver, seed = job
    site = G["site"]
    if lab == "real":
        g, secs = site.real(ver), np.nan
    else:
        c = pickle.loads((site.cache / f"{key_of(lab, ver)}_s{seed}.pkl").read_bytes())
        g, secs = c["graph"], c["seconds"]
    gc = crop(g, G["mask"], site)
    typ, nw = typicality(gc, np.random.default_rng([seed, 17]))
    row = {"method": lab, "version": ver, "seed": seed, "milestone": MILESTONE.get(lab, "reference"), "typicality": typ,
           "windows_scored": nw, **tas(gc), **city_stats(gc), "unbridged_water_m": water_unbridged_m(g), "seconds": secs}
    row["sparse"] = bool(row["street_km_per_km2"] < SPARSE_KM_PER_KM2)
    return row, gc


def stage_score(site: Site, workers: int, seeds):
    import pandas as pd
    from multiprocessing import Pool
    js = [("real", "vehicular", 0), ("real", "paths", 0)]
    js += [(lab, ver, s) for lab, ver, s in jobs(seeds) if (site.cache / f"{key_of(lab, ver)}_s{s}.pkl").exists()]
    missing = len(jobs(seeds)) + 2 - len(js)
    with Pool(workers, initializer=init, initargs=(site.name,)) as p:
        res = p.map(score_one, js, chunksize=1)
    (site.out / "city_graphs.pkl").write_bytes(pickle.dumps({(r["method"], r["version"], r["seed"]): gc for r, gc in res}))
    df = pd.DataFrame([r for r, _ in res])
    df.to_csv(site.out / "per_seed.csv", index=False)
    num = [c for c in df.columns if c not in ("method", "version", "seed", "milestone", "sparse")]
    agg = df.groupby(["method", "version", "milestone"], sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["sparse"] = agg["street_km_per_km2_mean"] < SPARSE_KM_PER_KM2
    agg = agg.sort_values("typicality_mean", na_position="last")
    agg.to_csv(site.out / "summary.csv", index=False)
    init(site.name)
    json.dump({"windows": G["wins"], "median_city_slope": G["med_slope"], "missing_runs": missing,
               "rows": agg.to_dict("records")}, open(site.out / "summary.json", "w"), indent=1, default=float)
    print(agg[["method", "version", "typicality_mean", "TAS_mean", "G_gentle_mean", "C_contour_mean", "S_siting_mean",
               "street_km_per_km2_mean", "coverage_60_mean"]].round(2).to_string(index=False))


# ---------------------------------------------------------------- figures
FAMILY = {"A": 0, "M*": 0, "M* TNse": 0, "M13-SMC": 0, "NA-myopic": 1, "NA-fixed": 1, "NA-adaptive": 1, "NA-random": 1,
          "NA-SMC": 1, "M* + grid": 2, "M* + organic": 2, "M* + radial": 2, "M* + culdesac": 2, "M12-HR": 3, "P": 3,
          "Tracer · grid": 3, "Tracer · organic": 3}
FAM = [("learned (M13 family)", "#2a78d6"), ("network-aware (M14 NA)", "#eb6834"), ("styles (M16)", "#1baf7a"),
       ("rules / procedural (M12, M15 tracer)", "#eda100")]                 # validated categorical slots 1-4
INK, INK2, MUTED, GRID_C, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# label offsets (points, ha) per site for the scatter, set after looking at the figure (layout only)
NUDGE = {"machu_picchu": {"M* + organic · F1+F4": (-8, -14, "right"), "M* + organic": (-30, 10, "left"), "M* + radial": (8, 6, "left"),
         "M* + radial · F1+F4": (8, -11, "left"), "M* + grid": (6, 5, "left"), "Tracer · organic": (-24, -14, "left"),
         "Tracer · grid": (8, 4, "left"), "NA-SMC": (-8, -6, "right"), "NA-SMC · F1+F4": (-8, 4, "right"),
         "NA-myopic": (6, -13, "left"), "NA-random": (8, 4, "left"), "NA-fixed": (-8, 0, "right"),
         "NA-fixed · F1+F4": (-6, 9, "right"), "NA-adaptive": (8, 4, "left"), "M13-SMC · F1+F4": (-8, 2, "right"),
         "M13-SMC": (8, 4, "left"), "A · F1+F4": (-8, 8, "right"), "A": (8, 3, "left"), "M* · F1+F4": (8, 4, "left"),
         "M*": (8, -9, "left"), "REAL Inca trails / citadel paths": (10, -15, "left")}}


def label(r, site=None):
    if r["method"] == "real":
        return (site or G["site"]).cfg["real_names"][r["version"]]
    return r["method"] + (" · F1+F4" if r["version"] == "F1 + F4" else "")


def fmt(v, n=2):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def city_map(ax, site, big, mask, g, view, title, fs=9.5, lw=0.7):
    from matplotlib.collections import LineCollection
    L = site.L
    ax.imshow(hillshade(site, big, mask), extent=(-MARGIN, L + MARGIN, -MARGIN, L + MARGIN))
    contours(ax, site, big, step=50, lw=0.25)
    from scipy import ndimage
    ax.contour(*site.xy_grid(), ndimage.binary_fill_holes(mask).astype(float), levels=[0.5], colors=site.cfg["outside"]["edge"], linewidths=1.0)
    if g is not None and len(g.edges):
        st = np.asarray(g.structs())
        ax.add_collection(LineCollection(g.nodes[g.edges[st == 0]], colors="#111", linewidths=lw, capstyle="round"))
        if (st != 0).any():
            ax.add_collection(LineCollection(g.nodes[g.edges[st != 0]], colors="#b04a2f", linewidths=lw * 1.6))
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=fs, color=INK, loc="left")


def stage_figures(site: Site):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    init(site.name)
    big, mask = G["big"], G["mask"]
    agg = pd.read_csv(site.out / "summary.csv")
    graphs = pickle.loads((site.out / "city_graphs.pkl").read_bytes())
    rows = agg.to_dict("records")
    gen = [r for r in rows if r["method"] != "real"]
    ref = [r for r in rows if r["method"] == "real"]
    ys, xs = np.nonzero(mask)
    view = (-MARGIN + xs.min() * RES - 40, -MARGIN + xs.max() * RES + 40, site.L + MARGIN - ys.max() * RES - 40,
            site.L + MARGIN - ys.min() * RES + 40)
    rank = {(r["method"], r["version"]): k + 1 for k, r in enumerate(gen)}
    sp = lambda r: "  · SPARSE" if r["sparse"] else ""
    # ---- overview sheet, seed 1
    panels = gen + ref
    nc = 6
    nr = int(np.ceil(len(panels) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.3 * nc, 4.75 * nr), facecolor=SURF)
    for ax, r in zip(axs.ravel(), panels):
        g = graphs.get((r["method"], r["version"], 0 if r["method"] == "real" else 1))
        k = rank.get((r["method"], r["version"]))
        head = f"#{k} {label(r)}" if k else label(r)
        city_map(ax, site, big, mask, g, view, f"{head}{sp(r)}\nrealism {fmt(r['typicality_mean'])} · TAS {fmt(r['TAS_mean'], 0)} · "
                                               f"{fmt(r['street_km_per_km2_mean'], 0)} km/km²", fs=9, lw=0.6)
    for ax in axs.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{site.cfg['short']}: every street generator inside the same city limits ({mask.sum() * RES ** 2 / 1e6:.2f} km², "
                 f"median slope {100 * G['med_slope']:.0f} %), seed 1, "
                 "ordered by realism (typicality, lower = more realistic). TAS = terrain adaptiveness 0-100 (higher = better). "
                 f"Black = streets, red = bridges / tunnels, 50 m contours, {site.cfg['outside']['legend']}", fontsize=12.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(site.out / "fig_maps_seed1.png", dpi=80, facecolor=SURF)
    plt.close(fig)
    # ---- one map per method and seed
    md = site.out / "maps"
    md.mkdir(exist_ok=True)
    per = pd.read_csv(site.out / "per_seed.csv").to_dict("records")
    for r in per:
        g = graphs.get((r["method"], r["version"], r["seed"]))
        k = rank.get((r["method"], r["version"]))
        fig, ax = plt.subplots(figsize=(9.5, 10.2), facecolor=SURF)
        name = label(r) + ("" if r["method"] == "real" else f"  (seed {r['seed']})")
        city_map(ax, site, big, mask, g, view,
                 f"{'#' + str(k) + ' ' if k else ''}{name}{sp(r)}\n"
                 f"realism {fmt(r['typicality'])} · TAS {fmt(r['TAS'], 0)} (gentle {fmt(r['G_gentle'])}, contour "
                 f"{fmt(r['C_contour'])}, siting {fmt(r['S_siting'])})\n{fmt(r['street_km_per_km2'], 1)} km/km² · coverage "
                 f"{fmt(100 * r['coverage_60'], 0)} % · junctions {fmt(r['junctions_per_km2'], 0)}/km² · > 15 %: "
                 f"{fmt(100 * r['over_15pct'], 1) if 'over_15pct' in r else '–'} % of length", fs=10.5, lw=0.9)
        fig.tight_layout()
        fn = ("real_" + r["version"]) if r["method"] == "real" else f"{k:02d}_{key_of(r['method'], r['version'])}_s{r['seed']}"
        fig.savefig(md / f"{fn}.png", dpi=110, facecolor=SURF)
        plt.close(fig)
    # ---- realism vs terrain adaptiveness
    fig, ax = plt.subplots(figsize=(13, 8.5), facecolor=SURF)
    ax.set_facecolor(SURF)
    for r in gen:
        col = FAM[FAMILY[r["method"]]][1]
        hollow = r["version"] == "F1 + F4"
        mk = "s" if r["method"].startswith("Tracer") else "o"
        ax.errorbar(r["typicality_mean"], r["TAS_mean"], xerr=r["typicality_std"], yerr=r["TAS_std"], fmt=mk, ms=8,
                    mfc=SURF if hollow else col, mec=col, mew=2, ecolor=col, elinewidth=0.8, alpha=0.95 if not r["sparse"] else 0.45,
                    zorder=3)
        dx, dy, ha = NUDGE.get(site.name, {}).get(label(r), (6, 4, "left"))
        ax.annotate(label(r) + (" (sparse)" if r["sparse"] else ""), (r["typicality_mean"], r["TAS_mean"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=INK2 if not r["sparse"] else MUTED, ha=ha, zorder=4)
    for r in ref:
        if np.isfinite(r["typicality_mean"]) and np.isfinite(r["TAS_mean"]):
            ax.plot(r["typicality_mean"], r["TAS_mean"], marker="*", ms=16, color=INK, zorder=5)
            dx, dy, ha = NUDGE.get(site.name, {}).get(label(r), (9, -3, "left"))
            ax.annotate(label(r), (r["typicality_mean"], r["TAS_mean"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=9, color=INK, ha=ha)
    ax.set_xlim(left=min(r["typicality_mean"] for r in gen) - 0.6)
    ax.axvline(1.0, color=AXIS, lw=1, ls="--", zorder=1)
    ax.text(1.0, ax.get_ylim()[1], " 1 = as typical as a real neighbourhood", color=MUTED, fontsize=8.5, va="top")
    ax.set_xlabel("realism: typicality vs 156 real neighbourhoods (lower = more realistic)", color=INK2)
    ax.set_ylabel("terrain adaptiveness score TAS (0-100, higher = better adapted)", color=INK2)
    ax.grid(color=GRID_C, lw=0.6, zorder=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=8, mfc=c, mec=c, label=n) for n, c in FAM]
    hs += [plt.Line2D([], [], marker="o", ls="", ms=8, mfc=SURF, mec=INK2, mew=2, label="with F1 + F4 (M17)"),
           plt.Line2D([], [], marker="s", ls="", ms=8, mfc=INK2, mec=INK2, label="field tracer (M15)"),
           plt.Line2D([], [], marker="*", ls="", ms=14, color=INK, label=f"real {site.cfg['short']} ways (not a city)")]
    ax.legend(handles=hs, frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    ax.set_title(f"{site.cfg['short']} city limits: realism vs terrain adaptiveness (mean of seeds 1-3, bars = seed SD; faded = sparse, "
                 "< 5 km/km²). Best = top left", color=INK, fontsize=11.5, loc="left")
    fig.savefig(site.out / "fig_realism_vs_terrain.png", dpi=120, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    readme(site, rows)


def readme(site, rows):
    st = json.loads((site.out / "site.json").read_text())
    pc = lambda v: "–" if v is None or not np.isfinite(v) else f"{100 * v:.0f} %"
    md = [f"# {site.cfg['short']} terrain benchmark: every street generator (EXPLORATORY)", "",
          "Protocol, terrain adaptiveness score and flags: [docs/SITE_BENCHMARKS.md](../../../docs/SITE_BENCHMARKS.md) "
          "(declared before scoring). Code: `scripts/site_benchmark.py`.", "",
          f"**City limits** {st['city_km2']:.2f} km² {site.cfg['limits_where']} ({st['elev_city_min_max'][0]:.0f}-"
          f"{st['elev_city_min_max'][1]:.0f} m; median slope {100 * st['slope_city_median']:.0f} %; "
          f"{100 * st['slope_city_share_gt']['45%']:.0f} % of the land steeper than 45 %), {site.cfg['outside']['name']} outside; GLO-30 DSM, "
          "no supplied roads; seeds 1-3 (means ± seed SD).", "",
          "* **Realism** = typicality vs 156 real neighbourhoods (1 = as typical as a real one; **lower = more realistic**).",
          "* **TAS** = terrain adaptiveness 0-100 (**higher = better**) = mean of G (street length ≤ 15 % grade), C (contour "
          "following: 1 along contours, 0.36 random, 0 straight downhill) and S (share of street length on the gentler half "
          "of the land; 0.5 = unrelated to slope).",
          f"* **SPARSE** = under {SPARSE_KM_PER_KM2:.0f} km of street per km²: no town was built there.", "",
          "Files: `fig_maps_seed1.png` (all maps), `maps/<rank>_<method>_s<seed>.png` (one map per method and seed), "
          "`fig_realism_vs_terrain.png`, `site_terrain.png`, `summary.csv`, `per_seed.csv`, `summary.json`.", "",
          "| # | Generator | Realism | TAS | G gentle | C contour | S siting | km/km² | Coverage 60 m | Junctions/km² | "
          "4-way | Dead ends/km | Largest piece | > 15 % | > 20 % | Water m | Seconds |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    k = 0
    for r in rows:
        real = r["method"] == "real"
        k += not real
        name = f"**{label(r)}**" if real else label(r) + (" (sparse)" if r["sparse"] else "")
        md.append(f"| {'' if real else k} | {name} | {fmt(r['typicality_mean'])} ± {fmt(r['typicality_std'])} | "
                  f"{fmt(r['TAS_mean'], 0)} ± {fmt(r['TAS_std'], 0)} | {fmt(r['G_gentle_mean'])} | {fmt(r['C_contour_mean'])} | "
                  f"{fmt(r['S_siting_mean'])} | {fmt(r['street_km_per_km2_mean'], 1)} | {pc(r['coverage_60_mean'])} | "
                  f"{fmt(r['junctions_per_km2_mean'], 0)} | {pc(r.get('share_4way_mean', np.nan))} | "
                  f"{fmt(r.get('deadends_per_km_mean', np.nan), 1)} | {pc(r.get('largest_piece_share_mean', np.nan))} | "
                  f"{pc(r.get('over_15pct_mean', np.nan))} | {pc(r.get('over_20pct_mean', np.nan))} | "
                  f"{fmt(r['unbridged_water_m_mean'], 0)} | {fmt(r['seconds_mean'], 0)} |")
    (site.out / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "generate", "score", "figures"])
    ap.add_argument("--site", default="machu_picchu")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    site = Site(a.site)
    if a.stage == "data":
        stage_data(site)
    elif a.stage == "generate":
        stage_generate(site, a.workers, a.seeds, a.only)
    elif a.stage == "score":
        stage_score(site, a.workers, a.seeds)
    else:
        stage_figures(site)
    return 0


if __name__ == "__main__":
    sys.exit(main())
