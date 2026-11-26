"""M18 external street-network baselines on the named terrain sites (protocol: docs/SITE_BENCHMARKS.md; report:
outputs/m18/external/README.md).

Two external methods run on exactly the inputs of the 24 internal generator versions (the 2048 m domain, the 4 m
GLO-30 grid, the water layer, the fixed city limits, seeds 1-3) and are stored in the benchmark cache, so the
unchanged score code of scripts/site_benchmark.py scores them. They use only the DEM, the water layer and the
city-limits mask, plus seeded internal choices:

* "leastcostpath (ext)" = leastcostpath (Lewis) routed between seeded anchors; merge by us.
  R part: scripts/m18_ext_leastcostpath.R (CRAN leastcostpath, GPL-3). Anchors: farthest-point sampling of K = 12
  cells inside the limits, not water, with a DEM slope below the city's median DEM slope; the first anchor is drawn
  with the run seed. Cost: "wheeled transport", critical slope 12 %, max_slope 20 %, 16 neighbours; water = barrier;
  routing restricted to the city limits (NA outside). All ordered anchor pairs (create_FETE_lcps). Merge (here,
  shapely): collinear vertices dropped, each path snapped (8 m) to the union of the longer ones, unary union (nodes
  every crossing and dissolves duplicate overlapping pieces), duplicate edges dropped; class 3, ground.
* "citygen P&M (ext)" = t-mw/citygen (MIT, third_party/citygen at a pinned commit), run headless in Node
  (scripts/m18_citygen/), with only Parish & Mueller 2001's local constraints added: a segment that enters water,
  leaves the city limits or has a DEM step steeper than 20 % is pruned, rotated within a maximal angle or rejected.
  Scale: 1 citygen unit = REAL_LINK_MEDIAN_M / 300 m (citygen's median junction-to-junction link at default
  settings is 300 units = its DEFAULT_SEGMENT_LENGTH; the real median link is 62.87 m). Start: the city-limits
  centroid. Stop: street length per km2 inside the limits = the median of the internal versions at the site, or the
  upstream segment budget (2000). Highways class 1, streets class 2, ground.

    .venv/bin/python scripts/site_benchmark.py generate --site S --only ext_leastcostpath,ext_citygen_PM
    .venv/bin/python scripts/m18_external.py sanity      # package example, citygen demo, scale calibration
    .venv/bin/python scripts/m18_external.py report      # per-run diagnostics -> outputs/m18/external/runs.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

RSCRIPT = shutil.which("Rscript") or "/usr/local/bin/Rscript"
NODE = shutil.which("node") or "/usr/local/bin/node"
R_SCRIPT = ROOT / "scripts" / "m18_ext_leastcostpath.R"
CITYGEN = ROOT / "third_party" / "citygen"
CITYGEN_COMMIT = "aec49d2efaa59a1eebffaacc322459c5219ab9a6"
RUNNER_SRC = ROOT / "scripts" / "m18_citygen"
RUNNER = RUNNER_SRC / "build" / "runner.cjs"
OUT = ROOT / "outputs" / "m18" / "external"

LCP_LABEL, CG_LABEL = "leastcostpath (ext)", "citygen P&M (ext)"
LCP = {"cost_function": "wheeled transport", "crit_slope_pct": 12, "max_slope_pct": 20, "neighbours": 16,
       "anchors_k": 12, "snap_m": 8.0, "class": 3}
REAL_LINK_MEDIAN_M = 62.87035481544801      # outputs/m18/distributions/summary.json, real/link p50 (GEN2 real tiles)
CITYGEN_LINK_MEDIAN_UNITS = 300.0            # measured by `sanity` (upstream default settings, seeds 1-3)
CG = {"scale_m_per_unit": REAL_LINK_MEDIAN_M / CITYGEN_LINK_MEDIAN_UNITS, "max_grade": 0.20, "sample_step_m": 4.0,
      "prune_min_fraction": 0.5, "rotate_max_deg": 90.0, "rotate_step_deg": 5.0, "segment_budget": 2000,
      "max_proposals": 200000, "axiom_dir_step_deg": 5.0, "class_highway": 1, "class_street": 2}


def workdir(site, key: str, seed: int | None = None) -> Path:
    d = site.data / "external" / (key if seed is None else f"{key}_s{seed}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def inputs(site):
    from site_benchmark import RES
    big = site.domain()
    mask = np.load(site.data / "city_mask.npz")["mask"]
    return big, mask, float(mask.sum() * RES ** 2 / 1e6)


def graph_from_segments(P: np.ndarray, Q: np.ndarray, cls: np.ndarray, L: float):
    """RoadGraph from segment end points (domain metres): identical points (1 mm) become one node; self-loops and
    repeated edges are dropped (the most major class is kept)."""
    from urbanmorph.graphs.road_graph import RoadGraph, is_border
    if len(P) == 0:
        return RoadGraph(np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, bool), edge_class=np.zeros(0, int),
                         edge_struct=np.zeros(0, int))
    key = np.round(np.vstack([P, Q]) * 1000).astype(np.int64)
    u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    nodes = np.vstack([P, Q])[first]
    inv = inv.ravel()
    E = np.c_[inv[:len(P)], inv[len(P):]]
    ok = E[:, 0] != E[:, 1]
    E, c = np.sort(E[ok], 1), np.asarray(cls)[ok]
    o = np.lexsort((c, E[:, 1], E[:, 0]))
    E, c = E[o], c[o]
    keep = np.r_[True, (np.diff(E, axis=0) != 0).any(1)]
    E, c = E[keep], c[keep]
    used, inv2 = np.unique(E, return_inverse=True)
    N = nodes[used]
    return RoadGraph(N, inv2.reshape(-1, 2), is_border(N, size=L), edge_class=c.astype(int),
                     edge_struct=np.zeros(len(c), int))


# ------------------------------------------------------------------------------------------- leastcostpath
def lcp_anchors(site, big, mask, seed: int, k: int = LCP["anchors_k"]):
    """Farthest-point sampling of k cell centres inside the limits, not water, with a DEM slope (central differences on
    the 4 m grid) below the city's median DEM slope; the first point is drawn with the run seed."""
    from site_benchmark import RES
    gy, gx = np.gradient(big["elev"].astype(float), RES)
    s = np.hypot(gx, gy)
    med = float(np.median(s[mask]))
    ok = mask & (s < med) & (big["water"] == 0)
    X, Y = site.xy_grid()
    P = np.c_[X[ok], Y[ok]]
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(P)))]
    d = np.hypot(*(P - P[idx[0]]).T)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.hypot(*(P - P[j]).T))
    return P[idx], med


def lcp_merge(lines: list[np.ndarray], L: float, snap: float = LCP["snap_m"]):
    """Planar graph from overlapping least-cost paths (domain metres).

    Paths are added longest first. A vertex of a new path within ``snap`` m of the network built so far duplicates it:
    each run of such vertices is dropped, and the path is joined to the network at the network vertex nearest to the
    run's first and last vertex (snap within ``snap`` m). Then a unary union nodes every crossing and dissolves
    overlapping pieces, and repeated edges are dropped. (Plain GEOS vertex snapping was tried first: on cell-scale
    zigzags it joins vertices of parallel zigzags and makes a mesh of spurious junctions.)"""
    import shapely
    from scipy.spatial import cKDTree
    paths = [np.asarray(c, float)[:, :2] for c in lines if len(c) >= 2]
    paths.sort(key=lambda c: -np.hypot(*np.diff(c, axis=0).T).sum())
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    verts = np.zeros((0, 2))
    for c in paths:
        if not segs:
            segs += [(c[i], c[i + 1]) for i in range(len(c) - 1)]
            verts = c.copy()
            continue
        tree = shapely.STRtree([shapely.LineString([a, b]) for a, b in segs])
        _, dist = tree.query_nearest(shapely.points(c), return_distance=True, all_matches=False)
        near = dist < snap
        kd = cKDTree(verts)
        prev = None                                          # last kept point of the path (or network vertex)
        for i in range(len(c)):
            if near[i]:
                if prev is not None and not near[i - 1]:     # entering the network: join at its nearest vertex
                    segs.append((prev, verts[kd.query(c[i])[1]]))
                    prev = None
                continue
            if i > 0 and near[i - 1]:                        # leaving the network
                prev = verts[kd.query(c[i - 1])[1]]
            if prev is not None:
                segs.append((prev, c[i]))
            prev = c[i]
        verts = np.vstack([verts, c[~near]])
    lines2 = [shapely.LineString([a, b]) for a, b in segs if np.hypot(*(b - a)) > 1e-6]
    net = shapely.unary_union(lines2) if lines2 else shapely.GeometryCollection()
    P, Q = [], []
    for part in getattr(net, "geoms", [net]):
        if part.geom_type != "LineString" or part.is_empty:
            continue
        c = np.asarray(part.coords)[:, :2]
        P.append(c[:-1])
        Q.append(c[1:])
    if not P:
        return graph_from_segments(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int), L)
    P, Q = np.vstack(P), np.vstack(Q)
    return graph_from_segments(P, Q, np.full(len(P), LCP["class"]), L)


def run_lcp(site, seed: int):
    import pandas as pd
    import rasterio
    from rasterio.transform import from_origin
    from site_benchmark import MARGIN, RES
    big, mask, area = inputs(site)
    wd = workdir(site, "ext_leastcostpath", seed)
    rows, cols = np.nonzero(mask)
    r0, r1 = max(rows.min() - 2, 0), min(rows.max() + 3, site.n)
    c0, c1 = max(cols.min() - 2, 0), min(cols.max() + 3, site.n)
    dem = big["elev"][r0:r1, c0:c1].astype(np.float32).copy()
    dem[~mask[r0:r1, c0:c1]] = np.nan                            # routing restricted to the city limits
    water = (big["water"][r0:r1, c0:c1] > 0).astype(np.uint8)
    tr = from_origin(site.x0 - MARGIN + c0 * RES, site.y0 + site.L + MARGIN - r0 * RES, RES, RES)
    for name, arr, dt, nd in (("dem.tif", dem, "float32", np.nan), ("water.tif", water, "uint8", None)):
        with rasterio.open(wd / name, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dt,
                           crs=site.crs, transform=tr, nodata=nd) as f:
            f.write(arr, 1)
    A, med = lcp_anchors(site, big, mask, seed)
    pd.DataFrame({"id": np.arange(1, len(A) + 1), "x": A[:, 0] + site.x0, "y": A[:, 1] + site.y0}).to_csv(
        wd / "anchors.csv", index=False)
    t0 = time.time()
    r = subprocess.run([RSCRIPT, str(R_SCRIPT), str(wd / "dem.tif"), str(wd / "water.tif"), str(wd / "anchors.csv"),
                        str(wd / "paths.csv"), LCP["cost_function"], str(LCP["crit_slope_pct"]),
                        str(LCP["max_slope_pct"]), str(LCP["neighbours"])], capture_output=True, text=True)
    (wd / "r.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"leastcostpath R step failed ({r.returncode}): {r.stderr[-2000:]}")
    r_sec = time.time() - t0
    df = pd.read_csv(wd / "paths.csv")
    lines = [g[["x", "y"]].to_numpy() - [site.x0, site.y0] for _, g in df.groupby("path", sort=True)]
    pairs = df.groupby("path")[["origin", "destination"]].first()
    raw_km = float(sum(np.hypot(*np.diff(c, axis=0).T).sum() for c in lines) / 1000)
    t1 = time.time()
    g = lcp_merge(lines, site.L)
    Le = np.linalg.norm(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]], axis=1) if len(g.edges) else np.zeros(0)
    log = r.stdout
    enterable = int(log.split("enterable")[1].split(";")[0]) if "enterable" in log else -1
    stats = {"anchors": len(A), "anchors_enterable": enterable, "anchor_slope_threshold": med,
             "paths": int(len(lines)), "ordered_pairs": len(A) * (len(A) - 1),
             "pairs_unordered_connected": int(len({tuple(sorted(p)) for p in pairs.itertuples(index=False)})),
             "raw_path_km": raw_km, "merged_km": float(Le.sum() / 1000), "r_seconds": r_sec,
             "merge_seconds": time.time() - t1, "city_km2": area}
    (wd / "stats.json").write_text(json.dumps(stats, indent=1))
    return g, stats


# ------------------------------------------------------------------------------------------- citygen
def build_runner(force: bool = False) -> Path:
    srcs = list(RUNNER_SRC.glob("*.ts"))
    if not force and RUNNER.exists() and RUNNER.stat().st_mtime > max(p.stat().st_mtime for p in srcs):
        return RUNNER
    if not (CITYGEN / "node_modules").exists():
        subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=CITYGEN, check=True, capture_output=True)
    env = {"NODE_PATH": str(CITYGEN / "node_modules"), "PATH": f"{Path(NODE).parent}:/usr/bin:/bin"}
    subprocess.run([str(CITYGEN / "node_modules" / ".bin" / "esbuild"), str(RUNNER_SRC / "runner.ts"), "--bundle",
                    "--platform=node", "--format=cjs", "--target=node22", f"--outfile={RUNNER}", "--log-level=warning"],
                   check=True, env=env, cwd=ROOT)
    return RUNNER


def internal_median_density(site) -> float:
    """Median street km per km2 (inside the limits) of the internal generator versions, from the site summary."""
    import pandas as pd
    d = pd.read_csv(site.out / "summary.csv")
    d = d[d.milestone.isin(["M12", "M13", "M14", "M15", "M16"])]
    assert len(d) == 24, len(d)
    return float(d.street_km_per_km2_mean.median())


def run_citygen(site, seed: int):
    from site_benchmark import MARGIN, RES
    big, mask, area = inputs(site)
    runner = build_runner()
    wd = workdir(site, "ext_citygen_PM")
    big["elev"].astype("<f4").tofile(wd / "elev.f32")
    (mask & (big["water"] == 0)).astype(np.uint8).tofile(wd / "legal.u8")
    X, Y = site.xy_grid()
    dens = internal_median_density(site)
    params = {"n": site.n, "res": RES, "margin": MARGIN, "L": site.L, "elev_path": str(wd / "elev.f32"),
              "legal_path": str(wd / "legal.u8"), "centroid_m": [float(X[mask].mean()), float(Y[mask].mean())],
              "target_length_m": dens * area * 1000.0, **{k: v for k, v in CG.items() if not k.startswith("class")}}
    (wd / "params.json").write_text(json.dumps(params, indent=1))
    out = wd / f"city_s{seed}.json"
    t0 = time.time()
    r = subprocess.run([NODE, str(runner), "--params", str(wd / "params.json"), "--seed", str(seed), "--out", str(out)],
                       capture_output=True, text=True)
    (wd / f"node_s{seed}.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"citygen runner failed ({r.returncode}): {r.stderr[-2000:]}")
    res = json.loads(out.read_text())
    S = np.asarray(res["segments"], float).reshape(-1, 5)
    cls = np.where(S[:, 4] > 0, CG["class_highway"], CG["class_street"])
    g = graph_from_segments(S[:, :2], S[:, 2:4], cls, site.L)
    stats = {**{k: v for k, v in res["stats"].items() if isinstance(v, (int, float))}, "stop_reason": res["stats"].get("stopReason"),
             "segments": int(len(S)), "length_km": res["length_m"] / 1000, "target_km_per_km2": dens,
             "target_km": params["target_length_m"] / 1000, "density_generated_km_per_km2": res["length_m"] / 1000 / area,
             "axiom_offset_m": (res["axiom"] or {}).get("offset_from_centroid_m"),
             "axiom_dir_deg": (res["axiom"] or {}).get("dir_deg"), "population_at_axiom": res["population_at_axiom"],
             "population_share_above_threshold": res["population_share_above_threshold"],
             "node_seconds": time.time() - t0, "city_km2": area}
    return g, stats


def generate(site, lab: str, seed: int):
    """Entry point of scripts/site_benchmark.generate_one for the external labels: (RoadGraph, stats)."""
    if lab == LCP_LABEL:
        return run_lcp(site, seed)
    if lab == CG_LABEL:
        return run_citygen(site, seed)
    raise KeyError(lab)


# ------------------------------------------------------------------------------------------- sanity / report
def max_step_grade(g, big, site, step: float = 4.0) -> dict:
    """Share of street length (ground edges) whose raw-DEM steps of <= ``step`` m are >= 20 % (the generators' own
    rule, on their own input; not the benchmark's bare-earth proxy)."""
    from site_benchmark import MARGIN, RES
    from urbanmorph.terrain.heightfield import Terrain
    T = Terrain(big["elev"].astype(np.float32), -MARGIN, site.L + MARGIN, RES)
    tot = bad = 0.0
    for a, b in g.edges:
        p, q = g.nodes[a], g.nodes[b]
        Ln = float(np.hypot(*(q - p)))
        if Ln <= 0:
            continue
        k = max(1, int(np.ceil(Ln / step)))
        t = np.linspace(0, 1, k + 1)
        P = p + np.outer(t, q - p)
        z = T.sample(P[:, 0], P[:, 1])
        gr = np.abs(np.diff(z)) / (Ln / k)
        tot += Ln
        bad += Ln / k * float((gr >= 0.20).sum())
    return {"km": tot / 1000, "share_steps_ge_20pct": bad / tot if tot else np.nan}


def sanity():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import rasterio
    from m18_audit_m14 import links
    sd = OUT / "sanity"
    sd.mkdir(parents=True, exist_ok=True)
    rep = {}
    # 1. leastcostpath: the package's create_FETE_lcps example
    r = subprocess.run([RSCRIPT, str(R_SCRIPT), "--example", str(sd / "lcp_example.csv")], capture_output=True, text=True)
    rep["lcp_example_stdout"] = [l for l in r.stdout.splitlines() if l.strip()]
    df = pd.read_csv(sd / "lcp_example.csv")
    tif = subprocess.run([RSCRIPT, "-e", 'cat(system.file("extdata/SICILY_1000m.tif", package="leastcostpath"))'],
                         capture_output=True, text=True).stdout.strip()
    with rasterio.open(tif) as f:
        z, b = f.read(1, masked=True), f.bounds
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    axs[0].imshow(z, extent=(b.left, b.right, b.bottom, b.top), cmap="terrain")
    for _, gg in df.groupby("path"):
        axs[0].plot(gg.x, gg.y, lw=1.6, color="#c0392b")
    axs[0].set_title("leastcostpath 2.0.13 example (?create_FETE_lcps): Sicily 1 km DEM, tobler, 4 neighbours,\n"
                     f"{df.path.nunique()} paths (locations 3 and 4 coincide, so the package drops them)", fontsize=9)
    # 2. citygen: upstream default settings (no terrain) vs the derived generator with terrain off
    build_runner()
    demo = []
    for s in (1, 2, 3):
        out = sd / f"citygen_demo_s{s}.json"
        subprocess.run([NODE, str(RUNNER), "--demo", "--seed", str(s), "--out", str(out)], check=True, capture_output=True)
        d = json.loads(out.read_text())
        S = np.asarray(d["upstream"], float)
        g = graph_from_segments(S[:, :2], S[:, 2:4], np.where(S[:, 4] > 0, 1, 2), 1e9)
        lk = links(g)
        demo.append({"seed": s, "segments": len(S), "identical_to_upstream": d["identical"],
                     "link_median_units": float(np.median(lk)), "highway_share": float(S[:, 4].mean())})
        if s == 1:
            for hw, lw, col in ((0, 0.5, "#333"), (1, 1.6, "#c0392b")):
                T = S[S[:, 4] == hw]
                axs[1].plot(np.c_[T[:, 0], T[:, 2], np.full(len(T), np.nan)].ravel(),
                            np.c_[T[:, 1], T[:, 3], np.full(len(T), np.nan)].ravel(), lw=lw, color=col)
            axs[1].set_aspect("equal")
            axs[1].set_title("citygen, upstream generateCity(seed 1), default settings, headless (2000 segments;\n"
                             "red = highways). Compare third_party/citygen/screen.jpg", fontsize=9)
    rep["citygen_demo"] = demo
    med = float(np.median([d["link_median_units"] for d in demo]))
    rep["scale"] = {"citygen_link_median_units": med, "real_link_median_m": REAL_LINK_MEDIAN_M,
                    "scale_m_per_unit": REAL_LINK_MEDIAN_M / med, "used": CG["scale_m_per_unit"]}
    assert abs(med - CITYGEN_LINK_MEDIAN_UNITS) < 1e-3, med
    fig.tight_layout()
    fig.savefig(sd / "fig_sanity.png", dpi=100)
    plt.close(fig)
    (sd / "sanity.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def report(sites=("machu_picchu", "petra")):
    """Per-run diagnostics of the cached external runs -> outputs/m18/external/runs.json."""
    import pickle
    from site_benchmark import Site, key_of
    out = {}
    for name in sites:
        site = Site(name)
        big = site.domain()
        for lab in (LCP_LABEL, CG_LABEL):
            for s in (1, 2, 3):
                f = site.cache / f"{key_of(lab)}_s{s}.pkl"
                if not f.exists():
                    continue
                c = pickle.loads(f.read_bytes())
                out[f"{name} | {lab} | {s}"] = {"seconds": c["seconds"], **c["stats"], **max_step_grade(c["graph"], big, site)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


def chains(g) -> list[np.ndarray]:
    """Polylines of a graph between nodes of degree != 2 (cycles of degree-2 nodes skipped)."""
    adj = g.adjacency()
    deg = np.array([len(a) for a in adj])
    seen, out = set(), []
    for s in np.flatnonzero(deg != 2):
        for v in adj[s]:
            if (min(s, v), max(s, v)) in seen:
                continue
            path, prev, cur = [s, v], s, v
            seen.add((min(s, v), max(s, v)))
            while deg[cur] == 2:
                nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
                if (min(cur, nxt), max(cur, nxt)) in seen:
                    break
                seen.add((min(cur, nxt), max(cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
            out.append(g.nodes[path])
    return out


def sensitivity(sites=("machu_picchu", "petra")):
    """Variants scored in memory with the benchmark's own functions (crop, typicality, TAS, statistics); nothing is
    written to the site tables. (a) leastcostpath: the merged graph re-built with the repository's polyline-to-graph
    convention (build_graph: resample <= 30 m, merge 8 m, as for the real OSM ways), which collapses the cell-scale
    zigzags. (b) citygen P&M with a 30 deg maximal rotation instead of 90 deg."""
    import pickle
    import site_benchmark as SB
    from urbanmorph.graphs.road_graph import RoadGraph, build_graph, is_border
    out = {}
    for name in sites:
        site = SB.Site(name)
        SB.init(name)
        for s in (1, 2, 3):
            variants = {}
            c = pickle.loads((site.cache / f"{SB.key_of(LCP_LABEL)}_s{s}.pkl").read_bytes())
            ch = chains(c["graph"])
            gg = build_graph(ch, classes=[LCP["class"]] * len(ch))
            variants["leastcostpath, zigzags collapsed (build_graph 30 m / 8 m)"] = RoadGraph(
                gg.nodes, gg.edges, is_border(gg.nodes, size=site.L), edge_class=gg.classes(),
                edge_struct=np.zeros(len(gg.edges), int))
            wd = workdir(site, "ext_citygen_PM")
            p = json.loads((wd / "params.json").read_text())
            p["rotate_max_deg"] = 30.0
            (wd / "params_rot30.json").write_text(json.dumps(p, indent=1))
            o = wd / f"city_rot30_s{s}.json"
            subprocess.run([NODE, str(build_runner()), "--params", str(wd / "params_rot30.json"), "--seed", str(s), "--out",
                            str(o)], check=True, capture_output=True)
            S = np.asarray(json.loads(o.read_text())["segments"], float).reshape(-1, 5)
            variants["citygen P&M, maximal rotation 30 deg"] = graph_from_segments(
                S[:, :2], S[:, 2:4], np.where(S[:, 4] > 0, CG["class_highway"], CG["class_street"]), site.L)
            for k, g in variants.items():
                gc = SB.crop(g, SB.G["mask"], site)
                typ, nw = SB.typicality(gc, np.random.default_rng([s, 17]))
                out.setdefault(name, {}).setdefault(k, []).append(
                    {"seed": s, "typicality": typ, "windows": nw, **SB.tas(gc), **SB.city_stats(gc)})
    summ = {n: {k: {m: float(np.nanmean([r[m] for r in v])) for m in ("typicality", "TAS", "G_gentle", "C_contour",
                                                                         "S_siting", "street_km_per_km2")}
                for k, v in d.items()} for n, d in out.items()}
    (OUT / "sensitivity.json").write_text(json.dumps({"per_seed": out, "mean": summ}, indent=1, default=float))
    print(json.dumps(summ, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sanity", "report", "sensitivity", "build"])
    a = ap.parse_args()
    if a.stage == "sanity":
        sanity()
    elif a.stage == "report":
        report()
    elif a.stage == "sensitivity":
        sensitivity()
    else:
        print(build_runner(force=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
