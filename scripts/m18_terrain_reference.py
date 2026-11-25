"""Milestone 18, D11 step 1: real-city terrain reference curves and the terrain-realism score.

Common physical smoothing (every network, real, generated or external, before any measurement): the graph is chained
into junction-to-junction polylines of GROUND edges (a chain is cut where the structure changes; bridges and tunnels
are left out), each polyline is resampled every 30 m of arc length (end points kept; 30 m = the spacing of the real
street build, scripts/m12 build_graph), and the 30 m chords are cut into pieces of at most 10 m. So a route that climbs
in 4-9 m zigzags is measured on its 30 m chords, as a real street is.

Per piece (inside the 512 m tile): street grade = |dz| / length on the raw DEM; terrain slope and fall line from the
DEM low-passed at sigma 16 m; contour index = |cos| to the fall line. Per tile and terrain-slope class (0-5, 5-10,
10-20, 20-30, > 30 %), sums are kept so that any set of tiles can be pooled or resampled by 3 km block:
street length, length x contour index, length above 10 / 15 / 20 %, a length-weighted grade histogram (0.5 %
bins), land area of the class, and land of the class within 60 m of a street (coverage).

Curves per class: density (street km per km2 of that land), coverage share, mean contour index, grade p50 / p90,
shares above 10 / 15 / 20 %.

Reference = the real networks of the M12 TRAIN tiles with LiDAR (Spain, USA); the sealed test split is never read.
Real-vs-real scale = the standard deviation of each curve value across the six training cities. Terrain-realism score
of a set = the mean over classes and statistics of |set - reference| / between-city SD (1 = as far from the reference
as a typical real training city; lower = closer). TAS and G / C / S are reported beside it on the same smoothed
networks.

    python scripts/m18_terrain_reference.py  ->  outputs/m18/terrain_reference/{curves.json, README.md, fig_reference.png}
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
from scipy import ndimage

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m18" / "terrain_reference"
TERRAIN = ROOT / "data" / "processed" / "m12" / "terrain"
CLASSES = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 1000)]
KEYS = ["0-5", "5-10", "10-20", "20-30", "30-inf"]
GBINS = np.r_[np.arange(0, 60.01, 0.5), 1e9]
TILE, N, RES, COVER_M = 512.0, 256, 4.0, 60.0
STATS = ["density", "coverage", "contour_index", "grade_p50", "grade_p90", "share_gt10", "share_gt15", "share_gt20"]
SCORED = ["density", "coverage", "contour_index", "grade_p90", "share_gt15"]


def ground_polylines(g):
    """Junction-to-junction polylines of ground edges (coordinates), cut where the structure changes."""
    E = np.asarray(g.edges)
    if len(E) == 0:
        return []
    P = np.asarray(g.nodes)[:, :2]
    st = np.asarray(g.structs()) if hasattr(g, "structs") else np.zeros(len(E), int)
    keep = st == 0
    E = E[keep]
    adj = [[] for _ in range(len(P))]
    for k, (a, b) in enumerate(E):
        adj[a].append((b, k))
        adj[b].append((a, k))
    deg = np.array([len(x) for x in adj])
    seen = np.zeros(len(E), bool)
    out = []
    starts = list(np.flatnonzero((deg != 2) & (deg > 0))) + list(np.flatnonzero(deg == 2))
    for s in starts:
        for v, k in adj[s]:
            if seen[k]:
                continue
            pts, cur, kk = [P[s], P[v]], v, k
            seen[kk] = True
            while deg[cur] == 2:
                nxt = [(w, j) for w, j in adj[cur] if j != kk and not seen[j]]
                if not nxt:
                    break
                cur, kk = nxt[0]
                seen[kk] = True
                pts.append(P[cur])
            out.append(np.array(pts))
    return out


def resample(poly, step=30.0):
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 1e-6:
        return None
    t = np.r_[np.arange(0, s[-1], step), s[-1]]
    return np.c_[np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])]


class Field:
    def __init__(self, d):
        from urbanmorph.graphs.terrain_generate import terrain_of
        from urbanmorph.terrain.heightfield import Terrain
        self.T = terrain_of(d)
        zs = ndimage.gaussian_filter(np.where(d["valid"], d["elev"], np.nanmean(d["elev"])).astype(np.float64), 4.0)
        self.S = Terrain(zs.astype(np.float32), -256.0, 768.0, RES)
        self.gx, self.gy = self.S.gradient()
        slope = np.hypot(self.gx, self.gy) * 100
        land = self.S.window(0, 0, TILE, TILE) & (np.asarray(d.get("water", np.zeros_like(d["valid"]))) == 0) & d["valid"]
        self.land, self.slope = land, slope
        self.cls = np.digitize(slope, [5, 10, 20, 30])


def tile_sums(g, F: Field):
    """Per-class sums for one network on one tile (dict of arrays of length 5, plus the grade histogram 5 x bins).
    A larger domain (a site) passes ``F.extent`` (its side in metres; grids keep the 256 m margin)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    import shapely
    ext, n_ = getattr(F, "extent", TILE), F.land.shape[0]
    z = {k: np.zeros(5) for k in ("L", "LCI", "gt10", "gt15", "gt20", "land", "covered")}
    z["hist"] = np.zeros((5, len(GBINS) - 1))
    for c in range(5):
        z["land"][c] = (F.land & (F.cls == c)).sum() * RES * RES / 1e6
    polys = [q for q in (resample(p) for p in ground_polylines(g)) if q is not None and len(q) >= 2]
    if not polys:
        return z
    A = np.concatenate([q[:-1] for q in polys])
    B = np.concatenate([q[1:] for q in polys])
    L = np.hypot(*(B - A).T)
    n = np.maximum(1, np.ceil(L / 10.0)).astype(int)
    k = np.repeat(np.arange(len(L)), n)
    j = np.concatenate([np.arange(m) for m in n])
    p0 = A[k] + (j / n[k])[:, None] * (B[k] - A[k])
    p1 = A[k] + ((j + 1) / n[k])[:, None] * (B[k] - A[k])
    mid, pl = (p0 + p1) / 2, L[k] / n[k]
    ok = (mid[:, 0] >= 0) & (mid[:, 0] <= ext) & (mid[:, 1] >= 0) & (mid[:, 1] <= ext) & (pl > 0.5)
    p0, p1, mid, pl = p0[ok], p1[ok], mid[ok], pl[ok]
    gr = np.abs(F.T.sample(p1[:, 0], p1[:, 1]) - F.T.sample(p0[:, 0], p0[:, 1])) / pl * 100
    r, c = F.S.rc(mid[:, 0], mid[:, 1])
    r = np.clip(np.round(r).astype(int), 0, n_ - 1)
    c = np.clip(np.round(c).astype(int), 0, n_ - 1)
    sx, sy = F.gx[r, c], F.gy[r, c]
    ci = np.abs((p1 - p0)[:, 0] / pl * sx + (p1 - p0)[:, 1] / pl * sy) / np.maximum(np.hypot(sx, sy), 1e-9)
    cl = F.cls[r, c]
    m = np.isfinite(gr)
    for q in range(5):
        s = m & (cl == q)
        z["L"][q] = pl[s].sum()
        z["LCI"][q] = (pl[s] * ci[s]).sum()
        for t in (10, 15, 20):
            z[f"gt{t}"][q] = pl[s][gr[s] > t].sum()
        z["hist"][q] = np.histogram(gr[s], GBINS, weights=pl[s])[0]
    ras = rasterize([(shapely.LineString(q), 1) for q in polys], out_shape=F.land.shape,
                    transform=from_origin(-256.0, ext + 256.0, RES, RES), fill=0, dtype="uint8").astype(bool)
    near = ndimage.distance_transform_edt(~ras) * RES <= COVER_M
    for q in range(5):
        z["covered"][q] = (F.land & (F.cls == q) & near).sum() * RES * RES / 1e6
    return z


def pool(sums):
    """Curves of a pooled set of tile sums."""
    t = {k: np.sum([s[k] for s in sums], axis=0) for k in sums[0]}
    out = {}
    for q, key in enumerate(KEYS):
        L = t["L"][q]
        h = t["hist"][q]
        cdf = np.cumsum(h) / max(h.sum(), 1e-9)
        qt = lambda p: float(GBINS[min(int(np.searchsorted(cdf, p)), len(GBINS) - 2)] + 0.25) if h.sum() > 0 else np.nan
        out[key] = {"density": float(L / 1000 / max(t["land"][q], 1e-9)) if t["land"][q] > 0.01 else np.nan,
                    "coverage": float(t["covered"][q] / t["land"][q]) if t["land"][q] > 0.01 else np.nan,
                    "contour_index": float(t["LCI"][q] / L) if L > 100 else np.nan,
                    "grade_p50": qt(0.5) if L > 100 else np.nan, "grade_p90": qt(0.9) if L > 100 else np.nan,
                    "share_gt10": float(t["gt10"][q] / L) if L > 100 else np.nan,
                    "share_gt15": float(t["gt15"][q] / L) if L > 100 else np.nan,
                    "share_gt20": float(t["gt20"][q] / L) if L > 100 else np.nan,
                    "street_km": float(L / 1000), "land_km2": float(t["land"][q])}
    return out


def tas(sums, fields_land_slopes):
    """TAS and G / C / S of a pooled set on the smoothed networks (C over land steeper than 5 %; S below the set's
    median land slope, computed from the per-class land as in the site benchmark's definition at class resolution)."""
    t = {k: np.sum([s[k] for s in sums], axis=0) for k in sums[0]}
    L = t["L"].sum()
    G = 1 - t["gt15"].sum() / max(L, 1e-9)
    C = 1 - t["LCI"][1:].sum() / max(t["L"][1:].sum(), 1e-9)
    land = t["land"]
    half = np.cumsum(land) / land.sum()
    k = int(np.searchsorted(half, 0.5))                      # the gentler half of the land, at class resolution
    S = (t["L"][:k].sum() + t["L"][k] * max(0.0, (0.5 - (half[k - 1] if k else 0)) / max(land[k] / land.sum(), 1e-9))) / max(L, 1e-9)
    return {"G": float(G), "C": float(C), "S": float(min(S, 1.0)), "TAS": float(100 * (G + C + min(S, 1.0)) / 3)}


def score(curves, ref, sd):
    vals = []
    for key in KEYS:
        for st in SCORED:
            a, b, s = curves[key][st], ref[key][st], sd[key][st]
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(s) and s > 0:
                vals.append(abs(a - b) / s)
    return float(np.mean(vals)) if vals else np.nan


def train_tiles():
    d = pickle.loads((ROOT / "data" / "processed" / "m12" / "graphs3d.pkl").read_bytes())
    for sid, _, g in d["train"]:                              # never the test split
        f = TERRAIN / f"{sid}.npz"
        if f.exists():
            z = dict(np.load(f))
            if z["valid"].all():
                yield sid, g, z


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    ref_sums, by_city = [], {}
    for sid, g, d in train_tiles():
        s = tile_sums(g, Field(d))
        ref_sums.append(s)
        by_city.setdefault(sid[:7], []).append(s)
    ref = pool(ref_sums)
    city = {c: pool(v) for c, v in by_city.items()}
    sd = {key: {st: float(np.nanstd([city[c][key][st] for c in city], ddof=1)) for st in STATS} for key in KEYS}
    out = {"reference": ref, "between_city_sd": sd, "cities": city, "n_tiles": len(ref_sums),
           "tiles_per_city": {c: len(v) for c, v in by_city.items()}, "sets": {}}
    # development sets on GEN2: the real networks and the stored M14 runs (TN) and the TNg runs
    from m14_build_gen2 import gen2_tiles
    from m14_report import get
    from urbanmorph.experiments.graph_io import graph_from_json
    T = gen2_tiles()
    F = {t["sid"]: Field(t["d"]) for t in T}
    real = [tile_sums(t["graph"], F[t["sid"]]) for t in T]
    out["sets"]["real GEN2"] = {"curves": pool(real), "tas": tas(real, None)}
    for fam in ("TN", "TNg"):
        for lab in ("A", "M*", "M*+repair", "NA-SMC", "NA-fixed", "P"):
            if fam == "TNg" and lab not in ("M*", "M*+repair", "NA-SMC"):
                continue
            sums = []
            for b in range(3):
                r = get("gen2", f"{fam}_s{b}", lab)
                if r is None or (lab == "P" and b > 0):
                    continue
                for sd_ in r["seeds"].values():
                    sums += [tile_sums(graph_from_json(gj), F[row["sid"]]) for row, gj in zip(sd_["rows"], sd_["graphs"])]
            if sums:
                name = f"{lab} [{fam}]" if lab != "P" else "P"
                out["sets"][name] = {"curves": pool(sums), "tas": tas(sums, None)}
                print(name, len(sums), flush=True)
    # leakage-free sets: the natural windows of leakage control (b) carry no street beds, so no generator can copy
    # streets there; the reference curves need no real streets on the same tile
    nat = json.loads((ROOT / "outputs" / "m18" / "natural" / "windows.json").read_text())["windows"]
    NF = [Field(dict(np.load(ROOT / "data" / "processed" / "m18" / "natural" / f"{w['name']}.npz"))) for w in nat]
    groups = {}
    for key, r in json.loads((ROOT / "outputs" / "m18" / "natural" / "runs.json").read_text())["runs"].items():
        lab, bb, _ = key.split("|")
        fam = "P" if bb == "-" else bb.split("_")[0]
        groups.setdefault(f"natural: {lab} [{fam}]" if lab != "P" else "natural: P", []).append(r)
    for name, rs in groups.items():
        sums = [tile_sums(graph_from_json(gj), f) for r in rs for gj, f in zip(r["graphs"], NF)]
        out["sets"][name] = {"curves": pool(sums), "tas": tas(sums, None)}
    for name, v in out["sets"].items():
        v["terrain_realism"] = score(v["curves"], ref, sd)
    out["reference_tas"] = tas(ref_sums, None)
    out["real_city_scores"] = {c: score(city[c], ref, sd) for c in city}
    atomic_json(OUT / "curves.json", out)

    ink, ink2, grid_c, ref_c = "#1f2328", "#59636e", "#e6e8eb", "#d1242f"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    panels = [("density", "Street km per km² of that land"), ("coverage", "Land within 60 m of a street"),
              ("contour_index", "Contour index (0 = along contours)"), ("share_gt15", "Street length steeper than 15 %")]
    show = {"real GEN2": ("#1f2328", "--"), "M* [TN]": ("#2a78d6", "-"), "NA-SMC [TN]": ("#1a5aa6", "-"),
            "NA-SMC [TNg]": ("#7b4fb0", "-"), "M*+repair [TN]": ("#9aa4ae", "-"), "P": ("#4c9a5f", "-")}
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
    xs = np.arange(5)
    for ax, (st, ttl) in zip(axs, panels):
        rv = np.array([ref[k][st] for k in KEYS])
        sv = np.array([sd[k][st] for k in KEYS])
        ax.fill_between(xs, rv - sv, rv + sv, color=ref_c, alpha=0.15, lw=0)
        ax.plot(xs, rv, "-", color=ref_c, lw=3, label="training cities (reference ± between-city SD)")
        for nm, (col, ls) in show.items():
            if nm in out["sets"]:
                ax.plot(xs, [out["sets"][nm]["curves"][k][st] for k in KEYS], ls, color=col, lw=1.5, marker="o", ms=4, label=nm)
        ax.set_xticks(xs)
        ax.set_xticklabels(["0–5", "5–10", "10–20", "20–30", "> 30"])
        ax.set_xlabel("terrain slope class (%)")
        ax.set_title(ttl, loc="left", fontweight="bold", fontsize=10)
        ax.grid(color=grid_c, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle(f"Real-city terrain reference curves ({len(ref_sums)} training tiles, 6 cities) and GEN2 sets, all networks "
                 "smoothed alike (30 m resampling)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_reference.png", dpi=130)
    plt.close(fig)
    md = ["# M18 D11 step 1: real-city terrain reference curves and the terrain-realism score", "",
          f"Reference: {len(ref_sums)} training tiles with LiDAR ({json.dumps(out['tiles_per_city'])}). Script and "
          "definitions: `scripts/m18_terrain_reference.py`. Terrain-realism: mean |set − reference| / between-city SD "
          "over 5 slope classes × (density, coverage, contour index, grade p90, share > 15 %); lower = closer.", "",
          "Real training cities scored against the pooled reference (yardstick): " +
          ", ".join(f"{c} {v:.2f}" for c, v in out["real_city_scores"].items()), "",
          "| Set (GEN2 city tiles; \"natural:\" = leakage-free natural windows) | terrain realism | TAS | G | C | S |", "|---|---|---|---|---|---|"]
    for name, v in sorted(out["sets"].items(), key=lambda kv: kv[1]["terrain_realism"]):
        t = v["tas"]
        md.append(f"| {name} | {v['terrain_realism']:.2f} | {t['TAS']:.1f} | {t['G']:.2f} | {t['C']:.2f} | {t['S']:.2f} |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
