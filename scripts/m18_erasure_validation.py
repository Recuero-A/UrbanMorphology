"""Milestone 18: validation of the street-erased DEM (TNse, the leakage control of M13; scripts/m13_street_erase.py).

Three tests (docs/M18_LOG.md D5). Development data only: M12 train / val / val_t tiles, never the sealed test split.

1. random  - Random-strip control. The raw-DEM backbone TN (seeds 0-2) generates on the development tiles (val + val_t)
             with three MODEL inputs: the raw DEM, the street-erased DEM (data/processed/m13/terrain_se/f4), and the raw
             DEM erased in the street corridors of ANOTHER tile of the same set (a seeded permutation: same corridor
             widths, shapes and area share, but not where this tile's streets are), with the same erase(). Rules and
             metrics keep the raw DEM. Samplers A and M*, inference seeds 1-2. If terrain fit (S_T, lower = closer to
             the real streets) worsens as much under random strips as under street strips, the TNse drop is lost
             terrain information; if it worsens much more under street strips, the model was reading the street beds.
2. probe   - Leakage probe. A small fully convolutional network is trained on the TRAIN tiles to find carriageway
             pixels (each street edge buffered by its class half-width) from the terrain alone, once per input
             version (raw, low-pass 16 m, street-erased, random-erased, landform only = low-pass 64 m), and scored on
             the development tiles (ROC AUC and average precision; also inside the erased corridors only).
             Street-erased near the landform-only reference = the erasure leaves nothing a network can find.
3. synth   - Synthetic imprint test. Eight natural mountain windows of the IGN 5 m LiDAR DTM (free WCS, the M12
             source) are resampled to the 4 m grid. Synthetic roads (smooth random centrelines, 6-10 m carriageways,
             a road surface smoothed along the centreline, cut and fill banks at 1:1.5) are stamped in, then erased
             with the TNse recipe (corridor 4 x half-width, biharmonic fill, transplanted micro-relief). The result is
             compared with the known natural surface (elevation and slope error in the corridors), with the 16 m
             low-pass (TNlp) as a second recipe; the probe of test 2 also scores the stamped and erased windows.

    python scripts/m18_erasure_validation.py random|probe|synth|report   ->  outputs/m18/erasure/
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from scipy import ndimage

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m18" / "erasure"
TERRAIN = ROOT / "data" / "processed" / "m12" / "terrain"
SE = ROOT / "data" / "processed" / "m13" / "terrain_se" / "f4"
NAT = ROOT / "data" / "raw" / "m18_natural"
WIDTH = np.array([9.75, 10.0, 8.0, 6.0])
N, RES, MARGIN = 256, 4.0, 256.0
PERM_SEED = 18


def seed_of(s: str) -> int:
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def street_mask(raw, se, valid):
    """The erased corridor of a tile, recovered exactly: where the street-erased DEM differs from the raw one."""
    return (np.abs(se.astype(np.float64) - raw.astype(np.float64)) > 1e-3) & valid


def erase(elev, valid, mask, key):
    from m13_street_erase import erase as _erase
    return _erase(elev, valid, mask, np.random.default_rng(seed_of(key)))


def random_inputs(tiles):
    """{sid: elev} erased in the corridors of another tile of the same set (seeded derangement)."""
    sids = [t["sid"] for t in tiles]
    rng = np.random.default_rng(PERM_SEED)
    while True:
        perm = rng.permutation(len(sids))
        if not (perm == np.arange(len(sids))).any():
            break
    masks = {t["sid"]: street_mask(t["d"]["elev"], np.load(SE / f"{t['sid']}.npz")["elev"], t["d"]["valid"]) for t in tiles}
    out, share = {}, {}
    for i, t in enumerate(tiles):
        m = masks[sids[perm[i]]] & t["d"]["valid"]
        out[t["sid"]] = erase(t["d"]["elev"], t["d"]["valid"], m, "random:" + t["sid"])
        share[t["sid"]] = (float(masks[t["sid"]].mean()), float(m.mean()))
    return out, share


# ---------------------------------------------------------------- 1. random-strip control
def stage_random(device="cpu", seeds=(1, 2)):
    from m12_evaluate import score
    from m13_experiments import dev_tiles, m12_rows
    from urbanmorph.experiments.backbones import load_backbone
    from urbanmorph.graphs.tc_potentials import Potentials
    from urbanmorph.graphs.tc_sampler import preset, sample_tc
    from urbanmorph.graphs.terrain_metrics import S_T_KEYS
    tiles = [t for t in dev_tiles() if (SE / f"{t['sid']}.npz").exists()]
    ds = [t["d"] for t in tiles]
    rnd, share = random_inputs(tiles)
    inputs = {"raw": None, "street-erased": [{**d, "elev": np.load(SE / f"{t['sid']}.npz")["elev"]} for t, d in zip(tiles, ds)],
              "random-erased": [{**d, "elev": rnd[t["sid"]]} for t, d in zip(tiles, ds)]}
    plain = lambda m: {k[4:] if k.startswith("m12_") else k: v for k, v in m.items()}   # score() reads bare names
    ref = [{"sid": t["sid"], **plain(m12_rows(t["graph"], t["d"]))} for t in tiles]
    pots = Potentials.load(ROOT / "configs" / "m13_potentials" / "TN.json")
    samplers = {"A": preset("standard"), "M*": preset("energy", K=8, lam_ratio=0.5)}
    f = OUT / "random_runs.json"
    res = json.loads(f.read_text()) if f.exists() else {"tiles": [t["sid"] for t in tiles], "mask_share": share, "runs": {}}
    for bb in ("TN_s0", "TN_s1", "TN_s2"):
        model, names, norm, _ = load_backbone(bb, device)
        for sname, cfg in samplers.items():
            for inp, crop in inputs.items():
                for s in seeds:
                    key = f"{bb}|{sname}|{inp}|{s}"
                    if key in res["runs"]:
                        continue
                    t0 = time.time()
                    gs, _, _ = sample_tc(model, ds, names, norm, pots, cfg, seed=s, device=device, crop_tiles=crop)
                    rows = [{"sid": t["sid"], **plain(m12_rows(g, d))} for t, g, d in zip(tiles, gs, ds)]
                    res["runs"][key] = {"S_T": float(score(rows, ref, S_T_KEYS)),
                                        "S_T_per_key": {k: float(score(rows, ref, [k])) for k in S_T_KEYS},
                                        "km_per_km2": float(np.mean([r["street_km_per_km2"] for r in rows])),
                                        "seconds": round(time.time() - t0, 1)}
                    atomic_json(f, res)
                    print(key, round(res["runs"][key]["S_T"], 3), flush=True)
    return 0


# ---------------------------------------------------------------- 2. leakage probe
def carriageway(g) -> np.ndarray:
    """Carriageway pixels of a tile graph on the 256 x 256 grid (tile interior = rows / cols 64..191)."""
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    E = np.asarray(g.edges)
    if len(E) == 0:
        return np.zeros((N, N), bool)
    P = np.asarray(g.nodes)[:, :2]
    cls = np.clip(np.asarray(g.classes()), 0, 3)
    geoms = [(shapely.LineString(P[e]).buffer(WIDTH[c] / 2), 1) for e, c in zip(E, cls)]
    return rasterize(geoms, out_shape=(N, N), transform=from_origin(-MARGIN, 512 + MARGIN, RES, RES), fill=0,
                     dtype="uint8").astype(bool)


def features(z, valid):
    """Probe input channels: local relief (z minus its 32 m Gaussian, m / 2) and slope (dimensionless x 5)."""
    z = np.where(valid, z, np.nanmean(np.where(valid, z, np.nan))).astype(np.float64)
    rel = (z - ndimage.gaussian_filter(z, 8.0)) / 2.0
    gy, gx = np.gradient(z, RES)
    return np.stack([np.clip(rel, -5, 5), np.clip(np.hypot(gx, gy) * 5.0, 0, 10)]).astype(np.float32)


def variant(name, t, rnd):
    d = t["d"]
    if name == "raw":
        return d["elev"]
    if name == "lowpass16":
        return ndimage.gaussian_filter(np.where(d["valid"], d["elev"], np.nanmean(d["elev"])), 4.0)
    if name == "landform64":
        return ndimage.gaussian_filter(np.where(d["valid"], d["elev"], np.nanmean(d["elev"])), 16.0)
    if name == "street-erased":
        return np.load(SE / f"{t['sid']}.npz")["elev"]
    if name == "random-erased":
        return rnd[t["sid"]]
    raise ValueError(name)


def roc_auc_score(y, p):
    """ROC AUC by the rank (Mann-Whitney) formula, ties averaged."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision_score(y, p):
    """Average precision: the mean of the precision at each positive, scores sorted high to low."""
    y = np.asarray(y, bool)[np.argsort(-np.asarray(p), kind="stable")]
    tp = np.cumsum(y)
    return float(np.mean((tp / np.arange(1, len(y) + 1))[y])) if y.any() else float("nan")


VARIANTS = ("raw", "lowpass16", "street-erased", "random-erased", "landform64")


def probe_net():
    import torch.nn as nn
    layers, ch = [], 2
    for dil in (1, 2, 4, 8, 16, 1):
        layers += [nn.Conv2d(ch, 32, 3, padding=dil, dilation=dil), nn.ReLU()]
        ch = 32
    return nn.Sequential(*layers, nn.Conv2d(32, 1, 1))


def train_tiles():
    d = pickle.loads((ROOT / "data" / "processed" / "m12" / "graphs3d.pkl").read_bytes())
    out = []
    for sid, _, g in d["train"]:                                # the sealed test split is never read
        f, s = TERRAIN / f"{sid}.npz", SE / f"{sid}.npz"
        if f.exists() and s.exists():
            z = dict(np.load(f))
            if z["valid"].all():
                out.append({"sid": sid, "graph": g, "d": z})
    return out


def stage_probe(device="mps", epochs=8):
    import torch
    from m13_experiments import dev_tiles
    tr, ev = train_tiles(), [t for t in dev_tiles() if (SE / f"{t['sid']}.npz").exists()]
    rnd_tr, _ = random_inputs(tr)
    rnd_ev, _ = random_inputs(ev)
    Y_tr = np.stack([carriageway(t["graph"]) for t in tr])
    Y_ev = np.stack([carriageway(t["graph"]) for t in ev])
    C_ev = np.stack([street_mask(t["d"]["elev"], np.load(SE / f"{t['sid']}.npz")["elev"], t["d"]["valid"]) for t in ev])
    inner = np.zeros((N, N), bool)
    inner[64:192, 64:192] = True
    pos = Y_tr[:, inner].mean()
    res, models = {"train_tiles": len(tr), "eval_tiles": len(ev), "street_pixel_share_train": float(pos), "variants": {}}, {}
    for v in VARIANTS:
        torch.manual_seed(0)
        X_tr = np.stack([features(variant(v, t, rnd_tr), t["d"]["valid"]) for t in tr])
        X_ev = np.stack([features(variant(v, t, rnd_ev), t["d"]["valid"]) for t in ev])
        net = probe_net().to(device)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1 - pos) / pos), dtype=torch.float32, device=device), reduction="none")
        W = torch.tensor(inner, device=device, dtype=torch.float32)
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // 8)):
                k = int(rng.integers(4))
                x = torch.tensor(np.rot90(X_tr[b], k, axes=(2, 3)).copy(), device=device)
                y = torch.tensor(np.rot90(Y_tr[b], k, axes=(1, 2)).copy(), device=device, dtype=torch.float32)
                w = torch.rot90(W, k, dims=(0, 1))
                loss = (lossf(net(x)[:, 0], y) * w).sum() / w.sum() / len(b)
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            P = np.concatenate([torch.sigmoid(net(torch.tensor(X_ev[i:i + 8], device=device))[:, 0]).cpu().numpy()
                                for i in range(0, len(ev), 8)])
        y, p, c = Y_ev[:, inner].ravel(), P[:, inner].ravel(), C_ev[:, inner].ravel()
        res["variants"][v] = {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
                              "base_rate": float(y.mean()),
                              "auc_inside_corridor": float(roc_auc_score(y[c], p[c])) if 0 < y[c].mean() < 1 else None,
                              "ap_inside_corridor": float(average_precision_score(y[c], p[c])) if y[c].any() else None,
                              "base_rate_inside_corridor": float(y[c].mean())}
        models[v] = {k: t.cpu() for k, t in net.state_dict().items()}
        print(v, json.dumps(res["variants"][v]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "probe.json", res)
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    torch.save(models, OUT / "cache" / "probe_models.pt")
    return 0


# ---------------------------------------------------------------- 3. synthetic imprint test
SITES = {"gredos": (40.255, -5.235), "guadarrama": (40.835, -3.965), "mariola": (38.725, -0.545),
         "montseny": (41.765, 2.415), "picos": (43.195, -4.835), "nevada": (37.095, -3.385),
         "cazorla": (37.915, -2.905), "moncayo": (41.775, -1.835)}


def natural_window(name, lat, lon):
    """1,024 m x 1,024 m (tile + 256 m margin) of the IGN 5 m DTM around (lat, lon), resampled to the 4 m grid."""
    import rasterio
    from pyproj import Transformer
    from m12_fetch_dem import fetch_spain
    NAT.mkdir(parents=True, exist_ok=True)
    tif = NAT / f"{name}.tif"
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True).transform(lon, lat)
    r = SimpleNamespace(crs="EPSG:25830", xmin=round(x) - 256, ymin=round(y) - 256, xmax=round(x) + 256, ymax=round(y) + 256)
    if not tif.exists():
        meta = fetch_spain(r, tif)
        (NAT / f"{name}.json").write_text(json.dumps({**meta, "lat": lat, "lon": lon}, indent=1))
    with rasterio.open(tif) as src:
        z = src.read(1).astype(np.float64)
        tr = src.transform
    xs = r.xmin - MARGIN + (np.arange(N) + 0.5) * RES
    ys = r.ymax + MARGIN - (np.arange(N) + 0.5) * RES
    cc = (xs - tr.c) / tr.a - 0.5
    rr = (ys - tr.f) / tr.e - 0.5
    R, C = np.meshgrid(rr, cc, indexing="ij")
    return ndimage.map_coordinates(z, [R, C], order=1, mode="nearest").astype(np.float32)


def stamp_roads(z, rng, n_roads=6):
    """Stamp synthetic roads: smooth random centrelines through the window, a road surface smoothed along the
    centreline (80 m running mean), the carriageway set to it, cut / fill banks at 1:1.5 back to the natural ground."""
    from scipy.interpolate import splev, splprep
    zs = z.astype(np.float64).copy()
    lines, widths = [], []
    for _ in range(n_roads):
        k = int(rng.integers(4, 7))
        pts = rng.uniform(-200, 712, size=(k, 2))
        pts = pts[np.argsort(pts[:, 0] if rng.random() < 0.5 else pts[:, 1])]
        tck, _ = splprep(pts.T, s=len(pts) * 400.0, k=min(3, k - 1))
        P = np.array(splev(np.linspace(0, 1, 1200), tck)).T
        seg = np.hypot(*np.diff(P, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        P = np.array([np.interp(np.arange(0, s[-1], 2.0), s, P[:, i]) for i in (0, 1)]).T
        lines.append(P)
        widths.append(float(rng.choice([6.0, 8.0, 10.0])))
    out = zs.copy()
    road = np.zeros(z.shape, bool)
    xs = -MARGIN + (np.arange(N) + 0.5) * RES
    ys = 512 + MARGIN - (np.arange(N) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    from scipy.spatial import cKDTree
    for P, w in zip(lines, widths):
        r = (512 + MARGIN - P[:, 1]) / RES - 0.5
        c = (P[:, 0] + MARGIN) / RES - 0.5
        zc = ndimage.map_coordinates(zs, [r, c], order=1, mode="nearest")
        zr = ndimage.uniform_filter1d(zc, 41, mode="nearest")             # 80 m running mean: the road surface
        d, i = cKDTree(P).query(np.c_[X.ravel(), Y.ravel()])
        d, i = d.reshape(z.shape), i.reshape(z.shape)
        surf = zr[i]
        carr = d <= w / 2
        bank = (d > w / 2) & (d <= w / 2 + 40.0)
        allow = (d - w / 2) / 1.5                                         # |dz| allowed by a 1:1.5 bank
        dz = out - surf
        new = np.where(carr, surf, np.where(bank & (np.abs(dz) > allow), surf + np.sign(dz) * allow, out))
        road |= carr
        out = new
    return out.astype(np.float32), road, lines, widths


def corridor(lines, widths, f=4.0):
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    geoms = [(shapely.LineString(P).buffer(f * w / 2), 1) for P, w in zip(lines, widths)]
    return rasterize(geoms, out_shape=(N, N), transform=from_origin(-MARGIN, 512 + MARGIN, RES, RES), fill=0,
                     dtype="uint8").astype(bool)


def stage_synth(device="mps", reps=4):
    import torch
    models = torch.load(OUT / "cache" / "probe_models.pt") if (OUT / "cache" / "probe_models.pt").exists() else None
    rows = []
    for name, (lat, lon) in SITES.items():
        try:
            nat = natural_window(name, lat, lon)
        except Exception as e:                                             # record the blocker, go on
            rows.append({"site": name, "error": f"{type(e).__name__}: {e}"})
            print("skip", name, e, flush=True)
            continue
        valid = np.isfinite(nat)
        gy, gx = np.gradient(ndimage.gaussian_filter(nat.astype(np.float64), 4.0), RES)
        for k in range(reps):
            rng = np.random.default_rng(seed_of(f"{name}:{k}"))
            st, road, lines, widths = stamp_roads(nat, rng)
            cor = corridor(lines, widths) & valid
            er = erase(st, valid, cor, f"synth:{name}:{k}")
            lp = ndimage.gaussian_filter(st.astype(np.float64), 4.0)
            inner = np.zeros_like(valid)
            inner[64:192, 64:192] = True
            m = cor & inner
            slope = lambda a: np.hypot(*np.gradient(a.astype(np.float64), RES))
            sn = slope(nat)

            def err(a, mask):
                return {"rmse_m": float(np.sqrt(np.mean((a[mask] - nat[mask]) ** 2))),
                        "slope_rmse_pct": float(np.sqrt(np.mean((slope(a)[mask] - sn[mask]) ** 2)) * 100)}
            row = {"site": name, "rep": k, "terrain_slope_med_pct": float(np.median(np.hypot(gx, gy)[inner]) * 100),
                   "corridor_share": float(m.mean() / inner.mean()), "stamped": err(st, m), "erased": err(er, m),
                   "lowpass16": err(lp, m), "lowpass16_off_corridor": err(lp, inner & ~cor)}
            if models is not None:
                from torch import nn  # noqa: F401
                y = road[inner]
                for v, arr in (("stamped", st), ("erased", er)):
                    net = probe_net()
                    net.load_state_dict(models["raw"])
                    net.eval()
                    with torch.no_grad():
                        p = torch.sigmoid(net(torch.tensor(features(arr, valid)[None])))[0, 0].numpy()
                    row[f"probe_auc_{v}"] = float(roc_auc_score(y, p[inner])) if 0 < y.mean() < 1 else None
            rows.append(row)
            print(name, k, json.dumps({k2: row[k2] for k2 in ("stamped", "erased")}), flush=True)
    atomic_json(OUT / "synth.json", {"sites": SITES, "rows": rows})
    return 0


def stage_report():
    """README.md and fig_erasure.png from the three stages' JSON."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rr = json.loads((OUT / "random_runs.json").read_text())
    pr = json.loads((OUT / "probe.json").read_text())
    sy = [r for r in json.loads((OUT / "synth.json").read_text())["rows"] if "error" not in r]
    ink, ink2, grid_c, accent, ref = "#1f2328", "#59636e", "#e6e8eb", "#2a78d6", "#d1242f"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    inputs = ("raw", "street-erased", "random-erased")
    summ = {}
    ax = axs[0]
    for j, smp in enumerate(("A", "M*")):
        for i, inp in enumerate(inputs):
            v = [r["S_T"] for k, r in rr["runs"].items() if k.split("|")[1] == smp and k.split("|")[2] == inp]
            summ[f"{smp}|{inp}"] = {"S_T_mean": float(np.mean(v)), "S_T_sd": float(np.std(v)), "runs": len(v),
                                    "km_per_km2": float(np.mean([r["km_per_km2"] for k, r in rr["runs"].items()
                                                                 if k.split("|")[1] == smp and k.split("|")[2] == inp]))}
            x = i + (j - 0.5) * 0.3
            ax.plot([x] * len(v), v, "o", color=accent if smp == "M*" else ink2, ms=4, alpha=0.6)
            ax.plot([x - 0.1, x + 0.1], [np.mean(v)] * 2, color=ink, lw=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["raw DEM", "street strips\nerased", "random strips\nerased"])
    ax.set_ylabel("S_T (terrain gap to real streets; lower = better)")
    ax.set_title("1. TN model, input erased in street vs random strips\n(grey A, blue M*; bar = mean of 3 backbones × 2 seeds)",
                 loc="left", fontsize=10, fontweight="bold")
    ax = axs[1]
    vs = list(pr["variants"])
    lab = {"raw": "raw", "lowpass16": "blur 16 m\n(TNlp)", "street-erased": "street-\nerased (TNse)", "random-erased": "random-\nerased",
           "landform64": "hills only\n(64 m)"}
    xs = np.arange(len(vs))
    ax.bar(xs - 0.18, [pr["variants"][v]["auc"] for v in vs], 0.36, color=accent, label="whole tile")
    ax.bar(xs + 0.18, [pr["variants"][v]["auc_inside_corridor"] for v in vs], 0.36, color=ink2, label="inside the erased corridors")
    ax.axhline(0.5, color=ref, ls="--", lw=1)
    ax.text(-0.45, 0.505, "chance", color=ref, ha="left", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab[v] for v in vs], fontsize=8.5)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("probe ROC AUC: finding street pixels from terrain")
    ax.set_title(f"2. Leakage probe (trained on {pr['train_tiles']} train tiles,\nscored on {pr['eval_tiles']} development tiles)",
                 loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axs[2]
    sites = sorted({r["site"] for r in sy}, key=lambda s: np.mean([r["terrain_slope_med_pct"] for r in sy if r["site"] == s]))
    for i, s in enumerate(sites):
        R = [r for r in sy if r["site"] == s]
        a, b = np.mean([r["stamped"]["rmse_m"] for r in R]), np.mean([r["erased"]["rmse_m"] for r in R])
        ax.plot([a, b], [i, i], color=grid_c, lw=3, zorder=1)
        ax.plot(a, i, "o", color=ink2, ms=7, zorder=2, label="road stamped" if i == 0 else None)
        ax.plot(b, i, "o", color=accent, ms=7, zorder=3, label="after erasure" if i == 0 else None)
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels([f"{s} ({np.mean([r['terrain_slope_med_pct'] for r in sy if r['site'] == s]):.0f} % slope)" for s in sites])
    pa, pe = np.mean([r["probe_auc_stamped"] for r in sy]), np.mean([r["probe_auc_erased"] for r in sy])
    ax.set_xlabel("elevation error in the corridors (RMSE, m)")
    ax.set_title(f"3. Synthetic roads on natural LiDAR terrain\n(probe AUC: stamped {pa:.2f} → erased {pe:.2f})",
                 loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for a in axs:
        a.grid(color=grid_c, lw=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("Is the street-erased terrain (TNse) a valid leakage control?", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_erasure.png", dpi=130)
    plt.close(fig)
    rec = [1 - r["erased"]["rmse_m"] / r["stamped"]["rmse_m"] for r in sy]
    md = ["# M18: validation of the street-erased DEM (TNse leakage control)", "",
          "Script `scripts/m18_erasure_validation.py` (docstring = method); figure `fig_erasure.png`. Development data only.", "",
          "## 1. Random-strip control (TN backbones s0-s2, inference seeds 1-2, 52 development tiles)", "",
          "| Sampler | Model input | S_T mean (sd) | km/km² |", "|---|---|---|---|"]
    for k, v in summ.items():
        smp, inp = k.split("|")
        md.append(f"| {smp} | {inp} | {v['S_T_mean']:.3f} ({v['S_T_sd']:.3f}) | {v['km_per_km2']:.1f} |")
    md += ["", "Mask area share, own street corridors vs the borrowed ones: "
           f"{np.mean([v[0] for v in rr['mask_share'].values()]):.2f} vs {np.mean([v[1] for v in rr['mask_share'].values()]):.2f}.",
           "", "## 2. Leakage probe", "", "| Input | AUC (whole tile) | AP (base rate) | AUC inside erased corridors | AP inside (base rate) |",
           "|---|---|---|---|---|"]
    for v, r in pr["variants"].items():
        md.append(f"| {v} | {r['auc']:.3f} | {r['ap']:.3f} ({r['base_rate']:.3f}) | {r['auc_inside_corridor']:.3f} | "
                  f"{r['ap_inside_corridor']:.3f} ({r['base_rate_inside_corridor']:.3f}) |")
    md += ["", "## 3. Synthetic imprint test (8 natural IGN windows × 4 road layouts)", "",
           f"* Elevation RMSE in the corridors: stamped {np.mean([r['stamped']['rmse_m'] for r in sy]):.2f} m, erased "
           f"{np.mean([r['erased']['rmse_m'] for r in sy]):.2f} m, low-pass 16 m {np.mean([r['lowpass16']['rmse_m'] for r in sy]):.2f} m; "
           f"recovery (1 − erased / stamped) median {np.median(rec):.2f}, range {np.min(rec):.2f} to {np.max(rec):.2f}.",
           f"* Slope RMSE in the corridors: stamped {np.mean([r['stamped']['slope_rmse_pct'] for r in sy]):.1f} %, erased "
           f"{np.mean([r['erased']['slope_rmse_pct'] for r in sy]):.1f} %; low-pass 16 m off the corridors "
           f"{np.mean([r['lowpass16_off_corridor']['slope_rmse_pct'] for r in sy]):.1f} % (it distorts natural terrain everywhere).",
           f"* Probe trained on raw real tiles, AUC for the synthetic roads: stamped {pa:.3f}, erased {pe:.3f}."]
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    atomic_json(OUT / "summary.json", {"random": summ, "probe": pr["variants"], "synth_recovery_median": float(np.median(rec)),
                                       "synth_probe_auc": {"stamped": float(pa), "erased": float(pe)}})
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["random", "probe", "synth", "report"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "random":
        return stage_random(a.device or "cpu")
    if a.stage == "probe":
        return stage_probe(a.device or "mps")
    if a.stage == "report":
        return stage_report()
    return stage_synth(a.device or "mps")


if __name__ == "__main__":
    sys.exit(main())
