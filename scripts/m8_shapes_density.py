"""Milestone 8 analysis: (1) can a small library of real footprint shapes replace the sampled shapes?
(2) is parcel building coverage spatially autocorrelated (dense parcels next to dense parcels)?

    python scripts/m8_shapes_density.py

(1) Train footprints (20 ray fractions in their minimum rotated rectangle) are clustered with k-means;
each prototype is the MEDOID (a real footprint), so the library contains only observed shapes. A val
footprint is matched to its nearest prototype, which is stretched to the footprint's own length and
width (scale + aspect = the "scaled / skewed" version); reported: IoU with the val footprint's own
ray representation, and the share of prototypes with an angle < 45 deg.
(2) Observed parcels (train + val, Spain + US): neighbours = parcels sharing a boundary. Reported per
city: Moran's I of coverage, correlation of a parcel's coverage with its neighbours' mean, and
P(coverage above city median | neighbours' mean above median). Only uncut parcels as focal parcels.
Writes outputs/m8/analysis/shapes_density.json.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import shapely
import torch

from m6_build_objects import load_objects  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.layout.frontage import interior_angles
from urbanmorph.layout.objects import decode

OUT = PROJECT_ROOT / "outputs" / "m8" / "analysis"


def kmeans(x, k, seed=0, iters=50):
    """Plain k-means with k-means++ initialisation (torch; no extra dependency). Returns (centres, labels)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32)
    c = x[torch.randint(len(x), (1,), generator=g)]
    for _ in range(k - 1):
        d = torch.cdist(x, c).min(1).values ** 2
        c = torch.cat([c, x[torch.multinomial(d / d.sum(), 1, generator=g)]])
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        c = torch.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c.numpy(), torch.cdist(x, c).argmin(1).numpy()


def footprints(tiles):
    r, lw = [], []
    for t in tiles:
        o = t["objects"]
        b = o["type"] == 0
        r.append(o["ratios"][b])
        lw.append(np.c_[o["length"][b], o["width"][b]])
    return np.concatenate(r), np.concatenate(lw)


def shape_library(data, ks=(10, 30, 100, 300), n_eval=3000, seed=0):
    rng = np.random.default_rng(seed)
    tr, _ = footprints(data["train"])
    va, va_lw = footprints(data["val"])
    sel = rng.choice(len(va), min(n_eval, len(va)), replace=False)
    lt = np.log(np.clip(tr, 1e-3, None))
    res = {"train_footprints": int(len(tr)), "val_footprints_evaluated": int(len(sel))}
    for k in ks:
        cen, lab = kmeans(lt, k, seed)
        med = np.array([np.flatnonzero(lab == c)[np.argmin(((lt[lab == c] - cen[c]) ** 2).sum(1))]
                        for c in range(k) if (lab == c).any()])
        proto = tr[med]
        acute = np.mean([interior_angles(decode(np.zeros(2), 0.0, 20.0, 12.0, p)).min() < 45 for p in proto])
        lp = np.log(np.clip(proto, 1e-3, None))
        ious = []
        for i in sel:
            L, W = va_lw[i]
            real = decode(np.zeros(2), 0.0, L, W, va[i])
            j = np.argmin(((lp - np.log(np.clip(va[i], 1e-3, None))) ** 2).sum(1))
            p = decode(np.zeros(2), 0.0, L, W, proto[j])
            u = real.union(p).area
            ious.append(real.intersection(p).area / u if u else 1.0)
        ious = np.array(ious)
        res[f"k{k}"] = {"iou_median": float(np.median(ious)), "iou_p10": float(np.percentile(ious, 10)),
                        "share_iou_ge_0.9": float((ious >= 0.9).mean()), "prototypes_with_acute_angle": float(acute)}
        print(k, res[f"k{k}"], flush=True)
    # baseline: plain rectangle (the footprint's own rotated rectangle)
    ious = [decode(np.zeros(2), 0.0, *va_lw[i], va[i]).area / (va_lw[i][0] * va_lw[i][1]) for i in sel]
    res["rectangle_only"] = {"iou_median": float(np.median(ious)), "share_iou_ge_0.9": float((np.array(ious) >= 0.9).mean())}
    return res


def density_autocorrelation():
    tiles = pickle.loads((PROJECT_ROOT / "data" / "processed" / "m8" / "parcels_observed.pkl").read_bytes())
    tab = pd.read_csv(PROJECT_ROOT / "outputs" / "m8" / "parcels" / "observed_parcel_table.csv")
    focal, nbmean, city = [], [], []
    for sid, d in tab.groupby("sample_id", sort=False):
        polys = tiles[sid]["parcels"]
        assert len(polys) == len(d)
        cov, cut = d.coverage.to_numpy(), d.cut.to_numpy()
        tree = shapely.STRtree(polys)
        for i, p in enumerate(polys):
            if cut[i]:
                continue
            nb = [j for j in tree.query(p.buffer(0.5), predicate="intersects") if j != i]
            if nb:
                focal.append(cov[i])
                nbmean.append(cov[nb].mean())
                city.append(d.city.iloc[0])
    df = pd.DataFrame({"city": city, "cov": focal, "nb": nbmean})
    res = {}
    for c, g in df.groupby("city"):
        x, y = g["cov"].to_numpy(), g["nb"].to_numpy()
        z = x - x.mean()
        med = np.median(x)
        hi_nb = y > np.median(y)
        res[c] = {"focal_parcels": int(len(g)),
                  "coverage_median": float(med),
                  # slope of neighbour mean on focal (Moran's I with row-standardised weights)
                  "morans_I": float((z * (y - x.mean())).sum() / (z ** 2).sum()),
                  "corr_with_neighbour_mean": float(np.corrcoef(x, y)[0, 1]),
                  "P_high_given_neighbours_high": float((x[hi_nb] > med).mean()),
                  "P_high_given_neighbours_low": float((x[~hi_nb] > med).mean())}
        print(c, res[c], flush=True)
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"density": density_autocorrelation(), "shape_library": shape_library(load_objects())}
    (OUT / "shapes_density.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
