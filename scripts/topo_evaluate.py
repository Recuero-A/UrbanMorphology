"""Road-topology experiment: identical evaluation of C2 (lambda = 0) and C2 + lambda * soft-clDice.

    python scripts/topo_evaluate.py C2_spatial16_roadx3 C2_spatial16_roadx3_seed1 T_cl0.1_s0 T_cl0.3_s0 T_cl1_s0 T_cl3_s0

Three separate blocks per run (results cached in outputs/topology/eval/<run>.json):
  1. RECONSTRUCTION, paired (19 validation tiles): road IoU, hard clDice, absolute errors of
     road pieces (>= 10 px), interior fragments, loops/km2, intersections/km2, dead-end ratio,
     orientation phi, road-coverage bias, and building/green IoU (constraint check).
  2. STRUCTURAL BIAS per country (reconstructions of the 565 TRAINING tiles + 19 validation
     tiles, labelled separately): mean paired shift (prediction - truth) of phi, orientation
     entropy, intersections, loops, dead-end ratio, interior fragments; spread ratio
     std(pred)/std(true); share of variance between countries (eta^2) for prediction vs truth.
  3. GENERATION, distributional (256 prior samples z ~ N(0, I), fixed seed): the same per-tile
     metrics compared with real validation (19) and real training (565) tiles: medians,
     5-95 % ranges, 1-D Wasserstein distances (in units of the real tiles' IQR), diversity
     (mean pairwise layout distance) and spread (IQR) of intersections, loops and phi.
     Unconditional samples have no country, so country effects are measured in block 2 only.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.metrics.road_raster import full_road_profile, hard_cldice
from urbanmorph.models import evaluate as ev
from urbanmorph.models.data import LayoutDataset
from urbanmorph.models.train import CHECKPOINT_DIR, RUNS_DIR, load_checkpoint, pick_device

OUT = PROJECT_ROOT / "outputs" / "topology"
THR = 0.5
N_GEN = 256
TOPO = ["road_coverage", "betti0_10", "interior_fragments", "loops_per_km2", "intersections_per_km2",
        "dead_end_ratio", "mean_segment_length_m", "mean_road_width_m", "orientation_entropy", "orientation_phi",
        "grid_strength_r4"]
BIAS = ["orientation_phi", "orientation_entropy", "intersections_per_km2", "loops_per_km2", "dead_end_ratio",
        "interior_fragments"]


def profiles(road_masks: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([full_road_profile(m) for m in road_masks])


def real_profiles(ds: LayoutDataset) -> pd.DataFrame:
    p = profiles(ds.x.numpy()[:, 0].astype(bool))
    p["country"] = ds.rows.country.values
    p["sample_id"] = ds.rows.sample_id.values
    return p


def wasserstein1(a, b) -> float:
    a, b = np.sort(np.asarray(a, float)[np.isfinite(a)]), np.sort(np.asarray(b, float)[np.isfinite(b)])
    if not len(a) or not len(b):
        return float("nan")
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def eta2(values: pd.Series, groups: pd.Series) -> float:
    """Share of variance explained by country (between / total)."""
    v = values.astype(float)
    ok = v.notna()
    v, g = v[ok], groups[ok]
    tot = ((v - v.mean()) ** 2).sum()
    between = sum(len(x) * (x.mean() - v.mean()) ** 2 for _, x in v.groupby(g))
    return float(between / tot) if tot else float("nan")


@torch.no_grad()
def reconstruct(model, x, device):
    probs, _ = ev.predict(model, x, device)
    return probs


def evaluate(name: str, device, refs: dict) -> dict:
    model, ck = load_checkpoint(CHECKPOINT_DIR / name / "best.pt", device)
    val, train = refs["val_ds"], refs["train_ds"]
    summary = json.loads((RUNS_DIR / name / "summary.json").read_text())

    # 1. paired reconstruction (validation)
    pv = reconstruct(model, val.x, device)
    mv = pv >= THR
    rec_metrics = ev.reconstruction_metrics(pv, val.x.numpy(), THR)
    pred_v = profiles(mv[:, 0])
    true_v = refs["val"]
    paired = {"road_iou": rec_metrics["roads"]["iou"], "building_iou": rec_metrics["buildings"]["iou"],
              "green_iou": rec_metrics["green"]["iou"], "road_coverage_bias": rec_metrics["roads"]["coverage_bias"],
              "hard_cldice": float(np.mean([hard_cldice(p, t) for p, t in zip(mv[:, 0], val.x.numpy()[:, 0])]))}
    for k in ("betti0_10", "interior_fragments", "loops_per_km2", "intersections_per_km2", "dead_end_ratio", "orientation_phi"):
        paired[f"abs_err_{k}"] = float(np.nanmean(np.abs(pred_v[k].to_numpy(float) - true_v[k].to_numpy(float))))

    # 2. per-country bias on reconstructions (train 565 + val 19, kept separate)
    bias = {}
    for label, ds, truth in (("train", train, refs["train"]), ("val", val, true_v)):
        pr = profiles(reconstruct(model, ds.x, device)[:, 0] >= THR)
        pr["country"] = truth["country"].values
        rows = {}
        for c, g in pr.groupby("country"):
            t = truth[truth.country == c]
            rows[c] = {"n": int(len(g)),
                       **{f"shift_{k}": float(np.nanmean(g[k].to_numpy(float) - t[k].to_numpy(float))) for k in BIAS},
                       **{f"spread_ratio_{k}": float(np.nanstd(g[k].to_numpy(float)) / (np.nanstd(t[k].to_numpy(float)) + 1e-9))
                          for k in ("intersections_per_km2", "loops_per_km2", "orientation_phi")}}
        rows["eta2_pred_vs_true"] = {k: [eta2(pr[k], pr.country), eta2(truth[k], truth.country)] for k in BIAS}
        bias[label] = rows

    # 3. generation (prior samples), distributional
    g = torch.Generator().manual_seed(123)
    s = model.sample(N_GEN, device=device, generator=g).cpu().numpy() >= THR
    gen = profiles(s[:, 0])
    cov = s.mean(axis=(2, 3))
    gen_report = {"coverage": {"roads": float(cov[:, 0].mean()), "buildings": float(cov[:, 1].mean()), "green": float(cov[:, 2].mean())},
                  "diversity_all_channels": ev.diversity(s[:96])}
    for k in TOPO:
        gk = gen[k].to_numpy(float)
        entry = {"median": float(np.nanmedian(gk)), "p05": float(np.nanpercentile(gk, 5)), "p95": float(np.nanpercentile(gk, 95)),
                 "iqr": float(np.nanpercentile(gk, 75) - np.nanpercentile(gk, 25))}
        for ref in ("val", "train"):
            rk = refs[ref][k].to_numpy(float)
            iqr = np.nanpercentile(rk, 75) - np.nanpercentile(rk, 25)
            entry[f"w1_vs_{ref}_iqr_units"] = wasserstein1(gk, rk) / (iqr if iqr else 1.0)
        gen_report[k] = entry

    res = {"run": name, "cldice_weight": ck["config"]["loss"].get("cldice_weight", 0.0), "seed": ck["config"]["experiment"]["seed"],
           "best_epoch": ck["epoch"], "train_seconds": summary["train_seconds"], "params": summary["n_params"],
           "reconstruction_val": paired, "bias": bias, "generation": gen_report}
    np.save(OUT / "eval" / f"{name}_samples.npy", np.packbits(s[:48], axis=-1))  # small cache for figures
    np.save(OUT / "eval" / f"{name}_recon_val.npy", np.packbits(mv, axis=-1))
    gen.to_csv(OUT / "eval" / f"{name}_gen_profiles.csv", index=False)
    return res


def main() -> int:
    runs = sys.argv[1:]
    device = pick_device("auto")
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    val = LayoutDataset("val")
    train = LayoutDataset("train", index=load_index_with_expansion())
    refs = {"val_ds": val, "train_ds": train, "val": real_profiles(val), "train": real_profiles(train)}
    refs["val"].to_csv(OUT / "eval" / "real_val_profiles.csv", index=False)
    refs["train"].to_csv(OUT / "eval" / "real_train_profiles.csv", index=False)
    for name in runs:
        print(f"evaluating {name}", flush=True)
        res = evaluate(name, device, refs)
        (OUT / "eval" / f"{name}.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
