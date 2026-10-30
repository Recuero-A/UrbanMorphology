"""Milestone 3b: identical validation-only evaluation for every experiment.

    python scripts/m3b_evaluate.py vae_z32_b0.5 vae_z32_b0.5_exp spatial16_b0.5 ...

For each run (checkpoint in checkpoints/m3/<run>/best.pt) on the 19 ORIGINAL
validation tiles:
  * reconstruction: per-channel BCE, IoU (p >= 0.5), coverage error/bias;
  * road networks (skeleton metrics) of reconstructions vs real tiles;
  * generation, three ways, kept separate:
      prior   z ~ N(0, I)                      (the VAE's actual generative model)
      fitted  z ~ diagonal Gaussian fitted to the training posterior means
              (a DIAGNOSTIC, not a valid way to sample from the trained VAE)
    with coverage, diversity, nearest-training-tile IoU and road-network metrics;
  * prior alignment: spread of posterior means, posterior sigma, KL of the fitted
    aggregate posterior to N(0, I) per dimension, active dimensions;
  * latent probes: kNN city/country accuracy, ridge R^2 of green coverage.
Writes outputs/m3b/eval/<run>.json and figures, plus outputs/m3b/comparison.{csv,md}.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index, load_index_with_expansion
from urbanmorph.metrics import road_raster as rr
from urbanmorph.models import evaluate as ev
from urbanmorph.models import plots
from urbanmorph.models.data import LayoutDataset
from urbanmorph.models.train import CHECKPOINT_DIR, RUNS_DIR, load_checkpoint, pick_device

OUT = PROJECT_ROOT / "outputs" / "m3b"
THR = 0.5
N_SAMPLES = 64


def prior_alignment(z_tr: np.ndarray, logvar_tr: np.ndarray) -> dict:
    m, s = z_tr.mean(0), z_tr.std(0) + 1e-8
    kl_dims = 0.5 * (s ** 2 + m ** 2 - 1 - np.log(s ** 2))  # KL(N(m, s^2) || N(0, 1)) per dimension
    return {"mu_std_per_dim_median": float(np.median(s)), "mu_abs_mean_median": float(np.median(np.abs(m))),
            "posterior_sigma_median": float(np.median(np.exp(0.5 * logvar_tr))),
            "aggregate_kl_per_dim_mean": float(kl_dims.mean()),
            "active_dims": int((z_tr.var(0) > 0.01).sum()), "latent_dim": int(z_tr.shape[1])}


@torch.no_grad()
def encode_all(model, x, device, batch=64):
    model.eval()
    mus, lvs = [], []
    for i in range(0, len(x), batch):
        _, mu, lv = model(x[i:i + batch].to(device))
        mus.append(mu.cpu().numpy())
        lvs.append(lv.cpu().numpy())
    return np.concatenate(mus), np.concatenate(lvs)


def gen_report(probs: np.ndarray, train_np: np.ndarray) -> dict:
    masks = probs >= THR
    cov = masks.mean(axis=(2, 3))
    near = ev.nearest_train_similarity(masks, train_np)
    return {"coverage": {c: float(cov[:, i].mean()) for i, c in enumerate(ev.CHANNELS)},
            "diversity": ev.diversity(masks), "nearest_train_iou_median": float(np.median(near)),
            "nearest_train_iou_max": float(near.max()), "road_network": rr.summarise(masks[:, 0])}


def evaluate_run(name: str, val: LayoutDataset, device) -> dict:
    model, ck = load_checkpoint(CHECKPOINT_DIR / name / "best.pt", device)
    cfg = ck["config"]
    train_index = load_index_with_expansion() if cfg["data"].get("include_expansion_m3b") else load_index()
    train = LayoutDataset("train", index=train_index)
    train_np, tgt = train.x.numpy(), val.x.numpy()

    probs, z_val = ev.predict(model, val.x, device)
    z_tr, lv_tr = encode_all(model, train.x, device)
    g = torch.Generator().manual_seed(123)
    s_prior = model.sample(N_SAMPLES, device=device, generator=g).cpu().numpy()
    rng = np.random.default_rng(123)
    z_fit = rng.normal(z_tr.mean(0), z_tr.std(0), size=(N_SAMPLES, z_tr.shape[1])).astype(np.float32)
    s_fit = ev.decode(model, z_fit, device)

    summary = json.loads((RUNS_DIR / name / "summary.json").read_text())
    res = {
        "run": name,
        "setup": {"train_tiles": len(train), "val_tiles": len(val), "model_type": cfg["model"].get("type", "global"),
                  "latent": (f"{cfg['model']['latent_channels']}x8x8" if cfg["model"].get("type") == "spatial"
                             else str(cfg["model"]["latent_dim"])),
                  "params": summary["n_params"], "beta": cfg["loss"]["beta"], "pos_weight": cfg["loss"]["pos_weight"],
                  "dice_weight": cfg["loss"].get("dice_weight", 0.0), "channel_weights": cfg["loss"].get("channel_weights"),
                  "augment": cfg["data"]["augment"], "best_epoch": ck["epoch"], "epochs_run": summary["epochs_run"],
                  "train_seconds": summary["train_seconds"], "device": summary["device"]},
        "reconstruction": ev.reconstruction_metrics(probs, tgt, THR),
        "road_network": {"real_val": rr.summarise(tgt[:, 0]), "reconstruction": rr.summarise(probs[:, 0] >= THR)},
        "generation": {"prior": gen_report(s_prior, train_np), "fitted_diagnostic": gen_report(s_fit, train_np)},
        "prior_alignment": prior_alignment(z_tr, lv_tr),
        "latent": {"knn_city": ev.knn_accuracy(z_tr, train.rows.city, z_val, val.rows.city),
                   "knn_country": ev.knn_accuracy(z_tr, train.rows.country, z_val, val.rows.country),
                   "r2_green": ev.ridge_r2(z_tr, train.rows.green_spaces_coverage.to_numpy(float), z_val,
                                           val.rows.green_spaces_coverage.to_numpy(float), lam=10.0)},
    }
    d = OUT / "figures" / name
    d.mkdir(parents=True, exist_ok=True)
    rows = val.rows
    pick = [rows.index[rows.country == c][k] for k in (0, 1) for c in ("spain", "united_states", "japan")]
    plots.reconstructions(tgt[pick], {name: probs[pick]}, rows.sample_id[pick].tolist(), THR, d / "reconstructions.png")
    plots.sample_grid(s_prior[:18], THR, f"{name}: prior samples z ~ N(0, I) (p ≥ {THR})", d / "samples_prior.png")
    plots.sample_grid(s_fit[:18], THR, f"{name}: DIAGNOSTIC samples from fitted aggregate posterior (p ≥ {THR})",
                      d / "samples_fitted_diagnostic.png")
    grid = {"recon": probs[pick[:3]], "prior": s_prior[:3], "fitted": s_fit[:3], "target": tgt[pick[:3]]}
    return res, grid


def comparison_grid(grids: dict, out: Path) -> Path:
    import matplotlib.pyplot as plt
    from urbanmorph.visualization.layout_plot import INK, SURFACE, composite_rgb
    names = list(grids)
    first = grids[names[0]]
    fig, axes = plt.subplots(len(names) + 1, 9, figsize=(16, 1.9 * (len(names) + 1) + 0.6), facecolor=SURFACE)
    cols = ["validation reconstruction"] * 3 + ["prior sample N(0, I)"] * 3 + ["fitted-posterior sample (diagnostic)"] * 3
    for j in range(9):
        ax = axes[0, j]
        if j < 3:
            ax.imshow(composite_rgb(first["target"][j].astype(np.uint8)), interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        if j % 3 == 0:
            ax.set_title(cols[j], fontsize=8, loc="left", color=INK)
    axes[0, 0].set_ylabel("real tile", fontsize=8, color=INK)
    for i, n in enumerate(names, start=1):
        g = grids[n]
        imgs = list(g["recon"]) + list(g["prior"]) + list(g["fitted"])
        for j, im in enumerate(imgs):
            axes[i, j].imshow(composite_rgb((im >= THR).astype(np.uint8)), interpolation="nearest")
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(n, fontsize=7, color=INK)
    fig.suptitle(f"Milestone 3b: same validation tiles and same random seeds for every model (p ≥ {THR})",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out


def flat(res: dict) -> dict:
    r = {"run": res["run"], **{k: res["setup"][k] for k in ("train_tiles", "model_type", "latent", "params", "beta",
                                                           "dice_weight", "best_epoch", "train_seconds")}}
    for c in ev.CHANNELS:
        r[f"iou_{c}"] = res["reconstruction"][c]["iou"]
        r[f"bce_{c}"] = res["reconstruction"][c]["bce"]
        r[f"covbias_{c}"] = res["reconstruction"][c]["coverage_bias"]
    rn = res["road_network"]["reconstruction"]
    r.update(rec_road_width_m=rn["mean_road_width_m"], rec_road_inter_km2=rn["intersections_per_km2"],
             rec_road_components=rn["skeleton_components"], rec_road_seg_m=rn["mean_segment_length_m"])
    for mode in ("prior", "fitted_diagnostic"):
        gr = res["generation"][mode]
        r.update({f"{mode}_cov_roads": gr["coverage"]["roads"], f"{mode}_cov_buildings": gr["coverage"]["buildings"],
                  f"{mode}_cov_green": gr["coverage"]["green"], f"{mode}_diversity": gr["diversity"],
                  f"{mode}_nearest_train_iou": gr["nearest_train_iou_median"],
                  f"{mode}_road_width_m": gr["road_network"]["mean_road_width_m"],
                  f"{mode}_road_inter_km2": gr["road_network"]["intersections_per_km2"]})
    pa = res["prior_alignment"]
    r.update(mu_std=pa["mu_std_per_dim_median"], post_sigma=pa["posterior_sigma_median"],
             agg_kl_per_dim=pa["aggregate_kl_per_dim_mean"], active=f"{pa['active_dims']}/{pa['latent_dim']}",
             knn_city=res["latent"]["knn_city"], r2_green=res["latent"]["r2_green"])
    return r


def main() -> int:
    runs = sys.argv[1:]
    if not runs:
        print(__doc__)
        return 1
    device = pick_device("auto")
    val = LayoutDataset("val")
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    table, grids = [], {}
    for name in runs:
        print(f"evaluating {name}", flush=True)
        res, grids[name] = evaluate_run(name, val, device)
        (OUT / "eval" / f"{name}.json").write_text(json.dumps(res, indent=2))
        table.append(flat(res))
    real = rr.summarise(val.x.numpy()[:, 0].astype(bool))
    real_cov = val.x.numpy().mean(axis=(0, 2, 3))
    comparison_grid(grids, OUT / "comparison_grid.png")
    df = pd.DataFrame(table)
    df.to_csv(OUT / "comparison.csv", index=False)
    print(df.round(3).T.to_string())
    print(f"\nreal validation: coverage roads/bldg/green {real_cov.round(3).tolist()}, road width {real['mean_road_width_m']:.1f} m, "
          f"raster intersections {real['intersections_per_km2']:.0f}/km2, components {real['skeleton_components']:.0f}, "
          f"segment {real['mean_segment_length_m']:.0f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
