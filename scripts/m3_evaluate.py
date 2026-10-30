"""Milestone 3: evaluate the VAE against the AE and trivial baselines, and make figures.

    python scripts/m3_evaluate.py                       # VALIDATION split (development)
    python scripts/m3_evaluate.py --split test --final  # one-off final test evaluation

Uses checkpoints/m3/<vae_name>/best.pt and checkpoints/m3/<ae_name>/best.pt.
Writes outputs/m3/evaluation_<split>.{json,md} and figures in outputs/m3/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.models import evaluate as ev
from urbanmorph.models import plots
from urbanmorph.models.data import LayoutDataset
from urbanmorph.models.train import CHECKPOINT_DIR, load_checkpoint, pick_device, set_seed

OUT = PROJECT_ROOT / "outputs" / "m3"
METRIC_TARGETS = ["buildings_coverage", "roads_coverage", "green_spaces_coverage",
                  "intersection_density_per_km2", "orientation_order_phi", "building_area_median_m2"]


def latent_report(z_tr, rows_tr, z_ev, rows_ev) -> dict:
    out = {"knn_city_accuracy": ev.knn_accuracy(z_tr, rows_tr.city, z_ev, rows_ev.city),
           "knn_country_accuracy": ev.knn_accuracy(z_tr, rows_tr.country, z_ev, rows_ev.country),
           "chance_city": float(rows_ev.city.value_counts(normalize=True).max()),
           "chance_country": float(rows_ev.country.value_counts(normalize=True).max()),
           "ridge_r2": {}}
    for m in METRIC_TARGETS:
        t_tr, t_ev = rows_tr[m].to_numpy(float), rows_ev[m].to_numpy(float)
        if m == "building_area_median_m2":
            t_tr, t_ev = np.log10(np.nan_to_num(t_tr, nan=np.nanmedian(t_tr))), np.log10(np.nan_to_num(t_ev, nan=np.nanmedian(t_ev)))
        ok_tr, ok_ev = ~np.isnan(t_tr), ~np.isnan(t_ev)
        out["ridge_r2"][m] = ev.ridge_r2(z_tr[ok_tr], t_tr[ok_tr], z_ev[ok_ev], t_ev[ok_ev], lam=10.0)
    return out


def generation_report(masks: np.ndarray, train: np.ndarray) -> dict:
    near = ev.nearest_train_similarity(masks, train)
    cov = masks.mean(axis=(2, 3))
    return {"diversity": ev.diversity(masks), "nearest_train_iou_median": float(np.median(near)),
            "nearest_train_iou_max": float(near.max()), **ev.road_connectivity(masks[:, 0]),
            "coverage_mean": {n: float(cov[:, c].mean()) for c, n in enumerate(ev.CHANNELS)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="required to evaluate on the test split")
    ap.add_argument("--vae", default="vae_z32")
    ap.add_argument("--ae", default="ae_z32")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        ap.error("the test split is reserved for the final evaluation; pass --final deliberately")

    device = pick_device("auto")
    vae, ck_vae = load_checkpoint(CHECKPOINT_DIR / args.vae / "best.pt", device)
    ae, ck_ae = load_checkpoint(CHECKPOINT_DIR / args.ae / "best.pt", device)
    thr = ck_vae["config"]["eval"]["threshold"]
    n_samples = ck_vae["config"]["eval"]["n_samples"]
    set_seed(ck_vae["config"]["experiment"]["seed"])

    train_ds, eval_ds = LayoutDataset("train"), LayoutDataset(args.split)
    x_tr, x_ev = train_ds.x, eval_ds.x
    tgt, train_np = x_ev.numpy(), x_tr.numpy()

    p_vae, z_vae_ev = ev.predict(vae, x_ev, device)
    p_ae, z_ae_ev = ev.predict(ae, x_ev, device)
    _, z_vae_tr = ev.predict(vae, x_tr, device)
    _, z_ae_tr = ev.predict(ae, x_tr, device)

    # random generation: VAE prior N(0, I); AE has no prior, so fit a Gaussian to its training codes
    g = torch.Generator().manual_seed(123)
    s_vae = vae.sample(n_samples, device=device, generator=g).cpu().numpy()
    rng = np.random.default_rng(123)
    z_ae_s = rng.multivariate_normal(z_ae_tr.mean(0), np.cov(z_ae_tr, rowvar=False), size=n_samples)
    s_ae = ev.decode(ae, z_ae_s.astype(np.float32), device)
    # control (no training): the SAME VAE decoder, but z drawn from a Gaussian fitted to its training
    # posterior means instead of the N(0, I) prior. Isolates prior/aggregate-posterior mismatch.
    z_vae_s = rng.multivariate_normal(z_vae_tr.mean(0), np.cov(z_vae_tr, rowvar=False), size=n_samples)
    s_vae_fit = ev.decode(vae, z_vae_s.astype(np.float32), device)

    results = {
        "split": args.split, "threshold": thr, "n_eval_tiles": len(tgt), "n_samples": n_samples,
        "checkpoints": {"vae": {"epoch": ck_vae["epoch"], **ck_vae["extra"]}, "ae": {"epoch": ck_ae["epoch"], **ck_ae["extra"]}},
        "reconstruction": {"VAE": ev.reconstruction_metrics(p_vae, tgt, thr), "AE": ev.reconstruction_metrics(p_ae, tgt, thr),
                           "trivial_mean_coverage": ev.trivial_baseline(train_np, tgt, thr)},
        "structure": {"real": ev.road_connectivity(tgt[:, 0]),
                      "VAE_reconstruction": ev.road_connectivity(p_vae[:, 0] >= thr),
                      "AE_reconstruction": ev.road_connectivity(p_ae[:, 0] >= thr)},
        "generation": {"real_eval_tiles": generation_report(tgt.astype(bool), train_np),
                       "VAE_prior_samples": generation_report(s_vae >= thr, train_np),
                       "VAE_fitted_gaussian_samples": generation_report(s_vae_fit >= thr, train_np),
                       "AE_gaussian_samples": generation_report(s_ae >= thr, train_np)},
        "latent": {"VAE": latent_report(z_vae_tr, train_ds.rows, z_vae_ev, eval_ds.rows),
                   "AE": latent_report(z_ae_tr, train_ds.rows, z_ae_ev, eval_ds.rows)},
    }
    # active latent dimensions: posterior means that actually vary across tiles
    var = z_vae_tr.var(axis=0)
    results["latent"]["VAE"]["active_dims(var_mu>0.01)"] = int((var > 0.01).sum())
    results["latent"]["VAE"]["aggregate_posterior"] = {"std_of_mu_per_dim_median": float(np.median(np.sqrt(var))),
                                                       "abs_mean_of_mu": float(np.abs(z_vae_tr.mean(0)).mean())}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"evaluation_{args.split}.json").write_text(json.dumps(results, indent=2))

    # ---- figures (development split only, so figures never leak test tiles into model choices)
    if args.split == "val":
        rows = eval_ds.rows
        pick = [rows.index[rows.country == c][0] for c in ("spain", "united_states", "japan")]
        pick += [rows.index[rows.country == c][1] for c in ("spain", "united_states", "japan")]
        ids = rows.sample_id[pick].tolist()
        plots.reconstructions(tgt[pick], {"VAE": p_vae[pick], "AE": p_ae[pick]}, ids, thr, OUT / "reconstructions_val.png")
        plots.per_channel(tgt[pick[:2]], p_vae[pick[:2]], ids[:2], thr, OUT / "per_channel_val.png")
        plots.sample_grid(s_vae[:24], thr, f"VAE: 24 random samples z ~ N(0, I), thresholded at p ≥ {thr}", OUT / "samples_vae.png")
        plots.sample_grid(s_vae_fit[:24], thr, f"VAE decoder, z from a Gaussian FITTED to training codes (p ≥ {thr})",
                          OUT / "samples_vae_fitted_gaussian.png")
        plots.sample_grid(s_ae[:24], thr, f"AE baseline: 24 samples from a Gaussian fitted to training codes (p ≥ {thr})",
                          OUT / "samples_ae.png")
        a, b = pick[0], pick[2]  # Spain -> Japan
        ts = np.linspace(0, 1, 8)
        frames = ev.decode(vae, np.stack([(1 - t) * z_vae_ev[a] + t * z_vae_ev[b] for t in ts]).astype(np.float32), device)
        plots.interpolation(frames, rows.sample_id[a], rows.sample_id[b], thr, OUT / "interpolation_vae.png")
        hist = {"VAE": json.loads((OUT / "runs" / args.vae / "history.json").read_text()),
                "AE": json.loads((OUT / "runs" / args.ae / "history.json").read_text())}
        plots.loss_curves(hist, OUT / "loss_curves.png")
        plots.latent_pca(z_vae_tr, train_ds.rows, z_vae_ev, rows, OUT / "latent_pca_vae.png",
                         "VAE latent means (PCA): train dots, validation triangles")

    # ---- compact markdown table
    r = results["reconstruction"]
    lines = [f"# Milestone 3 evaluation ({args.split}, {len(tgt)} tiles, threshold p ≥ {thr})", "",
             "| model | channel | BCE (nats/px) | IoU | coverage abs. error | coverage bias |", "|---|---|---|---|---|---|"]
    for m in ("VAE", "AE", "trivial_mean_coverage"):
        for c in ev.CHANNELS:
            v = r[m][c]
            lines.append(f"| {m} | {c} | {v['bce']:.3f} | {v['iou']:.3f} | {v['coverage_abs_error']:.3f} | {v['coverage_bias']:+.3f} |")
    lines += ["", "| set | diversity | nearest-train IoU (median / max) | road components (median) | largest road component share |",
              "|---|---|---|---|---|"]
    for k, v in results["generation"].items():
        lines.append(f"| {k} | {v['diversity']:.3f} | {v['nearest_train_iou_median']:.3f} / {v['nearest_train_iou_max']:.3f} | "
                     f"{v['components_median']:.0f} | {v['largest_component_share_median']:.2f} |")
    lines += ["", "| latent | kNN city acc. (chance) | kNN country acc. (chance) | " + " | ".join(f"R² {m}" for m in METRIC_TARGETS) + " |",
              "|---|---|---|" + "---|" * len(METRIC_TARGETS)]
    for m in ("VAE", "AE"):
        L = results["latent"][m]
        lines.append(f"| {m} | {L['knn_city_accuracy']:.2f} ({L['chance_city']:.2f}) | {L['knn_country_accuracy']:.2f} "
                     f"({L['chance_country']:.2f}) | " + " | ".join(f"{L['ridge_r2'][t]:.2f}" for t in METRIC_TARGETS) + " |")
    (OUT / f"evaluation_{args.split}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nVAE active latent dims: {results['latent']['VAE']['active_dims(var_mu>0.01)']} / {vae.latent_dim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
