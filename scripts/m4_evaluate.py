"""Milestone 4: compare priors on the SAME frozen C2 decoder with the SAME evaluation.

    python scripts/m4_evaluate.py

Priors (256 samples each, fixed seeds):
  A  N(0, I)                                   - the VAE's own prior
  B  full-covariance Gaussian fitted to training codes z ~ q(z|x) (565 x 8 symmetries), 10 % shrinkage
  C  Gated PixelCNN prior, training seeds 0 and 1
  ref decoded posterior means of the 565 training tiles - what a perfect prior could at best
      produce with this decoder (they ARE the training tiles, so they are not "generation")
Metrics: road topology (pieces, floating fragments, loops, intersections, dead ends, phi, entropy),
coverage of all channels, diversity, memorisation (image and latent nearest neighbours), prior NLL on
validation codes (reported, not used for ranking), country shares and per-country metrics via a
5-NN country classifier on latent codes (accuracy reported). Decision criteria fixed in advance.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from m4_train_prior import CKPT, load_codes  # noqa: E402
from topo_evaluate import TOPO, profiles, wasserstein1  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.models import evaluate as ev
from urbanmorph.models.data import LayoutDataset
from urbanmorph.models.prior import GatedPixelCNNPrior
from urbanmorph.models.train import pick_device
from urbanmorph.visualization.layout_plot import INK, SURFACE, composite_rgb

OUT = PROJECT_ROOT / "outputs" / "m4"
N, THR = 256, 0.5
KEY = ["betti0_10", "interior_fragments", "intersections_per_km2"]


@torch.no_grad()
def decode(vae, z, device, batch=64):
    out = []
    for i in range(0, len(z), batch):
        out.append(torch.sigmoid(vae.decode(z[i:i + batch].to(device))).cpu().numpy())
    return np.concatenate(out) >= THR


def load_prior(seed, D, device):
    ck = torch.load(CKPT / f"prior_pixelcnn_s{seed}" / "prior.pt", map_location=device, weights_only=False)
    pc = ck["config"]["prior"]
    p = GatedPixelCNNPrior(latent_channels=D, hidden=pc["hidden"], n_layers=pc["n_layers"], n_mix=pc["n_mix"]).to(device)
    p.load_state_dict(ck["state"])
    p.eval()
    return p


def knn_predict(train_z, train_y, z, k=5):
    mu, sd = train_z.mean(0), train_z.std(0) + 1e-8
    a, b = (train_z - mu) / sd, (z - mu) / sd
    d = ((b ** 2).sum(1)[:, None] - 2 * b @ a.T + (a ** 2).sum(1)[None])
    nn = np.argsort(d, axis=1)[:, :k]
    y = np.asarray(train_y)
    return np.array([pd.Series(y[r]).mode().iloc[0] for r in nn])


def main() -> int:
    device = pick_device("auto")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, vae, (mu_tr, lv_tr), (mu_va, lv_va) = load_codes(device)
    D, H, W = mu_tr.shape[1:]
    train = LayoutDataset("train", index=load_index_with_expansion())
    val = LayoutDataset("val")
    n_tr = len(train)
    mu_tr_plain = mu_tr[:n_tr]  # symmetry k = 0 comes first (unaugmented tiles)
    g = torch.Generator().manual_seed(1000)
    z_va = mu_va + (0.5 * lv_va).exp() * torch.randn(mu_va.shape, generator=g)
    z_tr_s = mu_tr + (0.5 * lv_tr).exp() * torch.randn(mu_tr.shape, generator=g)

    # ---- priors
    samples_z, nll = {}, {}
    gA = torch.Generator().manual_seed(123)
    samples_z["A: N(0,I)"] = torch.randn(N, D, H, W, generator=gA)
    flat = z_va.flatten(1).double()
    nll["A: N(0,I)"] = float((0.5 * flat ** 2 + 0.5 * np.log(2 * np.pi)).mean())
    Ztr = z_tr_s.flatten(1).double()
    m0 = Ztr.mean(0)
    cov = torch.cov(Ztr.T)
    cov = 0.9 * cov + 0.1 * torch.diag(torch.diag(cov))
    mvn = torch.distributions.MultivariateNormal(m0, covariance_matrix=cov)
    torch.manual_seed(123)
    samples_z["B: fitted Gaussian"] = mvn.sample((N,)).float().view(N, D, H, W)
    nll["B: fitted Gaussian"] = float(-mvn.log_prob(flat).mean() / flat.shape[1])
    for s in (0, 1):
        pr = load_prior(s, D, device)
        name = f"C: PixelCNN (seed {s})"
        samples_z[name] = pr.sample(N, H, W, generator=torch.Generator().manual_seed(123), device=device).cpu()
        with torch.no_grad():
            nll[name] = float(-pr.log_prob(z_va.to(device)).mean().item() / flat.shape[1])

    # ---- references
    real_val = pd.read_csv(PROJECT_ROOT / "outputs" / "topology" / "eval" / "real_val_profiles.csv")
    real_train = pd.read_csv(PROJECT_ROOT / "outputs" / "topology" / "eval" / "real_train_profiles.csv")
    masks = {k: decode(vae, z, device) for k, z in samples_z.items()}
    masks["ref: decoded training codes"] = decode(vae, mu_tr_plain[:N], device)
    train_np = train.x.numpy()
    real_val_masks = val.x.numpy().astype(bool)
    real_near = ev.nearest_train_similarity(real_val_masks, train_np)

    # ---- country classifier on latent codes
    ztr_flat = mu_tr_plain.flatten(1).numpy()
    country_tr = train.rows.country.to_numpy()
    acc = float((knn_predict(ztr_flat, country_tr, mu_va[:len(val)].flatten(1).numpy()) == val.rows.country.to_numpy()).mean())
    train_share = pd.Series(country_tr).value_counts(normalize=True).to_dict()

    results = {"prior_nll_val_nats_per_dim": nll, "country_classifier_val_accuracy": acc,
               "country_classifier_chance": float(val.rows.country.value_counts(normalize=True).max()),
               "train_country_share": train_share, "real_val_nearest_train_iou_median": float(np.median(real_near)),
               "sets": {}}
    lat_tr = ztr_flat
    lat_sd = lat_tr.std(0) + 1e-8
    def latent_nn(z):
        a = lat_tr / lat_sd
        b = z / lat_sd
        d = (b ** 2).sum(1)[:, None] - 2 * b @ a.T + (a ** 2).sum(1)[None]
        return np.sqrt(np.maximum(d.min(1), 0)) / np.sqrt(a.shape[1])
    real_lat_nn = float(np.median(latent_nn(mu_va[:len(val)].flatten(1).numpy())))
    results["real_val_latent_nn_median"] = real_lat_nn

    for name, m in masks.items():
        prof = profiles(m[:, 0])
        cov_ = m.mean(axis=(2, 3))
        entry = {"coverage": {c: float(cov_[:, i].mean()) for i, c in enumerate(("roads", "buildings", "green"))},
                 "diversity": ev.diversity(m[:96]),
                 "nearest_train_iou_median": float(np.median(ev.nearest_train_similarity(m[:128], train_np))),
                 "metrics": {}}
        for k in TOPO:
            v = prof[k].to_numpy(float)
            e = {"median": float(np.nanmedian(v)), "p05": float(np.nanpercentile(v, 5)), "p95": float(np.nanpercentile(v, 95)),
                 "iqr": float(np.nanpercentile(v, 75) - np.nanpercentile(v, 25))}
            for rn, ref in (("val", real_val), ("train", real_train)):
                r = ref[k].to_numpy(float)
                iqr = np.nanpercentile(r, 75) - np.nanpercentile(r, 25)
                e[f"w1_{rn}_iqr"] = wasserstein1(v, r) / (iqr if iqr else 1.0)
            entry["metrics"][k] = e
        if name in samples_z:
            zf = samples_z[name].flatten(1).numpy()
            entry["latent_nn_median"] = float(np.median(latent_nn(zf)))
            pred = knn_predict(ztr_flat, country_tr, zf)
            entry["country_share"] = pd.Series(pred).value_counts(normalize=True).to_dict()
            pc = {}
            for c in ("spain", "united_states", "japan"):
                sel = pred == c
                rt = real_train[real_train.country == c]
                pc[c] = {"n": int(sel.sum()), **{k: [float(prof.loc[sel, k].median()) if sel.any() else None, float(rt[k].median())]
                                                  for k in ("betti0_10", "interior_fragments", "intersections_per_km2",
                                                            "orientation_phi", "road_coverage")}}
            entry["per_country_median_vs_real_train"] = pc
        results["sets"][name] = entry
        prof.to_csv(OUT / f"profiles_{name.split(':')[0].strip()}_{name.split('(')[-1].strip(')').replace(' ', '') if '(' in name else ''}.csv", index=False)

    # ---- decision criteria (fixed before seeing results)
    A = results["sets"]["A: N(0,I)"]["metrics"]
    rv = {k: float(real_val[k].median()) for k in KEY}
    dec = {}
    for s in (0, 1):
        C = results["sets"][f"C: PixelCNN (seed {s})"]
        halfway = {k: abs(C["metrics"][k]["median"] - rv[k]) <= 0.5 * abs(A[k]["median"] - rv[k]) for k in KEY}
        real_div = ev.diversity(real_val_masks)
        dec[s] = {"halfway_towards_real": halfway,
                  "diversity_ok": C["diversity"] >= real_div,
                  "memorisation_ok": C["nearest_train_iou_median"] <= results["real_val_nearest_train_iou_median"] + 0.10,
                  "country_share_ok": all(abs(C["country_share"].get(c, 0) - train_share[c]) <= 0.15 for c in train_share)}
        dec[s]["substantial_improvement"] = all(halfway.values()) and dec[s]["diversity_ok"] and dec[s]["memorisation_ok"] and dec[s]["country_share_ok"]
    results["real_val_diversity"] = ev.diversity(real_val_masks)
    results["decision"] = {str(k): v for k, v in dec.items()}
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))

    # ---- figure: samples per prior
    names = list(masks)
    fig, axes = plt.subplots(len(names) + 1, 6, figsize=(13, 2.15 * (len(names) + 1)), facecolor=SURFACE)
    for j in range(6):
        axes[0, j].imshow(composite_rgb(real_val_masks[j * 3].astype(np.uint8)), interpolation="nearest")
        axes[0, j].set_xticks([]), axes[0, j].set_yticks([])
    axes[0, 0].set_ylabel("real validation", fontsize=8, color=INK)
    for i, n in enumerate(names, start=1):
        for j in range(6):
            axes[i, j].imshow(composite_rgb(masks[n][j].astype(np.uint8)), interpolation="nearest")
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(n, fontsize=7, color=INK)
    fig.suptitle("Milestone 4: samples from each prior through the SAME frozen C2 decoder (p ≥ 0.5)", fontsize=10,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "samples_by_prior.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)

    # ---- figure: metric distributions
    sets = {"real val": real_val, "real train": real_train}
    for n in names:
        sets[n] = profiles(masks[n][:, 0])
    metr = [("betti0_10", "road pieces (≥10 px)"), ("interior_fragments", "floating fragments"),
            ("intersections_per_km2", "intersections / km²"), ("loops_per_km2", "loops / km²"),
            ("orientation_phi", "orientation order φ"), ("road_coverage", "road coverage")]
    colors = ["#6b6b6b", "#a8a8a8", "#2a78d6", "#eb6834", "#1baf7a", "#1baf7a", "#8c8c8c"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), facecolor=SURFACE)
    for ax, (k, lab) in zip(axes.ravel(), metr):
        bp = ax.boxplot([s[k].dropna().to_numpy(float) for s in sets.values()], widths=0.55, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_xticks(range(1, len(sets) + 1))
        ax.set_xticklabels([n.replace(": ", ":\n").replace(" (", "\n(") for n in sets], fontsize=6.5, color=INK)
        ax.set_title(lab, fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", color="#ececec")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("Milestone 4: generated road-network metrics per prior vs real tiles", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "distributions_by_prior.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    summ = {n: {"pieces": e["metrics"]["betti0_10"]["median"], "floating": e["metrics"]["interior_fragments"]["median"],
                "intersections": e["metrics"]["intersections_per_km2"]["median"], "loops": e["metrics"]["loops_per_km2"]["median"],
                "phi": e["metrics"]["orientation_phi"]["median"], "cov_r/b/g": [round(v, 3) for v in e["coverage"].values()],
                "diversity": e["diversity"], "near_train_iou": e["nearest_train_iou_median"],
                "latent_nn": e.get("latent_nn_median"), "country_share": e.get("country_share")} for n, e in results["sets"].items()}
    print(json.dumps({"nll": nll, "knn_acc": acc, "real_val_diversity": results["real_val_diversity"],
                      "real_val_near_train": results["real_val_nearest_train_iou_median"], "real_val_latent_nn": real_lat_nn,
                      "sets": summ, "decision": results["decision"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
