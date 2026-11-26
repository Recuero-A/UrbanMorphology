"""Road-topology experiment, part 7: does the factorised N(0, I) prior cause discontinuities between latent cells?

    python scripts/topo_prior_diagnosis.py C2_spatial16_roadx3

1. Neighbour correlation of posterior means: for real training tiles (565, unaugmented), the
   Pearson correlation between mu at latent cell (i, j) and at (i+di, j+dj), averaged over the 16
   channels, for offsets 1-4. The prior N(0, I) assumes 0 at every offset.
2. Road breaks at latent-cell boundaries: each latent cell decodes a 16 x 16 px (64 m) patch.
   On every internal boundary line between patches, count road pixels whose neighbour across the
   line (within +-1 px) is background, divided by road pixels on the line ("break rate"). The same
   statistic on CONTROL lines through the middle of the patches gives the baseline.
   Boundary / control ratio ~ 1 means no seam; > 1 means roads break at patch seams.
   Also: road end points within +-1 px of a boundary vs the share expected by area.
3. Diagnostic sampler (NOT a VAE sample): a Gaussian with FULL covariance over all 1,024 latent
   elements, fitted to posterior means of the augmented training set (565 x 8 symmetries) with 10 %
   shrinkage to the diagonal. It captures linear correlations between neighbouring cells. If its
   samples are more continuous, a learned spatial prior is warranted.
Writes outputs/topology/prior_diagnosis_<run>.json and figures.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.metrics.road_raster import full_road_profile, neighbour_count, prune_spurs, zhang_suen_thin
from urbanmorph.models import evaluate as ev
from urbanmorph.models.data import LayoutDataset, dihedral
from urbanmorph.models.train import CHECKPOINT_DIR, load_checkpoint, pick_device
from urbanmorph.visualization.layout_plot import INK, INK_MUTED, SURFACE, composite_rgb

OUT = PROJECT_ROOT / "outputs" / "topology"
THR, N, CELL = 0.5, 256, 16
BOUNDS = [CELL * k for k in range(1, 8)]          # first pixel of patch k (seam between 16k-1 and 16k)
CONTROLS = [CELL * k + CELL // 2 for k in range(0, 8)][:7]


@torch.no_grad()
def encode_mu(model, x, device, batch=64) -> np.ndarray:
    out = []
    for i in range(0, len(x), batch):
        mu, _ = model.encode(x[i:i + batch].to(device))
        out.append(mu.cpu().numpy())
    return np.concatenate(out)  # (N, C, 8, 8)


def neighbour_correlation(mu: np.ndarray, max_off: int = 4) -> dict:
    out = {}
    for off in range(1, max_off + 1):
        cs = []
        for c in range(mu.shape[1]):
            a = np.concatenate([mu[:, c, :, :-off].ravel(), mu[:, c, :-off, :].ravel()])
            b = np.concatenate([mu[:, c, :, off:].ravel(), mu[:, c, off:, :].ravel()])
            if a.std() > 1e-6 and b.std() > 1e-6:
                cs.append(np.corrcoef(a, b)[0, 1])
        out[off] = float(np.mean(cs))
    return out


def seam_stats(roads: np.ndarray) -> dict:
    def break_rate(lines):
        broken = total = 0
        for m in roads:
            for x in lines:  # vertical line between columns x-1 and x, and horizontal between rows x-1 and x
                for a, b in ((m[:, x - 1], m[:, x]), (m[x - 1, :], m[x, :])):
                    for s, t in ((a, b), (b, a)):
                        idx = np.flatnonzero(s)
                        total += len(idx)
                        tt = np.pad(t, 1)
                        broken += int(sum(not (tt[i] or tt[i + 1] or tt[i + 2]) for i in idx))
        return broken / total if total else float("nan")

    ends_obs = ends_tot = 0
    band = np.zeros((128, 128), bool)
    for x in BOUNDS:
        band[:, x - 2:x + 2] = True
        band[x - 2:x + 2, :] = True
    interior = np.zeros((128, 128), bool)
    interior[3:-3, 3:-3] = True
    for m in roads:
        sk = prune_spurs(zhang_suen_thin(m), 3)
        e = (neighbour_count(sk) == 1) & interior
        ends_tot += int(e.sum())
        ends_obs += int((e & band).sum())
    expected = (band & interior).sum() / interior.sum()
    b, c = break_rate(BOUNDS), break_rate(CONTROLS)
    return {"break_rate_boundary": b, "break_rate_control": c, "break_ratio": b / c if c else float("nan"),
            "endpoint_share_at_boundaries": ends_obs / ends_tot if ends_tot else float("nan"),
            "endpoint_share_expected": float(expected),
            "endpoint_enrichment": (ends_obs / ends_tot) / expected if ends_tot else float("nan")}


def topo_summary(roads: np.ndarray) -> dict:
    p = pd.DataFrame([full_road_profile(m) for m in roads])
    return {k: float(p[k].median()) for k in ("betti0_10", "interior_fragments", "intersections_per_km2",
                                               "loops_per_km2", "orientation_phi", "road_coverage")}


def grid_figure(sets: dict, out: Path, n: int = 4) -> None:
    fig, axes = plt.subplots(len(sets), n, figsize=(2.4 * n, 2.5 * len(sets) + 0.4), facecolor=SURFACE)
    for i, (name, masks) in enumerate(sets.items()):
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(composite_rgb(masks[j].astype(np.uint8)), interpolation="nearest")
            for x in BOUNDS:
                ax.axvline(x - 0.5, color="#d0021b", lw=0.4, alpha=0.6)
                ax.axhline(x - 0.5, color="#d0021b", lw=0.4, alpha=0.6)
            ax.set_xticks([]), ax.set_yticks([])
        axes[i, 0].set_ylabel(name, fontsize=8, color=INK)
    fig.suptitle("Latent-cell boundaries (red, every 64 m) overlaid on layouts", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "C2_spatial16_roadx3"
    device = pick_device("auto")
    model, _ = load_checkpoint(CHECKPOINT_DIR / name / "best.pt", device)
    train = LayoutDataset("train", index=load_index_with_expansion())
    val = LayoutDataset("val")

    mu = encode_mu(model, train.x, device)
    corr = neighbour_correlation(mu)

    # full-covariance Gaussian over the 1,024 latent elements (augmented training codes)
    mu_aug = np.concatenate([encode_mu(model, dihedral(train.x, k), device) for k in range(8)]).reshape(-1, mu[0].size)
    m0 = mu_aug.mean(0)
    cov = np.cov(mu_aug, rowvar=False)
    cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov))
    rng = np.random.default_rng(123)
    z_full = rng.multivariate_normal(m0, cov, size=N, method="cholesky").astype(np.float32)
    s_full = ev.decode(model, z_full, device) >= THR
    g = torch.Generator().manual_seed(123)
    s_prior = model.sample(N, device=device, generator=g).cpu().numpy() >= THR
    rec = ev.predict(model, val.x, device)[0] >= THR
    real = val.x.numpy().astype(bool)

    sets = {"real validation": real, "reconstruction (val)": rec, "prior N(0,I)": s_prior,
            "full-cov Gaussian (diagnostic)": s_full}
    res = {"run": name, "neighbour_correlation_of_mu": corr, "prior_assumed_correlation": 0.0,
           "seams": {k: seam_stats(v[:, 0]) for k, v in sets.items()},
           "topology_medians": {k: topo_summary(v[:min(len(v), 128), 0]) for k, v in sets.items()},
           "coverage": {k: [float(v[:, c].mean()) for c in range(3)] for k, v in sets.items()}}
    (OUT / f"prior_diagnosis_{name}.json").write_text(json.dumps(res, indent=2))
    grid_figure({k: v for k, v in sets.items()}, OUT / f"prior_seams_{name}.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor=SURFACE)
    offs = list(corr)
    ax.plot(offs, [corr[o] for o in offs], marker="o", color="#2a78d6", label="real tiles (posterior means)")
    ax.plot(offs, [0] * len(offs), ls="--", color=INK_MUTED, label="assumed by the N(0, I) prior")
    ax.set_xlabel("distance between latent cells (cells of 64 m)", fontsize=8, color=INK)
    ax.set_ylabel("mean correlation of μ", fontsize=8, color=INK)
    ax.set_title(f"Spatial correlation in the latent grid ({name})", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(color="#ececec")
    fig.tight_layout()
    fig.savefig(OUT / f"latent_neighbour_correlation_{name}.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(json.dumps({k: res[k] for k in ("neighbour_correlation_of_mu", "seams", "topology_medians")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
