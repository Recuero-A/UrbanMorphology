"""Milestone 3 figures (the calling script chooses the Matplotlib backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from urbanmorph.visualization.compare import COUNTRY_COLORS, COUNTRY_LABELS, label
from urbanmorph.visualization.layout_plot import INK, INK_MUTED, LAYER_COLORS, LAYER_LABELS, SURFACE, composite_rgb

CH_NAMES = ("Roads", "Buildings", "Green")


def _legend(fig, y=0.005):
    handles = [Patch(color=LAYER_COLORS[k], label=LAYER_LABELS[k]) for k in ("roads", "buildings", "green_spaces")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, y))


def _show(ax, mask, title=None, fs=7):
    ax.imshow(composite_rgb(mask.astype(np.uint8)), interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d0d0d0")
    if title:
        ax.set_title(title, fontsize=fs, color=INK)


def loss_curves(histories: dict[str, list], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    styles = {"VAE": "#2a78d6", "AE": "#eb6834"}
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        c = styles.get(name, INK)
        axes[0].plot(ep, [r["train_recon"] for r in h], color=c, lw=1.2, alpha=0.55, label=f"{name} train")
        axes[0].plot(ep, [r["val_recon"] for r in h], color=c, lw=2, label=f"{name} validation")
        if name == "VAE":
            axes[1].plot(ep, [r["train_kl"] for r in h], color=c, lw=1.2, alpha=0.55, label="train KL")
            axes[1].plot(ep, [r["val_kl"] for r in h], color=c, lw=2, label="validation KL")
            ax2 = axes[1].twinx()
            ax2.plot(ep, [r["beta"] for r in h], color=INK_MUTED, lw=1, ls=":")
            ax2.set_ylabel("β (warm-up, dotted)", fontsize=8, color=INK_MUTED)
            ax2.tick_params(labelsize=7, colors=INK_MUTED)
    axes[0].set_yscale("log")
    axes[0].set_title("Reconstruction loss (weighted BCE, summed per tile)", fontsize=10, loc="left", color=INK)
    axes[1].set_title("VAE KL divergence (nats per tile)", fontsize=10, loc="left", color=INK)
    for ax in axes:
        ax.set_xlabel("epoch", fontsize=8, color=INK_MUTED)
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        ax.grid(color="#ececec")
        ax.legend(fontsize=7, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out


def reconstructions(target, recon_by_model: dict[str, np.ndarray], ids, threshold, out: Path) -> Path:
    n = len(target)
    cols = 1 + len(recon_by_model)
    fig, axes = plt.subplots(n, cols, figsize=(2.3 * cols, 2.3 * n + 0.6), facecolor=SURFACE)
    for i in range(n):
        _show(axes[i, 0], target[i], f"{ids[i]}\noriginal" if i == 0 else ids[i])
        for j, (name, probs) in enumerate(recon_by_model.items(), start=1):
            _show(axes[i, j], probs[i] >= threshold, f"{name} (p ≥ {threshold})" if i == 0 else None)
    fig.suptitle("Validation tiles: original vs reconstruction (posterior mean)", fontsize=10, color=INK, x=0.02, ha="left")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


def per_channel(target, probs, ids, threshold, out: Path) -> Path:
    n = len(target)
    fig, axes = plt.subplots(n * 3, 3, figsize=(7, 2.3 * 3 * n), facecolor=SURFACE)
    r = 0
    for i in range(n):
        for c in range(3):
            row = axes[r]
            row[0].imshow(target[i, c], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            im = row[1].imshow(probs[i, c], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            row[2].imshow(probs[i, c] >= threshold, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            row[0].set_ylabel(f"{ids[i]}\n{CH_NAMES[c]}", fontsize=7, color=INK)
            for ax in row:
                ax.set_xticks([]), ax.set_yticks([])
            r += 1
    for ax, t in zip(axes[0], ("original", "VAE probability", f"VAE mask (p ≥ {threshold})")):
        ax.set_title(t, fontsize=8, color=INK)
    fig.colorbar(im, ax=axes[:, 1], fraction=0.02, pad=0.01).ax.tick_params(labelsize=6)
    fig.savefig(out, dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def sample_grid(probs, threshold, title, out: Path, ncols=6) -> Path:
    n = len(probs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows + 0.7), facecolor=SURFACE)
    for k, ax in enumerate(np.ravel(axes)):
        if k < n:
            _show(ax, probs[k] >= threshold)
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=10, color=INK, x=0.02, ha="left")
    _legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out


def interpolation(frames, a_id, b_id, threshold, out: Path) -> Path:
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 2.6), facecolor=SURFACE)
    for k, ax in enumerate(axes):
        t = k / (n - 1)
        _show(ax, frames[k] >= threshold, f"t = {t:.2f}")
    axes[0].set_title(f"{a_id}\nt = 0 (decoded)", fontsize=7, color=INK)
    axes[-1].set_title(f"{b_id}\nt = 1 (decoded)", fontsize=7, color=INK)
    fig.suptitle("Linear interpolation between two validation tiles in latent space (z = (1-t)·μ_A + t·μ_B)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


def latent_pca(z_train, rows_train, z_val, rows_val, out: Path, title: str) -> Path:
    mu = z_train.mean(0)
    _, _, vt = np.linalg.svd(z_train - mu, full_matrices=False)
    pt, pv = (z_train - mu) @ vt[:2].T, (z_val - mu) @ vt[:2].T
    var = np.var((z_train - mu) @ vt.T, axis=0)
    fig, ax = plt.subplots(figsize=(6.6, 5.6), facecolor=SURFACE)
    for c, color in COUNTRY_COLORS.items():
        m = rows_train["country"].to_numpy() == c
        ax.scatter(pt[m, 0], pt[m, 1], s=22, color=color, alpha=0.75, edgecolor=SURFACE, lw=0.5)
        m = rows_val["country"].to_numpy() == c
        ax.scatter(pv[m, 0], pv[m, 1], s=50, marker="^", facecolor="none", edgecolor=color, lw=1.4)
    handles = [Line2D([], [], marker="o", ls="", color=COUNTRY_COLORS[c], label=COUNTRY_LABELS[c]) for c in COUNTRY_COLORS]
    handles += [Line2D([], [], marker="o", ls="", color=INK_MUTED, label="train"),
                Line2D([], [], marker="^", ls="", markerfacecolor="none", markeredgecolor=INK_MUTED, label="validation")]
    ax.legend(handles=handles, fontsize=8, frameon=False)
    ax.set_xlabel(f"PC1 ({var[0] / var.sum():.0%} of latent variance)", fontsize=8, color=INK)
    ax.set_ylabel(f"PC2 ({var[1] / var.sum():.0%})", fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(color="#ececec")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out
