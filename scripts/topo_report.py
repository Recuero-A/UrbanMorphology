"""Road-topology experiment: aggregate evaluations into tables and figures (outputs/topology/).

Groups (two seeds each where available):
  A = C2 baseline (lambda 0)          C2_spatial16_roadx3, C2_spatial16_roadx3_seed1
  B = weak clDice (lambda 0.1)        T_cl0.1_s0, T_cl0.1_s1
  C = strong clDice (lambda 1)        T_cl1_s0, T_cl1_s1
  pilot only: lambda 0.3, lambda 3 (degenerate failure case)
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

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.visualization.layout_plot import INK, INK_MUTED, SURFACE, composite_rgb

OUT = PROJECT_ROOT / "outputs" / "topology"
E = OUT / "eval"
GROUPS = {
    "A: C2 (λ=0)": ["C2_spatial16_roadx3", "C2_spatial16_roadx3_seed1"],
    "B: weak (λ=0.1)": ["T_cl0.1_s0", "T_cl0.1_s1"],
    "C: strong (λ=1)": ["T_cl1_s0", "T_cl1_s1"],
    "pilot λ=0.3": ["T_cl0.3_s0"],
    "pilot λ=3 (degenerate)": ["T_cl3_s0"],
}
COLORS = {"real val": "#6b6b6b", "real train": "#a8a8a8", "A: C2 (λ=0)": "#2a78d6", "B: weak (λ=0.1)": "#eb6834",
          "C: strong (λ=1)": "#1baf7a"}
GEN_METRICS = [("betti0_10", "road pieces (≥10 px)"), ("interior_fragments", "interior fragments"),
               ("intersections_per_km2", "intersections / km²"), ("loops_per_km2", "loops / km²"),
               ("dead_end_ratio", "dead-end ratio"), ("orientation_phi", "orientation order φ")]


def load(n):
    return json.loads((E / f"{n}.json").read_text())


def fmt(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "—"
    return f"{np.mean(vals):.3f}" + (f" (±{(max(vals) - min(vals)) / 2:.3f})" if len(vals) > 1 else "")


def recon_table() -> str:
    keys = ["road_iou", "hard_cldice", "abs_err_betti0_10", "abs_err_interior_fragments", "abs_err_loops_per_km2",
            "abs_err_intersections_per_km2", "abs_err_dead_end_ratio", "abs_err_orientation_phi", "road_coverage_bias",
            "building_iou", "green_iou"]
    lines = ["| group | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for g, runs in GROUPS.items():
        rs = [load(r)["reconstruction_val"] for r in runs if (E / f"{r}.json").exists()]
        lines.append(f"| {g} | " + " | ".join(fmt([r[k] for r in rs]) for k in keys) + " |")
    return "\n".join(lines)


def gen_table(real_val, real_train) -> str:
    keys = [k for k, _ in GEN_METRICS] + ["road_coverage", "orientation_entropy"]
    lines = ["| set | " + " | ".join(f"{k} median [p5–p95] / W1 vs train (IQR units)" for k in keys) + " | diversity |",
             "|---|" + "---|" * (len(keys) + 1)]
    for name, ref in (("real validation (19)", real_val), ("real training (565)", real_train)):
        cells = [f"{ref[k].median():.2f} [{ref[k].quantile(.05):.2f}–{ref[k].quantile(.95):.2f}]" for k in keys]
        lines.append(f"| {name} | " + " | ".join(cells) + " | — |")
    for g, runs in list(GROUPS.items()):
        rs = [r for r in runs if (E / f"{r}.json").exists()]
        prof = pd.concat([pd.read_csv(E / f"{r}_gen_profiles.csv") for r in rs])
        cells = []
        for k in keys:
            w1 = np.mean([load(r)["generation"][k]["w1_vs_train_iqr_units"] for r in rs])
            cells.append(f"{prof[k].median():.2f} [{prof[k].quantile(.05):.2f}–{prof[k].quantile(.95):.2f}] / {w1:.2f}")
        div = fmt([load(r)["generation"]["diversity_all_channels"] for r in rs])
        lines.append(f"| {g} | " + " | ".join(cells) + f" | {div} |")
    return "\n".join(lines)


def bias_table(split: str) -> str:
    keys = ["shift_orientation_phi", "shift_orientation_entropy", "shift_intersections_per_km2", "shift_loops_per_km2",
            "shift_dead_end_ratio", "shift_interior_fragments", "spread_ratio_intersections_per_km2",
            "spread_ratio_loops_per_km2", "spread_ratio_orientation_phi"]
    lines = ["| group | country | n | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    for g, runs in list(GROUPS.items())[:3]:
        bs = [load(r)["bias"][split] for r in runs if (E / f"{r}.json").exists()]
        for c in ("spain", "united_states", "japan"):
            lines.append(f"| {g} | {c} | {bs[0][c]['n']} | " + " | ".join(fmt([b[c][k] for b in bs]) for k in keys) + " |")
    lines += ["", "Share of variance between countries (eta², prediction vs truth):", "",
              "| group | " + " | ".join(k for k in ("orientation_phi", "orientation_entropy", "intersections_per_km2",
                                                       "loops_per_km2", "dead_end_ratio", "interior_fragments")) + " |",
              "|---|" + "---|" * 6]
    for g, runs in list(GROUPS.items())[:3]:
        bs = [load(r)["bias"][split] for r in runs if (E / f"{r}.json").exists()]
        e = bs[0]["eta2_pred_vs_true"]
        cells = [f"{np.mean([b['eta2_pred_vs_true'][k][0] for b in bs]):.2f} vs {e[k][1]:.2f}" for k in e]
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gen_figure(real_val, real_train) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), facecolor=SURFACE)
    sets = {"real val": real_val, "real train": real_train}
    for g in list(GROUPS)[:3]:
        rs = [r for r in GROUPS[g] if (E / f"{r}.json").exists()]
        sets[g] = pd.concat([pd.read_csv(E / f"{r}_gen_profiles.csv") for r in rs])
    for ax, (k, label) in zip(axes.ravel(), GEN_METRICS):
        data = [s[k].dropna().to_numpy(float) for s in sets.values()]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, name in zip(bp["boxes"], sets):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_xticks(range(1, len(sets) + 1))
        ax.set_xticklabels([n.replace(": ", ":\n") for n in sets], fontsize=7, color=INK)
        ax.set_title(label, fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", color="#ececec")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("GENERATION: distributions of road-network metrics, prior samples (256 per seed) vs real tiles",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "generation_distributions.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)


def sample_figure() -> None:
    names = [("A: C2 (λ=0)", "C2_spatial16_roadx3"), ("B: weak (λ=0.1)", "T_cl0.1_s0"),
             ("C: strong (λ=1)", "T_cl1_s0"), ("λ=3 (degenerate)", "T_cl3_s0")]
    val = np.unpackbits(np.load(E / "C2_spatial16_roadx3_recon_val.npy"), axis=-1)[..., :128]
    from urbanmorph.models.data import LayoutDataset
    real = LayoutDataset("val").x.numpy().astype(np.uint8)
    pick = [0, 7, 14]
    fig, axes = plt.subplots(len(names) + 1, 8, figsize=(16, 2.0 * (len(names) + 1) + 0.6), facecolor=SURFACE)
    for j in range(8):
        axes[0, j].axis("off")
    for j, p in enumerate(pick):
        axes[0, j].imshow(composite_rgb(real[p]), interpolation="nearest")
        axes[0, j].set_title("real validation tile" if j == 0 else "", fontsize=7, loc="left", color=INK)
    for i, (label, run) in enumerate(names, start=1):
        rec = np.unpackbits(np.load(E / f"{run}_recon_val.npy"), axis=-1)[..., :128]
        smp = np.unpackbits(np.load(E / f"{run}_samples.npy"), axis=-1)[..., :128]
        imgs = [rec[p] for p in pick] + [smp[k] for k in range(5)]
        for j, im in enumerate(imgs):
            axes[i, j].imshow(composite_rgb(im.astype(np.uint8)), interpolation="nearest")
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(label, fontsize=8, color=INK)
    axes[1, 0].set_title("reconstructions (paired with row 1)", fontsize=7, loc="left", color=INK)
    axes[1, 3].set_title("prior samples z ~ N(0, I), same seeds for every model", fontsize=7, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "samples_by_lambda.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    real_val = pd.read_csv(E / "real_val_profiles.csv")
    real_train = pd.read_csv(E / "real_train_profiles.csv")
    md = ["# Road-topology experiment: generated tables", "",
          "## Reconstruction (paired, 19 validation tiles; mean (± half-range over seeds))", "", recon_table(), "",
          "## Generation (prior samples vs real tiles)", "", gen_table(real_val, real_train), "",
          "## Structural bias per country: reconstructions of TRAINING tiles (prediction − truth)", "", bias_table("train"), "",
          "## Structural bias per country: reconstructions of VALIDATION tiles (small n)", "", bias_table("val"), ""]
    (OUT / "results_tables.md").write_text("\n".join(md))
    gen_figure(real_val, real_train)
    sample_figure()
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
