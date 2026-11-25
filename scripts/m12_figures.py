"""Milestone 12 result figures (development tiles only).

    python scripts/m12_figures.py   # -> outputs/m12/figures/{scores,by_class,examples}.png

1. scores.png: S_T (conditional real terrain) and S10 (unconditional) per model; one point per training seed.
2. by_class.png: S_T by the real tile's landform class (flat / moderate / steep).
3. examples.png: three San Diego val_t tiles (held-out city) - hillshade, water, real streets and the streets of N0,
   TN, TN-lp (low-passed terrain input), TN+HR and TN-lp+HR on the same terrain (bridges drawn thicker). Streets only: these are network figures, not city
   plans (the 3D cities' plans with parcels are in outputs/m12/cities/*/plan.png).
"""
import json
import pickle
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

from m12_evaluate import GRAPHS3D, OUT, SPECS, TERRAIN, TH, load_model  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.levels import BRIDGE
from urbanmorph.graphs.terrain_generate import TerrainPolicy

FIG = PROJECT_ROOT / "outputs" / "m12" / "figures"
INK, MUTED, GRID = "#2b2b2b", "#8a8a8a", "#e6e6e6"
ORDER = ["N0", "R", "P", "TN", "TN+R", "TN+HR", "TN+HRV", "TNlp", "TNlp+R", "TNlp+HR", "TNelev", "TNslope", "TNwater", "TNelevslope"]


def dot_panel(ax, values, title, ref=None):
    names = [n for n in ORDER if n in values]
    for i, n in enumerate(names):
        for s, v in values[n].items():
            ax.plot(v, i, "o", ms=7, mfc=INK if s == 0 else "white", mec=INK, mew=1.4)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    if ref is not None:
        ax.axvline(ref, color=MUTED, ls="--", lw=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    st = json.loads((OUT / "stats.json").read_text())
    vals = {"S_T": {}, "S10": {}}
    for k, v in st["A"].items():
        if "_s" in k:
            n, s = k.rsplit("_s", 1)
            vals["S_T"].setdefault(n, {})[int(s)] = v["S_T"]
    for k, v in st.get("B", {}).items():
        n, s = k.rsplit("_s", 1)
        vals["S10"].setdefault(n, {})[int(s)] = v["S10_vs_val"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    dot_panel(ax[0], vals["S_T"], f"S_T, conditional real terrain ({st['n_tiles']} development tiles); lower is better")
    dot_panel(ax[1], vals["S10"], "S10, unconditional (256 tiles vs 19 val tiles); lower is better")
    ax[0].set_xlabel("mean W1/IQR over 5 terrain metrics")
    ax[1].set_xlabel("mean W1/IQR over 10 street metrics")
    fig.text(0.01, 0.01, "filled = training seed 0, hollow = seed 1; P has one sampling seed", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "scores.png", dpi=150)
    # by terrain class
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    for a, c in zip(ax, ("flat", "moderate", "steep")):
        bc = st["by_class"].get(c, {})
        v = {}
        for k, x in bc.items():
            if "_s" in k:
                n, s = k.rsplit("_s", 1)
                v.setdefault(n, {})[int(s)] = x
        dot_panel(a, v, f"{c} tiles (n = {bc.get('n', 0)})")
        a.set_xlabel("S_T")
    fig.tight_layout()
    fig.savefig(FIG / "by_class.png", dpi=150)
    # examples on held-out San Diego terrain
    real = pickle.loads(GRAPHS3D.read_bytes())
    aud = pd.read_csv(PROJECT_ROOT / "outputs" / "m12" / "audits" / "terrain_audit.csv")
    vt = aud[aud.split == "val_t"].sort_values("slope_landform_p90")
    pick = [vt.sample_id.iloc[len(vt) // 2], vt.sample_id.iloc[-3], vt.sample_id.iloc[-1]]
    gmap = {sid: g for sp in real for sid, _, g in real[sp]}
    ds = [dict(np.load(TERRAIN / f"{s}.npz")) for s in pick]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cols = [("real", None)] + [(n, SPECS[n]) for n in ("N0", "TN", "TNlp", "TN+HR", "TNlp+HR")]
    gens = {}
    for name, spec in cols[1:]:
        model, names, norm = load_model(f"{spec[0]}_s0", device)
        from urbanmorph.graphs.terrain_sequence import lowpass
        crop = [lowpass(d, model.lowpass_m) for d in ds] if model.lowpass_m else None
        gens[name] = sample_graphs(model, len(ds), device=device, seed=1000, temperature=0.5,
                                   terrain=TerrainPolicy(ds, names, norm, mode=spec[1], th=TH, crop_tiles=crop))
    fig, ax = plt.subplots(len(pick), len(cols), figsize=(3.1 * len(cols), 3.2 * len(pick)))
    for r, (sid, d) in enumerate(zip(pick, ds)):
        z = d["elev"][64:192, 64:192]
        gy, gx = np.gradient(z, 4.0)
        shade = np.clip(0.6 + 0.5 * (gx - gy) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
        water = np.ma.masked_where(d["water"][64:192, 64:192] == 0, np.ones_like(z))
        for c, (name, _) in enumerate(cols):
            a = ax[r, c]
            a.imshow(shade, cmap="gray", extent=(0, 512, 0, 512), vmin=0, vmax=1.2)
            a.imshow(water, cmap="Blues", vmin=0, vmax=1.5, extent=(0, 512, 0, 512), alpha=0.9)
            g = gmap[sid] if name == "real" else gens[name][r]
            seg, st_ = g.segments(), g.structs()
            for s, b in zip(seg, st_):
                a.plot([s[0], s[2]], [s[1], s[3]], color="#c0392b" if b == BRIDGE else INK, lw=2.4 if b == BRIDGE else 0.9)
            a.set_xticks([]), a.set_yticks([])
            a.set_xlim(0, 512), a.set_ylim(0, 512)
            if r == 0:
                a.set_title(name, fontsize=11)
            if c == 0:
                cls = aud.set_index("sample_id").terrain_class_landform.get(sid)
                a.set_ylabel(f"{sid}\n{cls}", fontsize=8)
    fig.suptitle("Held-out San Diego tiles (val_t): hillshade, water (blue), streets (bridges red). Conditional generation, T = 0.5, seed 0",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "examples.png", dpi=130)
    print("figures written to", FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
