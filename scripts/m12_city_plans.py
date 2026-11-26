"""Milestone 12: 2D plans of the 3D cities (parcels drawn, as required for final city figures) and a contact sheet.

    python scripts/m12_city_plans.py   # -> outputs/m12/cities/<city>/plan.png, outputs/m12/figures/cities.png

Plan: hillshade of the terrain, water, BF parcels (thin outlines), buildings, streets (bridges and viaducts red).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.terrain.synthesis import generate, scenario

CITIES = PROJECT_ROOT / "outputs" / "m12" / "cities"
FIG = PROJECT_ROOT / "outputs" / "m12" / "figures"
TERRAIN = PROJECT_ROOT / "data" / "processed" / "m12" / "terrain"


def terrain_for(city):
    src = city["metadata"]["provenance"].get("terrain", "")
    if src.startswith("real:"):
        d = dict(np.load(TERRAIN / f"{src[5:]}.npz"))
        return d["elev"][64:192, 64:192], d["water"][64:192, 64:192]
    _, name, sd = src.split(":")
    t = generate(scenario(name, int(sd.replace("seed", ""))), -256, -256, 768, 768)
    return t.elev[64:192, 64:192], t.water_kind[64:192, 64:192]


def plan(d: Path, ax):
    city = json.loads((d / "city.json").read_text())
    z, w = terrain_for(city)
    gy, gx = np.gradient(z, 4.0)
    shade = np.clip(0.62 + 0.45 * (gx - gy) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
    ax.imshow(shade, cmap="gray", vmin=0, vmax=1.25, extent=(0, 512, 0, 512))
    ax.imshow(np.ma.masked_where(w == 0, w), cmap="Blues", vmin=0, vmax=4, extent=(0, 512, 0, 512), alpha=0.85)
    par = [np.asarray(p["polygon"][0])[:, :2] for p in city.get("parcels", []) if p.get("polygon")]
    ax.add_collection(PolyCollection(par, facecolors="none", edgecolors="#6d6d6d", linewidths=0.35))
    bld = [np.asarray(b["footprint"][0])[:, :2] for b in city["buildings"]]
    ax.add_collection(PolyCollection(bld, facecolors="#3a3a3a", edgecolors="none"))
    s = np.load(d / "streets.npy")
    seg = s[:, :4].reshape(-1, 2, 2)
    br = s[:, 5] == 1
    ax.add_collection(LineCollection(seg[~br], colors="#f4f1ea", linewidths=1.6))
    ax.add_collection(LineCollection(seg[br], colors="#c0392b", linewidths=2.6))
    ax.set_xlim(0, 512), ax.set_ylim(0, 512)
    ax.set_xticks([]), ax.set_yticks([])
    v = json.loads((d / "validation.json").read_text())
    st = v["streets"]["policy"]
    via = st.get("viaduct_rule", 0) + st.get("viaduct_model", 0)
    ax.set_title(f"{d.name[4:]}\n{len(bld)} buildings, {int(br.sum())} bridge edges"
                 + (f" ({via} viaduct steps)" if via else ""), fontsize=7.5)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in CITIES.iterdir() if p.name.startswith("m12_") and (p / "streets.npy").exists())
    for d in dirs:
        fig, ax = plt.subplots(figsize=(6, 6.3))
        plan(d, ax)
        fig.tight_layout()
        fig.savefig(d / "plan.png", dpi=150)
        plt.close(fig)
    n = len(dirs)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows * 2, cols, figsize=(4.2 * cols, 4.4 * rows * 2))
    for i, d in enumerate(dirs):
        r, c = divmod(i, cols)
        plan(d, ax[2 * r, c])
        img = plt.imread(d / "renders" / "oblique.png") if (d / "renders" / "oblique.png").exists() else None
        a = ax[2 * r + 1, c]
        if img is not None:
            a.imshow(img)
        a.set_axis_off()
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        ax[2 * r, c].set_axis_off(), ax[2 * r + 1, c].set_axis_off()
    fig.suptitle("Milestone 12 cities: plan (hillshade, water, parcels, buildings, streets; bridges red) and Blender render",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "cities.png", dpi=90)
    print(n, "plans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
