"""Milestone 13: aerial elevation maps of the demo cities (hypsometric tint + hillshade + contours, streets on top).

    python scripts/m13_elevation_maps.py   # -> outputs/m12/cities/<city>/elevation.png,
                                           #    outputs/m13/figures/elevation_maps.png (M* vs A per scenario)

Display only: the generated networks are drawn exactly as stored (streets.npy); nothing is edited.
Both samplers of a scenario share one terrain, so each pair uses the same colour scale and contour interval.
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
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LightSource, LinearSegmentedColormap

from m12_city_plans import terrain_for  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

CITIES = PROJECT_ROOT / "outputs" / "m12" / "cities"
FIG = PROJECT_ROOT / "outputs" / "m13" / "figures"
# lowland green -> olive -> tan -> brown -> pale rock; lightness rises monotonically with altitude
HYPSO = LinearSegmentedColormap.from_list(
    "hypso", ["#1f4d33", "#4f7a3f", "#93a35a", "#c8b77a", "#b8916a", "#d9c7b0", "#f5f1ea"])
EXT = (0, 512, 0, 512)


def contour_step(lo, hi, n=12):
    return next((s for s in (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100) if (hi - lo) / s <= n), 200)


def draw(d: Path, ax, lo, hi, step):
    city = json.loads((d / "city.json").read_text())
    z, w = terrain_for(city)
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(z, cmap=HYPSO, vmin=lo, vmax=hi, blend_mode="soft", vert_exag=2.0, dx=4.0, dy=4.0)
    ax.imshow(rgb, extent=EXT, origin="upper")
    ax.imshow(np.ma.masked_where(w == 0, w), cmap=LinearSegmentedColormap.from_list("w", ["#5b9bd5", "#2e6aa8"]),
              vmin=0, vmax=4, extent=EXT, origin="upper", alpha=0.9)
    yy, xx = np.mgrid[2:512:4, 2:512:4]
    yy = 512 - yy                                     # row 0 is the north edge (heightfield.py)
    lv = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    ax.contour(xx, yy, z, levels=lv, colors="#3b2f25", linewidths=0.35, alpha=0.55)
    major = lv[np.isclose(np.mod(lv, 5 * step), 0)] if len(lv) > 6 else lv
    cs = ax.contour(xx, yy, z, levels=major, colors="#3b2f25", linewidths=0.8, alpha=0.75)
    ax.clabel(cs, fmt=lambda v: f"{v:g} m", fontsize=5.5, inline_spacing=2)
    bld = [np.asarray(b["footprint"][0])[:, :2] for b in city["buildings"]]
    ax.add_collection(PolyCollection(bld, facecolors="#2b2b2b", edgecolors="none", alpha=0.55))
    s = np.load(d / "streets.npy")
    seg, br = s[:, :4].reshape(-1, 2, 2), s[:, 5] == 1
    ax.add_collection(LineCollection(seg[~br], colors="white", linewidths=2.6, capstyle="round"))
    ax.add_collection(LineCollection(seg[~br], colors="#1a1a1a", linewidths=1.2, capstyle="round"))
    ax.add_collection(LineCollection(seg[br], colors="white", linewidths=3.4, capstyle="round"))
    ax.add_collection(LineCollection(seg[br], colors="#c0392b", linewidths=2.0, capstyle="round"))
    ax.set_xlim(0, 512), ax.set_ylim(0, 512)
    ax.set_xticks([]), ax.set_yticks([])
    return len(bld), int(br.sum())


def scale_for(d: Path):
    z, w = terrain_for(json.loads((d / "city.json").read_text()))
    land = z[w == 0] if (w == 0).any() else z
    lo, hi = float(np.floor(land.min())), float(np.ceil(land.max()))
    return lo, max(hi, lo + 1.0)


def colorbar(fig, ax, lo, hi, step):
    sm = plt.cm.ScalarMappable(cmap=HYPSO, norm=plt.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"elevation (m), contours every {step:g} m", fontsize=7)
    cb.ax.tick_params(labelsize=6)


def label(d: Path):
    sc = d.name[4:].split("_TN_")[0]
    return sc, ("M* (energy, K=8, λ=0.5)" if "tc-energy" in d.name else "A (standard sampling)")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in CITIES.iterdir() if "_tc-" in p.name and (p / "streets.npy").exists())
    pairs = {}
    for d in dirs:
        pairs.setdefault(label(d)[0], []).append(d)
    pairs = {k: sorted(v, key=lambda p: "tc-standard" in p.name) for k, v in pairs.items()}
    scales = {k: scale_for(v[0]) for k, v in pairs.items()}
    for sc, ds in pairs.items():
        lo, hi = scales[sc]
        step = contour_step(lo, hi)
        for d in ds:
            fig, ax = plt.subplots(figsize=(6.6, 6))
            nb, nbr = draw(d, ax, lo, hi, step)
            ax.set_title(f"{sc}: {label(d)[1]}\n{nb} buildings, {nbr} bridge edges (red); 512 m tile", fontsize=8)
            colorbar(fig, ax, lo, hi, step)
            fig.savefig(d / "elevation.png", dpi=170, bbox_inches="tight")
            plt.close(fig)
    both = {k: v for k, v in pairs.items() if len(v) == 2}   # the sheet compares M* with A; singletons are smoke tests
    rows = len(both)
    fig, ax = plt.subplots(rows, 2, figsize=(9.2, 4.3 * rows), squeeze=False, layout="constrained")
    for r, (sc, ds) in enumerate(both.items()):
        lo, hi = scales[sc]
        step = contour_step(lo, hi)
        for c in range(2):
            nb, nbr = draw(ds[c], ax[r, c], lo, hi, step)
            ax[r, c].set_title(f"{sc}: {label(ds[c])[1]}\n{nb} buildings, {nbr} bridge edges", fontsize=7.5)
        colorbar(fig, list(ax[r]), lo, hi, step)
    fig.suptitle("Demo cities on their terrain: hypsometric tint + hillshade, contours; streets black, bridges red, "
                 "water blue (same sampling seed 2041, not cherry-picked)", fontsize=9)
    fig.savefig(FIG / "elevation_maps.png", dpi=110, bbox_inches="tight")
    print(len(dirs), "maps,", rows, "scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())
