"""Milestone 9 preparation: two cheap checks before writing the plan.

    python scripts/m9_checks.py     # -> outputs/m9/analysis/template_fit.json, templates.png, green_parcels.json

1. Standard shape templates. 7 parametric types drawn in a building's own minimum rotated rectangle
   (unit square, x = length): rectangle, L (one corner cut), T / wing (two corners of one side cut),
   Z (two opposite corners cut), U (notch in the middle of one side), H (notches in two opposite
   sides), cross (four corners cut); cut sizes a, b in {0.15, 0.25, 0.35, 0.45} of the side. Every
   orientation is included. Fit = best IoU of a VALIDATION footprint (4000, same sample as M8's library
   evaluation) with any template stretched to its own length and width. IoU is computed on 48 x 48
   masks of the unit square (stretching preserves area ratios, so it equals the metric IoU up to
   rasterisation). Same measure for the M8 100-shape library (nearest class) and the rectangle alone.
2. Real green spaces vs observed parcels (TRAIN tiles, Spain + US): share of green area inside parcels,
   and how much of it lies in parcels that are mostly green (green parcels, as in zoning).
"""
import json
import pickle
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Polygon, box

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from m8_parcel_records import OBS  # noqa: E402
from m8_shape_library import footprints  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.layout.objects import MIN_GREEN_M2, decode
from urbanmorph.layout.shape_library import ShapeLibrary
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m9" / "analysis"
LIB = PROJECT_ROOT / "outputs" / "m8" / "shape_library" / "library.npz"
R = 48
CUTS = (0.15, 0.25, 0.35, 0.45)
_xy = (np.arange(R) + 0.5) / R
GX, GY = np.meshgrid(_xy, _xy)


def mask(poly) -> np.ndarray:
    return shapely.contains_xy(poly, GX.ravel(), GY.ravel())


def templates():
    """(type, params, polygon in the unit square) for every type, cut size and orientation."""
    U = box(0, 0, 1, 1)
    out = [("rectangle", (), U)]
    corners = [(0, 0), (1, 0), (1, 1), (0, 1)]

    def corner(c, a, b):
        x0, y0 = c
        return box(min(x0, 1 - a) if x0 else 0, min(y0, 1 - b) if y0 else 0, a if not x0 else 1, b if not y0 else 1)

    def notch(side, a, b):   # side 0 bottom, 1 right, 2 top, 3 left; a = notch width, b = depth
        lo, hi = 0.5 - a / 2, 0.5 + a / 2
        return [box(lo, 0, hi, b), box(1 - b, lo, 1, hi), box(lo, 1 - b, hi, 1), box(0, lo, b, hi)][side]

    for a, b in product(CUTS, CUTS):
        for i, c in enumerate(corners):
            out.append(("L", (a, b, i), U.difference(corner(c, a, b))))
        for i in range(4):                                     # two corners of one side
            c1, c2 = corners[i], corners[(i + 1) % 4]
            out.append(("T / wing", (a, b, i), U.difference(corner(c1, a, b)).difference(corner(c2, a, b))))
            out.append(("U", (a, b, i), U.difference(notch(i, a, b))))
        for i in range(2):
            out.append(("Z", (a, b, i), U.difference(corner(corners[i], a, b)).difference(corner(corners[i + 2], a, b))))
            out.append(("H", (a, b, i), U.difference(notch(i, a, b)).difference(notch(i + 2, a, b))))
        p = U
        for c in corners:
            p = p.difference(corner(c, a, b))
        out.append(("cross", (a, b), p))
    return [(t, prm, p) for t, prm, p in out if p.geom_type == "Polygon" and p.area > 0.3]


def unit(poly, L, W):
    """Footprint decoded in its own frame (x = length) -> unit square."""
    return shapely.affinity.affine_transform(poly, [1 / L, 0, 0, 1 / W, 0.5, 0.5])


def iou(m, M):
    inter = (M & m).sum(1)
    return inter / ((M | m).sum(1))


def template_fit():
    data = load_objects()
    va, va_lw = footprints(data["val"])
    sel = np.random.default_rng(0).choice(len(va), min(4000, len(va)), replace=False)
    T = templates()
    M = np.stack([mask(p) for _, _, p in T])
    types = np.array([t for t, _, _ in T])
    lib = ShapeLibrary.load(LIB)
    cls = lib.nearest(va[sel])
    best_t, best_iou, rect_iou, lib_iou = [], [], [], []
    for i, c in zip(sel, cls):
        L, W = va_lw[i]
        m = mask(unit(decode(np.zeros(2), 0.0, L, W, va[i]), L, W))
        s = iou(m, M)
        k = int(s.argmax())
        best_t.append(types[k])
        best_iou.append(s[k])
        rect_iou.append(s[0])
        ml = mask(unit(decode(np.zeros(2), 0.0, L, W, lib.ratios(c)), L, W))
        lib_iou.append((m & ml).sum() / (m | ml).sum())
    best_t, best_iou = np.array(best_t), np.array(best_iou)
    res = {"val_footprints": int(len(sel)), "templates": len(T), "types": sorted(set(types)),
           "share_iou_ge_0.9": {"rectangle only": float(np.mean(np.array(rect_iou) >= 0.9)),
                                "7 standard types": float(np.mean(best_iou >= 0.9)),
                                "M8 100-shape library": float(np.mean(np.array(lib_iou) >= 0.9))},
           "iou_median": {"rectangle only": float(np.median(rect_iou)), "7 standard types": float(np.median(best_iou)),
                          "M8 100-shape library": float(np.median(lib_iou))},
           "best_type_share": {t: float(np.mean(best_t == t)) for t in sorted(set(types))},
           "rasterisation": f"{R} x {R} masks of the unit square"}
    (OUT / "template_fit.json").write_text(json.dumps(res, indent=1))
    order = ["rectangle", "L", "T / wing", "Z", "U", "H", "cross"]
    fig, axes = plt.subplots(1, 7, figsize=(10.5, 1.9), facecolor=SURFACE)
    for ax, t in zip(axes, order):
        p = next(p for tt, prm, p in T if tt == t and (not prm or prm[:2] == (0.35, 0.35)))
        x, y = p.exterior.xy
        ax.fill(np.array(x) * 1.6, y, color="#3c3c43")
        ax.set_aspect("equal"), ax.set_axis_off()
        ax.set_title(f"{t}\n{res['best_type_share'].get(t, 0):.0%} of val", fontsize=8, color=INK)
    fig.suptitle("Standard footprint types (one cut size shown; drawn at aspect 1.6) and share of validation "
                 "footprints best fitted by each", fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "templates.png", dpi=140, facecolor=SURFACE)
    return res


def green_parcels():
    tiles = pickle.loads(OBS.read_bytes())
    idx = load_index_with_expansion().set_index("sample_id")
    res = {}
    for c in ("spain", "united_states"):
        tot = inside = in_green_parcels = 0.0
        n_parc = n_green_parc = 0
        gp_area = []
        for sid, t in tiles.items():
            if t["split"] != "train" or t["country"] != c:
                continue
            r = idx.loc[sid]
            green = tile_polygons(sample_dir(r.country, r.city) / f"{sid}_vectors.gpkg", r.xmin, r.ymin, "green", MIN_GREEN_M2)
            if not green:
                n_parc += len(t["parcels"])
                continue
            G = shapely.union_all(green)
            tot += G.area
            P = shapely.union_all(t["parcels"])
            inside += G.intersection(P).area
            for p in t["parcels"]:
                n_parc += 1
                if not p.intersects(G):
                    continue
                g = p.intersection(G).area
                if g / p.area >= 0.5:
                    n_green_parc += 1
                    in_green_parcels += g
                    gp_area.append(p.area)
        res[c] = {"green_area_share_inside_parcels": inside / tot, "green_area_share_in_parcels_ge50pct_green": in_green_parcels / tot,
                  "share_of_parcels_ge50pct_green": n_green_parc / n_parc,
                  "green_parcel_area_m2_quantiles_25_50_75": [float(q) for q in np.quantile(gp_area, [.25, .5, .75])] if gp_area else None}
    (OUT / "green_parcels.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(template_fit(), indent=1))
    print(json.dumps(green_parcels(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
