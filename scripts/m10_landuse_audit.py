"""Milestone 10, Phase 4: what is the unbuilt land of real blocks? Tests the M9b rule "no street access -> green".

    python scripts/m10_landuse_audit.py      # -> outputs/m10/landuse/landuse_audit.json, landuse.png

TRAINING tiles only. Two views:
  1. observed cadastral parcels (Spain, US): each parcel is
       built         >= 1 building or building coverage >= 5 %,
       public green  >= 50 % covered by public green (see GREEN_CLASSES),
       private green >= 50 % covered by private / access-unknown gardens,
       unbuilt other everything else (vacant, paved yards, parking, untagged courtyards: OSM cannot tell them apart);
     cross-tabulated by street parcels vs parcels WITHOUT street frontage (the M9b "no access" case);
  2. every closed block of every training tile (all 9 cities): the block interior, i.e. the part farther than
     INNER_M from the block edge (courtyards / backyards), split into built, public green, private green and
     untagged unbuilt (area shares).
Green categories come from the OSM tags of the ``green`` layer:
  public  : leisure park / playground / pitch / recreation_ground / dog_park / nature_reserve / common /
            sports_centre / golf_course, landuse recreation_ground / village_green / cemetery / forest / meadow /
            grass, natural *, unless access is private / no;
  private : leisure garden with access private / no / customers, or any green with such an access tag;
  garden (access unknown): leisure garden without an access tag (counted with private: mostly courtyard gardens).
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely import affinity

from m6_build_objects import load_objects, tile_polygons  # noqa: E402
from m6_build_roads import class_widths  # noqa: E402
from m8_parcel_records import OBS, load_records  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
from urbanmorph.graphs.road_graph import TILE_M
from urbanmorph.layout.blocks import extract_blocks
from urbanmorph.layout.objects import MIN_BUILDING_M2
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m10" / "landuse"
INNER_M = 20.0
PRIVATE_ACCESS = {"private", "no", "customers"}
PUBLIC_LEISURE = {"park", "playground", "pitch", "recreation_ground", "dog_park", "nature_reserve", "common",
                  "sports_centre", "golf_course"}
PUBLIC_LANDUSE = {"recreation_ground", "village_green", "cemetery", "forest", "meadow", "grass"}
TILE = shapely.box(0, 0, TILE_M, TILE_M)


def green_by_class(gpkg, xmin, ymin):
    """(public, private) green unions in tile coordinates."""
    gdf = gpd.read_file(gpkg, layer="green")
    pub, pri = [], []
    for _, r in gdf.explode(index_parts=False).iterrows():
        g = r.geometry
        if g is None or g.is_empty or g.geom_type != "Polygon":
            continue
        p = affinity.translate(shapely.make_valid(g), -xmin, -ymin).intersection(TILE)
        if p.is_empty:
            continue
        acc = str(r.get("access") or "").lower()
        leis, lu = str(r.get("leisure") or ""), str(r.get("landuse") or "")
        if acc in PRIVATE_ACCESS or leis == "garden":
            pri.append(p)
        elif leis in PUBLIC_LEISURE or lu in PUBLIC_LANDUSE or str(r.get("natural") or "") not in ("", "None", "nan"):
            pub.append(p)
        else:
            pub.append(p)                                  # other green tags: treated as public open space
    U = lambda L: shapely.union_all(L) if L else shapely.Polygon()
    pri_u = U(pri)
    return U(pub).difference(pri_u), pri_u


def parcel_class(poly, B, PUB, PRI, n_b):
    a = poly.area
    if n_b >= 1 or poly.intersection(B).area >= 0.05 * a:
        return "built"
    if poly.intersection(PUB).area >= 0.5 * a:
        return "public_green"
    if poly.intersection(PRI).area >= 0.5 * a:
        return "private_green"
    return "unbuilt_other"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion().set_index("sample_id")
    data = load_objects()
    cw = class_widths()
    obs = pickle.loads(OBS.read_bytes())
    rec = load_records()
    cats = ("built", "public_green", "private_green", "unbuilt_other")
    parcels = {}                   # country -> {"street"/"no_frontage": counts}
    blocks = {}                    # city -> area sums
    for t in data["train"]:
        sid = t["sample_id"]
        r = idx.loc[sid]
        f = sample_dir(r.country, r.city) / f"{sid}_vectors.gpkg"
        bl = tile_polygons(f, r.xmin, r.ymin, "buildings", MIN_BUILDING_M2)
        B = shapely.union_all(bl) if bl else shapely.Polygon()
        PUB, PRI = green_by_class(f, r.xmin, r.ymin)
        # 1. observed parcels
        if sid in obs:
            rr = rec[rec.sample_id == sid]
            P = obs[sid]["parcels"]
            if len(P) == len(rr):
                d = parcels.setdefault(r.country, {k: {c: 0 for c in cats} for k in ("street", "no_frontage")})
                for poly, inter, nb, cut in zip(P, rr.interior, rr.n_buildings, rr.cut):
                    if cut:
                        continue                           # cut by the tile window: frontage unknown
                    d["no_frontage" if inter else "street"][parcel_class(poly, B, PUB, PRI, nb)] += 1
        # 2. block interiors
        acc = blocks.setdefault(r.city, {"n_blocks": 0, "interior_m2": 0.0, "built": 0.0, "public_green": 0.0,
                                         "private_green": 0.0, "untagged_unbuilt": 0.0})
        for b in extract_blocks(t["graph"], cw):
            if b["open"]:
                continue
            inner = b["polygon"].buffer(-INNER_M)
            if inner.is_empty:
                continue
            a = inner.area
            built = inner.intersection(B).area
            pub = inner.intersection(PUB).difference(B).area
            pri = inner.intersection(PRI).difference(B).difference(PUB).area
            acc["n_blocks"] += 1
            acc["interior_m2"] += a
            acc["built"] += built
            acc["public_green"] += pub
            acc["private_green"] += pri
            acc["untagged_unbuilt"] += max(a - built - pub - pri, 0.0)
    res = {"inner_m": INNER_M, "parcels": {}, "block_interiors": {}}
    for c, d in parcels.items():
        res["parcels"][c] = {k: {"n": sum(v.values()), **{x: v[x] / max(sum(v.values()), 1) for x in cats}} for k, v in d.items()}
    for city, a in blocks.items():
        T = max(a["interior_m2"], 1.0)
        res["block_interiors"][city] = {"n_blocks_with_interior": a["n_blocks"],
                                        **{k: a[k] / T for k in ("built", "public_green", "private_green", "untagged_unbuilt")}}
    (OUT / "landuse_audit.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=SURFACE)
    cols = {"built": "#6b6b6b", "public_green": "#3f8f4f", "private_green": "#9cc88f", "unbuilt_other": "#e0d6c2",
            "untagged_unbuilt": "#e0d6c2"}
    labels = [f"{c} / {k}" for c in res["parcels"] for k in ("street", "no_frontage")]
    left = np.zeros(len(labels))
    for x in cats:
        v = np.array([res["parcels"][c][k][x] for c in res["parcels"] for k in ("street", "no_frontage")])
        axes[0].barh(labels, v, left=left, color=cols[x], label=x)
        left += v
    axes[0].set_title("observed parcels (train): what is on them", fontsize=9, loc="left", color=INK)
    axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    cities = sorted(res["block_interiors"])
    left = np.zeros(len(cities))
    for x in ("built", "public_green", "private_green", "untagged_unbuilt"):
        v = np.array([res["block_interiors"][c][x] for c in cities])
        axes[1].barh(cities, v, left=left, color=cols[x], label=x)
        left += v
    axes[1].set_title(f"closed-block interiors (> {INNER_M:.0f} m from the street), area share", fontsize=9, loc="left", color=INK)
    axes[1].legend(fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    for a in axes:
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "landuse.png", dpi=130, facecolor=SURFACE)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
