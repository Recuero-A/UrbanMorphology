"""Synthetic roof and architectural-style parameters from LOCAL context, never from nationality alone
(docs/M11_PROCEDURAL_BUILDINGS.md).

Typology of a building = f(floors, footprint area, attachment to neighbours):
  detached_low  : <= 2 floors, not attached, < 400 m2
  attached_low  : <= 3 floors, attached or >= 400 m2
  midrise       : 4-8 floors
  highrise      : >= 9 floors
Roofs: P(pitched | typology) from PROFILE (a documented, configurable assumption: OSM roof:shape is tagged on
<= 3.4 % of buildings, outputs/m11/audit/height_audit.json), and only where the footprint is a near-rectangle
(mesh.rect_footprint). Materials: one family per block (controlled variation), a palette variant per building.
"""

from __future__ import annotations

import numpy as np

from urbanmorph.city3d.materials import FACADE_FAMILIES
from urbanmorph.city3d.mesh import rect_footprint

PROFILE = {
    "pitched_share": {"detached_low": 0.75, "attached_low": 0.35, "midrise": 0.10, "highrise": 0.0},
    "hip_share_of_pitched": 0.4,
    "pitch_deg": (25.0, 35.0),
    # facade grammar parameters per block typology: (P(balconies), P(cornice), P(shop ground floor), bay width range m)
    "facade": {"detached_low": (0.10, 0.15, 0.0, (2.8, 3.6)), "attached_low": (0.35, 0.45, 0.15, (3.0, 3.8)),
               "midrise": (0.65, 0.55, 0.55, (3.2, 4.2)), "highrise": (0.25, 0.20, 0.45, (3.4, 4.4))},
    "flip": 0.15,                             # chance a building departs from its block's binary facade choices
    "family_weights": {                       # facade material family per block typology
        "detached_low": {"plaster": 0.45, "brick": 0.25, "wood_siding": 0.25, "concrete": 0.05},
        "attached_low": {"plaster": 0.45, "brick": 0.35, "stone": 0.10, "concrete": 0.10},
        "midrise": {"plaster": 0.45, "brick": 0.20, "stone": 0.15, "concrete": 0.20},
        "highrise": {"concrete": 0.60, "plaster": 0.25, "stone": 0.15},
    },
}
ATTACH_M = 0.3


def typology(floors: int, area: float, attached: bool) -> str:
    if floors >= 9:
        return "highrise"
    if floors >= 4:
        return "midrise"
    if floors <= 2 and not attached and area < 400:
        return "detached_low"
    return "attached_low"


def attachment(polys) -> list[list[int]]:
    """Neighbours each footprint touches (within ATTACH_M)."""
    import shapely
    tree = shapely.STRtree(polys)
    out = [[] for _ in polys]
    for i, p in enumerate(polys):
        for j in tree.query(p.buffer(ATTACH_M)):
            if j != i and p.distance(polys[j]) <= ATTACH_M:
                out[i].append(int(j))
    return out


def assign_style(buildings: list[dict], rng, profile: dict = PROFILE) -> None:
    """In place: roof dict, typology, style dict (with source labels). Needs ``_poly``, ``block``, ``floors``,
    ``height_m``, ``height_source``, and the city storey height under ``_storey``."""
    polys = [b["_poly"] for b in buildings]
    nb = attachment(polys)
    by_block = {}
    for i, b in enumerate(buildings):
        b["_attached"] = nb[i]
        b["typology"] = typology(b["floors"], b["_poly"].area, bool(nb[i]))
        by_block.setdefault(b["block"], []).append(b)
    for members in by_block.values():
        major = max({m["typology"] for m in members}, key=lambda t: sum(m["typology"] == t for m in members))
        fam_w = profile["family_weights"][major]
        family = rng.choice(list(fam_w), p=np.array(list(fam_w.values())) / sum(fam_w.values()))
        base_variant = int(rng.integers(len(FACADE_FAMILIES[family])))
        p_bal, p_cor, p_shop, (bw0, bw1) = profile["facade"][major]
        blk = {"balconies": rng.uniform() < p_bal, "cornice": rng.uniform() < p_cor, "shop": rng.uniform() < p_shop,
               "bay": float(rng.uniform(bw0, bw1)),
               "bal_pattern": str(rng.choice(["all", "alternate", "first_floor"], p=[0.4, 0.3, 0.3])), "ww": float(rng.uniform(0.42, 0.6)), "wh": float(rng.uniform(0.5, 0.62))}
        flip = lambda v: (not v) if rng.uniform() < profile["flip"] else v
        for b in members:
            variant = (base_variant + int(rng.choice([-1, 0, 0, 1]))) % len(FACADE_FAMILIES[family])
            pitched = rng.uniform() < profile["pitched_share"][b["typology"]] and not b["_attached"]
            rect = rect_footprint(b["_poly"]) if pitched else None
            if rect is not None:
                kind = "hip" if rng.uniform() < profile["hip_share_of_pitched"] else "gable"
                pitch = float(rng.uniform(*profile["pitch_deg"]))
                xy = np.asarray(rect.exterior.coords)[:4]
                e = np.linalg.norm(np.diff(np.r_[xy, xy[:1]], axis=0), axis=1)
                short = float(min(e))
                rise = short / 2 * np.tan(np.radians(pitch))
                b["roof"] = {"type": kind, "pitch_deg": round(pitch, 1), "rise_m": round(rise, 2), "source": "synthetic"}
                b["_rect"] = rect
            else:
                b["roof"] = {"type": "flat", "pitch_deg": 0.0, "rise_m": 0.0, "source": "synthetic"}
            rise = b["roof"]["rise_m"]
            if b["height_source"] == "observed":            # OSM height is to the top of the roof
                b["eave_m"] = round(max(2.5, b["height_m"] - rise), 2)
            else:
                b["eave_m"] = round(b["height_m"], 2)
                b["height_m"] = round(b["height_m"] + rise, 2)
            b["style"] = {"facade_family": str(family), "palette": variant, "typology_of_block": major,
                          "attached_to": [buildings[j]["id"] for j in b["_attached"]],
                          "bay_width_m": round(blk["bay"] + float(rng.normal(0, 0.15)), 2),
                          "window_width_ratio": round(blk["ww"], 3), "window_height_ratio": round(blk["wh"], 3),
                          "balconies": bool(flip(blk["balconies"])), "balcony_pattern": blk["bal_pattern"],
                          "cornice": bool(flip(blk["cornice"])),
                          "ground_floor": "shop" if flip(blk["shop"]) else "residential", "source": "synthetic"}
