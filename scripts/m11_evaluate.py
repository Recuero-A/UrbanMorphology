"""Milestone 11, Phase F: evaluation (docs/MILESTONE_11.md §5). No criteria were pre-registered for M11 3D
quality; every number here is descriptive and reported as measured.

    python scripts/m11_evaluate.py      # -> outputs/m11/evaluation.json

1. Height baseline, holdout on all 19 VALIDATION tiles (test split never read): per tile, 20 % of the buildings
   with known floors (observed or derived from OSM tags) are hidden, 5 folds; their floors are predicted by
   (a) city3d.heights (block-correlated, conditioned on the block's known neighbours), (b) the prior median of the
   footprint-area bin, (c) an independent draw from the prior. MAE in floors per tile; the tile is the unit.
   Spatial correlation: median within-block SD of floors (blocks with >= 3 buildings) for observed floors vs
   all-synthetic floors from (a) and from (c).
2. Per rendered tile: geometry validity, architectural plausibility, environment, performance (validation.json,
   blender_timing*.json, godot_check.json, godot_fps.json).
3. Real vs generated layouts rendered by the same pipeline (identical code): descriptive differences.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from urbanmorph.city3d.convert import from_real
from urbanmorph.city3d.facade import floor_levels
from urbanmorph.city3d.format import polygon
from urbanmorph.city3d.heights import U_CLIP, assign_heights, floor_quantiles
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m11"
RENDERED = {"real": ["ESP_BCN_e837_n8952", "USA_LAX_e758_n7370", "USA_NYC_e1149_n8784", "JPN_TYO_e764_n7739"],
            "generated": ["gen_N_T05_s2026_00_spain", "gen_N_T05_s2026_01_united_states"]}
PAIRS = [("ESP_BCN_e837_n8952", "gen_N_T05_s2026_00_spain"), ("USA_LAX_e758_n7370", "gen_N_T05_s2026_01_united_states"),
         ("USA_NYC_e1149_n8784", "gen_N_T05_s2026_01_united_states")]


def _rows(city):
    out = []
    for b in city["buildings"]:
        known = b["floors_source"] in ("observed", "derived") and b["height_source"] in ("observed", "derived")
        out.append({"_poly": polygon(b["footprint"]), "block": b["block"], "floors_true": b["floors"], "known": known})
    return out


def _predict(rows, hidden, priors, key, storey, rng, noise=True):
    bs = [{"_poly": r["_poly"], "block": r["block"], "_obs_height": np.nan,
           "_obs_levels": float(r["floors_true"]) if (r["known"] and i not in hidden) else np.nan} for i, r in enumerate(rows)]
    assign_heights(bs, priors, key, storey, rng, noise)
    return np.array([b["floors"] for b in bs])


def _within_block_sd(rows, floors):
    by = {}
    for r, f in zip(rows, floors):
        by.setdefault(r["block"], []).append(f)
    sds = [np.std(v) for v in by.values() if len(v) >= 3]
    return float(np.median(sds)) if sds else None


def height_holdout(priors, storey_tab):
    from m6_build_objects import load_objects
    from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
    data = load_objects()
    idx = load_index_with_expansion().set_index("sample_id")
    res = {}
    for t in data["val"]:
        row = idx.loc[t["sample_id"]]
        assert row.split == "val"
        city = from_real(t, row, sample_dir(row.country, row.city) / f"{t['sample_id']}_vectors.gpkg", data["class_widths"],
                         priors, storey_tab[row.city], None, seed=0)
        rows = _rows(city)
        known = [i for i, r in enumerate(rows) if r["known"]]
        if len(known) < 50:
            res[t["sample_id"]] = {"n_known": len(known), "skipped": "fewer than 50 buildings with known floors"}
            continue
        rng = np.random.default_rng(0)
        perm = rng.permutation(known)
        err = {"block_conditioned": [], "block_conditioned_median": [], "prior_median": [], "independent_draw": []}
        for k in range(5):
            hidden = set(perm[k::5].tolist())
            truth = np.array([rows[i]["floors_true"] for i in sorted(hidden)])
            pred = _predict(rows, hidden, priors, row.city, storey_tab[row.city]["value"], np.random.default_rng(k))
            err["block_conditioned"] += list(np.abs(pred[sorted(hidden)] - truth))
            pm = _predict(rows, hidden, priors, row.city, storey_tab[row.city]["value"], np.random.default_rng(k), noise=False)
            err["block_conditioned_median"] += list(np.abs(pm[sorted(hidden)] - truth))
            med = np.array([max(1, round(float(np.interp(0.5, np.linspace(0, 1, len(q)), q))))
                            for q in (floor_quantiles(priors, row.city, rows[i]["_poly"].area) for i in sorted(hidden))])
            err["prior_median"] += list(np.abs(med - truth))
            r2 = np.random.default_rng(100 + k)
            ind = np.array([max(1, round(float(np.interp(r2.uniform(*U_CLIP), np.linspace(0, 1, len(q)), q))))
                            for q in (floor_quantiles(priors, row.city, rows[i]["_poly"].area) for i in sorted(hidden))])
            err["independent_draw"] += list(np.abs(ind - truth))
        # spatial correlation: all buildings synthetic
        all_h = set(range(len(rows)))
        syn = _predict([{**r, "known": False} for r in rows], set(), priors, row.city, storey_tab[row.city]["value"], np.random.default_rng(7))
        r3 = np.random.default_rng(8)
        ind_all = [max(1, round(float(np.interp(r3.uniform(*U_CLIP), np.linspace(0, 1, len(q)), q))))
                   for q in (floor_quantiles(priors, row.city, r["_poly"].area) for r in rows)]
        kr = [r for r in rows if r["known"]]
        res[t["sample_id"]] = {"city": row.city, "n_known": len(known), "n_buildings": len(rows),
                               "mae_floors": {k: float(np.mean(v)) for k, v in err.items()},
                               "within_block_sd_floors": {"observed_known_only": _within_block_sd(kr, [r["floors_true"] for r in kr]),
                                                          "synthetic_block_correlated": _within_block_sd(rows, syn),
                                                          "synthetic_independent": _within_block_sd(rows, ind_all)}}
        del all_h
        print(t["sample_id"], res[t["sample_id"]].get("mae_floors"), flush=True)
    ev = [v for v in res.values() if "mae_floors" in v]
    M = ("block_conditioned", "block_conditioned_median", "prior_median", "independent_draw")
    summ = {m: float(np.median([v["mae_floors"][m] for v in ev])) for m in M}
    wins = {f"{a}_better_than_{b}": int(sum(v["mae_floors"][a] < v["mae_floors"][b] for v in ev))
            for a, b in (("block_conditioned", "prior_median"), ("block_conditioned", "independent_draw"),
                         ("block_conditioned_median", "prior_median"))}
    sd = {k: float(np.median([v["within_block_sd_floors"][k] for v in ev if v["within_block_sd_floors"][k] is not None]))
          for k in ("observed_known_only", "synthetic_block_correlated", "synthetic_independent")}
    return {"tiles": res, "n_tiles_evaluated": len(ev), "median_tile_mae_floors": summ,
            "tile_wins": wins, "median_within_block_sd_floors": sd}


def tile_summary(t):
    d = OUT / t
    v = json.loads((d / "validation.json").read_text())
    city = json.loads((d / "city.json").read_text())
    s, f, e = v["summary"], v["facade"], v.get("environment", {})
    storey = (city["metadata"].get("storey_height_m") or {}).get("value", 3.0)
    ups, gfs = [], []
    for b in city["buildings"]:
        shop = b.get("typology") in ("midrise", "highrise") and b["style"].get("ground_floor") == "shop"
        lv = floor_levels(b["eave_m"], b["floors"], shop, storey)
        gfs.append(lv[0][1] - lv[0][0])
        ups += [z1 - z0 for z0, z1 in lv[1:]]
    ups = np.asarray(ups)
    fam = {}
    for b in city["buildings"]:
        fam.setdefault(b["block"], []).append(b["style"]["facade_family"])
    same = [max(v.count(x) for x in set(v)) / len(v) for v in fam.values() if len(v) >= 3]
    combos = {(b["style"]["facade_family"], b["style"]["palette"], b["style"].get("bay_width_m"), b["style"].get("balconies")) for b in city["buildings"]}
    out = {"kind": city["metadata"]["kind"],
           "geometry": {"buildings": s["n_buildings"], "non_watertight": s["n_buildings"] - s["watertight"],
                        "inward_normals": s["n_buildings"] - s["outward"], "nonmanifold_edges": s["nonmanifold_edges"],
                        "degenerate_tris": s["degenerate_tris"], "floating": s["floating"], "missing_roof": s["missing_roof"],
                        "roof_fallback_to_flat": s["roof_fallback_to_flat"], "intersecting_building_pairs": s["footprint_overlap_pairs_gt_0.5m2"],
                        "buildings_on_road": s["buildings_on_road_gt_1m2"], "road_surface_coverage": s["road_coverage"],
                        "road_down_facing_tris": s["road_down_facing"], "parapets_watertight": f"{f['parapets_watertight']}/{f['parapet_solids']}",
                        "sidewalks_watertight": f"{e.get('sidewalk_watertight')}/{e.get('sidewalk_solids')}"},
           "architecture": {"height_sources": s["height_sources"], "height_m_q10_50_90": s["height_m_quantiles"], "roofs": s["roofs"],
                            "upper_storey_m_q05_50_95": [float(x) for x in np.quantile(ups, [0.05, 0.5, 0.95])] if len(ups) else None,
                            "share_upper_storey_outside_2.4_4.5m": float(np.mean((ups < 2.4) | (ups > 4.5))) if len(ups) else None,
                            "ground_floor_m_median": float(np.median(gfs)),
                            "bay_width_m_q10_50_90": f["bay_width_m_quantiles"], "windows": f["windows"], "shops": f["shops"],
                            "balconies": f["balconies"], "doors": f["doors"], "buildings_with_door": f["buildings_with_door"],
                            "door_bottom_z_max_abs": f["door_bottom_z_max_abs"], "openings_outside_wall": f["openings_outside_wall"],
                            "party_wall_m": f["party_wall_m"], "doors_opening_onto_road": f.get("doors_opening_onto_road"),
                            "doors_facing_sidewalk": f.get("doors_facing_sidewalk"),
                            "block_facade_family_agreement_median": float(np.median(same)) if same else None,
                            "distinct_facade_configs_per_building": len(combos) / max(1, s["n_buildings"])},
           "environment": {k: e.get(k) for k in ("sidewalk_area_m2", "trees", "streetlights", "dashes", "crossings",
                                                 "trunk_or_pole_inside_building", "props_on_road", "canopy_within_80pct_radius_of_building")},
           "triangles": sum(s["triangles"].values()), "timing_python_s": v["timing"]}
    for name in ("blender_timing.json", "godot_check.json", "godot_fps.json", "blender_timing_cycles64.json",
                 "blender_timing_cycles256.json", "blender_timing_eevee256.json"):
        if (d / name).exists():
            out[name.removesuffix(".json")] = json.loads((d / name).read_text())
    return out


def main() -> int:
    priors = json.loads((OUT / "audit" / "height_priors.json").read_text())
    res = {"note": "descriptive; no pre-registered criteria for M11 3D quality", "height_baseline": height_holdout(priors, priors["storey_height_m"])}
    res["tiles"] = {t: tile_summary(t) for k in RENDERED for t in RENDERED[k]}
    comp = []
    for r, g in PAIRS:
        a, b = res["tiles"][r], res["tiles"][g]
        comp.append({"real": r, "generated": g,
                     "buildings": [a["geometry"]["buildings"], b["geometry"]["buildings"]],
                     "height_m_median": [a["architecture"]["height_m_q10_50_90"][1], b["architecture"]["height_m_q10_50_90"][1]],
                     "height_m_q90": [a["architecture"]["height_m_q10_50_90"][2], b["architecture"]["height_m_q10_50_90"][2]],
                     "pitched_share": [(a["architecture"]["roofs"]["gable"] + a["architecture"]["roofs"]["hip"]) / a["geometry"]["buildings"],
                                       (b["architecture"]["roofs"]["gable"] + b["architecture"]["roofs"]["hip"]) / b["geometry"]["buildings"]],
                     "windows_per_building": [a["architecture"]["windows"] / a["geometry"]["buildings"], b["architecture"]["windows"] / b["geometry"]["buildings"]],
                     "doors_opening_onto_road_share": [a["architecture"]["doors_opening_onto_road"] / max(1, a["architecture"]["doors"]),
                                                       b["architecture"]["doors_opening_onto_road"] / max(1, b["architecture"]["doors"])],
                     "party_wall_m_per_building": [a["architecture"]["party_wall_m"] / a["geometry"]["buildings"], b["architecture"]["party_wall_m"] / b["geometry"]["buildings"]],
                     "sidewalk_area_m2": [a["environment"]["sidewalk_area_m2"], b["environment"]["sidewalk_area_m2"]],
                     "trees": [a["environment"]["trees"], b["environment"]["trees"]]})
    res["real_vs_generated"] = comp
    (OUT / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"height": {k: res["height_baseline"][k] for k in ("n_tiles_evaluated", "median_tile_mae_floors",
                                                                          "tile_wins", "median_within_block_sd_floors")},
                      "real_vs_generated": comp}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
