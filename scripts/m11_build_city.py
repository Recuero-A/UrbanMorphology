"""Milestone 11: one tile -> city.json (urbanmorph-city v0.1) -> meshes.npz + validation.json.

    python scripts/m11_build_city.py --real ESP_BCN_e837_n8952            # a VALIDATION tile (test split refused)
    python scripts/m11_build_city.py --generated 2026 --index 0 --country spain

Generated tiles: GraphTurtle N (N_noise_s0) at T = 0.5 from sampling seed --generated, the --index-th tile with
>= 100 street edges (the same draw as scripts/m10_generate_cities.py), BF parcels and buildings for --country.
Output: outputs/m11/<tile_id>/ (city.json and meshes are git-ignored; validation.json and timing are committed).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from urbanmorph.city3d.convert import from_generated, from_real
from urbanmorph.city3d.format import save_city, validate_city
from urbanmorph.city3d.scene import build_scene, save_scene
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m11"
AUDIT = OUT / "audit"
MIN_EDGES = 100


def build(city: dict, timings: dict, t0: float) -> Path:
    d = OUT / city["metadata"]["tile_id"]
    d.mkdir(parents=True, exist_ok=True)
    err = validate_city(city)
    save_city(city, d / "city.json")
    timings["city_s"] = round(time.time() - t0, 2)
    t1 = time.time()
    groups, mats, rep = build_scene(city)
    timings["mesh_s"] = round(time.time() - t1, 2)
    save_scene(d / "meshes.npz", groups, mats)
    rep["format_errors"] = err
    rep["timing"] = timings
    rep["files_bytes"] = {f.name: f.stat().st_size for f in (d / "city.json", d / "meshes.npz")}
    (d / "validation.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "buildings"}, indent=1))
    (d / "validation_buildings.json").write_text(json.dumps(rep["buildings"]))
    print(json.dumps(rep["summary"], indent=1), "\nformat errors:", err[:5], "\n", timings, flush=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real")
    ap.add_argument("--generated", type=int)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--country", default="spain", choices=["spain", "united_states"])
    ap.add_argument("--seed", type=int, default=0, help="seed of the synthetic attributes (heights, roofs, styles)")
    a = ap.parse_args()
    priors = json.loads((AUDIT / "height_priors.json").read_text())
    storey = priors["storey_height_m"]
    t0 = time.time()
    from m6_build_objects import load_objects  # noqa: E402
    data = load_objects()
    cw = data["class_widths"]
    if a.real:
        from m8_parcel_records import OBS  # noqa: E402
        from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir
        idx = load_index_with_expansion().set_index("sample_id")
        row = idx.loc[a.real]
        if row.split != "val":
            raise SystemExit(f"{a.real} is in split '{row.split}': M11 renders validation tiles only")
        tile = next(t for t in data["val"] if t["sample_id"] == a.real)
        obs = pickle.loads(OBS.read_bytes()).get(a.real)
        parcels = obs["parcels"] if obs and obs.get("status") == "observed" else None
        city = from_real(tile, row, sample_dir(row.country, row.city) / f"{a.real}_vectors.gpkg", cw, priors,
                         storey[row.city], parcels, seed=a.seed)
    else:
        from m10_eval_streets import CK, load_model  # noqa: E402
        from m10_integrate import bf_layouts  # noqa: E402
        from urbanmorph.graphs.generate import sample_graphs
        model = load_model(CK / "m10" / "N_noise_s0")
        graphs = [g for g in sample_graphs(model, 2 * (a.index + 5), seed=a.generated, temperature=0.5) if len(g.edges) >= MIN_EDGES]
        g = graphs[a.index]
        (st, P), = bf_layouts([g], [a.country], cw, seed=0)
        key = {"spain": "spain", "united_states": "united_states"}[a.country]
        sh = {"value": 3.0, "source": "default 3.0 m (generated tile: no city; country-level floor priors)"}
        city = from_generated(f"gen_N_T05_s{a.generated}_{a.index:02d}_{a.country}", st.graph, st, P, cw, key, priors, sh, a.seed,
                              {"streets": "GraphTurtle N_noise_s0, T=0.5, sampling seed %d" % a.generated,
                               "parcels_buildings": "BF (M10), parcel net m8 s0, block nets m9 s0, country %s" % a.country,
                               "heights": "synthetic (city3d.heights, %s floor priors from train tiles)" % key,
                               "roofs_styles": "synthetic (city3d.style)"})
    build(city, {"generation_s": round(time.time() - t0, 2)}, time.time())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
