"""Milestone 2 samples: processing, metadata, storage and loading.

The raster pipeline is Milestone 1's ``build_layout`` unchanged; this module
adds metrics, quality flags and provenance around it.

Layout on disk:
    data/processed/m2/<country>/<city>/<sample_id>.npy            (3,128,128) uint8
    data/processed/m2/<country>/<city>/<sample_id>.json           full metadata
    data/processed/m2/<country>/<city>/<sample_id>_vectors.gpkg   projected, clipped vectors
    data/processed/m2/samples.csv                                 one row per sample
"""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from urbanmorph.config import M2_DIR, PROJECT_ROOT, AreaSpec
from urbanmorph.data.dataset import validate_layout, validate_metadata
from urbanmorph.metrics import morphology as mm
from urbanmorph.preprocessing.pipeline import ProcessedSample, build_layout, export_vectors, make_metadata
from urbanmorph.quality.checks import is_usable, layout_hash, sample_flags
from urbanmorph.sampling.ghsl import GHSL_ATTRIBUTION

INDEX_PATH = M2_DIR / "samples.csv"
SCHEMA_VERSION = "m2.1"


def software_versions() -> dict:
    pkgs = ["urbanmorph", "osmnx", "geopandas", "shapely", "numpy", "pyproj", "rasterio", "networkx", "matplotlib"]
    out = {"python": platform.python_version()}
    for p in pkgs:
        try:
            out[p] = version(p)
        except Exception:
            out[p] = "unknown"
    try:
        out["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                           capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        out["git_commit"] = "unknown"
    return out


def sample_dir(country: str, city: str, root: Path = M2_DIR) -> Path:
    return root / country / city


def count_invalid(*gdfs) -> int:
    return int(sum((~g.geometry.is_valid).sum() for g in gdfs if not g.empty))


def compute_metrics(sample: ProcessedSample, graph, crs: str, cfg: dict) -> dict:
    m = mm.raster_coverage(sample.layout)
    m["building_coverage_vector"] = mm.vector_coverage(sample.buildings, sample.bounds)
    m["green_coverage_vector"] = mm.vector_coverage(sample.green, sample.bounds)
    m["green_polygon_count"] = int(len(sample.green))
    m.update(mm.building_metrics(sample.buildings_full, sample.bounds))
    gp = mm.project_graph(graph, crs) if graph.number_of_nodes() > 0 else graph
    m.update(mm.intersection_metrics(gp, sample.bounds, cfg["metrics"]["intersection_consolidation_tolerance_m"]))
    m.update(mm.orientation_metrics(mm.street_segments(gp, sample.bounds), cfg["metrics"]["orientation_bins"]))
    return m


def process_sample(area: AreaSpec, raw: dict, plan_row: dict, cfg: dict, processing_date: str) -> tuple[np.ndarray, dict, ProcessedSample]:
    """Raw cached OSM layers -> (layout, metadata, processed vectors)."""
    sample = build_layout(area, raw["edges"], raw["buildings"], raw["green"], road_areas=raw["road_areas"])
    sample.buildings_full = raw["buildings"].to_crs(area.projected_crs) if not raw["buildings"].empty else raw["buildings"]
    if not sample.buildings_full.empty:
        sample.buildings_full = sample.buildings_full.set_geometry(sample.buildings_full.geometry.make_valid())

    meta = make_metadata(area, sample, raw["info"], processing_date)
    metrics = compute_metrics(sample, raw["graph"], area.projected_crs, cfg)
    ghsl_built = float(plan_row["ghsl_built_2020"])
    q = cfg["quality"]
    flags = sample_flags(
        sample.layout, meta, expected_crs=plan_row["crs"], expected_pixel_m=cfg["tile"]["size_m"] / cfg["tile"]["resolution"],
        ghsl_built=ghsl_built, osm_building_coverage=metrics["building_coverage_vector"],
        n_invalid_geometries=count_invalid(raw["buildings"], raw["green"], raw["road_areas"]),
        min_osm_to_ghsl_ratio=q["min_osm_to_ghsl_building_ratio"], max_overlap_fraction=q["max_channel_overlap_fraction"],
        min_builtup_fraction=cfg["study_area"]["min_builtup_fraction"],
    )
    meta.update({
        "schema_version": SCHEMA_VERSION,
        "country": area.country,
        "city": area.city,
        "city_code": plan_row["city_code"],
        "grid_index": {"e": int(plan_row["grid_e"]), "n": int(plan_row["grid_n"])},
        "attribution": [meta["attribution"], GHSL_ATTRIBUTION],
        "sampling": {k: plan_row[k] for k in ("role", "rank", "stratum", "density", "era", "p_inclusion",
                                              "ghsl_built_1975", "ghsl_built_2000", "ghsl_built_2020",
                                              "ghsl_nres_share_2020") if k in plan_row},
        "processing_config": {"config_file": Path(cfg["_path"]).name, "config_sha256_12": cfg["_hash"],
                              "tile": cfg["tile"], "metrics": cfg["metrics"], "quality": cfg["quality"]},
        "metrics": metrics,
        "quality_flags": flags,
        "usable": is_usable(flags),
        "layout_sha1": layout_hash(sample.layout),
        "provenance": {"raw_dir": str(area.raw_dir.relative_to(PROJECT_ROOT)),
                       "osm_retrieved_at_utc": raw["info"].get("retrieved_at_utc"),
                       "overpass_url": raw["info"].get("overpass_url"),
                       "software": software_versions(),
                       "note": "Live OSM changes over time; rebuild from the raw cache for identical results."},
        "split": None,
    })
    return sample.layout, meta, sample


def save_m2_sample(layout: np.ndarray, meta: dict, sample: ProcessedSample | None, root: Path = M2_DIR) -> Path:
    validate_layout(layout)
    validate_metadata(meta, layout)
    d = sample_dir(meta["country"], meta["city"], root)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{meta['sample_id']}.npy", layout)
    write_meta(meta, root)
    if sample is not None:
        export_vectors(sample, d / f"{meta['sample_id']}_vectors.gpkg")
    return d / f"{meta['sample_id']}.npy"


def write_meta(meta: dict, root: Path = M2_DIR) -> None:
    d = sample_dir(meta["country"], meta["city"], root)
    (d / f"{meta['sample_id']}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_index(path: Path = INDEX_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


EXPANSION_INDEX_PATH = M2_DIR / "samples_expansion_m3b.csv"


def load_index_with_expansion(path: Path = INDEX_PATH, expansion: Path = EXPANSION_INDEX_PATH) -> pd.DataFrame:
    """Original M2 index plus the Milestone 3b TRAIN-only expansion tiles (val/test unchanged)."""
    base = load_index(path)
    if not expansion.exists():
        raise FileNotFoundError(f"{expansion} missing; run scripts/m3b_build_expansion.py")
    return pd.concat([base, pd.read_csv(expansion)], ignore_index=True)


def load_m2_sample(sample_id: str, index: pd.DataFrame | None = None, root: Path = M2_DIR) -> tuple[np.ndarray, dict]:
    """Load and validate one Milestone 2 sample by ID."""
    index = load_index(root / "samples.csv") if index is None else index
    row = index.loc[index["sample_id"] == sample_id]
    if row.empty:
        raise KeyError(sample_id)
    row = row.iloc[0]
    d = sample_dir(row["country"], row["city"], root)
    layout = np.load(d / f"{sample_id}.npy")
    meta = json.loads((d / f"{sample_id}.json").read_text())
    validate_layout(layout)
    validate_metadata(meta, layout)
    return layout, meta


SCALAR_METRICS = [
    "roads_coverage", "buildings_coverage", "green_spaces_coverage", "building_coverage_vector",
    "green_coverage_vector", "green_polygon_count", "building_count", "building_density_per_ha",
    "building_area_mean_m2", "building_area_median_m2", "building_area_p10_m2", "building_area_p90_m2",
    "intersections_raw", "intersections", "intersection_density_per_km2", "street_length_m",
    "orientation_entropy", "orientation_order_phi", "dominant_bearing_deg", "dominant_axis_strength_r2",
    "grid_orientation_deg", "grid_strength_r4",
]


def index_row(meta: dict) -> dict:
    """Flatten one sample's metadata into a row of the consolidated table."""
    s, g, b = meta["sampling"], meta["geographic_bounds"]["bbox"], meta["projected_bounds_m"]
    row = {
        "sample_id": meta["sample_id"], "country": meta["country"], "city": meta["city"],
        "city_code": meta["city_code"], "role": s.get("role"), "rank": s.get("rank"),
        "stratum": s.get("stratum"), "density": s.get("density"), "era": s.get("era"),
        "p_inclusion": s.get("p_inclusion"), "design_weight": 1 / s["p_inclusion"] if s.get("p_inclusion") else np.nan,
        "ghsl_built_1975": s.get("ghsl_built_1975"), "ghsl_built_2000": s.get("ghsl_built_2000"),
        "ghsl_built_2020": s.get("ghsl_built_2020"), "ghsl_nres_share_2020": s.get("ghsl_nres_share_2020"),
        "crs": meta["projected_crs"], "grid_e": meta["grid_index"]["e"], "grid_n": meta["grid_index"]["n"],
        "xmin": b["xmin"], "ymin": b["ymin"], "xmax": b["xmax"], "ymax": b["ymax"],
        "center_lat": meta["center_wgs84"]["lat"], "center_lon": meta["center_wgs84"]["lon"],
        "bbox_south": g["south"], "bbox_west": g["west"], "bbox_north": g["north"], "bbox_east": g["east"],
        "size_m": meta["physical_size_m"], "resolution_px": meta["resolution_px"], "pixel_size_m": meta["pixel_size_m"],
    }
    row.update({k: meta["metrics"].get(k) for k in SCALAR_METRICS})
    flags = meta["quality_flags"]
    row.update({
        "usable": meta["usable"],
        "flags_error": ";".join(f["code"] for f in flags if f["severity"] == "error"),
        "flags_warning": ";".join(f["code"] for f in flags if f["severity"] == "warning"),
        "flags_info": ";".join(f["code"] for f in flags if f["severity"] == "info"),
        "layout_hash": meta["layout_sha1"], "layout_nonempty": bool(any(meta["statistics"]["coverage_fraction"].values())),
        "split": meta.get("split"), "block_id": meta.get("block_id"),
        "osm_retrieved_at_utc": meta["provenance"]["osm_retrieved_at_utc"],
        "processing_date_utc": meta["processing_date_utc"], "config_sha256_12": meta["processing_config"]["config_sha256_12"],
        "git_commit": meta["provenance"]["software"].get("git_commit"),
        "raw_dir": meta["provenance"]["raw_dir"], "attribution": " | ".join(meta["attribution"]),
    })
    return row
