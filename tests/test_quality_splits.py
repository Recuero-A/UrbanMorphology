"""Quality flags, dataset-level checks, geographic splits (synthetic)."""

import numpy as np
import pandas as pd
import pytest

from urbanmorph.quality.checks import dataset_flags, is_usable, sample_flags
from urbanmorph.splits.geographic import (EXCLUDED, assign_splits, city_holdout, edge_distance, enforce_buffer,
                                          split_violations)

CRS = "EPSG:32631"


def meta(**over):
    m = {"sample_id": "X", "source": "s", "attribution": "a", "geographic_bounds": {}, "projected_crs": CRS,
         "projected_bounds_m": {"xmin": 0, "ymin": 0, "xmax": 512, "ymax": 512}, "physical_size_m": 512.0,
         "resolution_px": 128, "pixel_size_m": 4.0, "channels": {"0": "roads", "1": "buildings", "2": "green_spaces"},
         "road_buffer": {}, "processing_date_utc": "2026-01-01"}
    m.update(over)
    return m


def layout(roads=True, buildings=True, green=True):
    a = np.zeros((3, 128, 128), np.uint8)
    a[0, :10] = roads
    a[1, 20:40] = buildings
    a[2, 50:60] = green
    return a


def flags_for(lay, m=None, ghsl=0.3, osm_cov=0.3, invalid=0):
    return sample_flags(lay, m or meta(), CRS, 4.0, ghsl, osm_cov, invalid, 0.5, 0.05, 0.15)


def codes(flags, sev=None):
    return {f["code"] for f in flags if sev is None or f["severity"] == sev}


def test_clean_sample_has_no_flags():
    assert flags_for(layout()) == []


def test_empty_green_is_info_not_error():
    f = flags_for(layout(green=False))
    assert codes(f) == {"green_unmapped_or_absent"} and codes(f, "info") == codes(f)
    assert is_usable(f)


def test_missing_core_in_builtup_area_is_error():
    f = flags_for(layout(roads=False, buildings=False), ghsl=0.4)
    assert "osm_core_missing" in codes(f, "error") and not is_usable(f)
    # same emptiness where satellites see little built-up: only a warning
    assert "osm_core_missing" in codes(flags_for(layout(roads=False, buildings=False), ghsl=0.05), "warning")


def test_incomplete_buildings_warning():
    assert "buildings_incomplete_suspected" in codes(flags_for(layout(), ghsl=0.4, osm_cov=0.1), "warning")
    # zero mapped buildings where satellites see 38% built-up: both flags
    f = flags_for(layout(buildings=False), ghsl=0.38, osm_cov=0.0)
    assert {"no_buildings", "buildings_incomplete_suspected"} <= codes(f, "warning")


def test_crs_resolution_and_raster_errors():
    assert "crs_unexpected" in codes(flags_for(layout(), meta(projected_crs="EPSG:32630")), "error")
    assert "pixel_resolution_inconsistent" in codes(flags_for(layout(), meta(pixel_size_m=8.0)), "error")
    assert "raster_invalid" in codes(flags_for(np.zeros((3, 64, 64), np.uint8)), "error")
    assert "metadata_invalid" in codes(flags_for(layout(), {"projected_crs": CRS, "pixel_size_m": 4.0}), "error")


def test_overlap_and_invalid_geometry_flags():
    lay = layout()
    lay[2, 20:40] = 1  # green on top of all buildings: 20/128 = 15.6% overlap
    assert "excessive_channel_overlap" in codes(flags_for(lay), "warning")
    assert "invalid_geometries_repaired" in codes(flags_for(layout(), invalid=3), "info")


def tiles(rows):
    """rows: (sample_id, city, grid_e, grid_n)"""
    out = []
    for sid, city, e, n in rows:
        out.append({"sample_id": sid, "city_code": city, "crs": CRS, "grid_e": e, "grid_n": n,
                    "xmin": e * 512.0, "ymin": n * 512.0, "xmax": (e + 1) * 512.0, "ymax": (n + 1) * 512.0,
                    "center_lat": 41.0 + n * 0.0046 + (0 if city == "A" else 5), "center_lon": 2.0 + e * 0.006,
                    "layout_hash": sid, "layout_nonempty": True})
    return pd.DataFrame(out)


def test_dataset_flags_overlap_duplicates_identical_rasters():
    t = tiles([("a", "A", 0, 0), ("b", "A", 0, 0), ("c", "A", 5, 5), ("c", "A", 9, 9), ("d", "A", 20, 20)])
    t.loc[t.sample_id == "d", "layout_hash"] = "a"
    f = dataset_flags(t, 512.0)
    assert {x["code"] for x in f["a"]} == {"tile_overlap", "duplicate_raster"}
    assert "duplicate_sample_id" in {x["code"] for x in f["c"]}
    assert {x["code"] for x in f["d"]} == {"duplicate_raster"}


def test_edge_distance():
    t = tiles([("a", "A", 0, 0), ("b", "A", 1, 0), ("c", "A", 3, 0), ("d", "A", 3, 4)])
    r = t.set_index("sample_id")
    assert edge_distance(r.loc["a"], r.loc["b"]) == 0.0          # touching
    assert edge_distance(r.loc["a"], r.loc["c"]) == 1024.0       # two empty cells between
    assert edge_distance(r.loc["a"], r.loc["d"]) == pytest.approx(np.hypot(1024, 1536))


def grid_tiles(n_side=12, step=3):
    rows = [(f"A_{e}_{n}", "A", e, n) for e in range(0, n_side * step, step) for n in range(0, n_side * step, step)]
    return tiles(rows)


def test_splits_deterministic_blockwise_and_buffered():
    t = grid_tiles()
    fr = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = assign_splits(t, fr, 3072.0, seed=1)
    b = assign_splits(t, fr, 3072.0, seed=1)
    assert a["split"].tolist() == b["split"].tolist()
    assert set(a["split"]) == {"train", "val", "test"}
    # every block is in exactly one split
    assert (a.groupby("block_id")["split"].nunique() == 1).all()
    # tiles 3 cells apart are 1024 m edge-to-edge -> 2000 m buffer forces exclusions
    buffered = enforce_buffer(a, 2000.0)
    assert (buffered["split"] == EXCLUDED).any()
    assert split_violations(buffered, 2000.0) == []
    assert split_violations(a, 2000.0) != []  # the check really detects violations
    # exclusions never come out of the test split
    assert (buffered.loc[a["split"] == "test", "split"] == "test").all()


def test_city_holdout():
    t = pd.concat([tiles([("a1", "A", 0, 0), ("a2", "A", 5, 5)]), tiles([("b1", "B", 0, 0)])])
    t["usable"] = [True, False, True]
    train, test = city_holdout(t, "B")
    assert train["sample_id"].tolist() == ["a1"] and test["sample_id"].tolist() == ["b1"]
