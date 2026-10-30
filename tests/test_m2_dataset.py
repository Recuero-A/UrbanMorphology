"""End-to-end Milestone 2 sample processing on SYNTHETIC geometries (no network)."""

import dataclasses

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from urbanmorph.data.m2_dataset import index_row, load_m2_sample, process_sample, save_m2_sample
from urbanmorph.data.osm_extract import STREET_VALUES, _tags_filter_expressions
from urbanmorph.data.osm import combined_tags, match_tags
from urbanmorph.preprocessing.roads import estimate_width, is_tunnel

CFG = {
    "_path": "configs/test.toml", "_hash": "abc",
    "tile": {"size_m": 512.0, "resolution": 128, "download_margin_m": 100.0},
    "study_area": {"min_builtup_fraction": 0.15},
    "quality": {"min_osm_to_ghsl_building_ratio": 0.5, "max_channel_overlap_fraction": 0.05},
    "metrics": {"intersection_consolidation_tolerance_m": 10.0, "orientation_bins": 18},
}


@pytest.fixture
def raw(area, synthetic_layers, bounds):
    edges, buildings, green = synthetic_layers
    G = nx.MultiDiGraph(crs=edges.crs)
    # E-W street with a crossing N-S street at the centre -> 1 intersection (4 streets)
    cx, cy = bounds.xmin + 256, bounds.ymin + 256
    pts = {"w": (bounds.xmin - 50, cy), "c": (cx, cy), "e": (bounds.xmax + 50, cy),
           "n": (cx, bounds.ymax + 50), "s": (cx, bounds.ymin - 50)}
    for k, (x, y) in pts.items():
        G.add_node(k, x=x, y=y)
    for u in "wens":
        G.add_edge(u, "c", length=1.0, oneway=False, reversed=False, osmid=0)
        G.add_edge("c", u, length=1.0, oneway=False, reversed=False, osmid=0)
    return {"graph": G, "edges": edges, "road_areas": edges.iloc[0:0], "buildings": buildings.to_crs(4326),
            "green": green.to_crs(4326), "info": {"source": "synthetic", "attribution": "synthetic",
                                                 "retrieved_at_utc": "2026-01-01T00:00:00Z"}}


@pytest.fixture
def plan_row(area, bounds):
    return {"sample_id": "SYN_e0_n0", "city_code": "SYN", "crs": area.projected_crs, "grid_e": 0, "grid_n": 0,
            "role": "primary", "rank": 0, "stratum": "high|by1975", "density": "high", "era": "by1975",
            "p_inclusion": 0.1, "ghsl_built_1975": 0.2, "ghsl_built_2000": 0.2, "ghsl_built_2020": 0.02,
            "ghsl_nres_share_2020": 0.1}


def test_process_save_load_roundtrip(tmp_path, area, raw, plan_row):
    area = dataclasses.replace(area, sample_id="SYN_e0_n0", country="testland", city="synthville")
    layout, meta, sample = process_sample(area, raw, plan_row, CFG, "2026-01-01T00:00:00+00:00")
    assert layout.shape == (3, 128, 128) and layout.dtype == np.uint8
    m = meta["metrics"]
    assert m["building_count"] == 1 and m["building_area_mean_m2"] == pytest.approx(1600.0)
    assert m["intersections"] == 1
    assert m["orientation_order_phi"] == pytest.approx(1.0)
    assert meta["usable"] is True
    for key in ("country", "city", "geographic_bounds", "projected_bounds_m", "projected_crs", "pixel_size_m",
                "physical_size_m", "attribution", "processing_config", "metrics", "quality_flags", "provenance",
                "sampling", "split"):
        assert key in meta
    meta["split"] = "train"
    save_m2_sample(layout, meta, None, root=tmp_path)
    row = index_row(meta)
    pd.DataFrame([row]).to_csv(tmp_path / "samples.csv", index=False)
    lay2, meta2 = load_m2_sample("SYN_e0_n0", root=tmp_path)
    np.testing.assert_array_equal(lay2, layout)
    assert meta2["split"] == "train" and row["usable"] and row["design_weight"] == pytest.approx(10.0)


def test_combined_tags_split_back_into_layers():
    tags = combined_tags()
    assert tags["building"] is True and "park" in tags["leisure"] and "pedestrian" in tags["highway"]
    exprs = _tags_filter_expressions(tags)
    assert "nwr/building" in exprs and any(e.startswith("nwr/leisure=") for e in exprs)
    assert "trunk_link" in STREET_VALUES and "residential" in STREET_VALUES and "footway" not in STREET_VALUES
    gdf = pd.DataFrame({"building": ["yes", None, None], "leisure": [None, "park", None],
                        "highway": [None, None, "pedestrian"]})
    assert match_tags(gdf, {"building": True}).index.tolist() == [0]
    assert match_tags(gdf, {"leisure": ["park"]}).index.tolist() == [1]


def test_merged_edge_rules_are_order_independent():
    a = {"highway": "residential", "lanes": "['2', '3']", "width": "7"}
    b = {"highway": "residential", "lanes": "['3', '2']", "width": "7"}
    assert estimate_width(a) == estimate_width(b) == (9.75, "lanes_tag")
    assert estimate_width({"highway": "['residential', 'primary']"}) == estimate_width({"highway": "['primary', 'residential']"})
    assert is_tunnel({"tunnel": "['yes', 'yes']"}) and not is_tunnel({"tunnel": "['yes', 'no']"})
