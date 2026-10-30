"""Checks on the real Milestone 2 pilot dataset. Offline; skipped if not built yet."""

import numpy as np
import pandas as pd
import pytest

from urbanmorph.data.m2_dataset import INDEX_PATH, load_index, load_m2_sample
from urbanmorph.quality.checks import dataset_flags
from urbanmorph.sampling.plan_io import PLAN_PATH, load_plan
from urbanmorph.sampling.study_area import load_config
from urbanmorph.splits.geographic import SPLITS, split_violations

pytestmark = [pytest.mark.cached_data,
              pytest.mark.skipif(not INDEX_PATH.exists(), reason="run scripts/m2_build_dataset.py first")]

REQUIRED_COLUMNS = ["sample_id", "country", "city", "bbox_south", "bbox_north", "xmin", "ymax", "crs",
                    "pixel_size_m", "size_m", "attribution", "osm_retrieved_at_utc", "config_sha256_12",
                    "usable", "split", "stratum", "p_inclusion", "raw_dir", "intersection_density_per_km2"]


@pytest.fixture(scope="module")
def index():
    return load_index()


def test_index_schema_and_identity(index):
    assert set(REQUIRED_COLUMNS) <= set(index.columns)
    assert index["sample_id"].is_unique
    assert set(index["country"]) == {c["country"] for c in load_config()["cities"]}
    assert (index["pixel_size_m"] == 4.0).all() and (index["size_m"] == 512.0).all()
    assert index["attribution"].str.contains("OpenStreetMap").all()
    assert index["attribution"].str.contains("GHS-BUILT-S").all()


def test_every_dataset_sample_loads_and_validates(index):
    ds = index[index["in_dataset"]]
    for sid in ds["sample_id"]:
        layout, meta = load_m2_sample(sid, index)
        assert layout.shape == (3, 128, 128) and layout.dtype == np.uint8
        assert set(np.unique(layout)) <= {0, 1}
        assert meta["split"] in SPLITS
        assert meta["projected_crs"] == index.loc[index.sample_id == sid, "crs"].iloc[0]


def test_real_splits_respect_buffer_and_blocks(index):
    cfg = load_config()
    ds = index[index["in_dataset"]]
    assert split_violations(ds, cfg["splits"]["buffer_m"]) == []
    assert (ds.groupby("block_id")["split"].nunique() == 1).all()
    for _, grp in ds.groupby("city_code"):
        assert set(grp["split"]) == set(SPLITS)


def test_no_overlaps_or_duplicates(index):
    ds = index[index["in_dataset"]]
    flags = dataset_flags(ds, 512.0)
    assert not any(flags.values()), {k: v for k, v in flags.items() if v}


def test_tiles_are_grid_aligned(index):
    assert np.allclose(index["xmin"] % 512, 0) and np.allclose(index["ymin"] % 512, 0)
    assert np.allclose(index["xmax"] - index["xmin"], 512)


@pytest.mark.skipif(not PLAN_PATH.exists(), reason="no sampling plan")
def test_sampling_plan_is_reproducible():
    """Re-running selection for one city from the cached boundary + GHSL gives identical tiles."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("plan", Path(__file__).parents[1] / "scripts" / "m2_plan_sampling.py")
    plan_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan_mod)
    from urbanmorph.sampling import strata as st
    from urbanmorph.sampling.select import city_rng, select_tiles
    from urbanmorph.sampling.study_area import cities

    cfg = load_config()
    city = next(c for c in cities(cfg) if c.code == "ESP_VLC")
    _, _, cands = plan_mod.candidates_for(city, cfg)
    cands["density"] = st.density_class(cands["ghsl_built_2020"], cfg["strata"]["density_thresholds"])
    cands["era"] = st.era_class(cands, cfg["study_area"]["ghsl_epochs"], cfg["strata"]["era_builtup_fraction"])
    cands["stratum"] = cands["density"] + "|" + cands["era"]
    el = cands[cands["eligible"]].reset_index(drop=True)
    sel = cfg["selection"]
    again = select_tiles(el, sel["per_city"], sel["reserves_per_city"], sel["min_center_distance_m"],
                         city_rng(cfg["seed"], "ESP_VLC"))
    stored = load_plan()
    stored = stored[stored.city_code == "ESP_VLC"].sort_values("rank")
    assert list(zip(again.grid_e, again.grid_n)) == list(zip(stored.grid_e, stored.grid_n))


def test_explorer_renders_headless(tmp_path, index):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("explorer", Path(__file__).parents[1] / "scripts" / "m2_explorer.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    ids = index[index["in_dataset"]]["sample_id"].tolist()
    out = ex.save_comparison(ids[0], ids[1], tmp_path / "cmp.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_tiles_rebuild_bit_identical_from_raw_cache(index):
    """One tile per country: reprocessing the cached raw OSM data reproduces the saved array exactly."""
    from urbanmorph.data.m2_dataset import process_sample
    from urbanmorph.data.osm import is_cached, load_raw
    from urbanmorph.sampling.plan_io import area_from_row

    cfg = load_config()
    plan = load_plan()
    ds = index[index["in_dataset"]]
    for country in sorted(set(ds["country"])):
        sid = ds[ds.country == country]["sample_id"].iloc[0]
        row = plan[plan.sample_id == sid].iloc[0]
        area = area_from_row(row, cfg)
        if not is_cached(area):
            pytest.skip("raw OSM cache absent")
        rebuilt, meta, _ = process_sample(area, load_raw(area), row.to_dict(), cfg, "rebuild-test")
        saved, saved_meta = load_m2_sample(sid, index)
        np.testing.assert_array_equal(rebuilt, saved)
        assert meta["layout_sha1"] == saved_meta["layout_sha1"]
