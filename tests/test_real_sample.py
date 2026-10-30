"""Checks on the real Barcelona sample. Offline: uses only files already on disk.

Skipped (not failed) when the raw cache / processed sample have not been
generated yet, so a fresh clone can still run the unit tests.
"""

import numpy as np
import pytest

from urbanmorph.config import BARCELONA_EIXAMPLE_001 as AREA
from urbanmorph.data.dataset import sample_paths, load_sample
from urbanmorph.data.osm import is_cached, load_raw
from urbanmorph.preprocessing.pipeline import build_layout

pytestmark = pytest.mark.cached_data

needs_processed = pytest.mark.skipif(
    not sample_paths(AREA.sample_id)["array"].exists(), reason="run scripts/build_raster.py first"
)
needs_raw = pytest.mark.skipif(not is_cached(AREA), reason="run scripts/download_osm.py first")


@needs_processed
def test_saved_sample_is_valid():
    arr, meta = load_sample(AREA.sample_id)  # validates shape, dtype, values, metadata
    assert arr.shape == (3, 128, 128) and arr.dtype == np.uint8
    for ch in range(3):
        assert arr[ch].any(), f"channel {ch} is empty"


@needs_processed
def test_metadata_geography_is_consistent():
    _, meta = load_sample(AREA.sample_id)
    b = meta["projected_bounds_m"]
    assert b["xmax"] - b["xmin"] == pytest.approx(512.0)
    assert b["ymax"] - b["ymin"] == pytest.approx(512.0)
    assert meta["pixel_size_m"] == pytest.approx(4.0)
    assert meta["projected_crs"] == "EPSG:25831"
    g = meta["geographic_bounds"]["bbox"]
    assert g["south"] < AREA.center_lat < g["north"]
    assert g["west"] < AREA.center_lon < g["east"]
    # 512 m north-south ~ 0.0046 degrees of latitude.
    assert g["north"] - g["south"] == pytest.approx(512 / 111_000, rel=0.05)
    assert "OpenStreetMap" in meta["attribution"]


@needs_raw
@needs_processed
def test_reproducible_from_cache():
    raw = load_raw(AREA)
    rebuilt = build_layout(AREA, raw["edges"], raw["buildings"], raw["green"], road_areas=raw["road_areas"])
    saved, _ = load_sample(AREA.sample_id)
    np.testing.assert_array_equal(rebuilt.layout, saved)
