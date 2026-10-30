"""End-to-end pipeline on SYNTHETIC geometries (no network, not real data)."""

import json

import numpy as np
import pytest

from urbanmorph.data.dataset import LayoutValidationError, load_sample, save_sample, validate_layout
from urbanmorph.preprocessing.pipeline import build_layout, make_metadata

FAKE_INFO = {"source": "synthetic test geometries", "attribution": "none (synthetic)"}


@pytest.fixture
def sample(area, synthetic_layers):
    return build_layout(area, *synthetic_layers)


def test_shape_dtype_binary(sample):
    validate_layout(sample.layout)
    assert sample.layout.shape == (3, 128, 128)
    assert sample.layout.dtype == np.uint8
    assert set(np.unique(sample.layout)) <= {0, 1}


def test_roads_channel(sample):
    roads = sample.layout[0]
    # 8 m wide E-W road centred on the middle => rows 63 and 64, full width.
    assert roads[63].all() and roads[64].all()
    assert roads.sum() == 2 * 128  # the N-S tunnel contributes nothing


def test_buildings_channel(sample):
    buildings = sample.layout[1]
    assert buildings.sum() == 100
    assert buildings[10:20, 10:20].all()


def test_green_channel_and_overlap(sample):
    green = sample.layout[2]
    assert green[113:123, 103:113].all()  # south-east park
    assert green.sum() == 100 + 4  # park + 2x2 roof garden
    # Channels are independent: the roof garden is ALSO a building pixel.
    assert (sample.layout[1] & green).sum() == 4


def test_deterministic(area, synthetic_layers):
    a = build_layout(area, *synthetic_layers).layout
    b = build_layout(area, *synthetic_layers).layout
    np.testing.assert_array_equal(a, b)


def test_bounds_are_metric_square(sample, area):
    assert sample.bounds.width == pytest.approx(area.size_m)
    assert sample.bounds.height == pytest.approx(area.size_m)
    assert area.pixel_size_m == pytest.approx(4.0)


def test_save_load_roundtrip(tmp_path, area, sample):
    meta = make_metadata(area, sample, FAKE_INFO, "2026-01-01T00:00:00+00:00")
    save_sample(sample.layout, meta, area.sample_id, tmp_path)
    arr, meta2 = load_sample(area.sample_id, tmp_path)
    np.testing.assert_array_equal(arr, sample.layout)
    assert meta2["channels"] == {"0": "roads", "1": "buildings", "2": "green_spaces"}
    json.dumps(meta2)  # fully JSON-serialisable


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((3, 64, 64), np.uint8),
        np.zeros((3, 128, 128), np.float32),
        np.full((3, 128, 128), 2, np.uint8),
    ],
)
def test_validation_rejects_bad_arrays(bad):
    with pytest.raises(LayoutValidationError):
        validate_layout(bad)
