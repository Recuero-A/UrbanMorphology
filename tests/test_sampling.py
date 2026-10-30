"""Sampling: grid, GHSL reading, strata, allocation, selection reproducibility (synthetic)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from urbanmorph.sampling import ghsl
from urbanmorph.sampling.grid import candidate_tiles, main_component, sample_id
from urbanmorph.sampling.select import allocate, city_rng, select_tiles
from urbanmorph.sampling.strata import density_class, era_class, weighted_quantiles

UTM = "EPSG:32631"


def test_candidate_tiles_snap_to_grid_and_lie_inside():
    region = gpd.GeoDataFrame(geometry=[box(1000, 1000, 3100, 2600)], crs=UTM).to_crs(4326)
    tiles = candidate_tiles(region, UTM, 512.0)
    # x: cells 2..5 fit fully in [1000, 3100]; y: cells 2..4 fit in [1000, 2600]
    assert sorted(tiles["grid_e"].unique()) == [2, 3, 4, 5]
    assert sorted(tiles["grid_n"].unique()) == [2, 3, 4]
    assert np.allclose(tiles["xmin"] % 512, 0) and np.allclose(tiles["ymin"] % 512, 0)
    assert (tiles["xmax"] - tiles["xmin"] == 512).all()
    assert sample_id("ESP_BCN", 3, 4) == "ESP_BCN_e3_n4"


def test_main_component_bridges_small_gaps_only():
    # cluster A: 5 cells in a row; cluster B: 2 cells 3 empty cells away (gap too big)
    e = [0, 1, 2, 3, 4, 8, 9]
    n = [0] * 7
    keep = main_component(e, n, max_gap_cells=2)
    assert keep.tolist() == [True] * 5 + [False] * 2
    # with a 3-cell gap allowed, everything is one cluster
    assert main_component(e, n, max_gap_cells=3).all()
    # keeping clusters >= 40% of the largest keeps the 2-cell cluster (2/5), 50% does not
    assert main_component(e, n, max_gap_cells=2, min_fraction=0.4).all()
    assert main_component(e, n, max_gap_cells=2, min_fraction=0.5).tolist() == [True] * 5 + [False] * 2


def test_tile_index_matches_verified_ghsl_tiles():
    # Verified against downloaded GHSL tile bounds: R5_C19 = x [-41 km, 959 km], y [4000 km, 5000 km]
    assert ghsl.tile_index(181_901, 4_944_167) == (5, 19)   # Barcelona
    assert ghsl.tile_index(-32_428, 4_730_072) == (5, 19)   # Valencia, 8.6 km inside C19
    assert ghsl.tile_index(-41_001, 4_730_072) == (5, 18)


def test_builtup_fraction_on_synthetic_raster(tmp_path):
    # 4x4 Mollweide raster of 100 m cells; left half fully built (10,000 m2), right half 2,500 m2
    data = np.array([[10000, 10000, 2500, 2500]] * 4, dtype=np.uint16)
    path = tmp_path / "built.tif"
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint16",
                       crs=ghsl.MOLLWEIDE, transform=from_origin(0, 400, 100, 100), nodata=65535) as dst:
        dst.write(data, 1)
    squares = np.array([[0, 0, 200, 400], [200, 0, 400, 400], [0, 0, 400, 400]])
    frac = ghsl.builtup_fraction(path, squares, ghsl.MOLLWEIDE, samples_per_side=8)
    assert frac == pytest.approx([1.0, 0.25, 0.625])


def test_density_and_era_classes():
    assert density_class([0.1, 0.3, 0.5], [0.27, 0.37]).tolist() == ["low", "medium", "high"]
    df = pd.DataFrame({"ghsl_built_1975": [0.2, 0.0, 0.0], "ghsl_built_2000": [0.3, 0.2, 0.0],
                       "ghsl_built_2020": [0.4, 0.4, 0.2]})
    assert era_class(df, [1975, 2000, 2020], 0.15).tolist() == ["by1975", "1975_2000", "2000_2020"]


def test_weighted_quantiles_equal_city_weighting():
    # city A: 3 cells at 0.2; city B: 1 cell at 0.8. Unweighted, the median is
    # 0.2 (city A dominates); with equal city weights it moves between the cities.
    v = np.array([0.2, 0.2, 0.2, 0.8])
    w = np.array([1 / 3, 1 / 3, 1 / 3, 1.0])
    assert weighted_quantiles(v, np.ones(4), [0.5])[0] == pytest.approx(0.2)
    assert 0.2 < weighted_quantiles(v, w, [0.5])[0] < 0.8


def test_allocate_minimum_one_and_caps():
    alloc = allocate({"a": 100, "b": 10, "c": 1}, 15)
    assert sum(alloc.values()) == 15
    assert alloc["c"] == 1 and alloc["b"] >= 1
    assert allocate({"a": 2, "b": 1}, 15) == {"a": 2, "b": 1}  # capped by stratum sizes


def _cands(n_side=30, size=512.0):
    e, nn = np.meshgrid(np.arange(n_side), np.arange(n_side))
    e, nn = e.ravel(), nn.ravel()
    strata = np.where(e < n_side // 2, "low|by1975", "high|2000_2020")
    return pd.DataFrame({"grid_e": e, "grid_n": nn, "xmin": e * size, "ymin": nn * size,
                         "xmax": (e + 1) * size, "ymax": (nn + 1) * size, "stratum": strata})


def test_selection_reproducible_and_separated():
    c = _cands()
    a = select_tiles(c, 15, 5, 1536.0, city_rng(42, "TEST"))
    b = select_tiles(c, 15, 5, 1536.0, city_rng(42, "TEST"))
    other = select_tiles(c, 15, 5, 1536.0, city_rng(43, "TEST"))
    assert a["grid_e"].tolist() == b["grid_e"].tolist() and a["grid_n"].tolist() == b["grid_n"].tolist()
    assert a["grid_e"].tolist() != other["grid_e"].tolist()
    assert (a["role"] == "primary").sum() == 15 and (a["role"] == "reserve").sum() == 5
    cx = (a.xmin + a.xmax).to_numpy() / 2
    cy = (a.ymin + a.ymax).to_numpy() / 2
    d = np.hypot(cx[:, None] - cx[None], cy[:, None] - cy[None])
    assert d[~np.eye(len(a), dtype=bool)].min() >= 1536.0
    prim = a[a.role == "primary"]
    assert set(prim["stratum"]) == {"low|by1975", "high|2000_2020"}
    # p_inclusion = realised n_h / N_h
    for h, grp in prim.groupby("stratum"):
        assert grp["p_inclusion"].iloc[0] == pytest.approx(len(grp) / (c["stratum"] == h).sum())
