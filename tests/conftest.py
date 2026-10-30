import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from urbanmorph.config import AreaSpec
from urbanmorph.preprocessing.projection import square_bounds

# Same centre/CRS as the real sample, but the geometries below are synthetic.
SYNTHETIC_AREA = AreaSpec(sample_id="synthetic_test", center_lat=41.3910, center_lon=2.1650)
CRS = SYNTHETIC_AREA.projected_crs


@pytest.fixture
def area():
    return SYNTHETIC_AREA


@pytest.fixture
def bounds():
    return square_bounds(SYNTHETIC_AREA)


@pytest.fixture
def synthetic_layers(bounds):
    """One E-W road, one N-S tunnel, one building, one park, one park on a roof.

    Coordinates are chosen so every polygon edge falls between pixel centres
    (pixel centres sit at 2, 6, 10, ... metres from the square's edges), which
    makes the expected pixel counts exact.
    """
    b = bounds
    cy = b.ymin + 256
    cx = b.xmin + 256
    edges = gpd.GeoDataFrame(
        {
            "highway": ["residential", "primary"],  # residential default = 8 m
            "width": [None, None],
            "lanes": [None, None],
            "tunnel": [None, "yes"],
        },
        geometry=[
            LineString([(b.xmin - 50, cy), (b.xmax + 50, cy)]),
            LineString([(cx, b.ymin - 50), (cx, b.ymax + 50)]),
        ],
        crs=CRS,
    )
    # 40 m x 40 m building: columns 10..19, rows 10..19 (north-west area).
    buildings = gpd.GeoDataFrame(
        {"building": ["yes"]},
        geometry=[box(b.xmin + 40, b.ymax - 80, b.xmin + 80, b.ymax - 40)],
        crs=CRS,
    )
    # 40 m x 40 m park in the south-east, plus a roof garden on the building.
    green = gpd.GeoDataFrame(
        {"leisure": ["park", "garden"]},
        geometry=[
            box(b.xmax - 100, b.ymin + 20, b.xmax - 60, b.ymin + 60),
            box(b.xmin + 40, b.ymax - 48, b.xmin + 48, b.ymax - 40),  # 2x2 px on the roof
        ],
        crs=CRS,
    )
    return edges, buildings, green
