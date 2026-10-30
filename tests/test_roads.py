import geopandas as gpd
import pytest
from shapely.geometry import LineString

from urbanmorph.preprocessing.roads import buffer_roads, estimate_width


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"highway": "residential", "width": "12"}, (12.0, "width_tag")),
        ({"highway": "residential", "width": "12 m", "lanes": "2"}, (12.0, "width_tag")),
        ({"highway": "tertiary", "width": "5", "lanes": "4"}, (13.0, "lanes_tag")),  # contradiction
        ({"highway": "tertiary", "lanes": "2"}, (6.5, "lanes_tag")),
        ({"highway": "primary"}, (14.0, "class_default")),
        ({"highway": "primary_link"}, (6.0, "class_default")),
        ({"highway": '["living_street", "pedestrian"]'}, (6.0, "class_default")),  # merged edge
        ({"highway": "residential", "width": "500"}, (8.0, "class_default")),  # implausible tag
        ({"highway": "something_new"}, (7.0, "class_default")),
    ],
)
def test_estimate_width(row, expected):
    assert estimate_width(row) == expected


def test_buffer_excludes_tunnels_and_has_expected_area():
    edges = gpd.GeoDataFrame(
        {"highway": ["residential", "residential"], "tunnel": [None, "yes"]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)])],
        crs="EPSG:25831",
    )
    out = buffer_roads(edges)
    assert len(out) == 1
    # 100 m x 8 m rectangle + two semicircular caps of radius 4 m.
    assert out.geometry.area.iloc[0] == pytest.approx(800 + 3.14159 * 16, rel=1e-2)


def test_buffer_refuses_geographic_crs():
    edges = gpd.GeoDataFrame(
        {"highway": ["residential"]}, geometry=[LineString([(2.16, 41.39), (2.17, 41.39)])], crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="projected"):
        buffer_roads(edges)
