"""Milestone 16, part 1: terrain units, resolution, interpolation and slope calculations against analytic terrain.

Every quantity the samplers, the model inputs and the evaluator derive from elevation is checked on planes with a
known grade and azimuth, plus a geographic (EPSG:4326) GeoTIFF reprojected to UTM the way GLO-30 is.
"""
import json
import math

import numpy as np
import pytest

from urbanmorph.graphs import fixtures as fx
from urbanmorph.graphs.road_graph import RoadGraph
from urbanmorph.graphs.style_metrics import grade_exposure
from urbanmorph.graphs.tc_potentials import TerrainField, edge_features
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.graphs.terrain_sequence import terrain_crops, tile_layers

NORM = {"elev_scale": 10.0, "slope_scale": 0.1}
CASES = [(g, az, rel) for g in (0.04, 0.08, 0.15, 0.25) for az in (0.0, 30.0, 117.0) for rel in (0.0, 45.0, 90.0, 180.0)]


def plane(g, az):
    return terrain_of(fx.terrain("incline", grade=g, az=az))


def unit(deg):
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


@pytest.mark.parametrize("g,az,rel", CASES)
def test_edge_grade_slope_alignment_on_planes(g, az, rel):
    """Directional grade = g |cos(rel)|, terrain slope = g, alignment with the gradient = |cos(rel)|."""
    tf = TerrainField(plane(g, az))
    a = np.array([256.0, 256.0])
    q = a + 40.0 * unit(az + rel)
    f = edge_features(tf, a, q)
    c = abs(math.cos(math.radians(rel)))
    assert f["grade"] == pytest.approx(g * c, abs=1e-4)
    assert f["mean_grade"] == pytest.approx(g * c, abs=1e-4)
    assert f["slope"] == pytest.approx(g, rel=1e-3)
    assert f["align"] == pytest.approx(c, abs=1e-3)


@pytest.mark.parametrize("g,az,rel", CASES[::3])
def test_evaluator_grade_exposure_on_planes(g, az, rel):
    a = np.array([256.0, 256.0])
    g_ = RoadGraph(np.array([a, a + 100.0 * unit(az + rel)]), np.array([[0, 1]]), np.zeros(2, bool),
                   edge_class=np.array([2]), edge_struct=np.array([0]))
    e = grade_exposure(g_, plane(g, az), span=10.0)
    d = g * abs(math.cos(math.radians(rel)))
    assert e["grade_p90"] == pytest.approx(d, abs=1e-4)
    assert e["share_gt15"] == (1.0 if d > 0.15 + 1e-9 else 0.0)
    assert e["ground_km"] == pytest.approx(0.1, abs=1e-6)


@pytest.mark.parametrize("g,az", [(0.08, 0.0), (0.15, 30.0), (0.25, 117.0)])
def test_model_slope_channel_and_crop_orientation(g, az):
    """Slope channel = g / slope_scale; the elevation crop rises AHEAD (row 0) when the turtle heads uphill, falls
    when it heads downhill, and rises to the LEFT (column 0) when uphill is on its left."""
    d = fx.terrain("incline", grade=g, az=az)
    lay = tile_layers(d, 0, ["elev", "slope", "water"], NORM)
    assert np.median(lay["slope"]) == pytest.approx(min(g / NORM["slope_scale"], 4.0), rel=1e-3)
    pos = np.array([[256.0, 256.0]])
    for heading, sign_rows, sign_cols in ((az, +1, 0), (az + 180, -1, 0), (az - 90, 0, +1)):
        cr = terrain_crops(lay, ["elev", "slope", "water"], pos, np.radians([heading]), NORM)[0, 0].astype(float)
        ahead, behind = cr[:12].mean(), cr[-12:].mean()
        left, right = cr[:, :12].mean(), cr[:, -12:].mean()
        if sign_rows:
            assert np.sign(ahead - behind) == sign_rows
            assert abs(left - right) < 0.05 * abs(ahead - behind)
        if sign_cols:
            assert left > right
            assert abs(ahead - behind) < 0.05 * abs(left - right)
        # 36 px (4 m each) between the centres of the 12-row bands: elevation difference = g * 144 m / elev_scale
        if sign_rows:
            assert abs(ahead - behind) == pytest.approx(min(g * 36 * 4 / NORM["elev_scale"], 6), rel=0.02)


def test_geographic_geotiff_reprojects_to_true_grade(tmp_path, monkeypatch):
    """A plane stored in EPSG:4326 (as GLO-30 is), reprojected bilinearly to UTM 52N on the 4 m tile grid, keeps its
    grade: horizontal units (degrees -> metres) and vertical units (metres) are handled by the loader."""
    rasterio = pytest.importorskip("rasterio")
    from pyproj import Transformer
    from rasterio.transform import from_origin
    import urbanmorph.terrain.data as data
    g, lat0, lon0 = 0.12, 36.37, 127.36                       # KAIST, Daejeon
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    res = 1 / 3600                                             # 1 arc-second, GLO-30 latitude spacing
    n = 3600 // 60                                             # 1 arc-minute square
    lon = lon0 - 0.5 / 60 + (np.arange(n) + 0.5) * res
    z = 100 + g * (lon - lon0) * m_per_deg_lon                 # rises eastward at grade g
    arr = np.tile(z, (n, 1)).astype(np.float32)
    d = tmp_path / "plane"
    d.mkdir()
    with rasterio.open(d / "x.tif", "w", driver="GTiff", height=n, width=n, count=1, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(lon0 - 0.5 / 60, lat0 + 0.5 / 60, res, res), nodata=-9999) as dst:
        dst.write(arr, 1)
    (d / "x.json").write_text(json.dumps({"provider": "test", "native_resolution_m": 30.0}))
    monkeypatch.setattr(data, "DEM_DIR", tmp_path)
    E, N = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True).transform(lon0, lat0)
    t = data.tile_terrain("plane", "x", "EPSG:32652", E - 256, N - 256, margin=0.0)
    gx, gy = t.gradient()
    inner = (slice(8, -8), slice(8, -8))
    s = np.hypot(gx, gy)[inner]
    assert np.nanmedian(s) == pytest.approx(g, rel=0.02)        # UTM grid north differs from true north by ~1 deg
    assert np.degrees(np.arctan2(np.nanmedian(gy[inner]), np.nanmedian(gx[inner]))) == pytest.approx(0.0, abs=3.0)
