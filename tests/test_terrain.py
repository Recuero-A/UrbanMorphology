"""Terrain heightfield (Milestone 12): analytic cases."""
import numpy as np

from urbanmorph.terrain.heightfield import Terrain, grade_stats


def plane(gx=0.0, gy=0.0, z0=100.0, n=64, res=4.0):
    """z = z0 + gx * x + gy * y on a grid whose south-west corner is (0, 0)."""
    xc = (np.arange(n) + 0.5) * res
    yc = n * res - (np.arange(n) + 0.5) * res
    return Terrain((z0 + gx * xc[None, :] + gy * yc[:, None]).astype(np.float32), 0.0, n * res, res)


def test_flat_terrain_has_zero_slope():
    t = plane()
    assert np.allclose(t.slope(), 0)
    d, z = t.profile((20, 20), (200, 150))
    assert grade_stats(d, z)["max_grade"] < 1e-12


def test_uniform_slope_gives_the_expected_grade_in_every_direction():
    t = plane(gx=0.06, gy=-0.08)                                 # |grad| = 0.10
    assert np.allclose(t.slope(), 0.10, atol=1e-5)
    gx, gy = t.gradient()
    assert np.allclose(gx, 0.06, atol=1e-5) and np.allclose(gy, -0.08, atol=1e-5)
    for ang in np.radians([0, 37, 90, 145, 230]):
        u = np.array([np.cos(ang), np.sin(ang)])
        a = np.array([128.0, 128.0])
        d, z = t.profile(a, a + 80 * u)
        s = grade_stats(d, z)
        assert abs(s["mean_grade"] - abs(0.06 * u[0] - 0.08 * u[1])) < 1e-4
        assert abs(s["max_grade"] - s["mean_grade"]) < 1e-4


def test_profile_sees_a_bump_that_endpoints_miss():
    t = plane()
    xc = (np.arange(64) + 0.5) * 4
    t.elev += (5 * np.exp(-((xc - 128) / 12) ** 2))[None, :].astype(np.float32)   # a 5 m ridge across x = 128
    d, z = t.profile((60, 100), (196, 100))
    s = grade_stats(d, z)
    assert abs(s["mean_grade"]) < 1e-3 and s["max_grade"] > 0.2


def test_bilinear_sampling_is_exact_on_planes_and_nan_outside():
    t = plane(gx=0.03, gy=0.01)
    x, y = np.array([10.3, 77.7, 200.1]), np.array([50.0, 13.2, 240.9])
    assert np.allclose(t.sample(x, y), 100 + 0.03 * x + 0.01 * y, atol=1e-4)
    assert np.isnan(t.sample(np.array([-50.0]), np.array([10.0]))).all()


def test_window_selects_the_tile_without_margin():
    t = Terrain(np.zeros((256, 256), np.float32), -256.0, 768.0, 4.0)
    m = t.window(0, 0, 512, 512)
    assert m.sum() == 128 * 128


def test_sea_is_on_the_right_of_the_coastline():
    from shapely.geometry import LineString, Point, box
    from urbanmorph.terrain.water import sea_polygons
    win = box(0, 0, 100, 100)
    coast = LineString([(-10, 50), (110, 50)])                 # heading east: land on the left (north)
    sea = sea_polygons([coast], win)
    assert len(sea) == 1 and sea[0].contains(Point(50, 20)) and not sea[0].contains(Point(50, 80))
