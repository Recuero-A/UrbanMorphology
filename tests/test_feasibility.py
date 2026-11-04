"""Terrain feasibility and road profiles (Milestone 12): cases with known answers."""
import numpy as np

from urbanmorph.terrain.heightfield import Terrain
from urbanmorph.terrain.road_feasibility import Thresholds, evaluate, far_bank
from urbanmorph.terrain.road_profiles import deck_profile, ground_profile

TH = Thresholds()


def grid(fz, water=None, n=128, res=4.0):
    xc = (np.arange(n) + 0.5) * res
    yc = n * res - (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xc, yc)
    t = Terrain(fz(X, Y).astype(np.float32), 0.0, n * res, res)
    if water is not None:
        t.water_kind = water(X, Y).astype(np.int8)
    return t


def test_flat_ground_is_feasible():
    f = evaluate(grid(lambda x, y: 0 * x + 50), (50, 50), (90, 80), 2, TH)
    assert f.kind == "ground" and f.max_grade < 1e-6 and f.buildable and f.structure == 0


def test_uniform_slope_grade_and_class_limits():
    t = grid(lambda x, y: 0.12 * y)
    assert evaluate(t, (100, 100), (100, 130), 2, TH).kind == "ground"           # 12 % <= local limit 15 %
    f = evaluate(t, (100, 100), (100, 130), 0, TH)                              # major roads: 8 %
    assert f.kind == "excessive_grade" and abs(f.max_grade - 0.12) < 1e-3
    assert evaluate(t, (100, 100), (130, 100), 0, TH).kind == "ground"          # along the contour


def test_river_crossing_is_a_possible_bridge_and_angle_matters():
    river = lambda x, y: ((y > 240) & (y < 260)).astype(int)                   # 20 m wide river along x
    t = grid(lambda x, y: 0 * x + 30, river)
    f = evaluate(t, (200, 200), (200, 300), 2, TH)
    assert f.kind == "possible_bridge" and abs(f.wet_length - 20) <= 4 and f.crossing_angle_deg > 60
    oblique = evaluate(t, (100, 235), (400, 265), 2, TH)                       # almost along the river
    assert oblique.kind == "impossible_crossing" and oblique.crossing_angle_deg < 30
    assert evaluate(t, (200, 200), (200, 250), 2, TH).kind == "ends_in_water"
    end = far_bank(t, (200, 200), (0, 1), 200)
    assert end is not None and 260 <= end[1] <= 270


def test_wide_river_exceeds_the_local_span_but_not_the_major_one():
    river = lambda x, y: ((y > 150) & (y < 350)).astype(int)                   # 200 m
    t = grid(lambda x, y: 0 * x + 30, river)
    assert evaluate(t, (250, 100), (250, 400), 2, TH).kind == "impossible_crossing"
    assert evaluate(t, (250, 100), (250, 400), 0, TH).kind == "possible_bridge"


def test_lake_is_an_obstruction_for_streets():
    lake = lambda x, y: (np.hypot(x - 256, y - 256) < 120).astype(int) * 2
    t = grid(lambda x, y: 0 * x + 30, lake)
    assert evaluate(t, (100, 256), (420, 256), 2, TH).kind == "water_obstruction"


def test_valley_viaduct_clears_the_ground():
    valley = lambda x, y: 100 - 40 * np.exp(-((x - 256) / 40) ** 2)
    t = grid(valley)
    f = evaluate(t, (150, 200), (362, 200), 1, TH)
    assert f.kind == "possible_viaduct" and f.clearance > 30 and f.deck_grade < 0.01
    d, z = t.profile((150, 200), (362, 200))
    deck = deck_profile(d, z, z[0], z[-1])
    assert (deck >= z - 1e-6).all() and np.ptp(deck) < 1.0             # the deck never intersects the terrain


def test_ground_profile_respects_the_grade_limit_and_records_cut_fill():
    d = np.arange(0, 200, 2.0)
    z = np.where(d < 100, 0.0, 30.0)                                   # a 30 m step
    zs, cf = ground_profile(d, z, max_grade=0.10)
    assert np.max(np.abs(np.diff(zs) / np.diff(d))) <= 0.10 + 1e-9
    assert abs(cf).max() > 5 and zs[0] == 0.0


def test_far_rim_finds_the_other_side_of_a_valley():
    from urbanmorph.terrain.road_feasibility import far_rim
    t = grid(lambda x, y: 100 - 40 * np.exp(-((x - 256) / 40) ** 2))
    p = far_rim(t, (150, 200), (1, 0), 1, TH)
    assert p is not None and p[0] > 300 and evaluate(t, (150, 200), p, 1, TH).kind == "possible_viaduct"
    assert far_rim(grid(lambda x, y: 0.3 * x), (150, 200), (1, 0), 1, TH) is None     # a slope, not a valley
