"""Procedural landscapes and hydrology (Milestone 12)."""
import numpy as np
import pytest

from urbanmorph.terrain import hydrology
from urbanmorph.terrain.synthesis import SCENARIOS, LandscapeParams, generate, height, scenario


def test_deterministic_for_a_seed_and_different_across_seeds():
    p = LandscapeParams(seed=3, relief_m=30)
    a, b = generate(p, 0, 0, 512, 512, hydrology_on=False), generate(p, 0, 0, 512, 512, hydrology_on=False)
    assert np.array_equal(a.elev, b.elev)
    c = generate(LandscapeParams(seed=4, relief_m=30), 0, 0, 512, 512, hydrology_on=False)
    assert not np.allclose(a.elev, c.elev)


def test_adjacent_windows_are_seamless():
    p = LandscapeParams(seed=1, relief_m=40, ridge=0.5)
    big = generate(p, 0, 0, 1024, 512, hydrology_on=False)
    left = generate(p, 0, 0, 512, 512, hydrology_on=False)
    right = generate(p, 512, 0, 1024, 512, hydrology_on=False)
    assert np.allclose(big.elev, np.hstack([left.elev, right.elev]), atol=1e-4)
    x = np.array([511.9, 512.1])                              # no jump across the seam
    assert abs(np.diff(height(x, np.array([200.0, 200.0]), p))[0]) < 0.05


def test_priority_flood_drains_everything_and_d8_flows_downhill():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (40, 40)).cumsum(0)
    f = hydrology.priority_flood(z)
    assert (f >= z - 1e-12).all()
    flow = hydrology.d8(f)
    interior = flow[1:-1, 1:-1]
    assert (interior >= 0).all()                               # no interior pits remain
    r, c = 20, 20
    for _ in range(200):                                       # follow the flow: strictly downhill to an outlet
        k = flow[r, c]
        if k < 0:
            break
        dr, dc = hydrology.NB[k]
        assert f[r + dr, c + dc] < f[r, c]
        r, c = r + dr, c + dc
    assert flow[r, c] < 0


def test_lake_surface_is_flat_and_small_pits_are_not_lakes():
    x = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, x)
    z = 10 + 5 * (X ** 2 + Y ** 2) - 6 * np.exp(-(X ** 2 + Y ** 2) / 0.1)   # a deep basin in a bowl rim
    z[5, 5] -= 0.3                                             # a tiny pit
    f = hydrology.priority_flood(z)
    lake, surf = hydrology.lakes(z, f, 4.0, min_area_m2=2000, min_depth_m=1.0)
    assert lake[50, 50] and not lake[5, 5]
    assert np.nanstd(surf[lake]) < 1e-9


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_generates_valid_terrain(name):
    t = generate(scenario(name, seed=7), -256, -256, 768, 768)
    assert np.isfinite(t.elev).all() and t.elev.shape == (256, 256)
    assert t.provenance["params"]["seed"] == 7
    if name in ("narrow_river", "wide_river"):
        assert (t.water_kind == 1).any()
    if name == "large_lake":
        assert (t.water_kind == 2).any()
    if name == "coastal_city":
        sea = t.water_kind == 3
        assert sea.any() and (t.elev[sea] < 56.0).all()
    if name == "flat_city":
        assert np.percentile(t.slope(), 90) < 0.05
    if name == "steep_hillside":
        assert np.median(t.slope()) > 0.12


def test_wide_river_is_wider_than_narrow_river():
    from scipy import ndimage
    def width(n):                                              # widest channel: twice the largest distance to land
        t = generate(scenario(n, 7), -256, -256, 768, 768)
        return 2 * ndimage.distance_transform_edt(t.water_kind == 1).max() * t.res
    assert width("wide_river") > 2 * width("narrow_river")
