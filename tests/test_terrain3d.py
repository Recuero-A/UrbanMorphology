"""Terrain 3D layer (Milestone 12): grid cutting keeps area, buildings never float, decks clear the terrain."""
import numpy as np

from urbanmorph.city3d.terrain3d import apply_terrain, bridge_runs, grid_cut, grid_mesh
from urbanmorph.graphs.levels import BRIDGE, GROUND
from urbanmorph.graphs.road_graph import build_graph
from urbanmorph.terrain.heightfield import Terrain


def cross2(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def area(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.abs(cross2(b[:, :2] - a[:, :2], c[:, :2] - a[:, :2])).sum()


def test_grid_cut_preserves_area_and_orientation():
    V = np.array([[1.0, 1.0, 0.0], [37.0, 3.0, 0.0], [10.0, 29.0, 0.0]])
    F = np.array([[0, 1, 2]])
    V2, F2, m2, o2 = grid_cut(V, F, np.array([3], np.int16), np.array([-1]))
    assert abs(area(V2, F2) - area(V, F)) < 1e-6 and len(F2) > 10 and (m2 == 3).all()
    e = V2[F2]
    cross = cross2(e[:, 1, :2] - e[:, 0, :2], e[:, 2, :2] - e[:, 0, :2])
    assert (cross > -1e-9).all()                               # all counter-clockwise (upward normals)


def test_grid_mesh_and_bridge_runs():
    V, F = grid_mesh(np.zeros((3, 4)), 0.0, 12.0, 4.0, np.ones((3, 4), bool))
    assert len(F) == 24 and abs(area(V, F) - 12 * 16) < 1e-6
    g = build_graph([np.array([[0, 256.0], [200, 256]]), np.array([[200, 256.0], [300, 256]]),
                     np.array([[300, 256.0], [512, 256]])], levels=[0, 1, 0], structs=[GROUND, BRIDGE, GROUND])
    runs = bridge_runs(g)
    assert len(runs) == 1 and np.allclose(sorted(g.nodes[[runs[0][0], runs[0][-1]]][:, 0]), [200, 300])


def test_buildings_sit_on_the_lowest_ground_and_the_deck_clears_the_valley():
    c = (np.arange(256) - 64 + 0.5) * 4.0
    X, Y = np.meshgrid(c, 512 - c)
    z = 100 - 30 * np.exp(-((X - 250) / 30) ** 2) + 0.05 * Y   # a valley along y on a gentle slope
    t = Terrain(z.astype(np.float32), -256.0, 768.0, 4.0)
    t.water_kind = np.zeros(z.shape, np.int8)
    fp = [[[400, 100], [420, 100], [420, 140], [400, 140], [400, 100]]]
    city = {"buildings": [{"footprint": fp}]}
    walls = {"V": np.array([[400, 100, 0], [420, 100, 0], [420, 140, 10]], np.float32), "F": np.array([[0, 1, 2]]),
             "mat": np.array([0], np.int16), "obj": np.array([0], np.int32), "UV": np.zeros((1, 3, 2), np.float32)}
    ground = {"V": np.zeros((4, 3), np.float32), "F": np.array([[0, 1, 2]]), "mat": np.array([1], np.int16),
              "obj": np.array([-1], np.int32), "UV": np.zeros((1, 3, 2), np.float32)}
    g = build_graph([np.array([[150, 300.0], [200, 300]]), np.array([[200, 300.0], [300, 300]]),
                     np.array([[300, 300.0], [350, 300]])], classes=[2, 2, 2], levels=[0, 1, 0], structs=[GROUND, BRIDGE, GROUND])
    out, mats, rep = apply_terrain({"walls": walls, "ground": ground}, [{}, {}], city, t, g)
    zb = out["walls"]["V"][:2, 2]
    lowest = min(100 + 0.05 * y for y in (100, 140))
    assert np.all(zb <= lowest + 0.06) and np.all(zb >= lowest - 0.5)     # base = lowest ground under the footprint
    deck = out["bridges"]["V"]
    xs = deck[:, 0]
    mid = (xs > 230) & (xs < 270)
    assert deck[mid, 2].min() > 100 - 30 + 0.05 * 300 + 5         # far above the valley floor
    assert rep["bridges"] == 1 and rep["piers"] >= 1 and "water" not in out
