"""Terrain inputs of TerrainGraphTurtle (Milestone 12): alignment under symmetries, crop orientation, the model."""
import numpy as np
import torch

from urbanmorph.graphs.road_graph import RoadGraph
from urbanmorph.graphs.terrain_sequence import MARGIN_PX, node_height, terrain_crops, tile_canvas, tile_layers, transform_raster
from urbanmorph.models.terrain_graph_turtle import TerrainGraphTurtle

NORM = {"elev_scale": 5.0, "slope_scale": 0.1}


def field(fn):
    c = (np.arange(256) - MARGIN_PX + 0.5) * 4.0              # pixel centres in tile coordinates
    X, Y = np.meshgrid(c, 512.0 - c)
    return fn(X, Y).astype(np.float32)


def test_raster_symmetries_follow_the_graph_symmetries():
    z = field(lambda x, y: 0.3 * x + 0.05 * y + 0.001 * x * y)  # no symmetry of its own
    pts = np.array([[37.0, 401.0], [250.0, 90.0], [481.3, 302.7]])     # off pixel edges (floor is direction-dependent there)
    g = RoadGraph(pts, np.array([[0, 1], [1, 2]]), np.zeros(3, bool))
    base = node_height(z, pts)
    for k in range(8):
        zk, gk = transform_raster(z, k), g.transform(k)
        assert np.allclose(node_height(zk, gk.nodes), base, atol=1e-4), k


def test_elevation_crop_points_uphill_ahead():
    d = {"elev": field(lambda x, y: 0.1 * x), "valid": np.ones((256, 256), bool), "water": np.zeros((256, 256), np.int8)}
    lay = tile_layers(d, 0, ["elev", "slope", "water"], NORM)
    east = terrain_crops(lay, ["elev", "slope", "water"], np.array([[256.0, 256.0]]), np.array([0.0]), NORM)[0]
    west = terrain_crops(lay, ["elev", "slope", "water"], np.array([[256.0, 256.0]]), np.array([np.pi]), NORM)[0]
    assert east.shape == (4, 48, 48)                          # elev, slope, water, valid
    assert east[0, 0].mean() > 0 > east[0, -1].mean()        # heading east (uphill): ahead is higher
    assert west[0, 0].mean() < 0 < west[0, -1].mean()
    assert np.allclose(east[1], 0.1 / NORM["slope_scale"], atol=0.02) and (east[3] == 1).all()


def test_missing_dem_gives_zero_terrain_but_keeps_water():
    d = {"elev": np.zeros((256, 256), np.float32), "valid": np.zeros((256, 256), bool),
         "water": np.ones((256, 256), np.int8)}
    lay = tile_layers(d, 3, ["elev", "water"], NORM)
    c = terrain_crops(lay, ["elev", "water"], np.array([[100.0, 100.0]]), np.array([1.0]), NORM)[0]
    assert (c[0] == 0).all() and (c[1] == 1).all() and (c[2] == 0).all()
    assert tile_canvas(lay, ["elev", "water"], NORM).shape == (3, 128, 128)


def test_model_trains_with_terrain_and_masked_structure_labels():
    torch.manual_seed(0)
    B, T = 6, 8
    batch = {"crops": torch.zeros(B, 3, 48, 48, dtype=torch.bool), "scalars": torch.zeros(B, 9),
             "arms": torch.zeros(B, 8, 3), "arm_mask": torch.zeros(B, 8, dtype=torch.bool),
             "targets": torch.randn(B, 7, 2) * 0.5, "n_targets": torch.tensor([0, 1, 2, 3, 1, 2]),
             "target_class": torch.randint(0, 4, (B, 7)), "target_struct": torch.tensor([[-1] * 7] * B),
             "tcrops": torch.randn(B, 4, 48, 48).half()}
    batch["target_struct"][2, :2] = torch.tensor([1, 0])
    for nt in (0, 4):
        m = TerrainGraphTurtle(64, 4, 4, n_terrain=nt)
        b = dict(batch) if nt else {k: v for k, v in batch.items() if k != "tcrops"}
        loss = m.pop_nll(b).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert m.last_struct_logit.shape == (B, T, 3)
