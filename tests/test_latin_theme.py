"""Latin Caribbean theme (themes/latin_caribbean/DESIGN.md): the rendering layer is consistent and cheap."""
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "themes" / "latin_caribbean" / "scripts"))

from urbanmorph.city3d.latin import kit, palette, plants, textures  # noqa: E402
from urbanmorph.city3d.latin.church import church  # noqa: E402
from urbanmorph.city3d.scene import build_scene  # noqa: E402


@pytest.fixture(scope="module")
def showcase():
    from preview_kit import showcase as make
    return make(3)


@pytest.fixture(scope="module")
def latin(showcase):
    return build_scene(showcase, theme="latin_caribbean")


def test_textures_written_and_cutouts_have_alpha():
    from PIL import Image
    d = textures.ensure()
    for name in textures.TEXTURES:
        assert (d / name).exists()
    for name in ("palm_frond.png", "balusters.png", "iron_railing.png", "fan_leaf.png"):
        a = np.asarray(Image.open(d / name))
        assert a.shape[2] == 4 and (a[..., 3] == 0).any() and (a[..., 3] == 255).any()


def test_every_material_resolves():
    for n, _, _ in palette.WALLS:
        assert palette.material(f"wall:{n}")["image"] == "plaster.png"
    for k in palette.BASE:
        p = palette.material(k)
        assert len(p["base_color"]) == 3 and max(p["base_color"]) <= 1.0


def test_groups_consistent(latin):
    groups, mats, rep = latin
    for name, a in groups.items():
        if "F" not in a:
            assert a["T"].shape[1] in (4, 5)
            continue
        assert len(a["F"]) == len(a["mat"]) == len(a["obj"]) == len(a["UV"]), name
        if not len(a["F"]):
            continue
        assert a["F"].max() < len(a["V"]) and np.isfinite(a["V"]).all() and np.isfinite(a["UV"]).all(), name
        assert a["mat"].max() < len(mats)
    for p in mats:
        if p.get("image"):
            assert Path(p["image"]).exists()


def test_houses_watertight_roofed_and_a_church(latin, showcase):
    groups, _, rep = latin
    s = rep["summary"]
    assert s["churches"] >= 1
    assert s["watertight"] == s["outward"] == s["n_buildings"] - s["churches"] + s["churches"]
    roofed = set(np.unique(groups["roofs"]["obj"]))
    assert len(roofed) >= s["n_buildings"] - 1                   # every house (and the church) has a roof
    assert rep["facade"]["windows"] > 0 and rep["facade"]["doors"] > 0 and rep["facade"]["balconies"] > 0
    assert sum(s["triangles"].values()) / s["n_buildings"] < 2500   # performance budget per building


def test_standard_theme_unchanged_shape(showcase):
    groups, _, rep = build_scene(showcase)
    assert "carpentry" not in groups and rep["summary"]["n_buildings"] == len(showcase["buildings"])
    for a in groups.values():
        if "F" in a:
            assert len(a["UV"]) == len(a["F"])


def test_skirt_roof_on_l_shape_and_slope_uvs():
    S = kit.Soup()
    L_shape = shapely.orient_polygons(shapely.Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 18), (0, 18)]))
    assert kit.skirt_shell(S, "roof:teja", "roof_flat", "wood:madera", L_shape, 7.0, 2.5, 0.5)
    V, F, UV = S.parts["roof:teja"][0]                             # the tiled band (then the hip caps)
    P = V[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    assert (n[:, 2] > 0).all()                                     # the band faces up
    uv = kit.slope_uvs(V, F)
    up = P[:, :, 2].argmax(1)
    low = P[:, :, 2].argmin(1)
    k = np.arange(len(F))
    assert (uv[k, up, 1] >= uv[k, low, 1] - 1e-6).all()            # v grows up the slope


def test_church_bounded():
    S, info = church((100.0, 100.0), (1.0, 0.0), 44.0, 22.0, np.random.default_rng(0))
    assert info["height_m"] < 60 and S.triangles() < 20000


def test_giant_tower_height_and_footprint():
    for H, W in ((165.0, 34.0), (70.0, 16.0)):
        S, info = church((0.0, 0.0), (0.0, 1.0), 44.0, W, np.random.default_rng(3), giant=H)
        V = np.vstack([v for lst in S.parts.values() for v, _, _ in lst])
        assert info["giant_m"] == pytest.approx(H, abs=1.0) and V[:, 2].max() == pytest.approx(H, abs=1.0)
        assert S.triangles() < 30000
        tx, ty = info["tower_xy"]
        top = V[V[:, 2] > 0.5 * H]                                  # the upper half stays over the tower's base
        assert np.abs(top[:, :2] - [tx, ty]).max() <= info["tower_side_m"] / 2 + 1.0
        assert all(0 < z < H for _, _, _, z, _ in info["plants"])


def test_skyscraper_height_and_footprint():
    from urbanmorph.city3d.latin.skyscraper import skyscraper
    for H, W, D in ((125.0, 24.0, 20.0), (50.0, 15.0, 13.0)):
        S, info = skyscraper((10.0, 20.0), (0.6, 0.8), W, D, H, np.random.default_rng(1))
        V = np.vstack([v for lst in S.parts.values() for v, _, _ in lst])
        assert abs(info["height_m"] - H) < 6.0 and V[:, 2].max() == pytest.approx(info["height_m"], abs=0.01)
        assert S.triangles() < 30000
        loc = (V[:, :2] - [10.0, 20.0]) @ np.array([[0.6, -0.8], [0.8, 0.6]])          # into the tower frame
        assert np.abs(loc[:, 0]).max() <= W / 2 + 1.0 and np.abs(loc[:, 1]).max() <= D / 2 + 1.0
        assert {"tower:rojo", "trimc:crema", "glass_warm", "dome:cupula_naranja"} <= set(S.parts)


def test_district_lots_get_towers_within_their_lot():
    from urbanmorph.city3d.latin.infill import lots_of, tower_site
    rng = np.random.default_rng(0)
    district = {"center": [50.0, 40.0], "radius": 150.0, "h": [45.0, 130.0]}
    lots, core = lots_of(shapely.box(0, 0, 100, 80), rng, front=(18.0, 26.0), depth=(18.0, 24.0))
    sites = [(p, tower_site(p, rng, np.zeros(2), district)) for p in lots]
    assert sum(sk is not None for _, sk in sites) >= 6
    for p, sk in sites:
        if sk is None:
            continue
        ax, ay = np.asarray(sk["axis"]), np.array([-sk["axis"][1], sk["axis"][0]])
        R = shapely.Polygon([np.asarray(sk["center"]) + sx * ax * sk["W"] / 2 + sy * ay * sk["D"] / 2
                             for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        assert p.buffer(0.01).contains(R) and min(sk["W"], sk["D"]) >= 11.0
        assert 0.8 * 45.0 <= sk["H"] <= min(130.0, 7.5 * min(sk["W"], sk["D"])) + 1e-9


def test_lanes_split_big_blocks_into_manzanas():
    from urbanmorph.city3d.latin.infill import LANE_M, lots_of, split_block
    rng = np.random.default_rng(0)
    P = shapely.Polygon([(0, 0), (310, 20), (290, 280), (-10, 260)])
    parts = split_block(P, rng, 60.0)
    assert len(parts) >= 16 and all(P.buffer(0.01).contains(q) for q in parts)
    assert shapely.union_all(parts).area == pytest.approx(sum(q.area for q in parts), rel=1e-6)
    assert all(a.distance(b) >= LANE_M - 0.05 for i, a in enumerate(parts) for b in parts[i + 1:])
    patios = [lots_of(q, rng)[1] for q in parts]
    assert max(p.area for p in patios if p is not None) < 2500
    assert split_block(shapely.box(0, 0, 50, 50), rng, 60.0)[0].equals(shapely.box(0, 0, 50, 50))


def test_plant_budgets():
    for name, (V, F, part, UV, mats) in plants.prototypes().items():
        assert len(F) <= 1600, name
        assert UV.shape == (len(F), 3, 2) and part.max() < len(mats)


def test_colonial_lots_fill_blocks_without_overlap():
    from urbanmorph.city3d.latin.infill import lots_of
    rng = np.random.default_rng(0)
    for P in (shapely.box(0, 0, 60, 45), shapely.Polygon([(0, 0), (40, 0), (55, 30), (10, 38)]), shapely.box(0, 0, 30, 9)):
        lots, patio = lots_of(P, rng)
        assert len(lots) >= 2
        A = sum(p.area for p in lots)
        assert shapely.union_all(lots).area == pytest.approx(A, rel=1e-6)          # no overlaps
        assert all(P.buffer(0.01).contains(p) for p in lots)
        covered = A + (patio.area if patio is not None else 0.0)
        assert covered > 0.8 * P.area                                              # dense: little left over


def test_landmarks_height_footprint_and_materials():
    from urbanmorph.city3d.latin.build import group_of
    from urbanmorph.city3d.latin.landmarks import DESIGNS, FORMS, landmark
    for d in DESIGNS:
        for f in FORMS:
            for H in (110.0, 150.0):
                S, info = landmark(d, f, (10.0, 20.0), (0.6, 0.8), H, np.random.default_rng(1),
                                   colours={"wall": "wall:rosa"} if d == "coral" else None)
                V = np.vstack([v for lst in S.parts.values() for v, _, _ in lst])
                assert np.isfinite(V).all() and V[:, 2].min() >= kit.Z_B - 1e-6
                assert abs(info["height_m"] - H) < 10.0 and V[:, 2].max() <= info["height_m"] + 0.01, (d, f, H)
                loc = (V[:, :2] - [10.0, 20.0]) @ np.array([[0.6, -0.8], [0.8, 0.6]])
                assert np.abs(loc).max() <= info["half_extent_m"] + 1e-6 < 0.95 * info["shaft_m"]
                assert S.triangles() < 60000
                for key in S.parts:
                    group_of(key)
                    palette.material(key)
            assert (d != "coral") or "wall:rosa" in S.parts
