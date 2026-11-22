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


def test_plant_budgets():
    for name, (V, F, part, UV, mats) in plants.prototypes().items():
        assert len(F) <= 1600, name
        assert UV.shape == (len(F), 3, 2) and part.max() < len(mats)
