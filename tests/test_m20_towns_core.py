"""Milestone 20 D4d city cores in scripts/m20_towns.py (synthetic polygons in UTM 33N; no town data is read): a city
gets a 4096 m domain on its rounded population-weighted centroid and the unit polygon ∩ 3496 m square; towns are
unchanged; an oversize town is refused and reported; a city never is. Also the D4d round cap of the global sampler."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
from pyproj import Transformer
from shapely import affinity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import m20_towns as T  # noqa: E402

CRS = "EPSG:32633"
C = (286595.4, 4775406.6)                                 # a centroid in UTM 33N (central Italy)


def frame(kind, poly_utm, pop=C):
    """A town frame as town_frame builds it (Mollweide polygon, its 25 m segmentized UTM projection)."""
    to_moll = Transformer.from_crs(CRS, "ESRI:54009", always_xy=True).transform
    pm = shapely.transform(shapely.segmentize(poly_utm, 25.0), lambda x, y: to_moll(x, y), interleaved=False)
    pu = T.poly_utm(pm, CRS)
    return {"id": "X", "crs": CRS, "poly_moll": pm, "poly_utm": pu, "pop_utm": pop, "bbox": pu.bounds, "kind": kind}


def box(w, h, cx=C[0], cy=C[1], dx=0.0):
    return shapely.box(cx - w / 2 + dx, cy - h / 2, cx + w / 2 + dx, cy + h / 2)


def site_of(fr, tmp_path):
    """A TownSite on a minimal config (the geometry keys site_config writes)."""
    L, cx, cy, _, _ = T.domain_geom(fr)
    cfg = {"L": L, "crs": CRS, "centre_utm": [cx, cy], "limits": T.limits_of(fr)}
    (tmp_path / "X.json").write_text(json.dumps(cfg))
    return T.TownSite("X", SimpleNamespace(sites=tmp_path, towns=tmp_path, out=tmp_path))


def test_city_domain_is_4096_on_the_rounded_centroid_whatever_its_size():
    for w in (3000.0, 12000.0, 30000.0):                  # never refused, never resized
        L, cx, cy, x0, y0 = T.domain_geom(frame("city", box(w, 3000.0, dx=2500.0)))
        assert (L, cx, cy) == (4096, 286595, 4775407) and (x0, y0) == (cx - 2048, cy - 2048)


def test_city_core_is_the_polygon_within_the_3496m_square():
    fr = frame("city", box(12000.0, 2000.0, dx=2500.0))   # off-centre: the bbox centre is not the centroid
    unit, sq = T.unit_geometry(fr)
    assert sq.bounds == (286595 - 1748, 4775407 - 1748, 286595 + 1748, 4775407 + 1748)
    assert unit.difference(sq).area < 1e-3 and unit.difference(fr["poly_utm"]).area < 1e-3
    assert unit.area == pytest.approx(3496.0 * 2000.0, rel=1e-3)
    full, _ = T.unit_geometry(frame("city", box(9000.0, 9000.0)))
    assert full.area == pytest.approx(3496.0 ** 2, rel=1e-6)   # a polygon covering the square: the square


def test_towns_unchanged_and_an_oversize_town_refused():
    fr = frame("town", box(1200.0, 1000.0, dx=300.0))
    L, cx, cy, _, _ = T.domain_geom(fr)
    X0, Y0, X1, Y1 = fr["bbox"]
    assert L == 2048 and (cx, cy) == (round((X0 + X1) / 2), round((Y0 + Y1) / 2))   # the bbox centre, as before
    unit, sq = T.unit_geometry(fr)
    assert sq is None and unit.equals(fr["poly_utm"])
    assert T.domain_geom(frame("village", box(3000.0, 2000.0)))[0] == 4096
    with pytest.raises(T.Oversize):
        T.domain_geom(frame("town", box(9000.0, 1000.0)))
    assert issubclass(T.Oversize, ValueError)


def test_oversize_is_reported(tmp_path):
    P = SimpleNamespace(towns=tmp_path)
    T.report_oversize(P, "DUC_1", T.Oversize("too big"))
    rec = json.loads((tmp_path / "refused_oversize.jsonl").read_text().splitlines()[0])
    assert rec["id"] == "DUC_1" and "too big" in rec["reason"]


def test_config_stores_the_full_polygon_and_the_core():
    fr = frame("city", box(12000.0, 2000.0, dx=2500.0))
    lim = T.limits_of(fr)
    assert lim["type"] == "city core" and shapely.from_wkt(lim["wkt_mollweide"]).equals(fr["poly_moll"])
    core = lim["core"]
    assert core["side_m"] == 3496.0 and core["area_km2"] == pytest.approx(3496 * 2000 / 1e6, rel=1e-3)
    assert core["polygon_area_km2"] == pytest.approx(24.0, rel=1e-3)
    assert shapely.from_wkt(core["wkt_utm"]).equals_exact(T.unit_geometry(fr)[0], 1e-6)
    assert "core" not in T.limits_of(frame("town", box(1200.0, 1000.0)))


def test_townsite_unit_and_city_mask_are_the_core(tmp_path):
    fr = frame("city", box(12000.0, 2000.0, dx=2500.0))
    site = site_of(fr, tmp_path)
    unit, _ = T.unit_geometry(fr)
    core = site.polygon()
    assert core.symmetric_difference(affinity.translate(unit, -site.x0, -site.y0)).area < 1e-3
    x0, y0, x1, y1 = core.bounds
    assert x0 >= 300 - 1e-6 and y0 >= 300 - 1e-6 and x1 <= 3796 + 1e-6 and y1 <= 3796 + 1e-6
    assert site.full_polygon().area == pytest.approx(24e6, rel=1e-3) and site.full_polygon().area > core.area
    m = site.city_mask({"water": np.zeros((site.n, site.n), np.int8)})
    assert m.sum() * 16 == pytest.approx(core.area, rel=0.01)       # the 4 m city mask is the core
    town = site_of(frame("town", box(1200.0, 1000.0)), tmp_path)
    assert town.polygon().equals(town.full_polygon())


def test_global_sampler_round_cap_scales_with_windows(monkeypatch):
    from urbanmorph.graphs import tc_global, tc_sampler
    assert [tc_global.round_cap(n) for n in (1, 9, 16, 64, 256)] == [6000, 6000, 6000, 24000, 96000]
    seen = []
    monkeypatch.setattr(tc_sampler.TCSampler, "run", lambda self, seed, max_rounds: seen.append(max_rounds))
    for L, cap in ((1536, 6000), (2048, 6000), (4096, 24000), (8192, 96000)):
        sm = tc_global.GlobalTCSampler.__new__(tc_global.GlobalTCSampler)
        n = int(L // 512)
        sm.windows = [(c * 512.0, r * 512.0) for r in range(n) for c in range(n)]
        sm.run(1)
        sm.run(1, max_rounds=7)                            # an explicit cap is kept
        assert seen[-2:] == [cap, 7]
