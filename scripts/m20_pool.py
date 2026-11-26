"""Milestone 20, D1: the hill-town pool and its split, before any town data is downloaded (docs/M20_LOG.md D1).

Inputs contain no streets: the GHSL Degree of Urbanisation clusters (GHS-SMOD E2020 R2023A V2.0: dense urban clusters =
towns, urban centres = cities, rural clusters = villages, class 13 of the 1 km grid) and Natural Earth admin-0.

    python scripts/m20_pool.py frame                  -> data/processed/m20/pool/frame.csv.gz, frame_manifest.json
    python scripts/m20_pool.py beacon --pushed <UTC>  -> salt.json: the NIST beacon pulse at the first full hour at
                                                         least 30 min after the D1 push (waits for it if in the future)
    python scripts/m20_pool.py split                  -> lists/<split>.csv.gz (ordered candidates), split_manifest.json

Split by 1 degree WGS84 block: u = sha256(salt + block) / 2**256; train-H < 0.30 <= dev-H < 0.55 <= hold-H < 0.92 <=
study-H. Region R5 (rest of the world) and villages are used only in hold-H / study-H (villages also in dev-H).
Settlements of dev-H / hold-H / study-H within 10 km of a used settlement of another split are dropped (buffer).
Candidates are ordered by sha256(salt + "order" + id). Eligibility (FABDEM slope) and the OSM gate come later (walk).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

GHSL = ROOT / "data" / "raw" / "ghsl" / "smod_e2020_r2023a_v2"
SMOD = "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000"
NE = ROOT / "data" / "raw" / "naturalearth" / "ne_50m_admin_0_countries"
OUT = ROOT / "data" / "processed" / "m20" / "pool"
NEAR_KM, BUFFER_KM = 50.0, 10.0
SHARES = (("train", 0.30), ("dev", 0.55), ("hold", 0.92), ("study", 1.0))
BEACON = "https://beacon.nist.gov/beacon/2.0/pulse/time/{ms}"

# regions by Natural Earth ADM0_A3; anything else is R5 (rest of the world, hold-H / study-H only)
REGIONS = {
    "R1": "ESP PRT ITA GRC MLT CYP FRA MCO AND SMR VAT SVN HRV BIH SRB KOS MNE ALB MKD BGR ROU",
    "R2": "MEX GTM BLZ SLV HND NIC CRI PAN CUB DOM HTI JAM PRI TTO BHS BRB DMA GRD KNA LCA VCT ATG ABW CUW "
          "COL VEN ECU PER BOL CHL ARG URY PRY BRA GUY SUR",
    "R3": "KOR PRK JPN TWN CHN HKG MAC MNG",
    "R4": "MAR SAH DZA TUN LBY EGY ISR PSX JOR LBN SYR IRQ IRN SAU YEM OMN ARE QAT BHR KWT TUR GEO ARM AZE",
}
REGION_OF = {c: r for r, cs in REGIONS.items() for c in cs.split()}

# places used so far, not in the configs below (lat, lon; public gazetteer coordinates)
USED_EXTRA = {
    "Lleida (M15 held-out)": (41.6176, 0.6200), "Alicante (M15 held-out)": (38.3452, -0.4810),
    "Daegu (M15 held-out)": (35.8714, 128.6014), "Trujillo (M15 held-out)": (-8.1091, -79.0215),
    "Santa Barbara (GEN2, M15 held-out)": (34.4208, -119.6982), "El Escorial (M16 pack)": (40.5833, -4.1167),
}
ANCHORS = {"Wadi Musa (dev anchor, Petra)": (30.3214, 35.4790),
           "Aguas Calientes (dev anchor, Machu Picchu)": (-13.1547, -72.5254)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def used_places():
    """(name, lat, lon, source) of every city used in M1-M19."""
    import geopandas as gpd
    out = []
    for f in sorted((ROOT / "data" / "raw" / "boundaries").glob("*.geojson")):
        c = gpd.read_file(f).to_crs(4326).geometry.union_all().centroid
        out.append((f.stem, c.y, c.x, f"data/raw/boundaries/{f.name}"))
    for m in json.loads((ROOT / "configs" / "m17_holdout.json").read_text())["metros"]:
        out.append((m["name"], m["lat"], m["lon"], "configs/m17_holdout.json"))
    for f in sorted((ROOT / "configs" / "sites").glob("*.json")):
        s = json.loads(f.read_text())
        lon, lat = s.get("treasury_lonlat") or s.get("citadel_lonlat")
        out.append((f.stem, lat, lon, f"configs/sites/{f.name}"))
    out += [(k, la, lo, "m20_pool.USED_EXTRA") for k, (la, lo) in USED_EXTRA.items()]
    return out


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def chord(km):
    return 2.0 * math.sin(km / 6371.0 / 2.0)


def villages():
    """Rural clusters: 8-connected components of class 13 on the 1 km grid -> (label, x, y, n_cells); labels in
    scan order (scipy.ndimage.label), so the ids are reproducible."""
    import rasterio
    from scipy import ndimage
    with rasterio.open(GHSL / f"{SMOD}_V2_0" / f"{SMOD}_V2_0.tif") as r:
        a = r.read(1)
        tr = r.transform
    lab, n = ndimage.label(a == 13, structure=np.ones((3, 3), int))
    rr, cc = np.nonzero(lab)
    k = lab[rr, cc]
    cnt = np.bincount(k, minlength=n + 1)
    my = np.bincount(k, rr + 0.5, n + 1) / np.maximum(cnt, 1)
    mx = np.bincount(k, cc + 0.5, n + 1) / np.maximum(cnt, 1)
    x, y = tr.c + mx * tr.a, tr.f + my * tr.e
    ids = np.arange(1, n + 1)
    return ids, x[1:], y[1:], cnt[1:], (lab, tr)


def stage_frame() -> int:
    import geopandas as gpd
    import pandas as pd
    from pyproj import Transformer
    OUT.mkdir(parents=True, exist_ok=True)
    to_ll = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True)
    rows, polys = [], {}
    for k, kind in (("DUC", "town"), ("UC", "city")):
        g = gpd.read_file(GHSL / f"{SMOD}_{k}_V2_0" / f"{SMOD}_{k}_V2_0.shp")
        idc = "ID_DUC_G0" if k == "DUC" else "ID_UC_G0"
        lon, lat = to_ll.transform(g.PWCentr_X.values, g.PWCentr_Y.values)
        d = pd.DataFrame({"id": [f"{k}_{int(i)}" for i in g[idc]], "kind": kind, "lon": lon, "lat": lat,
                          "x": g.PWCentr_X.values, "y": g.PWCentr_Y.values, "pop": g.POP_2020.values.round(0),
                          "bu_m2": g.BU_m2_2020.values, "area_km2": (g.area / 1e6).round(2).values})
        rows.append(d)
        polys[kind] = gpd.GeoDataFrame({"id": d.id.values}, geometry=g.geometry.values, crs=g.crs)
    vid, vx, vy, vn, (lab, tr) = villages()
    vlon, vlat = to_ll.transform(vx, vy)
    rows.append(pd.DataFrame({"id": [f"RC_{i}" for i in vid], "kind": "village", "lon": vlon, "lat": vlat, "x": vx,
                              "y": vy, "pop": np.nan, "bu_m2": np.nan, "area_km2": vn.astype(float)}))
    F = pd.concat(rows, ignore_index=True)
    # country and region from the population-weighted centroid (nearest country for points just offshore)
    ne = gpd.read_file(NE / "ne_50m_admin_0_countries.shp")[["ADM0_A3", "geometry"]].to_crs("ESRI:54009")
    pts = gpd.GeoDataFrame(F[["id"]], geometry=gpd.points_from_xy(F.x, F.y), crs="ESRI:54009")
    j = gpd.sjoin_nearest(pts, ne, how="left").drop_duplicates("id")
    F["iso3"] = j.set_index("id").loc[F.id, "ADM0_A3"].values
    F["region"] = [REGION_OF.get(c, "R5") for c in F.iso3]
    F["block"] = [f"{math.floor(la):+03d}{math.floor(lo):+04d}" for la, lo in zip(F.lat, F.lon)]
    # exclusions: near a used city, or touching an M18 natural window
    U = used_places()
    from scipy.spatial import cKDTree
    tree = cKDTree(unit([u[1] for u in U], [u[2] for u in U]))
    dist, idx = tree.query(unit(F.lat.values, F.lon.values))
    near = dist <= chord(NEAR_KM)
    F["excluded"] = near
    F["reason"] = np.where(near, ["near:" + U[i][0] for i in idx], "")
    W = json.loads((ROOT / "outputs" / "m18" / "natural" / "windows.json").read_text())["windows"]
    to_m = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    from shapely.geometry import box
    hit = set()
    for w in W:
        cx, cy = to_m.transform(w["lon"], w["lat"])
        sq = box(cx - 512, cy - 512, cx + 512, cy + 512)
        for kind in ("town", "city"):
            P = polys[kind]
            hit |= set(P.id.values[P.intersects(sq).values])
        c0, r0 = ~tr * (cx - 512, cy + 512)
        c1, r1 = ~tr * (cx + 512, cy - 512)
        sub = lab[int(r0):int(r1) + 1, int(c0):int(c1) + 1]
        hit |= {f"RC_{v}" for v in np.unique(sub[sub > 0])}
    nw = F.id.isin(hit) & ~F.excluded
    F.loc[nw, "excluded"] = True
    F.loc[nw, "reason"] = "natural_window"
    F.to_csv(OUT / "frame.csv.gz", index=False, float_format="%.6f")
    man = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "script_sha256": sha256(Path(__file__)),
           "inputs": {p.name: sha256(p) for p in sorted(GHSL.glob("*.zip"))
                      } | {"ne_50m_admin_0_countries.zip": sha256(NE.parent / "ne_50m_admin_0_countries.zip")},
           "near_km": NEAR_KM, "used_places": U, "anchors": ANCHORS,
           "counts": F.groupby(["kind", "region"]).size().unstack(fill_value=0).to_dict(),
           "excluded": F[F.excluded].groupby(["kind"]).size().to_dict(),
           "frame_sha256": sha256(OUT / "frame.csv.gz")}
    (OUT / "frame_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({k: man[k] for k in ("counts", "excluded", "frame_sha256")}, indent=1, default=str))
    return 0


def stage_beacon(pushed: str) -> int:
    """The declared salt: the pulse at the first full hour >= 30 min after the push (UTC)."""
    t = dt.datetime.fromisoformat(pushed.replace("Z", "+00:00")) + dt.timedelta(minutes=30)
    t = (t + dt.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0) if (t.minute or t.second) else t
    now = dt.datetime.now(dt.timezone.utc)
    if now < t:
        print(f"pulse time {t.isoformat()} is in the future: wait {(t - now).total_seconds() / 60:.0f} min")
        return 1
    url = BEACON.format(ms=int(t.timestamp() * 1000))
    raw = urllib.request.urlopen(url, timeout=30).read()
    pulse = json.loads(raw)["pulse"]
    (OUT / "salt.json").write_text(json.dumps({"pushed_utc": pushed, "pulse_time_utc": t.isoformat(), "url": url,
                                               "pulse": pulse}, indent=1))
    print(pulse["timeStamp"], pulse["outputValue"][:16] + "...")
    return 0


def stage_split() -> int:
    import pandas as pd
    from scipy.spatial import cKDTree
    salt = json.loads((OUT / "salt.json").read_text())["pulse"]["outputValue"]
    F = pd.read_csv(OUT / "frame.csv.gz", keep_default_na=False, na_values=[""])
    F = F[~F.excluded.astype(bool)].copy()
    hu = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) / 2 ** 256
    ub = {b: hu(salt + b) for b in F.block.unique()}
    F["u"] = F.block.map(ub)
    F["split"] = [next(n for n, s in SHARES if u < s) for u in F.u]
    # R5 is for transfer only (hold / study); villages are evaluation only (dev / hold / study)
    unused = ((F.region == "R5") & F.split.isin(["train", "dev"])) | ((F.kind == "village") & (F.split == "train"))
    F["use"] = np.where(unused, "unused", F.split)
    # buffer: drop evaluation settlements within 10 km of a used settlement of another split
    A = F[F.use != "unused"]
    P = unit(A.lat.values, A.lon.values)
    pairs = cKDTree(P).query_pairs(chord(BUFFER_KM), output_type="ndarray")
    s = A.use.values
    cross = pairs[s[pairs[:, 0]] != s[pairs[:, 1]]]
    drop = set()
    for i, j in cross:
        for k in (i, j):
            if s[k] != "train":
                drop.add(A.id.values[k])
    F["buffer_dropped"] = F.id.isin(drop)
    F["order_key"] = [hashlib.sha256((salt + "order" + i).encode()).hexdigest() for i in F.id]
    (OUT / "lists").mkdir(exist_ok=True)
    man = {"salt_pulse": json.loads((OUT / "salt.json").read_text())["pulse"]["uri"],
           "script_sha256": sha256(Path(__file__)), "shares": SHARES, "buffer_km": BUFFER_KM, "lists": {}}
    cols = ["id", "kind", "region", "iso3", "block", "u", "order_key", "lon", "lat", "pop", "area_km2"]
    for name in ("train", "dev", "hold", "study"):
        L = F[(F.use == name) & ~F.buffer_dropped].sort_values("order_key")[cols]
        L.insert(0, "rank", np.arange(len(L)))
        f = OUT / "lists" / f"{name}.csv.gz"
        L.to_csv(f, index=False, float_format="%.6f")
        man["lists"][name] = {"rows": len(L), "sha256": sha256(f),
                              "by_kind_region": L.groupby(["kind", "region"]).size().unstack(fill_value=0).to_dict()}
    man["buffer_dropped"] = int(F.buffer_dropped.sum())
    man["unused"] = int((F.use == "unused").sum())
    (OUT / "split_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({k: {kk: v for kk, v in man["lists"][k].items() if kk != "by_kind_region"} for k in man["lists"]},
                     indent=1), man["buffer_dropped"], "buffer-dropped,", man["unused"], "unused")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["frame", "beacon", "split"])
    ap.add_argument("--pushed", default=None, help="beacon: UTC time of the D1 push (ISO 8601)")
    a = ap.parse_args()
    if a.stage == "frame":
        return stage_frame()
    if a.stage == "beacon":
        return stage_beacon(a.pushed)
    return stage_split()


if __name__ == "__main__":
    sys.exit(main())
