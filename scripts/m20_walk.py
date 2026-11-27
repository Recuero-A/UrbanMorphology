"""Milestone 20, D1 walk: eligibility and the OSM completeness gate for the ordered hill-town lists (docs/M20_LOG.md).

    python scripts/m20_walk.py terrain --split dev|train   -> data/processed/m20/walk/<split>_terrain.csv
    python scripts/m20_walk.py osm --split dev|train       -> (after 2026-11-27T00:00:00Z) <split>_osm.csv, osm/<id>.json

Terrain (eligibility, terrain only): the median slope inside the settlement polygon, weighted by GHS-BUILT-S E2020
(100 m), on FABDEM V1-2 averaged to a 32 m local UTM grid; eligible if >= 12 % (band A 12-20 %, band B >= 20 %).
Candidates are evaluated in list order within each stratum (region x size; villages pooled over R1-R4) until each
stratum holds SPARE x its quota of eligible candidates, so that later gate failures can be replaced in order.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

POOL = ROOT / "data" / "processed" / "m20" / "pool"
WALK = ROOT / "data" / "processed" / "m20" / "walk"
FAB = ROOT / "data" / "raw" / "fabdem"
BUILT = ROOT / "data" / "raw" / "ghsl" / "built_s_e2020_100"
GHSL = ROOT / "data" / "raw" / "ghsl" / "smod_e2020_r2023a_v2"
SMOD = "GHS_SMOD_E2020_GLOBE_R2023A_54009_1000"
FAB_URL = "https://huggingface.co/datasets/pavelzaitsau/fabdem-v12/resolve/main/tiles/{g}/{t}"
BUILT_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/"
             "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100/V1-0/tiles/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0_{rc}.zip")
RES = 32.0
ELIG, BAND_B = 12.0, 20.0
QUOTA = {"dev": {"stratum": 3, "village": 6}, "train": {"stratum": 4, "village": 0}}
SPARE = 3
BAND_B_SPARE = 6                                # dev town strata: walk until 6 band-B eligible (2 slots x 3)


def stratum(row):
    return "village" if row["kind"] == "village" else f"{row['region']}:{row['kind']}"


# ---------------------------------------------------------------- polygons
_POLY = {}


def polygons(ids):
    """Mollweide polygons of the given settlement ids (DUC_/UC_ from the shapefiles, RC_ from the 1 km raster)."""
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union
    need = [i for i in ids if i not in _POLY]
    for k, col in (("DUC", "ID_DUC_G0"), ("UC", "ID_UC_G0")):
        want = {int(i.split("_")[1]) for i in need if i.startswith(k + "_")}
        if want:
            g = gpd.read_file(GHSL / f"{SMOD}_{k}_V2_0" / f"{SMOD}_{k}_V2_0.shp")
            for i, geom in zip(g[col], g.geometry):
                if int(i) in want:
                    _POLY[f"{k}_{int(i)}"] = geom
    want = {int(i.split("_")[1]) for i in need if i.startswith("RC_")}
    if want:
        import rasterio
        from scipy import ndimage
        with rasterio.open(GHSL / f"{SMOD}_V2_0" / f"{SMOD}_V2_0.tif") as r:
            a, tr = r.read(1), r.transform
        lab, _ = ndimage.label(a == 13, structure=np.ones((3, 3), int))      # as m20_pool.villages
        sl = ndimage.find_objects(lab)
        for v in want:
            s = sl[v - 1]
            rr, cc = np.nonzero(lab[s] == v)
            rr, cc = rr + s[0].start, cc + s[1].start
            _POLY[f"RC_{v}"] = unary_union([box(tr.c + c * tr.a, tr.f + (r_ + 1) * tr.e, tr.c + (c + 1) * tr.a,
                                                tr.f + r_ * tr.e) for r_, c in zip(rr, cc)])
    return {i: _POLY[i] for i in ids}


# ---------------------------------------------------------------- rasters
def fetch(url, dst, tries=4):
    """Download url to dst through a per-process temp file; a stalled read times out after 120 s and is retried
    (a FABDEM read once hung for 4.5 h without a timeout). HTTP errors are raised at once."""
    import socket
    import time
    socket.setdefaulttimeout(120)
    tmp = dst.with_suffix(f".part{os.getpid()}")
    for i in range(tries):
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dst)
            return
        except urllib.error.HTTPError:
            tmp.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError):
            tmp.unlink(missing_ok=True)
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def fab_tile(la, lo):
    """Path of the FABDEM 1 degree tile with SW corner (la, lo), downloaded once; None where FABDEM has no tile."""
    ns = lambda v: f"{'N' if v >= 0 else 'S'}{abs(v):02d}"
    ew = lambda v: f"{'E' if 0 <= v < 180 else 'W'}{abs(v) if v != 180 else 180:03d}"
    t = f"{ns(la)}{ew(lo)}_FABDEM_V1-2.tif"
    f = FAB / "tiles" / t
    if f.exists():
        return f
    if (FAB / "tiles" / (t + ".missing")).exists():
        return None
    g0, h0 = 10 * math.floor(la / 10), 10 * math.floor(lo / 10)
    grp = f"{ns(g0)}{ew(h0)}-{ns(g0 + 10)}{ew(h0 + 10)}_FABDEM_V1-2"
    f.parent.mkdir(parents=True, exist_ok=True)
    try:
        fetch(FAB_URL.format(g=grp, t=t), f)
        return f
    except urllib.error.HTTPError as e:
        if e.code == 404:
            (FAB / "tiles" / (t + ".missing")).write_text(FAB_URL.format(g=grp, t=t))
            return None
        raise


def built_tile(x, y):
    """GHS-BUILT-S E2020 100 m tile holding Mollweide (x, y) (1,000 km tiles R<row>_C<col>), downloaded once."""
    rc = f"R{int((9_000_000 - y) // 1_000_000) + 1}_C{int((x + 18_041_000) // 1_000_000) + 1}"
    d = BUILT / rc
    ok = d / "extracted.ok"
    if ok.exists():
        return next(d.glob("*.tif"))
    d.mkdir(parents=True, exist_ok=True)
    z = BUILT / f"{rc}.zip"
    if not z.exists():
        fetch(BUILT_URL.format(rc=rc), z)
    zipfile.ZipFile(z).extractall(d)
    ok.write_text("ok")
    return next(d.glob("*.tif"))


def prefetch(rows, polys):
    """Download, one file at a time, every FABDEM and GHS-BUILT-S tile the batch needs (no two workers ever fetch
    the same file)."""
    from pyproj import Transformer
    from shapely.ops import transform as shp_tf
    to_ll = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True).transform
    fk, bk = set(), set()
    for r in rows:
        p = polys[r["id"]]
        lo0, la0, lo1, la1 = shp_tf(to_ll, p).bounds
        fk |= {(la, lo) for la in range(math.floor(la0 - 0.01), math.floor(la1 + 0.01) + 1)
               for lo in range(math.floor(lo0 - 0.01), math.floor(lo1 + 0.01) + 1)}
        x0, y0, x1, y1 = p.bounds
        bk |= {(x, y) for x in (x0, x1) for y in (y0, y1)}
    for k in sorted(fk):
        fab_tile(*k)
    for k in sorted(bk):
        built_tile(*k)


def terrain_job(args):
    import warnings
    warnings.filterwarnings("ignore")
    import shapely
    r, wkb = args
    return terrain_one(r, shapely.from_wkb(wkb))


def terrain_one(row, poly):
    """Built-up-weighted median slope (%) of the settlement on FABDEM at 32 m (UTM)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.merge import merge
    from rasterio.warp import Resampling, reproject
    from shapely.ops import transform as shp_tf
    zone = int((row["lon"] + 180) // 6) + 1
    utm = f"EPSG:{(32600 if row['lat'] >= 0 else 32700) + zone}"
    to_utm = Transformer.from_crs("ESRI:54009", utm, always_xy=True).transform
    to_ll = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True).transform
    pu = shp_tf(to_utm, poly)
    x0, y0, x1, y1 = pu.bounds
    x0, y0 = RES * math.floor(x0 / RES) - 2 * RES, RES * math.floor(y0 / RES) - 2 * RES
    x1, y1 = RES * math.ceil(x1 / RES) + 2 * RES, RES * math.ceil(y1 / RES) + 2 * RES
    W, H = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
    dst_tr = rasterio.transform.from_origin(x0, y1, RES, RES)
    lo0, la0, lo1, la1 = shp_tf(to_ll, poly).bounds
    tiles = [fab_tile(la, lo) for la in range(math.floor(la0 - 0.01), math.floor(la1 + 0.01) + 1)
             for lo in range(math.floor(lo0 - 0.01), math.floor(lo1 + 0.01) + 1)]
    tiles = [t for t in tiles if t is not None]
    if not tiles:
        return {"status": "no_dem"}
    srcs = [rasterio.open(t) for t in tiles]
    dem, dtr = merge(srcs, bounds=(lo0 - 0.01, la0 - 0.01, lo1 + 0.01, la1 + 0.01), nodata=np.nan, dtype="float32")
    nod = srcs[0].nodata
    for s in srcs:
        s.close()
    dem = dem[0]
    if nod is not None:
        dem[dem == nod] = np.nan
    Z = np.full((H, W), np.nan, np.float32)
    reproject(dem, Z, src_transform=dtr, src_crs="EPSG:4326", dst_transform=dst_tr, dst_crs=utm,
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    gy, gx = np.gradient(Z.astype(float), RES)
    slope = 100.0 * np.hypot(gx, gy)
    # built-up weights: GHS-BUILT-S (m2 built per 100 m cell) averaged onto the same grid
    bx0, by0, bx1, by1 = poly.bounds
    B = np.zeros((H, W), np.float32)
    seen = set()
    for x in (bx0, bx1):
        for y in (by0, by1):
            f = built_tile(x, y)
            if f in seen:
                continue
            seen.add(f)
            with rasterio.open(f) as r:
                win = rasterio.windows.from_bounds(bx0 - 500, by0 - 500, bx1 + 500, by1 + 500, r.transform)
                win = win.round_offsets().round_lengths().intersection(rasterio.windows.Window(0, 0, r.width, r.height))
                a = r.read(1, window=win).astype(np.float32)
                a[a == r.nodata] = 0
                part = np.zeros((H, W), np.float32)
                reproject(a, part, src_transform=r.window_transform(win), src_crs=r.crs, dst_transform=dst_tr,
                          dst_crs=utm, resampling=Resampling.average, src_nodata=None, dst_nodata=0)
                B = np.maximum(B, part)
    inside = rasterize([pu], out_shape=(H, W), transform=dst_tr, fill=0, default_value=1).astype(bool)
    ok = inside & np.isfinite(slope)
    if ok.sum() < 4:
        return {"status": "no_dem"}
    s, w = slope[ok], B[ok]
    med = float(np.median(s))
    if w.sum() > 0:
        o = np.argsort(s)
        cw = np.cumsum(w[o]) / w.sum()
        wmed = float(s[o][np.searchsorted(cw, 0.5)])
    else:
        wmed = med
    return {"status": "ok", "wmedian_slope": round(wmed, 2), "median_slope": round(med, 2),
            "built_m2": float(w.sum() * (RES * RES) / 10_000.0), "cells": int(ok.sum()), "utm": utm,
            "dem_tiles": ";".join(t.name for t in tiles)}


def stage_terrain(split, workers=6):
    import pandas as pd
    L = pd.read_csv(POOL / "lists" / f"{split}.csv.gz")
    L = L[L.region.isin(["R1", "R2", "R3", "R4"])]
    L["stratum"] = [stratum(r) for _, r in L.iterrows()]
    q = QUOTA[split]
    target = {s: SPARE * (q["village"] if s == "village" else q["stratum"]) for s in L.stratum.unique()}
    target = {s: t for s, t in target.items() if t > 0}
    need_b = {s: BAND_B_SPARE for s in target if split == "dev" and s.endswith(":town")}   # D1 implementation note
    WALK.mkdir(parents=True, exist_ok=True)
    out = WALK / f"{split}_terrain.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["id"])
    have = {r["id"]: r for r in done.to_dict("records")}
    rows = list(have.values())
    for s, tgt in sorted(target.items()):
        C = L[L.stratum == s].sort_values("rank")
        n_ok = sum(1 for r in rows if r.get("stratum") == s and bool(r.get("eligible")))
        recs, k = C.to_dict("records"), 0
        n_b = lambda: sum(1 for r in rows if r.get("stratum") == s and bool(r.get("eligible")) and r.get("band") == "B")
        while (n_ok < tgt or n_b() < need_b.get(s, 0)) and k < len(recs):
            batch = []
            while len(batch) < workers and k < len(recs):
                if recs[k]["id"] not in have:
                    batch.append(recs[k])
                k += 1
            if not batch:
                continue
            P = polygons([r["id"] for r in batch])
            prefetch(batch, P)
            import shapely
            with ProcessPoolExecutor(workers) as ex:
                res = list(ex.map(terrain_job, [(r, shapely.to_wkb(P[r["id"]])) for r in batch]))
            for r, t in zip(batch, res):
                e = t.get("status") == "ok" and t["wmedian_slope"] >= ELIG
                row = {"id": r["id"], "rank": r["rank"], "stratum": s, "kind": r["kind"], "region": r["region"],
                       "iso3": r["iso3"], "block": r["block"], "lon": r["lon"], "lat": r["lat"], **t, "eligible": bool(e),
                       "band": ("B" if t["wmedian_slope"] >= BAND_B else "A") if e else ""}
                rows.append(row)
                have[r["id"]] = row
                n_ok += int(e)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(s, f"evaluated {sum(1 for x in rows if x['stratum'] == s)}, eligible {n_ok}/{tgt}", flush=True)
    D = pd.DataFrame(rows)
    print(D.groupby("stratum").agg(evaluated=("id", "size"), eligible=("eligible", "sum"),
                                   band_B=("band", lambda b: (b == "B").sum())).to_string())
    return 0


# ---------------------------------------------------------------- OSM completeness gate
SNAPSHOT = "2026-11-27T00:00:00Z"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
VEHICULAR = set("motorway trunk primary secondary tertiary unclassified residential living_street pedestrian".split())
MIN_KM = {"village": 4.0}
UA = "UrbanMorph-research/0.20 (university research project; sequential low-volume queries)"


def osm_ways(sid, bounds):
    """All highway ways in the lon/lat bounds at the pinned snapshot; raw JSON kept (gzip) with its sha256."""
    import gzip
    import hashlib
    f = WALK / "osm" / f"{sid}.json.gz"
    if f.exists():
        raw = gzip.decompress(f.read_bytes())
    else:
        w, s, e, n = bounds
        q = (f'[out:json][timeout:300][maxsize:268435456][date:"{SNAPSHOT}"];'
             f'(way["highway"]({s:.6f},{w:.6f},{n:.6f},{e:.6f}););out geom;')
        data = urllib.parse.urlencode({"data": q}).encode()
        raw, err = None, None
        for url in OVERPASS * 3:
            try:
                req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
                raw = urllib.request.urlopen(req, timeout=400).read()
                break
            except Exception as ex:                                   # busy server: try the next / again
                err = ex
                import time
                time.sleep(20)
        if raw is None:
            raise RuntimeError(f"Overpass failed for {sid}: {err}")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(gzip.compress(raw))
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def osm_gate(row, poly):
    """Completeness (D1 as amended): >= 90 % of built-up 100 m cells (built share >= 0.2) inside the polygon within
    100 m of ANY highway way; vehicular km (training classes) inside the polygon >= 15 (villages 4)."""
    import rasterio
    import shapely
    from pyproj import Transformer
    from shapely.ops import transform as shp_tf
    to_ll = Transformer.from_crs("ESRI:54009", "EPSG:4326", always_xy=True).transform
    to_moll = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True).transform
    lo0, la0, lo1, la1 = shp_tf(to_ll, poly).bounds
    J, digest = osm_ways(row["id"], (lo0 - 0.003, la0 - 0.003, lo1 + 0.003, la1 + 0.003))
    ways = [e for e in J.get("elements", []) if e.get("type") == "way" and e.get("geometry")]
    lines, veh = [], []
    for e in ways:
        xy = np.array([to_moll(p["lon"], p["lat"]) for p in e["geometry"]])
        if len(xy) < 2:
            continue
        ln = shapely.LineString(xy)
        lines.append(ln)
        hw = e.get("tags", {}).get("highway", "")
        if hw in VEHICULAR or hw.replace("_link", "") in VEHICULAR:
            veh.append(ln)
    veh_km = float(sum(l.intersection(poly).length for l in veh)) / 1000.0
    # built-up cells inside the polygon
    bx0, by0, bx1, by1 = poly.bounds
    f = built_tile((bx0 + bx1) / 2, (by0 + by1) / 2)
    with rasterio.open(f) as r:
        win = rasterio.windows.from_bounds(bx0, by0, bx1, by1, r.transform).round_offsets().round_lengths()
        a = r.read(1, window=win, boundless=True, fill_value=0).astype(float)
        tr = r.window_transform(win)
    rr, cc = np.nonzero((a >= 2000) & (a != 65535))
    cx, cy = tr.c + (cc + 0.5) * tr.a, tr.f + (rr + 0.5) * tr.e
    pts = shapely.points(cx, cy)
    inside = shapely.contains(poly, pts)
    pts = pts[inside]
    if len(pts) == 0:
        cov = float("nan")
    elif not lines:
        cov = 0.0
    else:
        tree = shapely.STRtree(lines)
        near = tree.query(pts, predicate="dwithin", distance=100.0)
        cov = float(len(np.unique(near[0])) / len(pts))
    need = MIN_KM.get(row["kind"], 15.0)
    ok = bool(np.isfinite(cov) and cov >= 0.9 and veh_km >= need)
    return {"osm_sha256": digest, "ways": len(ways), "vehicular_km": round(veh_km, 2), "built_cells": int(len(pts)),
            "coverage_any_way": round(cov, 4) if np.isfinite(cov) else None, "gate": ok}


def stage_osm(split):
    """Round robin over strata (sorted), each taking its next terrain-eligible candidate in list order until one passes
    the gate and the 2-per-block cap; dev-H keeps a band-B slot per stratum; train-H stops at 30 settlements; then
    the D1 fallback for unfilled quotas. Raw OSM responses are cached, so a re-run reproduces every earlier outcome."""
    import datetime as dt
    import pandas as pd
    if dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(SNAPSHOT.replace("Z", "+00:00")):
        print("the pinned OSM snapshot", SNAPSHOT, "has not passed yet")
        return 1
    T = pd.read_csv(WALK / f"{split}_terrain.csv")
    T = T[T.eligible.astype(bool)].sort_values("rank")
    q = QUOTA[split]
    quota = {s: (q["village"] if s == "village" else q["stratum"]) for s in T.stratum.unique()}
    quota = {s: n for s, n in quota.items() if n > 0}
    cap_total = 30 if split == "train" else None
    queues = {s: T[T.stratum == s].to_dict("records") for s in sorted(quota)}
    pos = {s: 0 for s in queues}
    acc = {s: [] for s in queues}
    per_block, log = {}, []
    P = polygons(list(T.id))
    while True:
        moved = False
        for s in sorted(queues):
            if len(acc[s]) >= quota[s] or (cap_total and sum(map(len, acc.values())) >= cap_total):
                continue
            while pos[s] < len(queues[s]):
                r = queues[s][pos[s]]
                pos[s] += 1
                if per_block.get(r["block"], 0) >= 2:
                    log.append({**r, "outcome": "block_cap"})
                    continue
                n_a = sum(1 for x in acc[s] if x["band"] == "A")
                merged = not s.endswith(":town")                     # cities: fewer than 3 x quota eligible -> merged
                if split == "dev" and s != "village" and not merged and r["band"] == "A" and n_a >= 1:
                    log.append({**r, "outcome": "band_A_full"})
                    continue
                g = osm_gate(r, P[r["id"]])
                rec = {**r, **g, "outcome": "accepted" if g["gate"] else "gate_failed"}
                log.append(rec)
                pd.DataFrame(log).to_csv(WALK / f"{split}_osm.csv", index=False)
                print(s, r["id"], r["iso3"], rec["outcome"], g["vehicular_km"], g["coverage_any_way"], flush=True)
                if g["gate"]:
                    acc[s].append(rec)
                    per_block[r["block"]] = per_block.get(r["block"], 0) + 1
                    moved = True
                    break
        if not moved:
            break
    # D1 fallback: a quota a stratum cannot fill passes to the next stratum of the same region and size (the other
    # band: a dev town stratum's band-A limit is lifted), failing that to the same region (the other size), failing
    # that it is reported unfilled. Candidates are revisited in list order (only those never gate-tested; the block
    # cap still applies); an accepted fallback keeps its own stratum and is credited to the unfilled one. On dev-H a
    # band-A fallback is taken only while band B stays >= a third of the towns and cities (D1; logged band_B_third).
    tested = {x["id"] for x in log if x["outcome"] in ("accepted", "gate_failed")}
    unfilled = {}

    def third_b_after_a():
        """D1 hard constraint (statistics, F4 review): a band-A fallback is taken only if band B stays >= a third of
        the dev-H towns and cities with it."""
        tc = [x for k, v in acc.items() if k != "village" for x in v]
        return 3 * sum(x["band"] == "B" for x in tc) >= len(tc) + 1
    for s in sorted(queues):
        need = quota[s] - len(acc[s])
        if need <= 0 or s == "village":
            unfilled[s] = max(need, 0)
            continue
        reg, size = s.split(":")
        for src, step in ((s, "other_band"), (f"{reg}:{'city' if size == 'town' else 'town'}", "same_region")):
            for r in queues.get(src, []):
                if need <= 0 or (cap_total and sum(map(len, acc.values())) >= cap_total):
                    break
                if r["id"] in tested:
                    continue
                if per_block.get(r["block"], 0) >= 2:
                    continue
                if split == "dev" and r["band"] == "A" and not third_b_after_a():
                    log.append({**r, "outcome": "band_B_third", "fallback": step, "credited_to": s})
                    continue
                g = osm_gate(r, P[r["id"]])
                tested.add(r["id"])
                rec = {**r, **g, "outcome": "accepted" if g["gate"] else "gate_failed", "fallback": step,
                       "credited_to": s}
                log.append(rec)
                pd.DataFrame(log).to_csv(WALK / f"{split}_osm.csv", index=False)
                print(s, "fallback", step, r["id"], r["iso3"], rec["outcome"], g["vehicular_km"], g["coverage_any_way"],
                      flush=True)
                if g["gate"]:
                    acc[s].append(rec)
                    per_block[r["block"]] = per_block.get(r["block"], 0) + 1
                    need -= 1
        unfilled[s] = need
    pd.DataFrame(log).to_csv(WALK / f"{split}_osm.csv", index=False)      # skips logged after the last gate query
    print("unfilled quota after the D1 fallback:", {k: v for k, v in unfilled.items() if v > 0} or "none")
    A = pd.DataFrame([x for v in acc.values() for x in v])
    A.to_csv(WALK / f"{split}_accepted.csv", index=False)
    print(A.groupby("stratum").size().to_string() if len(A) else "none accepted")
    return 0


def stage_anchors():
    """The two declared dev anchors (D1: outside every split, development references only; m20_pool.ANCHORS): the
    frame settlement whose polygon contains the declared coordinate (of the 5 nearest population centroids). Terrain
    and completeness are computed with the walk's code and REPORTED, never used to admit or drop an anchor.
    -> data/processed/m20/walk/anchor_accepted.csv (outcome = declared_anchor)."""
    import datetime as dt
    import pandas as pd
    import shapely
    from pyproj import Transformer
    from m20_pool import ANCHORS
    if dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(SNAPSHOT.replace("Z", "+00:00")):
        print("the pinned OSM snapshot", SNAPSHOT, "has not passed yet")
        return 1
    F = pd.read_csv(POOL / "frame.csv.gz")
    to_m = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True).transform
    rows = []
    for name, (la, lo) in ANCHORS.items():
        x, y = to_m(lo, la)
        near = F.assign(d=np.hypot(F.x - x, F.y - y)).nsmallest(5, "d")
        P = polygons(list(near.id))
        hit = [r for r in near.to_dict("records") if shapely.contains_xy(P[r["id"]], x, y)]
        if len(hit) != 1:
            raise SystemExit(f"{name}: {len(hit)} frame polygons contain the declared coordinate")
        r = hit[0]
        r.update(anchor=name, stratum="anchor", rank=-1)
        t = terrain_one(r, P[r["id"]])
        g = osm_gate(r, P[r["id"]])
        e = t.get("status") == "ok" and t["wmedian_slope"] >= ELIG
        rows.append({k: r[k] for k in ("id", "anchor", "rank", "stratum", "kind", "region", "iso3", "block", "lon", "lat")}
                    | t | {"eligible": bool(e), "band": ("B" if t["wmedian_slope"] >= BAND_B else "A") if e else ""}
                    | g | {"outcome": "declared_anchor"})
        print(name, r["id"], r["kind"], f"slope {t.get('wmedian_slope')}", f"gate {g['gate']}", g["vehicular_km"],
              g["coverage_any_way"], flush=True)
    pd.DataFrame(rows).to_csv(WALK / "anchor_accepted.csv", index=False)
    return 0


def md_table(D) -> str:
    """A pandas frame as a Markdown table (no tabulate in the venv)."""
    cols = [str(c) for c in D.columns]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in D.itertuples(index=False)])


def stage_flow(split):
    """D1: the CONSORT-style flow per stratum and the balance table (accepted vs eligible-not-accepted), from the walk
    outputs only (no street metric) -> outputs/m20/walk/flow_<split>.md (+ .json)."""
    import pandas as pd
    L = pd.read_csv(POOL / "lists" / f"{split}.csv.gz")
    L = L[L.region.isin(["R1", "R2", "R3", "R4"])]
    L["stratum"] = [stratum(r) for _, r in L.iterrows()]
    T = pd.read_csv(WALK / f"{split}_terrain.csv")
    O = pd.read_csv(WALK / f"{split}_osm.csv")
    A = pd.read_csv(WALK / f"{split}_accepted.csv")
    if "fallback" not in O:
        O["fallback"] = ""
    O["fallback"] = O["fallback"].fillna("")
    km_min = O["kind"].map(lambda k: MIN_KM.get(k, 15.0))
    O["why"] = np.where(O.outcome != "gate_failed", "",
                        np.where(O.coverage_any_way.fillna(0) < 0.9, np.where(O.vehicular_km < km_min, "both", "coverage"),
                                 "km"))
    rows = []
    for s in sorted(L.stratum.unique()):
        t, o, a = T[T.stratum == s], O[O.stratum == s], A[A.stratum == s]
        tested = o[o.outcome.isin(["accepted", "gate_failed"])]
        rows.append({"stratum": s, "listed": int((L.stratum == s).sum()), "terrain_evaluated": len(t),
                     "no_dem": int((t.status != "ok").sum()), "eligible": int(t.eligible.astype(bool).sum()),
                     "eligible_B": int((t.band == "B").sum()), "osm_tested": len(tested),
                     "failed_km": int((o.why == "km").sum()), "failed_coverage": int((o.why == "coverage").sum()),
                     "failed_both": int((o.why == "both").sum()), "block_cap": int((o.outcome == "block_cap").sum()),
                     "band_A_full": int((o.outcome == "band_A_full").sum()),
                     "band_B_third": int((o.outcome == "band_B_third").sum()),
                     "accepted": len(a), "accepted_B": int((a.band == "B").sum()),
                     "accepted_as_fallback": int((a.get("fallback", pd.Series(dtype=str)).fillna("") != "").sum())})
    F = pd.DataFrame(rows)
    acc_ids = set(A.id)
    E = T[T.eligible.astype(bool)].copy()
    E["group"] = np.where(E.id.isin(acc_ids), "accepted", "eligible, not accepted")
    E = E.merge(O[["id", "vehicular_km", "coverage_any_way"]], on="id", how="left")
    E = E.merge(L[["id", "pop", "area_km2"]], on="id", how="left")
    B = E.groupby("group").agg(n=("id", "size"), slope_med=("wmedian_slope", "median"),
                               band_B_share=("band", lambda b: float((b == "B").mean())),
                               area_km2_med=("area_km2", "median"), pop_med=("pop", "median"),
                               vehicular_km_med=("vehicular_km", "median")).round(2)
    tested = O[O.outcome.isin(["accepted", "gate_failed"])]
    C = tested.groupby("iso3").agg(tested=("id", "size"), passed=("outcome", lambda x: int((x == "accepted").sum())))
    C["pass_rate"] = (C.passed / C.tested).round(2)
    C = C.sort_values("tested", ascending=False)
    out = ROOT / "outputs" / "m20" / "walk"
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# {split}: flow per stratum (D1)", "", md_table(F), "",
          "Gate failures: km = under the vehicular minimum (15 km; villages 4 km); coverage = under 90 % of built-up "
          "cells within 100 m of any way; both = both.", "", f"# {split}: balance, terrain-eligible settlements", "",
          md_table(B.reset_index()), "", f"# {split}: OSM gate pass rate by country (selection on mapping quality)", "",
          md_table(C.reset_index()), ""]
    (out / f"flow_{split}.md").write_text("\n".join(md))
    (out / f"flow_{split}.json").write_text(json.dumps({"flow": rows, "balance": B.reset_index().to_dict("records"),
                                                        "by_country": C.reset_index().to_dict("records")}, indent=1,
                                                       default=float))
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["terrain", "osm", "anchors", "flow"])
    ap.add_argument("--split", choices=["dev", "train"], default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.stage == "anchors":
        return stage_anchors()
    if a.stage == "flow":
        return stage_flow(a.split)
    if a.split is None:
        ap.error("--split is required")
    return stage_terrain(a.split, a.workers) if a.stage == "terrain" else stage_osm(a.split)


if __name__ == "__main__":
    sys.exit(main())
