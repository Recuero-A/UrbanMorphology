"""Milestone 20: the per-town data build for the accepted hill towns (docs/M20_LOG.md D1, D3 / D3a / D3b, D4 inputs).

For each ACCEPTED dev-H (or train-H) settlement of data/processed/m20/walk/<split>_accepted.csv:
  (a) the REAL inputs of the metric suite (src/urbanmorph/metrics/hilltown.py): the real vehicular network (training
      classes plus links, D3), the pedestrian layer (steps, footways and paths, D1), FABDEM averaged to 32 m with its
      slope raster, the unit and land masks (settlement polygon minus water and sea), the 512 m windows with their real
      networks;
  (b) the GENERATOR inputs of the frozen whole-town sampler: a site_benchmark.Site-compatible config and its domain
      (GLO-30 on the 4 m grid with the OSM water layer, as the trained model receives it, D4) plus the city mask.
The site-benchmark machinery is reused unchanged (``TownSite`` is a thin subclass: config path, OSM clip, the city
limits are the settlement polygon).

    python scripts/m20_towns.py osm   --split dev   Overpass at the pinned snapshot -> data/raw/osm_overpass/m20/<id>.osm.gz
                                                    (+ .json: query, sha256) and data/raw/osm_extracts/cities/M20_<id>.osm.pbf
    python scripts/m20_towns.py site  --split dev   configs/sites/m20/<id>.json; data/processed/m20/towns/<id>/terrain.npz
                                                    (+ terrain_glo30_4m.tif, manifest.json, city_mask.npz) = generator inputs
    python scripts/m20_towns.py real  --split dev   data/processed/m20/towns/<id>/real.pkl (+ fabdem_32m.tif, windows/*.npz,
                                                    town_manifest.json with the sha256 of every input and output)
    python scripts/m20_towns.py check --split dev --id ID [--window]
                                                    the D3 suite on the REAL network of one built town (no generator, no z,
                                                    no composite: P2 stays sealed, D4)
Options: ``--ids A,B`` restricts the towns; ``--scratch`` = development only, BEFORE the pinned snapshot: ONE town (the
first terrain-eligible candidate of <split>_terrain.csv in the walk's order unless ``--ids``), the CURRENT OSM (no
date) and every file under data/processed/m20/scratch_unpinned/ (never the real paths). The non-scratch stages refuse
to run before 2026-11-27T00:00:00Z, and the osm stage also refuses a response whose osm_base precedes it.

Declared choices (for the committee; nothing here is fitted to data):
 1. Overpass bbox = the generator domain (the L box plus the 256 m margin of the 4 m grid, choice 2) + 300 m, as
    lon/lat bounds of the densified box (Site.clip_osm's pad), so water and streets cover the whole model domain. Query
    (one per town, sequential, retries with back-off, the m20_walk User-Agent): ways and relations with highway=*;
    natural=water|coastline, landuse=reservoir|basin and water=* (nodes, ways, relations); waterway=* ways and waterway
    multipolygon relations (a type=waterway relation is a whole river; its member ways carry the waterway tags);
    recursion down (``(._;>;)``) so every way and multipolygon is complete; ``out meta``.
 2. Site: crs = the UTM zone of the population-weighted centroid (as the terrain walk); centre = the centre of the
    polygon's UTM bbox (rounded to 1 m); L = the smallest of 2048 / 4096 / 8192 m with L >= bbox side + 2 x 300 m (a
    larger town is refused and reported); dem = the GLO-30 mosaic (m17_build_holdout.glo_mosaic); limits = the GHSL
    polygon (Mollweide WKT in the config; segmentized every 25 m, then projected).
 3. Real networks, in domain metres (site_benchmark.Site: x east, y north from the SW corner of the L box): the OSM
    ways of the clip, highway in urbanmorph.data.osm_extract.STREET_VALUES (the training classes plus links), cut with
    levels.tile_pieces (bridges, tunnels, layers from the way tags, as m17_build_holdout) and built twice:
      * ``graph``: road_graph.build_graph (30 m resample, 8 m merge, 3D-aware): the training / site-pipeline build;
      * ``native``: the same pieces with their OSM vertices, joined only where vertices coincide (a shared OSM node,
        1 mm), with class, level and structure per edge. hilltown derives both its native geometry (D3a: densify 5 m,
        Douglas-Peucker 1 m) and its convention from a graph, so the suite's input is ``native``: from a 30 m build,
        native would equal the convention and M7 and the native hairpins would be blind on real towns.
    ``unit_*`` = the same pieces clipped to the polygon (networks are cropped to the unit, hilltown choice 9).
 4. Pedestrian layer (D1, D3): highway = steps, footway, path (``pedestrian`` is a training class, so vehicular). Every
    other highway way (service, track, cycleway, bridleway, ...) is stored as ``excluded`` and enters no network.
 5. FABDEM 32 m: the FABDEM V1-2 tiles merged and averaged (rasterio Resampling.average, as m20_walk.terrain_one) onto
    a 32 m UTM grid whose lines pass through the population centroid rounded to the 4 m domain grid, inside the domain
    grid, so every 512 m window is exactly 16 x 16 cells and every 32 m cell exactly 8 x 8 cells of the 4 m water
    raster. Slope = |gradient| of that grid (heightfield.Terrain.gradient, the HillField slope). Water at 32 m = at
    least half of the cell's 4 m cells wet (the Site water layer, kinds river / lake / sea); kind = the majority.
    unit = pixel centres inside the polygon (HillField.from_polygon); land = unit, dry, with FABDEM data.
 6. Windows (D1): SW corners at the rounded centroid + 512 m x (i, j), inside the domain box; kept if the area share
    inside the polygon is >= 0.7, its 16 x 16 FABDEM cells all have data and its GLO-30 tile (window + 256 m margin,
    the generator's tile input, written as windows/<wid>.npz like m17_build_holdout) is all valid. No street minimum.
    Window networks are cut from the clip with tile_pieces + build_graph exactly as m17_build_holdout (window-local
    metres, 0..512), plus their ``native`` version and the pedestrian lines; the window unit is the 512 m square.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import pickle
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import site_benchmark as sb
from m20_walk import OVERPASS, SNAPSHOT, UA, WALK, fab_tile, polygons

LISTS = ROOT / "configs" / "m20_pool" / "lists"
SCRATCH = ROOT / "data" / "processed" / "m20" / "scratch_unpinned"
SIZES = (2048, 4096, 8192)
OSM_PAD, LIMIT_PAD = 300.0, 300.0                    # OSM: domain (with margin) + 300 m; limits: bbox + 300 m
WIN, FRES, COVER_MIN = 512.0, 32.0, 0.7
PED = ("steps", "footway", "path")
MAXSIZE, PAUSE = 1 << 28, 20.0                      # Overpass: declared 256 MiB; 20 s between towns
GLO_NOTE = "Copernicus DEM GLO-30 DSM (EGM2008), bilinear to the 4 m domain grid: the frozen TNg input (D4)"
FAB_NOTE = ("FABDEM V1-2 (Hawker et al. 2022), bare earth, CC BY-NC-SA 4.0, averaged to 32 m: rules, router and "
            "metrics (D1, D4)")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def snapshot_passed() -> bool:
    return now_utc() >= dt.datetime.fromisoformat(SNAPSHOT.replace("Z", "+00:00"))


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


# ---------------------------------------------------------------- paths (pinned or scratch)
class Paths:
    def __init__(self, scratch: bool):
        from urbanmorph.terrain import water
        self.scratch = scratch
        b = SCRATCH
        self.raw = b / "osm_raw" if scratch else ROOT / "data" / "raw" / "osm_overpass" / "m20"
        self.cities = b / "cities" if scratch else water.CLIPS
        self.osm_cache = b / "osm_cache" if scratch else water.CACHE
        self.sites = b / "sites" if scratch else ROOT / "configs" / "sites" / "m20"
        self.towns = b / "towns" if scratch else ROOT / "data" / "processed" / "m20" / "towns"
        self.out = b / "outputs" if scratch else ROOT / "outputs" / "m20" / "towns"
        if scratch:                                   # city_features reads these module globals at call time
            water.CLIPS, water.CACHE = self.cities, self.osm_cache
            for d in (self.raw, self.cities, self.osm_cache, self.sites, self.towns):
                d.mkdir(parents=True, exist_ok=True)

    def code(self, sid: str) -> str:
        return f"M20SCRATCH_{sid}" if self.scratch else f"M20_{sid}"

    def clip(self, sid: str) -> Path:
        return self.cities / f"{self.code(sid)}.osm.pbf"


# ---------------------------------------------------------------- towns and the sealed-split guard
def guard(ids, split: str):
    """Only settlements of ``split`` (dev or train); never one of the hold-H or study-H lists."""
    import pandas as pd
    own = set(pd.read_csv(LISTS / f"{split}.csv.gz", usecols=["id"]).id)
    sealed = set()
    for s in ("hold", "study"):
        sealed |= set(pd.read_csv(LISTS / f"{s}.csv.gz", usecols=["id"]).id)
    bad = [i for i in ids if i in sealed or i not in own]
    if bad:
        raise SystemExit(f"refused: {bad} not in the {split} list or in a sealed (hold-H / study-H) list")


def towns(split: str, ids, P: Paths) -> list[dict]:
    import pandas as pd
    if P.scratch:
        T = pd.read_csv(WALK / f"{split}_terrain.csv")
        T = T[T.eligible.astype(bool)].sort_values(["stratum", "rank"], kind="stable")   # the walk's first pick
        rows = T[T.id.isin(ids)] if ids else T.head(1)
        if len(rows) != 1:
            raise SystemExit("--scratch builds exactly ONE terrain-eligible town")
    else:
        f = WALK / f"{split}_accepted.csv"
        if not f.exists():
            raise SystemExit(f"{rel(f)} does not exist: run scripts/m20_walk.py osm --split {split} first")
        rows = pd.read_csv(f)
        if ids:
            rows = rows[rows.id.isin(ids)]
    rows = rows.to_dict("records")
    guard([r["id"] for r in rows], split)
    return rows


def require_pinned(P: Paths):
    if not P.scratch and not snapshot_passed():
        raise SystemExit(f"refused: the pinned OSM snapshot {SNAPSHOT} has not passed yet (now {now_utc():%Y-%m-%dT%H:%M:%SZ})"
                         "; use --scratch for development")


# ---------------------------------------------------------------- geometry of one town
def utm_of(lon: float, lat: float) -> str:
    return f"EPSG:{(32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1}"


def poly_utm(poly_moll, crs):
    import shapely
    from pyproj import Transformer
    f = Transformer.from_crs("ESRI:54009", crs, always_xy=True).transform
    return shapely.transform(shapely.segmentize(poly_moll, 25.0), lambda x, y: f(x, y), interleaved=False)


def ll_bounds(crs, X0, Y0, X1, Y1, n: int = 24):
    from pyproj import Transformer
    f = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    t = np.linspace(0, 1, n)
    xs = np.r_[X0 + (X1 - X0) * t, np.full(n, X1), X1 - (X1 - X0) * t, np.full(n, X0)]
    ys = np.r_[np.full(n, Y0), Y0 + (Y1 - Y0) * t, np.full(n, Y1), Y1 - (Y1 - Y0) * t]
    lo, la = f(xs, ys)
    return float(np.min(lo)), float(np.min(la)), float(np.max(lo)), float(np.max(la))


def town_frame(row: dict) -> dict:
    """Polygon (Mollweide and UTM), CRS, population centroid (UTM), polygon bbox (UTM)."""
    from pyproj import Transformer
    sid = row["id"]
    crs = utm_of(row["lon"], row["lat"])
    if isinstance(row.get("utm"), str) and row["utm"] != crs:
        raise ValueError(f"{sid}: UTM {crs} differs from the walk's {row['utm']}")
    pm = polygons([sid])[sid]
    pu = poly_utm(pm, crs)
    PX, PY = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(row["lon"], row["lat"])
    return {"id": sid, "crs": crs, "poly_moll": pm, "poly_utm": pu, "pop_utm": (float(PX), float(PY)), "bbox": pu.bounds}


# ---------------------------------------------------------------- stage osm
def overpass_query(bounds, date: str | None, maxsize: int = MAXSIZE) -> str:
    w, s, e, n = bounds
    bb = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
    d = f'[date:"{date}"]' if date else ""
    return (f'[out:xml][timeout:300][maxsize:{maxsize}]{d};\n(\n'
            f'  way["highway"]({bb});\n  relation["highway"]({bb});\n'
            f'  nwr["natural"~"^(water|coastline)$"]({bb});\n'
            f'  way["waterway"]({bb});\n  relation["waterway"]["type"="multipolygon"]({bb});\n'
            f'  nwr["landuse"~"^(reservoir|basin)$"]({bb});\n  nwr["water"]({bb});\n);\n(._;>;);\nout meta;')


def overpass(bounds, date, tries: int = 8, wait0: float = 60.0):
    """(raw bytes, endpoint, osm_base, query); sequential, alternating the m20_walk endpoints, exponential back-off. A
    200 response carrying an Overpass runtime-error remark is a failure; after an out-of-memory remark the declared
    maxsize doubles (up to 1 GiB). Declared resources stay small on purpose: with [timeout:900][maxsize:1 GiB] both
    public servers refused the query three times in a row (504 / 502), with 300 s / 256 MiB it ran at once."""
    err, maxsize = None, MAXSIZE
    for k in range(tries):
        url = OVERPASS[k % len(OVERPASS)]
        q = overpass_query(bounds, date, maxsize)
        data = urllib.parse.urlencode({"data": q}).encode()
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            raw = urllib.request.urlopen(req, timeout=1000).read()
            if not raw.lstrip().startswith(b"<?xml"):
                raise RuntimeError(f"not XML: {raw[:200]!r}")
            rem = re.findall(rb"<remark>(.*?)</remark>", raw[:4096] + raw[-8192:], re.S)
            if rem:
                raise RuntimeError(f"Overpass remark: {rem[0][:300]!r}")
            m = re.search(rb'osm_base="([^"]+)"', raw[:4096])
            print(f"  {url}: {len(raw) / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
            return raw, url, m.group(1).decode() if m else None, q
        except Exception as ex:                                   # busy server: back off, try the other endpoint
            err = ex
            if "out of memory" in str(ex):
                maxsize = min(2 * maxsize, 1 << 30)
            w = min(wait0 * 2 ** k, 900.0)
            print(f"  {url} failed after {time.time() - t0:.0f} s ({str(ex)[:160]}); retry in {w:.0f} s", flush=True)
            time.sleep(w)
    raise RuntimeError(f"Overpass failed: {err}")


def stage_osm(rows, P: Paths, tries: int):
    require_pinned(P)
    P.raw.mkdir(parents=True, exist_ok=True)
    done = 0
    for r in rows:
        sid = r["id"]
        meta_f, gz, clip = P.raw / f"{sid}.json", P.raw / f"{sid}.osm.gz", P.clip(sid)
        if meta_f.exists() and clip.exists():
            print(sid, "osm: cached")
            continue
        t0 = time.time()
        fr = town_frame(r)
        L, _, _, x0, y0 = domain_geom(fr)
        pad = sb.MARGIN + OSM_PAD
        bounds = ll_bounds(fr["crs"], x0 - pad, y0 - pad, x0 + L + pad, y0 + L + pad)
        date = None if P.scratch else SNAPSHOT
        if done:
            time.sleep(PAUSE)                                   # polite: one query at a time, with a pause
        print(sid, "osm: querying", "the CURRENT database (scratch, unpinned)" if P.scratch else f"date {SNAPSHOT}", flush=True)
        raw, url, base, q = overpass(bounds, date, tries)
        done += 1
        if not P.scratch and (base is None or base < SNAPSHOT):
            raise SystemExit(f"{sid}: the server's osm_base {base} precedes the snapshot {SNAPSHOT}; not saved")
        gz.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        tmp = clip.with_suffix(".tmp.pbf")
        subprocess.run(["osmium", "sort", "-O", "-o", str(tmp), str(gz)], check=True, capture_output=True)
        tmp.rename(clip)
        meta = {"id": sid, "snapshot": date, "pinned": not P.scratch,
                "note": None if not P.scratch else "SCRATCH: current database, not the pinned snapshot; development only",
                "endpoint": url, "osm_base": base, "fetched_utc": now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bbox_lonlat": bounds, "bbox_rule": f"generator domain (L {L} m + {sb.MARGIN:.0f} m margin) + {OSM_PAD:.0f} m", "query": q,
                "xml_bytes": len(raw), "xml_sha256": sha_bytes(raw), "xml_gz": rel(gz), "xml_gz_sha256": sha_file(gz),
                "clip": rel(clip), "clip_sha256": sha_file(clip), "seconds": round(time.time() - t0, 1),
                "attribution": "(c) OpenStreetMap contributors, ODbL 1.0"}
        meta_f.write_text(json.dumps(meta, indent=1))
        print(sid, f"osm: {len(raw) / 1e6:.1f} MB XML, clip {clip.stat().st_size / 1e6:.1f} MB, {meta['seconds']:.0f} s", flush=True)


# ---------------------------------------------------------------- the Site subclass and stage site
class TownSite(sb.Site):
    """site_benchmark.Site for an M20 town: config from ``Paths.sites``, data in ``Paths.towns/<id>``, the OSM clip
    from Overpass (never extracted from a regional file), the city limits = the settlement polygon."""

    def __init__(self, sid: str, P: Paths):
        self.name, self.P = sid, P
        self.cfg = json.loads((P.sites / f"{sid}.json").read_text())
        self.L = float(self.cfg["L"])
        self.crs = self.cfg["crs"]
        cx, cy = self.cfg["centre_utm"]
        self.x0, self.y0 = int(round(cx - self.L / 2)), int(round(cy - self.L / 2))
        self.n = int(self.L / sb.RES + 2 * sb.MARGIN / sb.RES)
        self.data = P.towns / sid
        self.out = P.out / sid
        self.cache = self.out / "cache"

    def clip_osm(self) -> Path:
        f = self.P.cities / f"{self.cfg['osm_clip']}.osm.pbf"
        if not f.exists():
            raise FileNotFoundError(f"{rel(f)}: run the osm stage first")
        return f

    def polygon(self):
        """The settlement polygon in domain metres."""
        import shapely
        from shapely import affinity
        pu = poly_utm(shapely.from_wkt(self.cfg["limits"]["wkt_mollweide"]), self.crs)
        return affinity.translate(pu, -self.x0, -self.y0)

    def city_mask(self, big) -> np.ndarray:
        """4 m pixel centres inside the polygon, dry, inside the L box (no largest-component rule: a polygon is the
        limit as it is)."""
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        m = rasterize([self.polygon()], out_shape=(self.n, self.n), fill=0, default_value=1, dtype="uint8",
                      transform=from_origin(-sb.MARGIN, self.L + sb.MARGIN, sb.RES, sb.RES)).astype(bool)
        X, Y = self.xy_grid()
        return m & (big["water"] == 0) & (X >= 0) & (X <= self.L) & (Y >= 0) & (Y <= self.L)


def domain_geom(fr: dict):
    """(L, centre x, centre y, x0, y0) of the site domain (choice 2), in the town's UTM."""
    X0, Y0, X1, Y1 = fr["bbox"]
    side = max(X1 - X0, Y1 - Y0) + 2 * LIMIT_PAD
    fits = [s for s in SIZES if s >= side]
    if not fits:
        raise ValueError(f"{fr['id']}: polygon bbox {X1 - X0:.0f} x {Y1 - Y0:.0f} m + 2 x {LIMIT_PAD:.0f} m exceeds {SIZES[-1]} m")
    L = fits[0]
    cx, cy = round((X0 + X1) / 2), round((Y0 + Y1) / 2)
    return L, cx, cy, int(round(cx - L / 2)), int(round(cy - L / 2))


def site_config(r: dict, fr: dict, P: Paths) -> dict:
    from m17_build_holdout import GLO_DIR, glo_mosaic, glo_name
    sid, crs = r["id"], fr["crs"]
    L, cx, cy, x0, y0 = domain_geom(fr)
    pad = sb.MARGIN + 100.0
    lo0, la0, lo1, la1 = ll_bounds(crs, x0 - pad, y0 - pad, x0 + L + pad, y0 + L + pad)
    dem = glo_mosaic(la0, la1, lo0, lo1)
    glo = [rel(GLO_DIR / f"{glo_name(la, lo)}.tif") for la in range(math.floor(la0), math.floor(la1) + 1)
           for lo in range(math.floor(lo0), math.floor(lo1) + 1)]                  # the mosaic's sources
    fab = [fab_tile(la, lo) for la in range(math.floor(la0), math.floor(la1) + 1) for lo in range(math.floor(lo0), math.floor(lo1) + 1)]
    fab = [rel(f) for f in fab if f is not None]
    if not fab:
        raise ValueError(f"{sid}: no FABDEM tile")
    meta = json.loads((P.raw / f"{sid}.json").read_text())
    ob = meta["bbox_lonlat"]
    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    bx, by = to_utm(np.array([ob[0], ob[2], ob[2], ob[0]]), np.array([ob[1], ob[1], ob[3], ob[3]]))
    PX, PY = fr["pop_utm"]
    code = P.code(sid)
    return {
        "id": code, "name": f"{sid} ({r.get('iso3')}, {r.get('stratum')}; M20 {'scratch' if P.scratch else 'pinned'})",
        "short": sid, "limits_where": "inside the settlement polygon",
        "outside": {"name": "outside the settlement", "legend": "grey = outside the settlement polygon", "rgb": [0.7, 0.7, 0.7],
                    "edge": "#555555"},
        "terrain_title": f"{sid}: GLO-30 terrain, settlement polygon (yellow), real OSM ways",
        "m20": {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()
                if k in ("id", "split", "rank", "stratum", "kind", "region", "iso3", "block", "lon", "lat", "band",
                         "wmedian_slope")}
               | {"split_file": rel(WALK / f"{r['split']}_{'terrain' if P.scratch else 'accepted'}.csv"),
                  "osm_snapshot": meta["snapshot"], "osm_pinned": meta["pinned"], "osm_base": meta["osm_base"]},
        "crs": crs, "crs_rule": "UTM zone of the population-weighted centroid",
        "centre_utm": [cx, cy], "centre_note": "centre of the settlement polygon's UTM bounding box (1 m)",
        "L": L, "L_rule": f"smallest of {SIZES} m with L >= polygon bbox side + 2 x {LIMIT_PAD:.0f} m",
        "dem": rel(dem), "dem_note": GLO_NOTE, "glo30_tiles": glo,
        "fabdem": fab, "fabdem_note": FAB_NOTE,
        "osm_extract": rel(P.clip(sid)), "osm_clip": code,
        "osm_note": "Overpass clip (stage osm); there is no regional extract",
        "osm_bbox_lonlat": ob, "osm_bbox_domain": [float(min(bx) - x0), float(min(by) - y0), float(max(bx) - x0), float(max(by) - y0)],
        "dry_wadis": False,
        "limits": {"type": "settlement polygon", "source": "GHS-SMOD E2020 R2023A V2.0 (m20_walk.polygons)",
                   "crs_source": "ESRI:54009", "wkt_mollweide": fr["poly_moll"].wkt,
                   "projection": "segmentized every 25 m, then projected to crs"},
        "city_limits": {"centre": [PX - x0, PY - y0],
                        "note": "M20: the limits are limits.wkt_mollweide; centre = the population-weighted centroid "
                                "(domain metres), read only by the radial style"},
        "population_centroid_utm": [PX, PY],
        "real": {"vehicular": STREET(), "paths": list(PED)},
        "real_names": {"vehicular": f"REAL streets ({sid})", "paths": f"REAL steps / footways / paths ({sid})"},
    }


def STREET() -> list:
    from urbanmorph.data.osm_extract import STREET_VALUES
    return list(STREET_VALUES)


def stage_site(rows, P: Paths):
    require_pinned(P)
    P.sites.mkdir(parents=True, exist_ok=True)
    for r in rows:
        sid = r["id"]
        t0 = time.time()
        if not P.clip(sid).exists():
            raise SystemExit(f"{sid}: no OSM clip; run the osm stage first")
        meta = json.loads((P.raw / f"{sid}.json").read_text())
        if not P.scratch and not (meta["pinned"] and meta["snapshot"] == SNAPSHOT):
            raise SystemExit(f"{sid}: the OSM clip is not the pinned snapshot")
        cf = P.sites / f"{sid}.json"
        if not cf.exists():
            cf.write_text(json.dumps(site_config(r, town_frame(r), P), indent=1))
        site = TownSite(sid, P)
        big = site.domain()                                  # GLO-30 4 m + water: site_benchmark.Site.domain unchanged
        mask = site.city_mask(big)
        np.savez_compressed(site.data / "city_mask.npz", mask=mask)
        tm = {"site": time.time() - t0}
        print(sid, f"site: L {site.L:.0f} m, domain {site.n} x {site.n} @ 4 m, city mask {mask.sum() * 16 / 1e6:.2f} km2, "
              f"GLO valid {big['valid'].mean():.4f}, water cells {int((big['water'] > 0).sum())}, {tm['site']:.0f} s", flush=True)
        _timing(site, tm)


def _timing(site, tm: dict):
    f = site.data / "timing.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d.update({k: round(v, 1) for k, v in tm.items()})
    f.write_text(json.dumps(d, indent=1))


# ---------------------------------------------------------------- networks
def osm_tables(site: TownSite):
    """(street GeoDataFrame in the site CRS with ``osmid``, structure way table) from the clip, as m17_build_holdout:
    one export of every highway way (attributes type, id) and one of the bridge / tunnel / layer / ford ways."""
    import geopandas as gpd
    from urbanmorph.graphs.levels import load_structure_ways
    clip = site.clip_osm()
    out = {}
    for name, filt in (("highways", ["w/highway"]), ("struct", ["w/bridge", "w/tunnel", "w/layer", "w/ford"])):
        seq = site.data / f"_{name}.geojsonseq"
        if not seq.exists():
            tmp = site.data / f"_{name}.osm.pbf"
            subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(clip), *filt], check=True, capture_output=True)
            subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "--attributes=type,id", "-o", str(seq), str(tmp)],
                           check=True, capture_output=True)
            tmp.unlink()
        out[name] = seq
    ways = load_structure_ways(out["struct"])
    g = gpd.read_file(out["highways"]).set_crs(4326, allow_override=True).to_crs(site.crs)
    g = g[g.geom_type.isin(["LineString", "MultiLineString"]) & (g["@type"] == "way")].copy()
    for c in ("bridge", "tunnel", "layer"):
        if c not in g:
            g[c] = None
    g["osmid"] = g["@id"]
    return g, ways


def cut(rc, ways, to_utm, xmin, ymin, box):
    """Pieces (lines in local metres, class, level, struct) of the ways in ``rc`` inside ``box`` (site CRS):
    levels.tile_pieces, as m17_build_holdout."""
    from urbanmorph.graphs.levels import tile_pieces
    from urbanmorph.graphs.road_graph import highway_class
    rc = rc[rc.intersects(box)].copy()
    if not len(rc):
        return [], [], [], []
    rc["geometry"] = rc.geometry.intersection(box)
    lines, cls, lev, st, _ = tile_pieces(rc, ways, to_utm, xmin, ymin, highway_class)
    return lines, cls, lev, st


def native_graph(lines, cls, lev, st, size: float):
    """The pieces with their own vertices: equal vertices (1 mm) are one node, consecutive vertices an edge; a duplicate
    edge keeps its most major class (choice 3)."""
    from urbanmorph.graphs.road_graph import RoadGraph, is_border
    items = [(np.asarray(l_, float)[:, :2], c, lv, s) for l_, c, lv, s in zip(lines, cls, lev, st) if len(l_) >= 2]
    if not items:
        z = np.zeros(0, int)
        return RoadGraph(np.zeros((0, 2)), np.zeros((0, 2), int), np.zeros(0, bool), edge_class=z, edge_level=z, edge_struct=z)
    allp = np.concatenate([p for p, *_ in items])
    _, rep, inv = np.unique(np.round(allp * 1000).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    off = np.cumsum([0] + [len(p) for p, *_ in items])
    E = np.concatenate([np.c_[inv[a:b - 1], inv[a + 1:b]] for a, b in zip(off[:-1], off[1:])])
    attr = np.concatenate([np.tile([c, lv, s], (len(p) - 1, 1)) for p, c, lv, s in items]).astype(int)
    ok = E[:, 0] != E[:, 1]
    E, attr = np.sort(E[ok], 1), attr[ok]
    o = np.lexsort((attr[:, 0], E[:, 1], E[:, 0]))                # per edge, the most major class first
    E, attr = E[o], attr[o]
    E, first = np.unique(E, axis=0, return_index=True)
    attr = attr[first]
    used, inv2 = np.unique(E.ravel(), return_inverse=True)
    N = allp[rep][used]
    return RoadGraph(N, inv2.reshape(-1, 2), is_border(N, size=size), edge_class=attr[:, 0], edge_level=attr[:, 1],
                     edge_struct=attr[:, 2])


def built_graph(lines, cls, lev, st, size: float):
    """road_graph.build_graph (30 m, 8 m, 3D-aware), border flags for a ``size`` box (as site_benchmark.Site.real)."""
    from urbanmorph.graphs.road_graph import RoadGraph, build_graph, is_border
    g = build_graph(lines, classes=cls, levels=lev, structs=st)
    return RoadGraph(g.nodes, g.edges, is_border(g.nodes, size=size) if g.n else np.zeros(0, bool), edge_class=g.classes(),
                     edge_level=g.levels(), edge_struct=g.structs())


def clip_pieces(lines, attrs, poly, min_len: float = 0.01):
    """Pieces cut to ``poly`` (a piece inside it is kept with its vertices untouched)."""
    import shapely
    from shapely.prepared import prep
    pp = prep(poly)
    out_l, out_a = [], []
    for l_, a in zip(lines, attrs):
        ls = shapely.LineString(l_)
        if pp.contains(ls):
            out_l.append(np.asarray(l_, float)[:, :2]), out_a.append(a)
            continue
        if not pp.intersects(ls):
            continue
        for part in shapely.get_parts(ls.intersection(poly)):
            if part.geom_type == "LineString" and part.length > min_len:
                out_l.append(np.asarray(part.coords)[:, :2]), out_a.append(a)
    return out_l, out_a


def line_layer(gdf, box, xmin, ymin):
    """(lines in local metres, highway, struct) of a way table cut to ``box`` (site CRS)."""
    from urbanmorph.graphs.levels import structure_of
    g = gdf[gdf.intersects(box)]
    lines, hw, st = [], [], []
    for geom, h, b, t in zip(g.geometry.intersection(box), g.highway, g.bridge, g.tunnel):
        for part in getattr(geom, "geoms", [geom]):
            if part.geom_type == "LineString" and part.length > 0:
                lines.append(np.asarray(part.coords)[:, :2] - [xmin, ymin])
                hw.append(str(h)), st.append(structure_of(b, t))
    return lines, hw, st


def km(g) -> float:
    return float(np.hypot(*(g.nodes[g.edges[:, 0]] - g.nodes[g.edges[:, 1]]).T).sum() / 1000) if len(g.edges) else 0.0


# ---------------------------------------------------------------- FABDEM at 32 m
def fab32(site: TownSite, big, pc):
    """FABDEM averaged to the 32 m grid through ``pc`` (domain metres, on the 4 m grid), inside the domain grid
    (choice 5): elevation, slope (%), water share / kind from the 4 m water raster, and the grid in domain metres."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    from urbanmorph.terrain.heightfield import Terrain
    M, L, R4 = sb.MARGIN, site.L, sb.RES
    k = int(FRES / R4)
    gx0 = pc[0] - FRES * math.floor((pc[0] + M) / FRES)
    gy1 = pc[1] + FRES * math.floor((L + M - pc[1]) / FRES)
    W, H = int((L + M - gx0) // FRES), int((gy1 + M) // FRES)
    X0, Y1 = site.x0 + gx0, site.y0 + gy1
    lo0, la0, lo1, la1 = ll_bounds(site.crs, X0, Y1 - H * FRES, X0 + W * FRES, Y1)
    tiles = [ROOT / t for t in site.cfg["fabdem"]]
    srcs = [rasterio.open(t) for t in tiles]
    dem, dtr = merge(srcs, bounds=(lo0 - 0.01, la0 - 0.01, lo1 + 0.01, la1 + 0.01), nodata=np.nan, dtype="float32")
    nod = srcs[0].nodata
    for s in srcs:
        s.close()
    dem = dem[0]
    if nod is not None:
        dem[dem == nod] = np.nan
    Z = np.full((H, W), np.nan, np.float32)
    reproject(dem, Z, src_transform=dtr, src_crs="EPSG:4326", dst_transform=from_origin(X0, Y1, FRES, FRES), dst_crs=site.crs,
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    T = Terrain(Z, float(gx0), float(gy1), FRES, native_res=30.0,
                provenance={"source": FAB_NOTE, "tiles": site.cfg["fabdem"]})
    gx, gy = T.gradient()
    slope = (np.hypot(gx, gy) * 100).astype(np.float32)
    r4, c4 = int(round((L + M - gy1) / R4)), int(round((gx0 + M) / R4))
    w4 = big["water"][r4:r4 + k * H, c4:c4 + k * W].reshape(H, k, W, k)
    share = (w4 > 0).mean((1, 3)).astype(np.float32)
    counts = np.stack([(w4 == q).sum((1, 3)) for q in (1, 2, 3)], -1)
    kind = np.where(share >= 0.5, 1 + counts.argmax(-1), 0).astype(np.int8)
    return {"elev": Z, "slope_pct": slope, "x0": float(gx0), "y1": float(gy1), "res": FRES, "shape": (H, W),
            "water": share >= 0.5, "water_kind": kind, "water_share": share, "tiles": site.cfg["fabdem"],
            "grid_rule": "lines through the population centroid rounded to the 4 m domain grid; inside the domain grid",
            "transform_utm": [X0, Y1, FRES], "source": FAB_NOTE}


def terrain_of(t32):
    from urbanmorph.terrain.heightfield import Terrain
    T = Terrain(t32["elev"], t32["x0"], t32["y1"], t32["res"], native_res=30.0)
    T.water = t32["water"]
    T.water_kind = t32["water_kind"]
    return T


def unit_masks(t32, poly):
    import shapely
    H, W = t32["shape"]
    X, Y = np.meshgrid(t32["x0"] + (np.arange(W) + 0.5) * FRES, t32["y1"] - (np.arange(H) + 0.5) * FRES)
    unit = shapely.contains_xy(poly, X, Y)                     # as HillField.from_polygon
    return unit, unit & ~t32["water"] & np.isfinite(t32["slope_pct"])


# ---------------------------------------------------------------- stage real
def stage_real(rows, P: Paths):
    require_pinned(P)
    import shapely
    from pyproj import Transformer
    for r in rows:
        sid = r["id"]
        t0 = time.time()
        site = TownSite(sid, P)
        big = site.domain()
        L, x0, y0 = site.L, site.x0, site.y0
        poly = site.polygon()
        PX, PY = site.cfg["population_centroid_utm"]
        pc = (4.0 * round((PX - x0) / 4.0), 4.0 * round((PY - y0) / 4.0))
        G, ways = osm_tables(site)
        to_utm = Transformer.from_crs("EPSG:4326", site.crs, always_xy=True).transform
        street_set = set(STREET())
        veh, ped = G[G.highway.isin(street_set)], G[G.highway.isin(PED)]
        oth = G[~G.highway.isin(street_set | set(PED))]
        dom = shapely.box(x0, y0, x0 + L, y0 + L)
        lines, cls, lev, st = cut(veh, ways, to_utm, x0, y0, dom)
        attrs = list(zip(cls, lev, st))
        ul, ua = clip_pieces(lines, attrs, poly)
        unz = lambda a: [list(x) for x in zip(*a)] if a else [[], [], []]
        uc, ulv, ust = unz(ua)
        vehicular = {"graph": built_graph(lines, cls, lev, st, L), "native": native_graph(lines, cls, lev, st, L),
                     "unit_graph": built_graph(ul, uc, ulv, ust, L), "unit_native": native_graph(ul, uc, ulv, ust, L),
                     "pieces": {"lines": lines, "class": cls, "level": lev, "struct": st},
                     "note": "OSM highway in STREET_VALUES (training classes + links); graph / unit_graph = build_graph "
                             "(30 m, 8 m, 3D-aware); native / unit_native = OSM vertices (hilltown input); unit_* = clipped "
                             "to the polygon"}
        t_net = time.time()
        pl, phw, pst = line_layer(ped, dom, x0, y0)
        upl, upa = clip_pieces(pl, list(zip(phw, pst)), poly)
        pedestrian = {"lines": pl, "highway": phw, "struct": pst, "steps": [h == "steps" for h in phw],
                      "unit_lines": upl, "unit_highway": [a[0] for a in upa], "unit_steps": [a[0] == "steps" for a in upa],
                      "note": "D1 pedestrian layer: highway = steps, footway, path; hilltown ped = (unit_lines, unit_steps)"}
        ol, ohw, _ = line_layer(oth, dom, x0, y0)
        excluded = {"lines": ol, "highway": ohw, "note": "every other highway way (service, track, cycleway, ...): in no network"}
        # FABDEM 32 m, unit and land
        t32 = fab32(site, big, pc)
        unit, land = unit_masks(t32, poly)
        t32.update(unit=unit, land=land)
        from m16_build_terrain_pack import write_tif
        from rasterio.transform import from_origin
        X0, Y1, _ = t32["transform_utm"]
        write_tif(site.data / "fabdem_32m.tif", t32["elev"], from_origin(X0, Y1, FRES, FRES), site.crs)
        t_fab = time.time()
        # windows
        wins = windows(site, big, t32, poly, pc, veh, ped, ways, to_utm)
        t_win = time.time()
        real = {"id": sid, "split": site.cfg["m20"].get("split"), "scratch": P.scratch, "crs": site.crs,
                "origin_utm_sw": [x0, y0], "L": L,
                "frame": "domain metres: x east, y north from the SW corner of the L box (site_benchmark.Site); windows "
                         "in window-local metres 0..512 with their SW corner x0, y0 in domain metres",
                "osm": json.loads((P.raw / f"{sid}.json").read_text()) | {"query": None},
                "polygon": poly, "polygon_mollweide_wkt": site.cfg["limits"]["wkt_mollweide"],
                "population_centroid_domain": [PX - x0, PY - y0], "window_grid_origin": list(pc),
                "vehicular": vehicular, "pedestrian": pedestrian, "excluded": excluded, "terrain32": t32, "windows": wins,
                "built_utc": now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"), "script": rel(Path(__file__))}
        f = site.data / "real.pkl"
        f.write_bytes(pickle.dumps(real))
        tm = {"real_networks": t_net - t0, "real_fabdem32": t_fab - t_net, "real_windows": t_win - t_fab,
              "real_total": time.time() - t0}
        _timing(site, tm)
        man = manifest(site, real, f)
        kept = [w for w in wins if w["keep"]]
        print(sid, f"real: vehicular {km(vehicular['graph']):.1f} km (unit {km(vehicular['unit_native']):.1f} km native), "
              f"pedestrian {sum(np.hypot(*np.diff(l_, axis=0).T).sum() for l_ in upl) / 1000:.1f} km in the unit "
              f"({sum(pedestrian['unit_steps'])} steps lines), unit land {land.sum() * FRES ** 2 / 1e6:.2f} km2, "
              f"windows {len(kept)} kept / {len(wins)} candidates, {tm['real_total']:.0f} s", flush=True)
        print("  manifest:", rel(site.data / "town_manifest.json"), f"({len(man['inputs'])} inputs, {len(man['outputs'])} outputs)")


def windows(site, big, t32, poly, pc, veh, ped, ways, to_utm):
    """512 m windows on the grid through ``pc`` (choice 6), with their real networks and GLO-30 tiles."""
    import shapely
    L, M, R4 = site.L, sb.MARGIN, sb.RES
    k32 = int(WIN / FRES)
    (site.data / "windows").mkdir(exist_ok=True)
    out = []
    i_lo, i_hi = math.ceil(-pc[0] / WIN), math.floor((L - WIN - pc[0]) / WIN)
    j_lo, j_hi = math.ceil(-pc[1] / WIN), math.floor((L - WIN - pc[1]) / WIN)
    for j in range(j_lo, j_hi + 1):
        for i in range(i_lo, i_hi + 1):
            x, y = pc[0] + i * WIN, pc[1] + j * WIN
            b = shapely.box(x, y, x + WIN, y + WIN)
            cover = float(b.intersection(poly).area / WIN ** 2)
            if cover <= 0:
                continue
            C0, R0 = int(round((x - t32["x0"]) / FRES)), int(round((t32["y1"] - y - WIN) / FRES))
            fab_ok = bool(np.isfinite(t32["elev"][R0:R0 + k32, C0:C0 + k32]).all())
            c0, r0 = int(round(x / R4)), int(round((L - WIN - y) / R4))
            n_t = int((WIN + 2 * M) / R4)
            tile = {k_: big[k_][r0:r0 + n_t, c0:c0 + n_t] for k_ in ("elev", "valid", "water")}
            glo_ok = bool(tile["valid"].all())
            wid = f"{site.name}_x{i:+d}_y{j:+d}"
            w = {"id": wid, "i": i, "j": j, "x0": x, "y0": y, "x0_utm": site.x0 + x, "y0_utm": site.y0 + y, "cover": cover,
                 "fab_valid": fab_ok, "glo_valid": glo_ok, "keep": cover >= COVER_MIN and fab_ok and glo_ok,
                 "fab_rc": [R0, C0], "glo_rc": [r0, c0]}
            if w["keep"]:
                X0, Y0 = site.x0 + x, site.y0 + y
                wb = shapely.box(X0, Y0, X0 + WIN, Y0 + WIN)
                lines, cls, lev, st = cut(veh, ways, to_utm, X0, Y0, wb)
                pl, phw, _ = line_layer(ped, wb, X0, Y0)
                f = site.data / "windows" / f"{wid}.npz"
                np.savez_compressed(f, elev=tile["elev"].astype(np.float32), valid=tile["valid"], water=tile["water"])
                w.update(graph=built_graph(lines, cls, lev, st, WIN), native=native_graph(lines, cls, lev, st, WIN),
                         ped_lines=pl, ped_steps=[h == "steps" for h in phw], tile_npz=rel(f), tile_npz_sha256=sha_file(f))
                w["street_km"] = km(w["graph"])
            out.append(w)
    return out


def manifest(site, real, f_real) -> dict:
    """sha256 of every input and output of one town (town_manifest.json)."""
    P, cfg = site.P, site.cfg
    sid = site.name
    split = cfg["m20"].get("split") or "dev"
    ins = {"overpass_xml_gz": rel(P.raw / f"{sid}.osm.gz"), "overpass_meta": rel(P.raw / f"{sid}.json"),
           "osm_clip": rel(site.clip_osm()), "site_config": rel(P.sites / f"{sid}.json"), "glo30_mosaic": cfg["dem"]}
    for t in cfg["fabdem"]:
        ins[f"fabdem:{Path(t).name}"] = t
    ins.update({f"glo30_tile:{Path(t).name}": t for t in cfg["glo30_tiles"]})
    for s in ("dev", "train", "hold", "study"):
        ins[f"list:{s}"] = rel(LISTS / f"{s}.csv.gz")
    for n in (f"{split}_terrain.csv", f"{split}_accepted.csv"):
        if (WALK / n).exists():
            ins[f"walk:{n}"] = rel(WALK / n)
    outs = {"terrain_npz": site.data / "terrain.npz", "terrain_glo30_4m_tif": site.data / "terrain_glo30_4m.tif",
            "city_mask_npz": site.data / "city_mask.npz", "fabdem_32m_tif": site.data / "fabdem_32m.tif", "real_pkl": f_real}
    outs.update({f"window:{w['id']}": ROOT / w["tile_npz"] for w in real["windows"] if w["keep"]})
    pm = real["polygon_mollweide_wkt"].encode()
    man = {"id": sid, "scratch": P.scratch, "osm_snapshot": real["osm"]["snapshot"], "osm_base": real["osm"]["osm_base"],
           "osm_xml_sha256": real["osm"]["xml_sha256"], "polygon_mollweide_wkt_sha256": sha_bytes(pm),
           "inputs": {k: {"path": v, "sha256": sha_file(ROOT / v)} for k, v in ins.items()},
           "outputs": {k: {"path": rel(v), "sha256": sha_file(v)} for k, v in outs.items()},
           "counts": {"vehicular_km_domain": km(real["vehicular"]["graph"]), "vehicular_km_unit_native": km(real["vehicular"]["unit_native"]),
                      "vehicular_km_unit_built": km(real["vehicular"]["unit_graph"]),
                      "pedestrian_lines_unit": len(real["pedestrian"]["unit_lines"]),
                      "unit_land_km2": float(real["terrain32"]["land"].sum() * FRES ** 2 / 1e6),
                      "windows_candidates": len(real["windows"]), "windows_kept": sum(w["keep"] for w in real["windows"])},
           "timing_s": json.loads((site.data / "timing.json").read_text()),
           "attribution": ["(c) OpenStreetMap contributors, ODbL 1.0", GLO_NOTE + "; (c) DLR e.V. 2010-2014 and (c) Airbus "
                           "Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA",
                           FAB_NOTE, "GHSL GHS-SMOD E2020 R2023A V2.0, CC BY 4.0"]}
    (site.data / "town_manifest.json").write_text(json.dumps(man, indent=1, default=float))
    return man


# ---------------------------------------------------------------- stage check (real network only)
def load_town(sid: str, P: Paths) -> dict:
    f = P.towns / sid / "real.pkl"
    if not f.exists():
        raise SystemExit(f"{rel(f)} does not exist: run the real stage first{' (with --scratch)' if P.scratch else ''}")
    return pickle.loads(f.read_bytes())


def hill_field(real: dict, window: dict | None = None):
    """The HillField of the town (unit = the polygon) or of one window (unit = its 512 m square), on the 32 m FABDEM."""
    import shapely
    from urbanmorph.metrics.hilltown import HillField
    t32 = real["terrain32"]
    S = terrain_of(t32)
    if window is None:
        return HillField(S, t32["land"], T=S, unit=t32["unit"], uid=real["id"])
    return HillField.from_polygon(S, shapely.box(window["x0"], window["y0"], window["x0"] + WIN, window["y0"] + WIN),
                                  T=S, water=t32["water"], uid=window["id"])


def shifted(g, dx, dy):
    from urbanmorph.graphs.road_graph import RoadGraph
    return RoadGraph(g.nodes + [dx, dy], g.edges, g.border, **g._meta())


def stage_check(sid: str, P: Paths, window: bool, access: bool):
    import pandas as pd
    from urbanmorph.metrics import hilltown as H
    real = load_town(sid, P)
    print(f"{sid}: {'SCRATCH (current OSM, unpinned)' if real['scratch'] else 'pinned ' + str(real['osm']['snapshot'])}; "
          "the REAL network only: raw metric values, no z, no composite")
    units = [("town", hill_field(real), real["vehicular"]["unit_native"],
              (real["pedestrian"]["unit_lines"], real["pedestrian"]["unit_steps"]))]
    if window:
        kept = sorted((w for w in real["windows"] if w["keep"]), key=lambda w: -w["cover"])
        if kept:
            w = kept[0]
            units.append((w["id"], hill_field(real, w), shifted(w["native"], w["x0"], w["y0"]),
                          ([np.asarray(l_) + [w["x0"], w["y0"]] for l_ in w["ped_lines"]], w["ped_steps"])))
    for name, F, g, ped in units:
        t0 = time.time()
        m = H.unit_metrics(g, F, ped=ped, access=access)
        sec = time.time() - t0
        outc = m.pop("_outcomes", {})
        cells = list(H.KEYS) + [H.POOLED]
        rows = {}
        for k, v in m.items():
            met, cell = k.split("@")
            rows.setdefault(met, {})[cell] = v
        D = pd.DataFrame(rows).T
        per = D[[c for c in cells if c in D.columns]].dropna(how="all")
        other = D[[c for c in D.columns if c not in cells]].dropna(how="all")
        pd.set_option("display.width", 200)
        print(f"\n== {name}: unit land {F.land.sum() * F.res ** 2 / 1e6:.2f} km2, street {m['street_km@unit']:.1f} km, "
              f"{sec:.0f} s ==")
        print(per.to_string(float_format=lambda x: f"{x:.3g}"))
        print(other.to_string(float_format=lambda x: f"{x:.4g}", na_rep=""))
        if outc:
            print("access outcomes (served, not joined, disconnected, detour, steep):", json.dumps(outc))
        f = P.towns / sid / f"check_{name}.json"
        f.write_text(json.dumps({"unit": name, "seconds": sec, "metrics": m, "outcomes": outc}, indent=1, default=float))


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["osm", "site", "real", "check", "all"])
    ap.add_argument("--split", choices=["dev", "train"], required=True)
    ap.add_argument("--ids", default="", help="comma-separated settlement ids (default: every accepted town)")
    ap.add_argument("--id", default="", help="check: the town")
    ap.add_argument("--scratch", action="store_true", help="development: ONE town, current OSM, scratch paths")
    ap.add_argument("--tries", type=int, default=8)
    ap.add_argument("--window", action="store_true", help="check: also the window with the largest polygon share")
    ap.add_argument("--no-access", action="store_true", help="check: skip M6 / G1 / pedestrian reach")
    a = ap.parse_args()
    P = Paths(a.scratch)
    if a.stage == "check":
        guard([a.id], a.split)
        stage_check(a.id, P, a.window, not a.no_access)
        return 0
    require_pinned(P)
    rows = towns(a.split, [i for i in a.ids.split(",") if i], P)
    for r in rows:
        r["split"] = a.split
    if a.stage in ("osm", "all"):
        stage_osm(rows, P, a.tries)
    if a.stage in ("site", "all"):
        stage_site(rows, P)
    if a.stage in ("real", "all"):
        stage_real(rows, P)
    return 0


if __name__ == "__main__":
    sys.exit(main())
