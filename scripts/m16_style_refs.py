"""Milestone 16 real style references: labelled 512 m street-network windows (grid / organic / radial / cul-de-sac).

    python scripts/m16_style_refs.py --stage fetch     # raw OSM (OSM API 0.6 map call; Overpass as fallback)
    python scripts/m16_style_refs.py --stage build     # RoadGraphs in tile metres (same street definition as training)
    python scripts/m16_style_refs.py --stage metrics   # outcome_row + morphology_row per window
    python scripts/m16_style_refs.py --stage figure    # 6 x 6 overview of all windows (black on white, id only)
    python scripts/m16_style_refs.py --stage all

Purpose: (a) calibrate M16 recognisability thresholds from REAL between-style differences and within-style
variability ("calib" subset, 5 windows per style); (b) real examples for the blinded style-sorting test ("sort"
subset, 4 windows per style). The two subsets are disjoint (different cities / areas).

Labels come from documented typology, never from our metrics: every window cites a source (Wikipedia, UNESCO,
Britannica) that describes the area as a gridiron / organic-medieval / radial (spoke-and-ring, star, trident) /
cul-de-sac (Radburn layout) street pattern. Candidates whose typology could not be verified were dropped.

Exclusions (declared before any result): no window in the held-out metros Lleida, Alicante, Seoul, Daegu, Lima,
Trujillo, and none in the USA (held-out external country). M15 development metros are not used.

Street definition = the training graphs: ways whose ``highway`` is in osm_extract.STREET_VALUES (ROAD_HIGHWAY_CLASSES
+ *_link; area=yes ways kept, as the osmium/OSMnx path keeps them), split at every node shared by two street ways and at
way ends (what OSMnx simplification produces), projected to the WGS84 UTM zone of the window centre, clipped to the
512 m square centred on the declared point (tile-local metres, origin SW corner, x east, y north), and built with
road_graph.build_graph with highway_class classes and the M12 levels / structures (bridge, tunnel, layer from the way
tags themselves).

Radial windows are centred on the documented focus (the declared centre is the radial focus); metrics use
centre = that focus, other styles the tile centre (256, 256).

Outputs: data/raw/m16/style_refs/<id>.osm (+ <id>.json: query, bbox, sha256, fetch time, OSM data timestamp);
data/processed/m16/style_refs/{graphs.pkl, metrics.csv, manifest.csv, manifest.json}; published copies in
configs/m16_style_refs/; outputs/m16/figures/style_refs.png; outputs/m16/style_refs_summary.md.
Data licence: (c) OpenStreetMap contributors, ODbL 1.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from urbanmorph.config import PROJECT_ROOT

RAW = PROJECT_ROOT / "data" / "raw" / "m16" / "style_refs"
PROC = PROJECT_ROOT / "data" / "processed" / "m16" / "style_refs"
CONF = PROJECT_ROOT / "configs" / "m16_style_refs"
OUT = PROJECT_ROOT / "outputs" / "m16"
VERSION = "m16-style-refs-1"
TILE = 512.0
MARGIN_M = 100.0                      # fetched around the window (the map call returns complete ways)
UA = "UrbanMorph-research (street-pattern reference windows; low request rate)"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
OSM_API = "https://api.openstreetmap.org/api/0.6/map?bbox={w},{s},{e},{n}"
LICENCE = "(c) OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright)"

HELD_OUT_METROS = ["Lleida", "Alicante", "Seoul", "Daegu", "Lima", "Trujillo"]
HELD_OUT_COUNTRIES = ["USA"]
RADBURN = "https://en.wikipedia.org/wiki/Radburn_design_housing"

# id, name, city, country, style, subset, centre lat, lon, source URL, source quote / note.
# Radial windows: the centre IS the documented radial focus.
WINDOWS = [
    # ---- grid
    ("G01", "Dreta de l'Eixample", "Barcelona", "Spain", "grid", "calib", 41.3930, 2.1660,
     "https://en.wikipedia.org/wiki/Eixample", "Cerda's Eixample: 'a strict grid pattern crossed by wide avenues'"),
    ("G02", "Quadrate (centre)", "Mannheim", "Germany", "grid", "calib", 49.4875, 8.4665,
     "https://en.wikipedia.org/wiki/Mannheim", "'central Mannheim's streets and avenues are laid out in a grid pattern'; blocks named by letter and number"),
    ("G03", "Hoddle Grid (CBD)", "Melbourne", "Australia", "grid", "calib", -37.8150, 144.9620,
     "https://en.wikipedia.org/wiki/Hoddle_Grid", "1.61 x 0.80 km grid of streets; blocks 10 chains (200 m) square"),
    ("G04", "Balvanera", "Buenos Aires", "Argentina", "grid", "calib", -34.6080, -58.4030,
     "https://en.wikipedia.org/wiki/Balvanera", "'As in most of Buenos Aires, the streets are laid out in a checkboard pattern'"),
    ("G05", "Nakagyo-ku (Rakuchu)", "Kyoto", "Japan", "grid", "calib", 35.0070, 135.7620,
     "https://en.wikipedia.org/wiki/Kyoto", "'The streets in the modern-day wards of Kamigyo-ku, Nakagyo-ku, and Shimogyo-ku still follow a grid pattern'"),
    ("G06", "City centre (Light's plan), east of Victoria Square", "Adelaide", "Australia", "grid", "sort", -34.9256, 138.6043,
     "https://en.wikipedia.org/wiki/Adelaide_city_centre", "'following a grid layout, with streets running at right angles to each other'"),
    ("G07", "Chuo-ku (south of Odori)", "Sapporo", "Japan", "grid", "sort", 43.0570, 141.3530,
     "https://en.wikipedia.org/wiki/Sapporo", "'Roadways in the urban district are laid out in a grid pattern'"),
    ("G08", "Ensanche (Pla del Remei / Gran Via)", "Valencia", "Spain", "grid", "sort", 39.4666, -0.3681,
     "https://es.wikipedia.org/wiki/Ensanche_(Valencia)", "'manzanas cuadradas (de unos 100 metros de largo), achaflanadas'; 'trama de ensanche' around Gran Via"),
    ("G09", "San Salvario", "Turin", "Italy", "grid", "sort", 45.0570, 7.6800,
     "https://en.wikipedia.org/wiki/Turin", "San Salvario: 'just as the city centre it is characterised by the grid plan'"),
    # ---- organic
    ("O01", "Casco historico", "Toledo", "Spain", "organic", "calib", 39.8575, -4.0230,
     "https://www.britannica.com/place/Toledo-Spain", "old city 'traversed by narrow, winding streets', centring on the Plaza de Zocodover"),
    ("O02", "Historic centre (Juderia)", "Cordoba", "Spain", "organic", "calib", 37.8800, -4.7810,
     "https://en.wikipedia.org/wiki/Historic_centre_of_C%C3%B3rdoba", "'comprises the buildings and narrow winding streets around the cathedral'"),
    ("O03", "Medina", "Tunis", "Tunisia", "organic", "calib", 36.7985, 10.1700,
     "https://en.wikipedia.org/wiki/Medina_of_Tunis", "'not obeying geometrical layouts or formal compositions such as gridlines'; main streets, secondary streets and small cul-de-sacs"),
    ("O04", "Albaicin", "Granada", "Spain", "organic", "calib", 37.1815, -3.5925,
     "https://en.wikipedia.org/wiki/Albaic%C3%ADn", "retains 'its medieval street plan dating back to the Nasrid period (13th to 15th centuries)'"),
    ("O05", "Barri Gotic", "Barcelona", "Spain", "organic", "calib", 41.3825, 2.1753,
     "https://en.wikipedia.org/wiki/Gothic_Quarter,_Barcelona", "'retains a labyrinthine street plan, with many small streets opening out into squares'"),
    ("O06", "Medina", "Marrakesh", "Morocco", "organic", "sort", 31.6300, -7.9874,
     "https://en.wikipedia.org/wiki/Marrakesh", "medina: 'a myriad of narrow winding streets that lead through a series of smaller markets'"),
    ("O07", "Medina", "Kairouan", "Tunisia", "organic", "sort", 35.6785, 10.0990,
     "https://whc.unesco.org/en/list/499", "UNESCO criterion (v): the medina 'has preserved its network of winding streets and courtyard houses'"),
    ("O08", "Alfama", "Lisbon", "Portugal", "organic", "sort", 38.71245, -9.13046,
     "https://en.wikipedia.org/wiki/Alfama", "'has remained a labyrinth of narrow streets and small squares'"),
    ("O09", "Barrio de Santa Cruz", "Seville", "Spain", "organic", "sort", 37.3870, -5.9890,
     "https://es.wikipedia.org/wiki/Barrio_de_Santa_Cruz_(Sevilla)", "'mantiene una irregularidad en su trazado ... con vias especialmente estrechas y frecuentes cambios de direccion'"),
    # ---- radial (centre = radial focus)
    ("R01", "Piazza Grande (star fortress town)", "Palmanova", "Italy", "radial", "calib", 45.9054, 13.3100,
     "https://en.wikipedia.org/wiki/Palmanova", "nine-pointed star fortress, 'three nine-sided ring roads intersecting in the main military radiating streets'"),
    ("R02", "Place Charles de Gaulle (Etoile)", "Paris", "France", "radial", "calib", 48.87389, 2.29472,
     "https://en.wikipedia.org/wiki/Place_Charles_de_Gaulle", "'Twelve avenues radiate from the Place Charles de Gaulle ... forming a star'"),
    ("R03", "Markt (old fortress town)", "Coevorden", "Netherlands", "radial", "calib", 52.66119, 6.74103,
     "https://en.wikipedia.org/wiki/Coevorden", "'The old part still has the star-shaped radial streets'; 'laid out in a radial pattern within polygonal fortifications'"),
    ("R04", "Piazza Carafa (hexagonal plan)", "Grammichele", "Italy", "radial", "calib", 37.21472, 14.63639,
     "https://it.wikipedia.org/wiki/Grammichele", "hexagonal plan: 'Cinque arterie anulari ... attorno alla piazza centrale; da questa si irradiano altre sei arterie'"),
    ("R05", "Connaught Place", "New Delhi", "India", "radial", "calib", 28.63278, 77.21972,
     "https://en.wikipedia.org/wiki/Connaught_Place,_New_Delhi", "concentric Inner/Middle/Outer Circle with 'radial roads beaming in all directions'"),
    ("R06", "Place de la Nation", "Paris", "France", "radial", "sort", 48.84833, 2.39583,
     "https://fr.wikipedia.org/wiki/Place_de_la_Nation_(Paris)", "twelve streets and avenues start from the square (listed clockwise from avenue du Trone)"),
    ("R07", "City Hill (Griffin plan)", "Canberra", "Australia", "radial", "sort", -35.28194, 149.12861,
     "https://en.wikipedia.org/wiki/City_Hill,_Canberra", "Vernon Circle and the concentric London Circuit; Griffin's six avenues radiating from City Hill"),
    ("R08", "Raatihuoneentori (circular plan)", "Hamina", "Finland", "radial", "sort", 60.5697, 27.1978,
     "https://en.wikipedia.org/wiki/Hamina", "'the circular town plan, designed by Axel Lowen' (Renaissance ideal-city fortress plan)"),
    ("R09", "Piazza del Popolo (Tridente)", "Rome", "Italy", "radial", "sort", 41.9107, 12.4763,
     "https://en.wikipedia.org/wiki/Piazza_del_Popolo", "three streets 'branch out from the piazza into the city, forming the so-called trident'"),
    # ---- cul-de-sac (Radburn layouts; "Living streets mostly take the form of culs-de-sac and turning loops")
    ("C01", "Craigshill", "Livingston", "United Kingdom", "culdesac", "calib", 55.8967, -3.4913,
     RADBURN, "'Much of the early new town of Livingston ... was built to a Radburn design plan'"),
    ("C02", "Charnwood", "Canberra", "Australia", "culdesac", "calib", -35.2002, 149.0341,
     RADBURN, "Radburn model in 1960s Canberra suburbs: Charnwood, Curtin, Garran, part of Hughes, Rivett"),
    ("C03", "Crestwood Estate, Thornlie", "Perth", "Australia", "culdesac", "calib", -32.0616, 115.9625,
     RADBURN, "Crestwood Estate in Thornlie: 'highly successful implementation of the Radburn principles' (1970s)"),
    ("C04", "Great Hollands", "Bracknell", "United Kingdom", "culdesac", "calib", 51.3982, -0.7677,
     RADBURN, "Bracknell: model used particularly in Great Hollands, Wildridings and Hanworth"),
    ("C05", "Bransholme", "Kingston upon Hull", "United Kingdom", "culdesac", "calib", 53.78995, -0.32512,
     RADBURN, "Hull: 'several other council estates in the city following that pattern - most notably Bransholme'"),
    ("C06", "Rivett", "Canberra", "Australia", "culdesac", "sort", -35.3471, 149.0380,
     RADBURN, "Radburn model in 1960s Canberra suburbs (Rivett cited as an example)"),
    ("C07", "Kildala", "Kitimat", "Canada", "culdesac", "sort", 54.0496, -128.6619,
     RADBURN, "Radburn design principles incorporated into the plan of the Alcan company town Kitimat"),
    ("C08", "Stanshawe Estate", "Yate", "United Kingdom", "culdesac", "sort", 51.5389, -2.4182,
     RADBURN, "'The Stanshawe Estate in Yate ... was developed using the Radburn model'"),
    ("C09", "Nechako", "Kitimat", "Canada", "culdesac", "sort", 54.0558, -128.6273,
     RADBURN, "Radburn design principles incorporated into the plan of the Alcan company town Kitimat"),
]
COLS = ["id", "name", "city", "country", "style", "subset", "lat", "lon", "source_url", "source_note"]
# Candidates considered and dropped (recorded; never silently): typology not verifiable, or the documented pattern is
# not a street network under the training definition inside a 512 m window.
DROPPED = {
    "Fez, Fes el-Bali": "medina lanes are mapped highway=footway/path: almost no streets under STREET_VALUES (fetched, preview)",
    "Genoa, centro storico": "caruggi mostly footway / pedestrian areas: sparse, polygon-dominated graph (fetched, preview)",
    "Karlsruhe, Schloss": "the documented fan starts ~250 m south of the palace tower; radials in the palace garden are "
                          "footways, so a window centred on the focus shows no fan (fetched, preview)",
    "Siena, Toledo (Wikipedia)": "Wikipedia pages do not describe the street layout; Toledo kept with Britannica instead, Siena dropped",
    "Madrid, Barrio de Salamanca": "source pages name the ensanche but do not describe a grid",
    "Versailles, Place d'Armes": "Wikipedia describes a 'rational and symmetrical grid', not the trident",
    "Bourtange / Nordlingen": "Wikipedia pages do not describe radial streets",
    "La Plata": "grid crossed by diagonals: mixed grid/radial, not a clean grid label",
    "Winnipeg, Wildwood Park": "its bays are mapped highway=service: no streets inside the window (fetched, preview)",
    "Kitimat downtown / Minto (Sydney) / Hull Orchard Park": "window fell on a commercial centre or arterials with few "
                                                            "culs-de-sac (Orchard Park also tried 400 m north); replaced "
                                                            "by Kitimat Kildala and Nechako and Hull Bransholme",
    "Spanish urbanizaciones (Las Rozas, Pozuelo, ...)": "no source describing a cul-de-sac / Radburn layout for a specific area",
}



def windows() -> pd.DataFrame:
    w = pd.DataFrame(WINDOWS, columns=COLS)
    assert w.id.is_unique and len(w) == 36
    assert not w.city.isin(HELD_OUT_METROS).any() and not w.country.isin(HELD_OUT_COUNTRIES + ["United States"]).any()
    assert (w.groupby(["style", "subset"]).size().unstack() == pd.DataFrame(
        {"calib": 5, "sort": 4}, index=sorted(w["style"].unique()))).all().all()
    w["utm_crs"] = [utm_crs(r.lat, r.lon) for r in w.itertuples()]
    rng = np.random.default_rng(16)                         # neutral codes for blinded use (no style information)
    w["blind_code"] = [f"W{i:02d}" for i in rng.permutation(len(w)) + 1]
    return w


def utm_crs(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def square(r):
    """Window (xmin, ymin) in the UTM CRS and the lat/lon bbox of window + margin."""
    from pyproj import Transformer
    fwd = Transformer.from_crs("EPSG:4326", r.utm_crs, always_xy=True)
    inv = Transformer.from_crs(r.utm_crs, "EPSG:4326", always_xy=True)
    cx, cy = fwd.transform(r.lon, r.lat)
    xmin, ymin = cx - TILE / 2, cy - TILE / 2
    h = TILE / 2 + MARGIN_M
    xs, ys = np.array([cx - h, cx + h, cx + h, cx - h]), np.array([cy - h, cy - h, cy + h, cy + h])
    lon, lat = inv.transform(xs, ys)
    return xmin, ymin, (float(min(lat)), float(min(lon)), float(max(lat)), float(max(lon)))


# ------------------------------------------------------------------ fetch
def _get(url: str, data: bytes | None = None, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return f.read()


def _api_map(s, w, n, e, depth: int = 0) -> list[tuple[str, bytes]]:
    """OSM API 0.6 map call; a box over the 50 000-node limit (HTTP 400) is split into 4 and re-requested."""
    url = OSM_API.format(s=f"{s:.6f}", w=f"{w:.6f}", n=f"{n:.6f}", e=f"{e:.6f}")
    try:
        body = _get(url)
        time.sleep(2)
        return [(url, body)]
    except urllib.error.HTTPError as ex:
        if ex.code != 400 or depth >= 3:
            raise
    time.sleep(2)
    ms, mw = (s + n) / 2, (w + e) / 2
    out = []
    for bb in ((s, w, ms, mw), (s, mw, ms, e), (ms, w, n, mw), (ms, mw, n, e)):
        out += _api_map(*bb, depth=depth + 1)
    return out


def _merge(parts: list[bytes], swne) -> bytes:
    """One OSM XML document from split map calls (elements de-duplicated by type and id)."""
    if len(parts) == 1:
        return parts[0]
    roots = [ET.fromstring(b) for b in parts]
    out = ET.Element("osm", dict(roots[0].attrib))
    ET.SubElement(out, "bounds", {"minlat": str(swne[0]), "minlon": str(swne[1]), "maxlat": str(swne[2]),
                                  "maxlon": str(swne[3])})
    seen = set()
    for kind in ("node", "way", "relation"):
        for r in roots:
            for el in r.findall(kind):
                k = (kind, el.get("id"))
                if k not in seen:
                    seen.add(k)
                    out.append(el)
    return ET.tostring(out, encoding="utf-8", xml_declaration=True)


def stage_fetch(force: bool = False):
    """Raw OSM per window: the OSM API 0.6 map call on window + MARGIN_M (complete ways; all tags), Overpass
    (street ways only) as the fallback. The street filter is applied at build time, identically for both sources."""
    from urbanmorph.data.osm_extract import STREET_VALUES
    RAW.mkdir(parents=True, exist_ok=True)
    regex = "^(" + "|".join(STREET_VALUES) + ")$"
    log = json.loads((RAW / "fetch_log.json").read_text()) if (RAW / "fetch_log.json").exists() else []
    for r in windows().itertuples():
        f, meta_f = RAW / f"{r.id}.osm", RAW / f"{r.id}.json"
        _, _, (s, w, n, e) = square(r)
        if f.exists() and meta_f.exists() and not force:
            m = json.loads(meta_f.read_text())
            bs, bw, bn, be = m["bbox_swne"]
            if bs <= s and bw <= w and bn >= n and be >= e:
                continue                                    # cached and still covers the (possibly re-centred) window
        body, how, err, parts = None, None, [], []
        try:
            parts = _api_map(s, w, n, e)
            body = _merge([b for _, b in parts], (s, w, n, e))
            how = "https://api.openstreetmap.org/api/0.6/map"
        except Exception as ex:                               # recorded, then Overpass
            err.append(f"osm api: {ex!r}"[:200])
        q = f'[out:xml][timeout:120];(way["highway"~"{regex}"]({s:.6f},{w:.6f},{n:.6f},{e:.6f}););(._;>;);out meta;'
        if body is None:
            for url in OVERPASS:
                try:
                    body = _get(url, urllib.parse.urlencode({"data": q}).encode())
                    if b"<osm" not in body[:500] or b"runtime error" in body[-2000:]:
                        raise RuntimeError("Overpass returned no OSM document / runtime error")
                    how = url
                    break
                except Exception as ex:
                    err.append(f"{url}: {ex!r}"[:200])
                    body = None
                    time.sleep(10)
        if body is None:
            log.append({"id": r.id, "ok": False, "errors": err,
                        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            print(r.id, "FAILED", err[-1:], flush=True)
            continue
        f.write_bytes(body)
        root = ET.fromstring(body)
        mt = root.find("meta")
        info = {"id": r.id, "source": how, "requests": [u for u, _ in parts] if parts else [q],
                "request_sha256": [hashlib.sha256(b).hexdigest() for _, b in parts],
                "bbox_swne": [s, w, n, e], "margin_m": MARGIN_M, "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body), "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                # the OSM API serves the live database: the data state is the fetch time; Overpass reports osm_base
                "osm_data_timestamp": mt.get("osm_base") if mt is not None else datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "max_element_timestamp": max((x.get("timestamp", "") for x in root), default=""),
                "generator": root.get("generator"), "licence": LICENCE, "errors": err}
        meta_f.write_text(json.dumps(info, indent=1))
        log.append({"id": r.id, "ok": True, "source": how, "utc": info["fetched_utc"]})
        print(r.id, r.city, how, len(parts), len(body), flush=True)
        time.sleep(3)                                         # low request rate (OSM API / Overpass usage policies)
    (RAW / "fetch_log.json").write_text(json.dumps(log, indent=1))


# ------------------------------------------------------------------ build
def osm_lines(path: Path):
    """Street pieces (lon/lat arrays) with highway tag and vertical structure, split at shared nodes and way ends."""
    from urbanmorph.data.osm_extract import STREET_VALUES
    from urbanmorph.graphs.levels import level_of, structure_of
    root = ET.parse(path).getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    ways = []
    for w in root.iter("way"):
        t = {x.get("k"): x.get("v") for x in w.iter("tag")}
        if t.get("highway") not in STREET_VALUES:
            continue
        nd = [x.get("ref") for x in w.iter("nd")]
        nd = [i for i in nd if i in nodes]
        if len(nd) >= 2:
            ways.append((nd, t))
    use = {}
    for nd, _ in ways:
        for i in set(nd):
            use[i] = use.get(i, 0) + 1
        for i in (nd[0], nd[-1]):
            use[i] = use.get(i, 0) + 1                        # way ends are always graph nodes
    out = []
    for nd, t in ways:
        st = structure_of(t.get("bridge"), t.get("tunnel"))
        lv = level_of(t.get("layer"), st)
        cut = [0] + [k for k in range(1, len(nd) - 1) if use[nd[k]] >= 2 or nd[k] in nd[:k]] + [len(nd) - 1]
        for a, b in zip(cut[:-1], cut[1:]):
            out.append((np.array([nodes[i] for i in nd[a:b + 1]]), t["highway"], lv, st))
    return out


def window_graph(r):
    import shapely
    from pyproj import Transformer
    from shapely.geometry import LineString, box
    from urbanmorph.graphs.road_graph import build_graph, highway_class
    xmin, ymin, _ = square(r)
    fwd = Transformer.from_crs("EPSG:4326", r.utm_crs, always_xy=True)
    win = box(xmin, ymin, xmin + TILE, ymin + TILE)
    lines, cls, lev, st = [], [], [], []
    for ll, hw, lv, s in osm_lines(RAW / f"{r.id}.osm"):
        x, y = fwd.transform(ll[:, 0], ll[:, 1])
        geom = LineString(np.c_[x, y])
        if geom.length <= 0 or not geom.intersects(win):
            continue
        clip = geom.intersection(win)
        for part in shapely.get_parts(shapely.line_merge(clip) if clip.geom_type == "MultiLineString" else clip):
            if part.geom_type != "LineString" or part.length <= 0:
                continue
            lines.append(np.asarray(part.coords)[:, :2] - [xmin, ymin])
            cls.append(highway_class(hw)), lev.append(int(lv)), st.append(int(s))
    return build_graph(lines, classes=cls, levels=lev, structs=st), xmin, ymin


def centre_of(r) -> tuple:
    """Radial windows: the declared radial focus (= window centre by construction); other styles: tile centre."""
    return (TILE / 2, TILE / 2)


def stage_build():
    PROC.mkdir(parents=True, exist_ok=True)
    w = windows()
    graphs, rows = [], []
    for r in w.itertuples():
        row = r._asdict()
        row.pop("Index")
        meta_f = RAW / f"{r.id}.json"
        if not meta_f.exists():
            row.update(status="failed", reason="no raw OSM download")
            rows.append(row)
            continue
        m = json.loads(meta_f.read_text())
        g, xmin, ymin = window_graph(r)
        row.update(xmin=round(xmin, 2), ymin=round(ymin, 2), osm_file=str((RAW / f"{r.id}.osm").relative_to(PROJECT_ROOT)),
                   osm_sha256=m["sha256"], osm_source=m["source"], osm_data_timestamp=m["osm_data_timestamp"],
                   osm_max_element_timestamp=m["max_element_timestamp"],
                   fetched_utc=m["fetched_utc"], licence=LICENCE, n_nodes=int(g.n), n_edges=int(len(g.edges)),
                   centre_x=centre_of(r)[0], centre_y=centre_of(r)[1])
        reason = "empty street graph" if len(g.edges) == 0 else ""
        row.update(status="excluded" if reason else "ok", reason=reason)
        rows.append(row)
        graphs.append((r.id, r.style, r.subset, g))
        print(r.id, r.city, g.n, len(g.edges), flush=True)
    man = pd.DataFrame(rows)
    man.to_csv(PROC / "manifest.csv", index=False)
    (PROC / "graphs.pkl").write_bytes(pickle.dumps(graphs))
    meta = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": "M16 real style references: calib (threshold calibration) and sort (blinded style-sorting) subsets",
            "styles": ["grid", "organic", "radial", "culdesac"], "per_style": {"calib": 5, "sort": 4},
            "excluded_held_out": {"metros": HELD_OUT_METROS, "countries": HELD_OUT_COUNTRIES},
            "label_rule": "documented typology of the named area (source URL + quote per window); never our metrics",
            "window_rule": "512 m square in the WGS84 UTM zone, centred on the declared point; radial windows centred "
                           "on the documented radial focus",
            "street_definition": "highway in osm_extract.STREET_VALUES, split at shared nodes / way ends, clipped, "
                                 "road_graph.build_graph(classes=highway_class, levels/structs from way tags)",
            "metric_centre": "radial: declared focus (tile centre by construction); others: tile centre (256, 256)",
            "dropped_candidates": DROPPED, "licence": LICENCE, "windows": man.to_dict(orient="records")}
    (PROC / "manifest.json").write_text(json.dumps(meta, indent=1, default=str))


# ------------------------------------------------------------------ metrics, figure, summary
KEY = ["block_rect_share", "block_phi_med", "block_logA_med", "block_logA_sd", "jangle_right_share", "through_share",
       "stroke_xjunc_lw", "tree_len_share", "term_len_med", "deadend_per_km", "loops_per_km2", "share_4way",
       "spokes_long", "ring_cover_max", "phi_boeing", "orth_order", "street_km"]


def stage_metrics():
    from urbanmorph.graphs.network_outcomes import outcome_row
    from urbanmorph.graphs.style_metrics import morphology_row
    w = windows().set_index("id")
    rows = []
    for gid, style, subset, g in pickle.loads((PROC / "graphs.pkl").read_bytes()):
        r = w.loc[gid]
        c = centre_of(r)
        o = outcome_row(g, centre=c, rng=np.random.default_rng(0))
        mo = morphology_row(g, centre=c, rng=np.random.default_rng(0))
        rows.append({"id": gid, "city": r.city, "style": style, "subset": subset, **o,
                     **{k: v for k, v in mo.items() if k not in o}})
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "metrics.csv", index=False)
    print(df[["id", "style"] + KEY].round(3).to_string())


def stage_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    graphs = pickle.loads((PROC / "graphs.pkl").read_bytes())
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(6, 6, figsize=(12, 12.6))
    for ax in axs.ravel():
        ax.set_axis_off()
    for ax, (gid, _, _, g) in zip(axs.ravel(), graphs):
        ax.add_collection(LineCollection(g.segments().reshape(-1, 2, 2), colors="black", linewidths=0.7))
        ax.set_xlim(0, TILE), ax.set_ylim(0, TILE), ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((0, 0), TILE, TILE, fill=False, lw=0.4, ec="0.6"))
        ax.set_title(gid, fontsize=8, pad=2)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.98, 0.04, 0.12)
    fig.savefig(OUT / "figures" / "style_refs.png", dpi=150, facecolor="white")
    plt.close(fig)


def stage_summary():
    df = pd.read_csv(PROC / "metrics.csv")
    man = pd.read_csv(PROC / "manifest.csv")
    cal = df[df.subset == "calib"]
    order = ["grid", "organic", "radial", "culdesac"]
    mean = cal.groupby("style")[KEY].mean().reindex(order)
    sd = cal.groupby("style")[KEY].std(ddof=1).reindex(order)
    lines = [f"# M16 real style references ({VERSION})", "",
             "36 real 512 m street windows (9 per style: 5 `calib`, 4 `sort`), labelled from documented typology "
             "(source per window in `configs/m16_style_refs/manifest.csv`). Held-out metros (Lleida, Alicante, Seoul, Daegu, "
             "Lima, Trujillo) and the USA are excluded. Street definition = training graphs (STREET_VALUES, build_graph, "
             "highway classes, M12 levels). Radial metrics use the declared focus (= tile centre), others the tile centre.",
             f"OSM data {LICENCE}; fetched {man.fetched_utc.min()} to {man.fetched_utc.max()}.", "",
             "## Calibration subset: mean (SD) per style, n = 5 each", "",
             "| metric | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    cnt = cal.groupby("style")[KEY].count().reindex(order)
    for k in KEY:
        lines.append(f"| {k} | " + " | ".join(f"{mean.loc[s, k]:.3g} ({sd.loc[s, k]:.2g})"
                                             + (f" n={cnt.loc[s, k]}" if cnt.loc[s, k] < 5 else "") for s in order) + " |")
    lines += ["", "NaN values are skipped (n shown when < 5): a window with no interior block or no interior dead end "
              "has no block / terminal statistics.", "",
              "Notes: closed pedestrian ways (area=yes squares) are kept, as in the training graphs (osmium/OSMnx path), "
              "and appear as small loops. Radburn 'turning loops' (Kitimat C07, C09; Crestwood C03) are part of the "
              "documented cul-de-sac typology ('Living streets mostly take the form of culs-de-sac and turning loops') "
              "but have few dead ends. Raw data: OSM API 0.6 map call (live database; data timestamp = fetch time)."]
    lines += ["", "## Windows", "", "| id | city | area | style | subset | source |", "|---|---|---|---|---|---|"]
    for r in man.itertuples():
        lines.append(f"| {r.id} | {r.city} | {r.name} | {r.style} | {r.subset} | {r.source_url} |")
    bad = man[man.status != "ok"]
    lines += ["", "## Failures / exclusions", ""] + ([f"- {r.id}: {r.reason}" for r in bad.itertuples()] or ["- none among the 36"])
    lines += ["", "Candidates dropped before the final list:", ""] + [f"- {k}: {v}" for k, v in DROPPED.items()]
    lines += ["", "Figure: `outputs/m16/figures/style_refs.png` (all 36 windows, ids only)."]
    (OUT / "style_refs_summary.md").write_text("\n".join(lines) + "\n")


def publish():
    import shutil
    CONF.mkdir(parents=True, exist_ok=True)
    for f in ("manifest.csv", "manifest.json", "metrics.csv"):
        if (PROC / f).exists():
            shutil.copy2(PROC / f, CONF / f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "build", "metrics", "figure", "summary", "all"])
    ap.add_argument("--force", action="store_true", help="re-download raw OSM even when cached")
    a = ap.parse_args()
    if a.stage in ("fetch", "all"):
        stage_fetch(a.force)
    for st, f in (("build", stage_build), ("metrics", stage_metrics), ("figure", stage_figure), ("summary", stage_summary)):
        if a.stage in (st, "all"):
            f()
    publish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
