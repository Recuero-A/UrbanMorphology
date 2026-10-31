"""Milestone 8: fetch observed cadastral parcels (raw data stays in data/raw/parcels/, git-ignored).

    python scripts/m8_fetch_parcels.py spain 08900 28900 46900
    python scripts/m8_fetch_parcels.py us            # NYC + LA, our TRAIN and VAL tiles only (per-tile queries)

Spain: Dirección General del Catastro, INSPIRE Cadastral Parcels, ATOM download per municipality.
Licence (v1.0, 2016): use incl. commercial allowed for TRANSFORMED products; the original data may
not be redistributed; derived products must not be presented as cadastral cartography.
US: NYC Department of City Planning MapPLUTO (tax lots; free public data since 2013) and Los Angeles
County Assessor "Parcel Boundaries" (County GIS; terms grant copy / publish / adapt with citation, but the
service also notes records are for sale, so we treat it conservatively: local research use and derived
statistics only). Chicago (Cook County) is EXCLUDED: its parcel licence is unspecified / personal use.
Only tiles of the train and val splits are queried; the held-out test tiles are never touched.
TLS is verified with the certifi bundle (the system store lacks the issuing CA).
A provenance record (URL, time, size, sha256, licence) is written next to every file.
"""
import hashlib
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "parcels"
CTX = ssl.create_default_context(cafile=certifi.where())
SPAIN = {"08900": "08/08900-BARCELONA", "28900": "28/28900-MADRID", "46900": "46/46900-VALENCIA"}
SPAIN_LICENCE = ("Dirección General del Catastro, INSPIRE CadastralParcels. Licencia de acceso y uso v1.0 (2016): "
                 "use of transformed information allowed (incl. commercial); original data not redistributable; "
                 "derived products must not be called 'cartografía catastral'.")


def fetch(url, dest: Path, licence: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    req = urllib.request.Request(url.replace(" ", "%20"), headers={"User-Agent": "UrbanMorph research (non-commercial)"})
    with urllib.request.urlopen(req, context=CTX, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    data = dest.read_bytes()
    prov = {"url": url, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "licence": licence, "seconds": round(time.time() - t0, 1)}
    dest.with_suffix(dest.suffix + ".provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(dest.name, f"{len(data) / 1e6:.1f} MB in {prov['seconds']} s", flush=True)


US_SERVICES = {
    "new_york": ("https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/MAPPLUTO/FeatureServer/0",
                 "NYC Department of City Planning, MapPLUTO (Digital Tax Map tax lots). Free public data."),
    "los_angeles": ("https://dpw.gis.lacounty.gov/dpw/rest/services/GIS_Web_Services/MapServer/2",
                    "County of Los Angeles, Office of the Assessor, Parcel Boundaries (County GIS). Local research use; "
                    "cite 'Data from County of Los Angeles'; raw data not redistributed."),
}


def fetch_us():
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from urbanmorph.data.m2_dataset import load_index_with_expansion
    idx = load_index_with_expansion()
    idx = idx[idx["city"].isin(US_SERVICES) & idx["split"].isin(["train", "val"])]
    log = []
    for _, r in idx.iterrows():
        base, licence = US_SERVICES[r.city]
        dest = ROOT / "us" / r.city / f"{r.sample_id}.geojson"
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        epsg = int(str(r.crs).split(":")[1])
        feats, offset = [], 0
        while True:
            q = {"geometry": f"{r.xmin},{r.ymin},{r.xmax},{r.ymax}", "geometryType": "esriGeometryEnvelope",
                 "inSR": epsg, "outSR": epsg, "spatialRel": "esriSpatialRelIntersects", "outFields": "*",
                 "returnGeometry": "true", "f": "geojson", "resultOffset": offset, "resultRecordCount": 1000}
            url = base + "/query?" + urllib.parse.urlencode(q)
            req = urllib.request.Request(url, headers={"User-Agent": "UrbanMorph research (non-commercial)"})
            j = json.loads(urllib.request.urlopen(req, context=CTX, timeout=120).read())
            batch = j.get("features", [])
            feats += batch
            if len(batch) < 1000:
                break
            offset += 1000
            time.sleep(0.5)
        fc = {"type": "FeatureCollection", "crs_epsg": epsg, "features": feats}
        dest.write_text(json.dumps(fc))
        prov = {"service": base, "sample_id": r.sample_id, "split": r.split, "bbox_tile_crs": [r.xmin, r.ymin, r.xmax, r.ymax],
                "epsg": epsg, "features": len(feats), "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "licence": licence}
        dest.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))
        log.append(prov)
        print(r.city, r.sample_id, len(feats), flush=True)
        time.sleep(0.5)
    return log


def main() -> int:
    country, codes = sys.argv[1], sys.argv[2:]
    if country == "us":
        fetch_us()
        return 0
    if country == "spain":
        for c in codes:
            path = SPAIN[c]
            url = f"https://www.catastro.hacienda.gob.es/INSPIRE/CadastralParcels/{path}/A.ES.SDGC.CP.{c}.zip"
            fetch(url, ROOT / "spain" / f"A.ES.SDGC.CP.{c}.zip", SPAIN_LICENCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
