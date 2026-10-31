"""Milestone 11, Phase A: can building heights and roof shapes come from data we already have?

    python scripts/m11_height_audit.py      # -> outputs/m11/audit/height_audit.json

1. Tile buildings (train + val tiles only; the test split is never read): share of buildings with an OSM
   `height` tag, a `building:levels` tag, or both; their distributions; the implied storey height
   (height / levels) where both are present; all per city.
2. Roof shapes: `roof:shape` counts over all buildings of each cached city clip (data/raw/osm_extracts/cities,
   osmium CLI). City clips cover more than our tiles, so this is an indicator of tag availability only.
Nothing is downloaded.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pyogrio

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion, sample_dir

OUT = PROJECT_ROOT / "outputs" / "m11" / "audit"
CLIPS = PROJECT_ROOT / "data" / "raw" / "osm_extracts" / "cities"
CITY_CLIP = {"barcelona": "ESP_BCN", "madrid": "ESP_MAD", "valencia": "ESP_VLC", "chicago": "USA_CHI",
             "los_angeles": "USA_LAX", "new_york": "USA_NYC", "tokyo": "JPN_TYO", "osaka": "JPN_OSA", "nagoya": "JPN_NGO"}


def to_m(v) -> float:
    """OSM height string -> metres (plain number, 'm' suffix or feet); NaN if unparseable."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower().replace(",", ".")
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(m|ft|')?$", s)
    if not m:
        return np.nan
    x = float(m.group(1))
    return x * 0.3048 if m.group(2) in ("ft", "'") else x


def to_levels(v) -> float:
    try:
        x = float(str(v).strip().replace(",", "."))
    except (TypeError, ValueError):
        return np.nan
    return x if 0 < x < 200 else np.nan


def q(a, ps=(10, 25, 50, 75, 90)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return {f"p{p}": float(np.percentile(a, p)) for p in ps} if len(a) else None


def roof_shapes(clip: Path) -> dict:
    """roof:shape value counts over buildings of one city clip (osmium tags-filter + tags-count)."""
    tmp = OUT / f"_{clip.stem}.osm.pbf"
    subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(clip), "nwr/building"], check=True, capture_output=True)
    n = subprocess.run(["osmium", "tags-count", str(tmp), "building"], capture_output=True, text=True).stdout
    r = subprocess.run(["osmium", "tags-count", str(tmp), "roof:shape=*"], capture_output=True, text=True).stdout
    tmp.unlink()
    n_bld = sum(int(line.split("\t")[0]) for line in n.splitlines() if line.strip())
    shapes = {}
    for line in r.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            shapes[parts[2].strip('"')] = int(parts[0])
    tagged = sum(shapes.values())
    return {"buildings": n_bld, "with_roof_shape": tagged, "share": tagged / max(n_bld, 1),
            "top": dict(sorted(shapes.items(), key=lambda kv: -kv[1])[:8])}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index_with_expansion()
    idx = idx[idx.in_dataset & idx.split.isin(["train", "val"])]
    res = {"note": "train + val tiles only; heights in metres", "cities": {}}
    for city, rows in idx.groupby("city"):
        H, L, HL = [], [], []
        n = 0
        for _, r in rows.iterrows():
            f = sample_dir(r.country, r.city) / f"{r.sample_id}_vectors.gpkg"
            if not f.exists():
                continue
            df = pyogrio.read_dataframe(f, layer="buildings", read_geometry=False)
            n += len(df)
            col = lambda c: df[c] if c in df.columns else [None] * len(df)     # tag column absent in some tiles
            h = np.array([to_m(v) for v in col("height")], float)
            lv = np.array([to_levels(v) for v in col("building:levels")], float)
            H += list(h)
            L += list(lv)
            both = np.isfinite(h) & np.isfinite(lv) & (h > 0)
            HL += list(h[both] / lv[both])
        H, L = np.array(H), np.array(L)
        res["cities"][city] = {
            "country": rows.country.iloc[0], "tiles": int(len(rows)), "buildings": n,
            "share_height": float(np.isfinite(H).mean()) if n else 0.0,
            "share_levels": float(np.isfinite(L).mean()) if n else 0.0,
            "share_height_or_levels": float((np.isfinite(H) | np.isfinite(L)).mean()) if n else 0.0,
            "height_m": q(H), "levels": q(L), "storey_height_m_from_both": q(HL), "n_both": len(HL)}
        print(city, {k: v for k, v in res["cities"][city].items() if k.startswith("share")}, flush=True)
    res["roof_shape_city_clips"] = {}
    for city, clip in CITY_CLIP.items():
        p = CLIPS / f"{clip}.osm.pbf"
        if p.exists():
            res["roof_shape_city_clips"][city] = roof_shapes(p)
            print(city, "roof:shape", res["roof_shape_city_clips"][city]["share"], flush=True)
    (OUT / "height_audit.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
