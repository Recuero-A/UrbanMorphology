"""Step 2: cached raw OSM data -> data/processed/<sample_id>.npy (+ .json, _vectors.gpkg).

Works offline: reads only the cache written by download_osm.py.
"""

import json
import logging
import sys
from datetime import datetime, timezone

from pathlib import Path

# Make `src/` importable even when the editable install is not picked up
# (iCloud-synced folders get the macOS "hidden" flag, and Python skips hidden .pth files).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanmorph.config import BARCELONA_EIXAMPLE_001
from urbanmorph.data.dataset import sample_paths, save_sample
from urbanmorph.data.osm import load_raw
from urbanmorph.preprocessing.pipeline import build_layout, export_vectors, make_metadata


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    area = BARCELONA_EIXAMPLE_001
    try:
        raw = load_raw(area)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return 1

    sample = build_layout(area, raw["edges"], raw["buildings"], raw["green"], road_areas=raw["road_areas"])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = make_metadata(area, sample, raw["info"], now)
    paths = save_sample(sample.layout, meta, area.sample_id)
    export_vectors(sample, sample_paths(area.sample_id)["vectors"])

    print(f"Saved {paths['array']}  shape={sample.layout.shape} dtype={sample.layout.dtype}")
    print(f"Saved {paths['metadata']}")
    print(json.dumps(meta["statistics"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
