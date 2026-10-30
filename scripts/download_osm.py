"""Step 1: download roads, buildings and green spaces for the sample area.

Usage:
    python scripts/download_osm.py           # uses the cache if present
    python scripts/download_osm.py --force   # re-download
"""

import argparse
import json
import logging
import sys

from pathlib import Path

# Make `src/` importable even when the editable install is not picked up
# (iCloud-synced folders get the macOS "hidden" flag, and Python skips hidden .pth files).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanmorph.config import BARCELONA_EIXAMPLE_001
from urbanmorph.data.osm import download_area


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="ignore the cache and download again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    area = BARCELONA_EIXAMPLE_001
    try:
        paths = download_area(area, force=args.force)
    except Exception as exc:
        logging.error("Download FAILED: %s", exc)
        logging.error("No real data was saved. Check your connection and retry later.")
        return 1

    info = json.loads(paths["info"].read_text())
    print(f"Raw OSM data for {area.sample_id} in {area.raw_dir}")
    print(json.dumps(info["counts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
