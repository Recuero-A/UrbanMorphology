"""Milestone 12: OSM street ways with vertical metadata (bridge / tunnel / layer / ford), one table per city.

    python scripts/m12_build_structures.py     # -> data/processed/m12/structures/<CITY>.geojsonseq (git-ignored)

osmium tags-filter on the cached city clips (same snapshot as the tiles), exported with way ids so that tile
centerlines (OSMnx edges with `osmid`) can be matched to their ways (graphs/levels.py). Nothing is downloaded.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanmorph.config import PROJECT_ROOT

CLIPS = PROJECT_ROOT / "data" / "raw" / "osm_extracts" / "cities"
OUT = PROJECT_ROOT / "data" / "processed" / "m12" / "structures"
CITIES = ["ESP_BCN", "ESP_MAD", "ESP_VLC", "USA_CHI", "USA_LAX", "USA_NYC", "USA_SAN", "JPN_TYO", "JPN_OSA", "JPN_NGO"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CITIES:
        tmp = OUT / f"_{c}.osm.pbf"
        subprocess.run(["osmium", "tags-filter", "-O", "-o", str(tmp), str(CLIPS / f"{c}.osm.pbf"),
                        "w/bridge", "w/tunnel", "w/layer", "w/ford"], check=True, capture_output=True)
        subprocess.run(["osmium", "export", "-O", "-f", "geojsonseq", "--attributes=type,id", "-o", str(OUT / f"{c}.geojsonseq"),
                        str(tmp)], check=True, capture_output=True)
        tmp.unlink()
        print(c, sum(1 for _ in open(OUT / f"{c}.geojsonseq")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
