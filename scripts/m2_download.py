"""Milestone 2, step B: build the raw OSM cache for planned tiles (resumable).

Source is set in the config ([osm] source):
* "geofabrik" (default): tiles are clipped locally from dated regional
  extracts in data/raw/osm_extracts/ - no network requests at all;
* "overpass": 2 Overpass requests per tile (street graph + one combined
  polygon request), with retries and pauses.
Tiles already cached are skipped, so the script can be interrupted and rerun.
Progress is appended to data/raw/osm/download_log.csv.

Usage:
    python scripts/m2_download.py --dry-run              # show workload only
    python scripts/m2_download.py                        # all primaries
    python scripts/m2_download.py --replacements         # reserves listed in replacements_needed.csv
    python scripts/m2_download.py --city PER_CUZ --limit 2
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from urbanmorph.config import M2_DIR, RAW_DIR
from urbanmorph.data.osm import download_area, is_cached
from urbanmorph.data.osm_extract import build_tile_from_extract, city_pbf, region_pbf, snapshot_info
from urbanmorph.sampling.study_area import cities, fetch_boundary
from urbanmorph.sampling.plan_io import area_from_row, load_plan
from urbanmorph.sampling.study_area import load_config

LOG_PATH = RAW_DIR / "osm" / "download_log.csv"
REPLACEMENTS_PATH = M2_DIR / "replacements_needed.csv"
PAUSE_S = 2.0  # extra courtesy pause between tiles (OSMnx also waits for Overpass slots)


def log_row(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replacements", action="store_true")
    ap.add_argument("--city", help="restrict to one city code")
    ap.add_argument("--limit", type=int, help="max tiles to download in this run")
    ap.add_argument("--plan", type=Path, help="alternative plan CSV (e.g. the M3b expansion); all its rows are fetched")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    cfg = load_config()
    plan = load_plan(args.plan) if args.plan else load_plan()
    if args.plan:
        todo = plan
    elif args.replacements:
        wanted = set(pd.read_csv(REPLACEMENTS_PATH)["replacement_id"]) if REPLACEMENTS_PATH.exists() else set()
        todo = plan[plan["sample_id"].isin(wanted)]
    if not args.plan and not args.replacements:
        todo = plan[plan["role"] == "primary"]
    if args.city:
        todo = todo[todo["city_code"] == args.city]

    # Round-robin over cities (rank 0 of every city, then rank 1, ...) so an
    # interrupted run still leaves a balanced partial dataset.
    todo = todo.sort_values(["rank", "city_code"])
    areas = [area_from_row(r, cfg) for _, r in todo.iterrows()]
    code_of = dict(zip(todo["sample_id"], todo["city_code"]))
    pending = [a for a in areas if not is_cached(a)]
    n_cached = len(areas) - len(pending)
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(areas)} tiles requested, {n_cached} cached, {len(pending)} to download in this run "
          + (f"(~{2 * len(pending)} Overpass requests)" if cfg["osm"]["source"] == "overpass" else "(local extracts, no requests)"))
    if args.dry_run or not pending:
        return 0

    use_extracts = cfg["osm"]["source"] == "geofabrik"
    if use_extracts:
        regions = cfg["osm"]["extracts"]
        missing = sorted({regions[c] for c in code_of.values() if not region_pbf(regions[c]).exists()})
        if missing:
            print(f"Missing Geofabrik extracts: {missing}. Download them into data/raw/osm_extracts/ first.")
            return 1
        snapshots = {r: snapshot_info(r) for r in sorted(set(regions[c] for c in code_of.values()))}
        bad = [r for r, s in snapshots.items() if s["md5_verified"] is False]
        if bad:
            print(f"MD5 mismatch for {bad}; re-download these extracts.")
            return 1
        city_by_code = {c.code: c for c in cities(cfg)}
        city_files = {code: city_pbf(code, regions[code], fetch_boundary(city_by_code[code]).total_bounds)
                      for code in sorted(set(code_of.values()))}

    failures = 0
    for k, area in enumerate(pending, 1):
        t0 = time.time()
        status, error = "ok", ""
        try:
            if use_extracts:
                code = code_of[area.sample_id]
                build_tile_from_extract(area, city_files[code], snapshots[cfg["osm"]["extracts"][code]])
            else:
                download_area(area)
        except Exception as exc:  # keep going; failures are logged and can be retried
            status, error, failures = "failed", repr(exc)[:300], failures + 1
        dt = time.time() - t0
        log_row({"sample_id": area.sample_id, "status": status, "seconds": round(dt, 1),
                 "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": error})
        print(f"[{k}/{len(pending)}] {area.sample_id}: {status} in {dt:.0f}s {error}", flush=True)
        if not use_extracts:
            time.sleep(PAUSE_S)
    print(f"done; {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
