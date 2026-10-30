"""Read the sampling plan and turn plan rows into ``AreaSpec`` objects."""

from __future__ import annotations

import pandas as pd

from urbanmorph.config import M2_DIR, AreaSpec

PLAN_PATH = M2_DIR / "sampling" / "plan.csv"


def load_plan(path=PLAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def area_from_row(row, cfg: dict) -> AreaSpec:
    t = cfg["tile"]
    return AreaSpec(
        sample_id=row["sample_id"],
        center_lat=float(row["center_lat"]),
        center_lon=float(row["center_lon"]),
        size_m=t["size_m"],
        resolution=t["resolution"],
        projected_crs=row["crs"],
        download_margin_m=t["download_margin_m"],
        description=f"{row['city'].title()} ({row['country']}), grid tile {row['grid_e']}/{row['grid_n']}",
        country=row["country"],
        city=row["city"],
        projected_center=((row["xmin"] + row["xmax"]) / 2, (row["ymin"] + row["ymax"]) / 2),
        raw_subdir=f"osm/{row['country']}/{row['city']}",
    )
