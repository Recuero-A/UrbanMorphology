"""Data-quality checks.

Every check returns flags of the form {"code", "severity", "detail"} where
severity is one of:

* ``error``   - the sample is not usable (corrupt output, wrong CRS, or OSM
                clearly missing in an area that satellites show as built-up);
* ``warning`` - usable, but probably incomplete or unusual; keep and analyse;
* ``info``    - a property worth knowing that is NOT a defect. In particular an
                empty green channel is ``info``: it may be a genuinely grey
                neighbourhood or simply unmapped greenery - OSM cannot tell us.
"""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
from pyproj import CRS

from urbanmorph.data.dataset import LayoutValidationError, validate_layout, validate_metadata

SEVERITIES = ("error", "warning", "info")


def flag(code: str, severity: str, detail: str = "") -> dict:
    assert severity in SEVERITIES
    return {"code": code, "severity": severity, "detail": detail}


def sample_flags(
    layout: np.ndarray,
    meta: dict,
    expected_crs: str,
    expected_pixel_m: float,
    ghsl_built: float,
    osm_building_coverage: float,
    n_invalid_geometries: int,
    min_osm_to_ghsl_ratio: float,
    max_overlap_fraction: float,
    min_builtup_fraction: float,
) -> list[dict]:
    flags: list[dict] = []

    try:
        validate_layout(layout)
    except LayoutValidationError as e:
        flags.append(flag("raster_invalid", "error", str(e)))
        return flags
    try:
        validate_metadata(meta, layout)
    except LayoutValidationError as e:
        flags.append(flag("metadata_invalid", "error", str(e)))

    crs = meta.get("projected_crs")
    if crs != expected_crs or CRS.from_user_input(crs).is_geographic:
        flags.append(flag("crs_unexpected", "error", f"{crs} != {expected_crs}"))
    if not np.isclose(meta.get("pixel_size_m", np.nan), expected_pixel_m):
        flags.append(flag("pixel_resolution_inconsistent", "error", f"{meta.get('pixel_size_m')} m/px"))

    roads, buildings, green = (layout[i].any() for i in range(3))
    if not roads and not buildings:
        sev = "error" if ghsl_built >= min_builtup_fraction else "warning"
        flags.append(flag("osm_core_missing", sev, f"no roads and no buildings; GHSL built-up {ghsl_built:.2f}"))
    else:
        if not roads:
            flags.append(flag("no_roads", "warning", "no mapped roads of the selected classes"))
        if not buildings:
            flags.append(flag("no_buildings", "warning", f"GHSL built-up {ghsl_built:.2f}"))
    if not green:
        flags.append(flag("green_unmapped_or_absent", "info", "empty green channel: absence OR missing OSM tags"))

    if ghsl_built > 0:  # also when OSM has zero buildings: ratio 0 is the most incomplete case
        ratio = osm_building_coverage / ghsl_built
        if ratio < min_osm_to_ghsl_ratio:
            flags.append(flag("buildings_incomplete_suspected", "warning",
                              f"OSM building coverage {osm_building_coverage:.2f} vs GHSL {ghsl_built:.2f} (ratio {ratio:.2f})"))

    if n_invalid_geometries > 0:
        flags.append(flag("invalid_geometries_repaired", "info", f"{n_invalid_geometries} repaired with make_valid"))

    names = ("roads", "buildings", "green_spaces")
    for a, b in combinations(range(3), 2):
        frac = float((layout[a].astype(bool) & layout[b].astype(bool)).mean())
        if frac > max_overlap_fraction:
            flags.append(flag("excessive_channel_overlap", "warning", f"{names[a]}&{names[b]} = {frac:.3f}"))
    return flags


def is_usable(flags: list[dict]) -> bool:
    return not any(f["severity"] == "error" for f in flags)


def layout_hash(layout: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(layout).tobytes()).hexdigest()


def dataset_flags(table: pd.DataFrame, tile_size_m: float) -> dict[str, list[dict]]:
    """Cross-sample checks: duplicate IDs, overlapping tiles, identical rasters.

    Tiles in the same CRS are on one snapped grid, so two tiles overlap iff
    their grid indices are equal. Tiles in different CRSs belong to different
    cities (>100 km apart); their lat/lon centres are compared as a safeguard.
    """
    out: dict[str, list[dict]] = {sid: [] for sid in table["sample_id"]}
    for sid, n in table["sample_id"].value_counts().items():
        if n > 1:
            out[sid].append(flag("duplicate_sample_id", "error", f"{n} rows"))
    for (crs, e, nn), grp in table.groupby(["crs", "grid_e", "grid_n"]):
        if len(grp) > 1:
            for sid in grp["sample_id"]:
                out[sid].append(flag("tile_overlap", "error", f"same grid cell as {list(grp['sample_id'])}"))
    # different-CRS safeguard: centres closer than a tile size => overlap
    lat = np.radians(table["center_lat"].to_numpy())
    lon = np.radians(table["center_lon"].to_numpy())
    crs = table["crs"].to_numpy()
    ids = table["sample_id"].to_numpy()
    for i in range(len(table)):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
        d = 2 * 6_371_000 * np.arcsin(np.sqrt(a))
        for j in np.flatnonzero((d < tile_size_m) & (crs != crs[i])):
            out[ids[i]].append(flag("tile_overlap", "error", f"centre {d[j]:.0f} m from {ids[j]} (other CRS)"))
    if "layout_hash" in table:
        nonempty = table[table["layout_nonempty"]] if "layout_nonempty" in table else table
        for h, grp in nonempty.groupby("layout_hash"):
            if len(grp) > 1:
                for sid in grp["sample_id"]:
                    out[sid].append(flag("duplicate_raster", "error", f"identical raster to {list(grp['sample_id'])}"))
    return out
