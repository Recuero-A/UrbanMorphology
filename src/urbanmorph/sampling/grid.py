"""Candidate tiles: a fixed 512 m grid in each city's UTM CRS.

The grid is snapped to multiples of the tile size (cell (i, j) covers
x in [i*512, (i+1)*512), y in [j*512, (j+1)*512)). Consequences:
* tiles can never partially overlap - they are identical or disjoint;
* a tile's identity (and its sample ID) is stable across runs and configs.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely


def candidate_tiles(boundary_wgs84: gpd.GeoDataFrame, crs: str, size_m: float) -> gpd.GeoDataFrame:
    """All grid cells lying entirely inside the boundary."""
    region = shapely.union_all(boundary_wgs84.to_crs(crs).geometry.make_valid().values)
    xmin, ymin, xmax, ymax = region.bounds
    ii = np.arange(int(np.floor(xmin / size_m)), int(np.ceil(xmax / size_m)))
    jj = np.arange(int(np.floor(ymin / size_m)), int(np.ceil(ymax / size_m)))
    I, J = (a.ravel() for a in np.meshgrid(ii, jj))
    boxes = shapely.box(I * size_m, J * size_m, (I + 1) * size_m, (J + 1) * size_m)
    shapely.prepare(region)
    keep = shapely.within(boxes, region)
    I, J, boxes = I[keep], J[keep], boxes[keep]
    return gpd.GeoDataFrame(
        {
            "grid_e": I,
            "grid_n": J,
            "xmin": I * size_m,
            "ymin": J * size_m,
            "xmax": (I + 1) * size_m,
            "ymax": (J + 1) * size_m,
        },
        geometry=boxes,
        crs=crs,
    )


def main_component(grid_e, grid_n, max_gap_cells: int, min_fraction: float = 1.0) -> np.ndarray:
    """Boolean mask of cells in the main connected cluster(s).

    Two cells are connected when their grid offsets differ by at most
    ``max_gap_cells + 1`` in both directions (Chebyshev distance), i.e. when no
    more than ``max_gap_cells`` empty cells separate them. Every cluster with at
    least ``min_fraction`` x (size of the largest cluster) cells is kept; with
    the default 1.0 only the largest cluster is kept. Large parts of a city
    separated by water (Staten Island, ~1.6 km from Brooklyn) survive a small
    fraction such as 0.05, while small detached settlements do not.
    """
    cells = list(zip(np.asarray(grid_e).tolist(), np.asarray(grid_n).tolist()))
    index = {c: k for k, c in enumerate(cells)}
    reach = max_gap_cells + 1
    offsets = [(di, dj) for di in range(-reach, reach + 1) for dj in range(-reach, reach + 1) if (di, dj) != (0, 0)]
    label = np.full(len(cells), -1)
    comps = []
    for start in range(len(cells)):
        if label[start] >= 0:
            continue
        cid, stack, members = len(comps), [start], []
        label[start] = cid
        while stack:
            k = stack.pop()
            members.append(k)
            i, j = cells[k]
            for di, dj in offsets:
                n = index.get((i + di, j + dj))
                if n is not None and label[n] < 0:
                    label[n] = cid
                    stack.append(n)
        comps.append(members)
    if not comps:
        return np.zeros(0, dtype=bool)
    largest = max(len(c) for c in comps)
    keep = [cid for cid, c in enumerate(comps) if len(c) >= min_fraction * largest]
    return np.isin(label, keep)


def sample_id(code: str, grid_e: int, grid_n: int) -> str:
    return f"{code}_e{int(grid_e)}_n{int(grid_n)}"
