"""Geometric validity checks on triangle meshes, no external dependencies (docs/M11_3D_ARCHITECTURE.md §4.3)."""

from __future__ import annotations

import numpy as np


def tri_areas(V, F) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def signed_volume(V, F) -> float:
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


def solid_report(V, F) -> dict:
    """Closed-solid checks: every directed edge appears once and its reverse once (watertight AND consistently
    oriented), no edge shared by > 2 faces (manifold), no degenerate triangles, positive volume (outward)."""
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    und = np.sort(e, axis=1)
    _, cnt = np.unique(und, axis=0, return_counts=True)
    dirs = {tuple(x) for x in e.tolist()}
    unmatched = sum((b, a) not in dirs for a, b in e.tolist())
    dup_dir = len(e) - len(dirs)
    area = tri_areas(V, F)
    vol = signed_volume(V, F)
    return {"watertight": bool(unmatched == 0 and (cnt == 2).all()), "nonmanifold_edges": int((cnt > 2).sum()),
            "boundary_edges": int((cnt == 1).sum()), "inconsistent_orientation": int(dup_dir),
            "degenerate_tris": int((area < 1e-8).sum()), "volume_m3": vol, "outward": bool(vol > 0),
            "min_z": float(V[:, 2].min()) if len(V) else 0.0}


def surface_report(V, F, target_area: float) -> dict:
    """Open upward surfaces (roads, ground): coverage vs. the source polygons, normals up, no degenerate faces."""
    if not len(F):
        return {"area_m2": 0.0, "target_m2": target_area, "coverage": 0.0, "down_facing": 0, "degenerate_tris": 0}
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = tri_areas(V, F)
    return {"area_m2": float(a.sum()), "target_m2": float(target_area),
            "coverage": float(a.sum() / target_area) if target_area else 0.0,
            "down_facing": int((n[:, 2] <= 0).sum()), "degenerate_tris": int((a < 1e-8).sum())}


def overlap_pairs(polys, min_area: float = 0.5) -> list[tuple[int, int, float]]:
    import shapely
    tree = shapely.STRtree(polys)
    out = []
    for i, p in enumerate(polys):
        for j in tree.query(p):
            if j > i:
                a = p.intersection(polys[j]).area
                if a > min_area:
                    out.append((i, int(j), float(a)))
    return out
