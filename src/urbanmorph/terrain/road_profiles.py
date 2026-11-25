"""Road elevation profiles (Milestone 12): what a street's surface height is, not just the DEM under it.

* Ground streets: the terrain is smoothed along the street (moving average over ``smooth_m``) and the grade is then
  limited to ``max_grade`` by forward / backward passes (the cut-and-fill of a real road). The change from the raw
  terrain is returned, so any geometric adjustment is recorded rather than hidden.
* Bridges and viaducts: a straight deck between the two abutments (the ground profile at the first and last dry /
  supported points), at least ``clearance`` above water or ground; the deck never dips below the ground.
"""

from __future__ import annotations

import numpy as np


def ground_profile(d: np.ndarray, z: np.ndarray, max_grade: float, smooth_m: float = 20.0):
    """(road height, cut(-) / fill(+) relative to the terrain) along distances ``d``."""
    if len(d) < 2:
        return z.copy(), np.zeros_like(z)
    step = float(np.median(np.diff(d)))
    k = max(1, int(round(smooth_m / step)))
    pad = np.pad(z, k, mode="edge")
    zs = np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    zs[0], zs[-1] = z[0], z[-1]                                 # keep the junction heights
    for _ in range(3):                                          # grade limit, both directions
        for i in range(1, len(zs)):
            dz = max_grade * (d[i] - d[i - 1])
            zs[i] = np.clip(zs[i], zs[i - 1] - dz, zs[i - 1] + dz)
        for i in range(len(zs) - 2, -1, -1):
            dz = max_grade * (d[i + 1] - d[i])
            zs[i] = np.clip(zs[i], zs[i + 1] - dz, zs[i + 1] + dz)
    return zs, zs - z


def deck_profile(d: np.ndarray, z_ground: np.ndarray, z_a: float, z_b: float, clearance: float = 0.0,
                 water_surface: np.ndarray | None = None) -> np.ndarray:
    """Straight deck between abutment heights, lifted where needed to clear water / ground by ``clearance``."""
    L = max(d[-1], 1e-9)
    deck = z_a + d / L * (z_b - z_a)
    floor = z_ground if water_surface is None else np.fmax(z_ground, np.nan_to_num(water_surface, nan=-np.inf))
    lift = max(0.0, float(np.max(floor + clearance - deck)))
    if lift > 0:                                                # a raised, level-ish deck with the same end grade
        deck = deck + lift * np.sin(np.pi * d / L) ** 0.5
    return np.maximum(deck, z_ground)
