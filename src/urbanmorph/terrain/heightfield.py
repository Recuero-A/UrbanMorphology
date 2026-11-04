"""A heightfield on a regular metric grid in tile-local coordinates (Milestone 12).

Coordinates follow the street graphs: x east, y north, metres, origin at the tile's south-west corner. The grid
may extend beyond the tile (a margin, so that crops near the edge see terrain). Row 0 is the NORTH edge, as in a
GeoTIFF; ``elev[r, c]`` is the value at the pixel centre (x0 + (c + 0.5) res, y1 - (r + 0.5) res).

Slope is |grad z| as a fraction (0.1 = 10 % = 5.7 deg), from central differences on the grid. Street grades use
``profile``: the terrain is sampled along the whole segment (every ``step`` metres), never from the endpoints only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Terrain:
    elev: np.ndarray                    # (H, W) float32 metres; NaN = no data
    x0: float                           # west edge of the grid, tile-local metres
    y1: float                           # north edge of the grid, tile-local metres
    res: float                          # grid spacing, metres
    native_res: float | None = None     # resolution of the source data (resampling never adds detail)
    provenance: dict = field(default_factory=dict)
    water: np.ndarray | None = None     # (H, W) bool, optional

    @property
    def shape(self):
        return self.elev.shape

    @property
    def y0(self) -> float:
        return self.y1 - self.elev.shape[0] * self.res

    @property
    def x1(self) -> float:
        return self.x0 + self.elev.shape[1] * self.res

    def rc(self, x, y):
        """Fractional (row, col) of points, pixel centres at integers."""
        return (self.y1 - np.asarray(y, float)) / self.res - 0.5, (np.asarray(x, float) - self.x0) / self.res - 0.5

    def sample(self, x, y) -> np.ndarray:
        """Bilinear elevation at points; NaN outside the grid or next to no-data."""
        r, c = self.rc(x, y)
        H, W = self.elev.shape
        r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
        ok = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < H) & (c0 + 1 < W)
        out = np.full(np.shape(r), np.nan)
        if ok.any():
            rr, cc, fr, fc = r0[ok], c0[ok], (r - r0)[ok], (c - c0)[ok]
            e = self.elev
            out[ok] = ((1 - fr) * (1 - fc) * e[rr, cc] + (1 - fr) * fc * e[rr, cc + 1]
                       + fr * (1 - fc) * e[rr + 1, cc] + fr * fc * e[rr + 1, cc + 1])
        return out

    def gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """(dz/dx east, dz/dy north) per pixel, dimensionless."""
        dzdr, dzdc = np.gradient(self.elev.astype(float), self.res)
        return dzdc, -dzdr                                        # rows run south

    def slope(self) -> np.ndarray:
        gx, gy = self.gradient()
        return np.hypot(gx, gy)

    def window(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Pixels whose centres lie inside [x0, x1) x [y0, y1) as a boolean mask (e.g. the tile without margin)."""
        H, W = self.elev.shape
        xc = self.x0 + (np.arange(W) + 0.5) * self.res
        yc = self.y1 - (np.arange(H) + 0.5) * self.res
        return ((yc >= y0) & (yc < y1))[:, None] & ((xc >= x0) & (xc < x1))[None, :]

    def profile(self, a, b, step: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """Distances along a->b and terrain elevation there (every ``step`` m, both ends included)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.hypot(*(b - a)))
        n = max(1, int(np.ceil(L / step)))
        t = np.linspace(0, 1, n + 1)
        p = a + t[:, None] * (b - a)
        return t * L, self.sample(p[:, 0], p[:, 1])


def grade_stats(d: np.ndarray, z: np.ndarray, window: float = 10.0) -> dict:
    """Grades of one terrain profile: mean |dz|/L, max over ``window``-metre sub-spans, elevation change."""
    if len(d) < 2 or not np.isfinite(z).all():
        return {"length": float(d[-1]) if len(d) else 0.0, "mean_grade": np.nan, "max_grade": np.nan, "dz": np.nan}
    L = d[-1]
    mean = abs(z[-1] - z[0]) / L if L > 0 else np.nan
    k = max(1, int(round(window / max(d[1] - d[0], 1e-9))))
    if len(d) > k:
        mx = float(np.max(np.abs(z[k:] - z[:-k]) / (d[k:] - d[:-k])))
    else:
        mx = mean
    return {"length": float(L), "mean_grade": float(mean), "max_grade": mx, "dz": float(z[-1] - z[0])}
