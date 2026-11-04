"""Procedural landscapes from scratch (Milestone 12): seeded, metric, seamless across tiles.

The height is a sum of terms evaluated in WORLD coordinates (metres), so any window of the same landscape gives the
same values and adjacent tiles join without seams:

    z(x, y) = relief * [ (1 - ridge) * fBm(p / scale) + ridge * ridged(p / scale) ]      coherent noise
            + regional slope . p                                                          a tilted plain
            - valley_depth * exp(-(d_valley / valley_width)^2)                            a Gaussian trough
            + basin / pass terms (scenarios)

``fBm`` is gradient (Perlin-type) noise with ``octaves`` octaves of persistence ``roughness``; ``ridged`` is
1 - |noise|, which gives sharp crests. Water is added by ``terrain.hydrology`` on the whole landscape at once (rivers
and lakes stay continuous over tiles): rivers from drainage area, lakes from deep depressions, sea below
``sea_level``. Every parameter and the seed are stored in the Terrain provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage

from urbanmorph.terrain import hydrology
from urbanmorph.terrain.heightfield import Terrain


@dataclass(frozen=True)
class LandscapeParams:
    seed: int = 0
    relief_m: float = 20.0          # amplitude of the noise terms
    scale_m: float = 600.0          # dominant horizontal wavelength
    octaves: int = 5
    roughness: float = 0.5          # fBm persistence
    ridge: float = 0.0              # 0 = rounded hills, 1 = ridged mountains
    ridge_angle_deg: float = 0.0    # anisotropy of ridges (stretches the noise across this direction)
    anisotropy: float = 1.0         # >1 elongates ridges along ridge_angle
    slope: tuple = (0.0, 0.0)       # regional gradient dz/dx, dz/dy
    valley_depth_m: float = 0.0
    valley_width_m: float = 200.0
    valley_angle_deg: float = 90.0  # direction of the valley axis
    valley_offset_m: float = 0.0    # signed distance of the axis from the origin
    basin_depth_m: float = 0.0      # a closed basin (lake scenario) centred at basin_xy
    basin_radius_m: float = 300.0
    basin_xy: tuple = (0.0, 0.0)
    sea_level: float | None = None
    river_threshold_km2: float = 0.5
    river_a: float = 8.0            # width at 1 km^2 drainage area (m)
    river_b: float = 0.5
    river_carve_m: float = 2.0      # channel depth
    lake_min_area_m2: float = 20000.0
    lake_min_depth_m: float = 1.0
    base_m: float = 50.0


def _hash(i, j, seed):
    h = (i.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (j.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
    h ^= np.uint64(seed) * np.uint64(0x165667B19E3779F9)
    h ^= h >> np.uint64(33)
    h *= np.uint64(0xFF51AFD7ED558CCD)
    h ^= h >> np.uint64(33)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def gradient_noise(x, y, seed):
    """Perlin-type noise in [-~0.7, ~0.7] at lattice coordinates (x, y)."""
    with np.errstate(over="ignore"):
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        out = 0.0
        u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        corners = {}
        for di in (0, 1):
            for dj in (0, 1):
                a = 2 * np.pi * _hash((x0 + di).astype(np.int64), (y0 + dj).astype(np.int64), seed)
                corners[di, dj] = np.cos(a) * (fx - di) + np.sin(a) * (fy - dj)
        top = corners[0, 0] + u * (corners[1, 0] - corners[0, 0])
        bot = corners[0, 1] + u * (corners[1, 1] - corners[0, 1])
        out = top + v * (bot - top)
    return out


def fbm(x, y, p: LandscapeParams, ridged: bool = False):
    a = np.radians(p.ridge_angle_deg)
    xr = (np.cos(a) * x + np.sin(a) * y) / p.anisotropy
    yr = -np.sin(a) * x + np.cos(a) * y
    total, amp, norm = 0.0, 1.0, 0.0
    for o in range(p.octaves):
        f = 2.0 ** o / p.scale_m
        n = gradient_noise(xr * f, yr * f, p.seed * 131 + o)
        total = total + amp * ((1 - np.abs(n) * 1.4) if ridged else n * 1.4)
        norm += amp
        amp *= p.roughness
    return total / norm


def height(x, y, p: LandscapeParams) -> np.ndarray:
    z = p.base_m + p.slope[0] * x + p.slope[1] * y
    if p.relief_m:
        z = z + p.relief_m * ((1 - p.ridge) * fbm(x, y, p) + p.ridge * fbm(x, y, p, ridged=True))
    if p.valley_depth_m:
        a = np.radians(p.valley_angle_deg)
        d = -np.sin(a) * x + np.cos(a) * y - p.valley_offset_m
        z = z - p.valley_depth_m * np.exp(-(d / p.valley_width_m) ** 2)
    if p.basin_depth_m:
        r = np.hypot(x - p.basin_xy[0], y - p.basin_xy[1])
        z = z - p.basin_depth_m * np.exp(-(r / p.basin_radius_m) ** 2)
    return z


def generate(p: LandscapeParams, x0: float, y0: float, x1: float, y1: float, res: float = 4.0,
             hydrology_on: bool = True) -> Terrain:
    """Terrain over [x0, x1] x [y0, y1] (world metres) with rivers, lakes and sea when ``hydrology_on``."""
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    xc = x0 + (np.arange(nx) + 0.5) * res
    yc = y1 - (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xc, yc)
    z = height(X, Y, p)
    water = np.zeros(z.shape, bool)
    kind = np.zeros(z.shape, np.int8)                           # 0 land, 1 river, 2 lake, 3 sea
    surface = np.full(z.shape, np.nan)
    if hydrology_on:
        sea = z < p.sea_level if p.sea_level is not None else None
        # hydrology on a coarser grid (8 m or the grid itself) for speed, then upsampled
        k = max(1, int(round(8.0 / res)))
        zc = z[::k, ::k]
        sc = sea[::k, ::k] if sea is not None else None
        filled = hydrology.priority_flood(zc, sc)
        flow = hydrology.d8(filled)
        area = hydrology.drainage_area(filled, flow, (res * k) ** 2)
        if sc is not None:
            area[sc] = 0
        riv, _ = hydrology.river_mask(area, res * k, p.river_threshold_km2, p.river_a, p.river_b)
        lk, lsurf = hydrology.lakes(zc, filled, res * k, p.lake_min_area_m2, p.lake_min_depth_m)
        up = lambda m: np.repeat(np.repeat(m, k, 0), k, 1)[:ny, :nx]
        riv, lk, lsurf = up(riv), up(lk), up(lsurf)
        if p.river_carve_m:
            carve = ndimage.gaussian_filter(riv.astype(float), 1.0) * p.river_carve_m
            z = z - carve
        kind[riv] = 1
        surface[riv] = ndimage.minimum_filter(np.where(riv, z, np.inf), size=3)[riv] + 0.3
        kind[lk] = 2
        surface[lk] = lsurf[lk]
        if sea is not None:
            kind[sea] = 3
            surface[sea] = p.sea_level
        water = kind > 0
    t = Terrain(z.astype(np.float32), x0, y1, res, native_res=res,
                provenance={"source": "procedural", "params": asdict(p), "bounds": [x0, y0, x1, y1]}, water=water)
    t.water_kind, t.water_surface = kind, surface
    return t


SCENARIOS = {
    "flat_city": LandscapeParams(relief_m=1.5, scale_m=800),
    "rolling_hills": LandscapeParams(relief_m=25, scale_m=500),
    "steep_hillside": LandscapeParams(relief_m=8, scale_m=300, slope=(0.0, 0.18)),
    "mountain_ridges": LandscapeParams(relief_m=120, scale_m=900, ridge=0.8, anisotropy=2.0, ridge_angle_deg=30),
    "narrow_river": LandscapeParams(relief_m=6, scale_m=700, slope=(0.004, 0.0), valley_depth_m=6, valley_width_m=120,
                                    river_threshold_km2=0.05, river_a=12),
    "wide_river": LandscapeParams(relief_m=6, scale_m=700, valley_depth_m=10, valley_width_m=300,
                                  river_threshold_km2=0.05, river_a=60, river_b=0.4),
    "deep_valley": LandscapeParams(relief_m=15, scale_m=600, valley_depth_m=60, valley_width_m=120,
                                   river_threshold_km2=5.0),
    # a ridge along x (an inverted trough) with a notch at the centre: a saddle the road must cross
    "mountain_pass": LandscapeParams(relief_m=20, scale_m=700, ridge=0.5, valley_depth_m=-80, valley_width_m=250,
                                     valley_angle_deg=0, valley_offset_m=256, basin_depth_m=45, basin_radius_m=160,
                                     basin_xy=(256, 256), lake_min_area_m2=1e12),
    "coastal_city": LandscapeParams(relief_m=12, scale_m=600, slope=(0.0, 0.03), sea_level=56.0),
    "large_lake": LandscapeParams(relief_m=8, scale_m=700, basin_depth_m=25, basin_radius_m=350, basin_xy=(256, 256),
                                  lake_min_area_m2=30000),
}


def scenario(name: str, seed: int = 0) -> LandscapeParams:
    return replace(SCENARIOS[name], seed=seed)
