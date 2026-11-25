"""Procedural texture images for the Latin Caribbean theme (no downloaded assets; themes/latin_caribbean/DESIGN.md §3).

Every surface texture tiles seamlessly (periodic noise from filtered white noise in the Fourier domain; periodic
Voronoi cells from a KD-tree with ``boxsize``). Grey images are tinted by the material colour; coloured ones are
used as they are. Cut-out images (fronds, leaves, balusters, iron railings, flowers) carry alpha. Deterministic:
one fixed seed per image.

    python -m urbanmorph.city3d.latin.textures [out_dir]    # writes themes/latin_caribbean/textures/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parents[4] / "themes" / "latin_caribbean" / "textures"


def hexrgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255


def pnoise(shape, sigma, rng) -> np.ndarray:
    """Periodic smooth noise, zero mean, unit std; sigma in pixels."""
    h, w = shape
    W = np.fft.fft2(rng.standard_normal(shape))
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None]
    out = np.real(np.fft.ifft2(W * np.exp(-2 * (np.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))))
    return (out - out.mean()) / (out.std() + 1e-12)


def voronoi(n: int, cells: int, rng, jitter: float = 0.8, aspect=(1.0, 1.0)):
    """Periodic Voronoi on an n x n image with cells x cells jittered seeds: (id, d1, d2) per pixel, d in pixels."""
    g = (np.stack(np.meshgrid(np.arange(cells), np.arange(cells)), -1).reshape(-1, 2) + 0.5
         + (rng.uniform(-0.5, 0.5, (cells * cells, 2)) * jitter)) / cells
    g %= 1.0
    t = cKDTree(g * aspect, boxsize=np.array(aspect, float))
    yy, xx = np.mgrid[0:n, 0:n] / n
    d, i = t.query(np.c_[xx.ravel(), yy.ravel()] * aspect, 2)
    return i[:, 0].reshape(n, n), (d[:, 0] * n).reshape(n, n), (d[:, 1] * n).reshape(n, n)


def save(a: np.ndarray, path: Path) -> None:
    """a: (h, w, 3|4) in [0, 1] with row 0 = v 0 (bottom); written flipped (image row 0 = top)."""
    Image.fromarray((np.clip(a, 0, 1)[::-1] * 255 + 0.5).astype(np.uint8)).save(path, optimize=True)


def grey(v: np.ndarray) -> np.ndarray:
    return np.repeat(v[..., None], 3, -1)


# ---------------------------------------------------------------------------------------------- surfaces
def plaster(n=512):
    r = np.random.default_rng(1)
    v = 0.93 + 0.035 * pnoise((n, n), 40, r) + 0.02 * pnoise((n, n), 7, r) + 0.012 * pnoise((n, n), 1.0, r)
    patch = pnoise((n, n), 14, r) > 2.1                          # small repaired patches
    v = np.where(patch, v - 0.035, v)
    return grey(np.clip(v, 0.78, 1.0))


def roof_tiles(n=512, aged=False, cols=10, courses=6):
    """Spanish barrel tiles, columns running down the slope (v), courses along u. 2.4 m per repeat."""
    r = np.random.default_rng(2 + aged)
    tones = ["#c0643c", "#b4573a", "#cc7446", "#a94f31", "#d0814f", "#b86040"] if not aged else \
        ["#9c5638", "#8a4a33", "#a86448", "#7d4a3a", "#b0603e", "#946048"]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    cw = n / cols
    col = (xx // cw).astype(int)
    xc = (xx % cw) / cw * 2 - 1                                  # -1..1 across a column
    stagger = r.uniform(0, 1, cols)
    cl = n / courses
    vv = (yy / cl + stagger[col]) % courses
    course = np.floor(vv).astype(int)
    fv = vv - course                                             # 0 at the lower end of a tile, 1 at the top
    cover = np.abs(xc) < 0.68                                    # convex cover tile
    prof = np.where(cover, np.sqrt(np.clip(1 - (xc / 0.68) ** 2, 0, 1)), 0.0)
    shade = np.where(cover, 0.5 + 0.55 * prof ** 0.7 + 0.08 * xc, 0.38 + 0.1 * np.abs(xc))
    shade *= np.where(fv < 0.06, 0.65 + 0.35 * fv / 0.06, 1.0)   # shadow under the tile above
    shade *= np.where((fv > 0.94) & cover, 1.08, 1.0)
    tid = (col * 131 + course * 17) % 997
    pal = np.array([hexrgb(h) for h in tones])
    tone = pal[r.integers(len(pal), size=997)[tid]] * 0.5 + pal.mean(0) * 0.5      # gentle tile-to-tile variation
    tone = tone * r.uniform(0.95, 1.05, 997)[tid, None]
    burnt = r.uniform(0, 1, 997)[tid] < 0.02
    tone = np.where(burnt[..., None], tone * 0.72, tone)
    out = tone * shade[..., None] * (1 + 0.04 * pnoise((n, n), 2, r))[..., None]
    if aged:
        moss = np.clip(pnoise((n, n), 18, r) - 0.6, 0, 1)[..., None] * 0.6
        out = out * (1 - moss) + moss * hexrgb("#55573a") * shade[..., None]
    return out


def dome_tiles(n=512):
    """Glazed fish-scale tiles (domes, cupolas). Rows along u, courses along v; 2 m per repeat."""
    r = np.random.default_rng(4)
    rows, cols = 16, 14
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    ch, cw = n / rows, n / cols
    row = (yy // ch).astype(int)
    xs = (xx / cw + 0.5 * (row % 2)) % cols
    fx = xs - np.floor(xs) - 0.5
    fy = yy / ch - row
    scale = np.hypot(fx * 1.0, (fy - 0.0) * 0.9) < 0.62
    base = np.where(scale[..., None], hexrgb("#c2603a"), hexrgb("#9a4a2e"))
    tone = r.uniform(0.88, 1.08, (rows + 1, cols + 1))[row % rows, np.floor(xs).astype(int) % cols]
    sh = 0.8 + 0.25 * fy
    return base * (tone * sh)[..., None] * (1 + 0.03 * pnoise((n, n), 2, r))[..., None]


def cobble(n=512):
    """Rounded river stones / setts in mortar; 3 m per repeat."""
    r = np.random.default_rng(5)
    i, d1, d2 = voronoi(n, 20, r, 0.7)
    edge = d2 - d1
    stones = ["#9b9085", "#8a8076", "#a8998a", "#7f776f", "#b09c86", "#968b7f"]
    pal = np.array([hexrgb(h) for h in stones])
    tone = pal[r.integers(len(pal), size=i.max() + 1)[i]] * r.uniform(0.88, 1.1, i.max() + 1)[i, None]
    dome = np.clip(edge / 7.0, 0, 1) ** 0.5
    out = tone * (0.7 + 0.3 * dome)[..., None] * (1 + 0.05 * pnoise((n, n), 1.5, r))[..., None]
    mortar = edge < 2.2
    return np.where(mortar[..., None], hexrgb("#5f564c") * (1 + 0.1 * pnoise((n, n), 3, r))[..., None], out)


def pavers(n=512):
    """Warm stone slabs, running bond; 2.4 m per repeat (6 x 6 slabs)."""
    r = np.random.default_rng(6)
    k = 6
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    s = n / k
    row = (yy // s).astype(int)
    xs = (xx / s + 0.5 * (row % 2)) % k
    ci = np.floor(xs).astype(int)
    joint = (np.minimum(xs - ci, 1 - (xs - ci)) * s < 1.6) | (np.minimum(yy % s, s - yy % s) < 1.6)
    pal = np.array([hexrgb(h) for h in ("#e8d5bd", "#f0dfc9", "#dcc4a8", "#e5cdb6", "#d8c0a5")])
    tone = pal[r.integers(len(pal), size=(k, k))[row % k, ci % k]] * r.uniform(0.94, 1.04, (k, k))[row % k, ci % k, None]
    out = tone * (1 + 0.035 * pnoise((n, n), 5, r) + 0.02 * pnoise((n, n), 1, r))[..., None]
    return np.where(joint[..., None], hexrgb("#a8998a"), out)


def brick_plaza(n=512):
    """Terracotta brick paving in basket weave; 2.4 m per repeat."""
    r = np.random.default_rng(7)
    sq = 12
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    s = n / sq
    bx, by = (xx // s).astype(int), (yy // s).astype(int)
    fx, fy = xx % s, yy % s
    horiz = (bx + by) % 2 == 0
    half = np.where(horiz, fy < s / 2, fx < s / 2)
    joint = (np.minimum(fx, s - fx) < 1.3) | (np.minimum(fy, s - fy) < 1.3) | \
            np.where(horiz, np.abs(fy - s / 2) < 0.9, np.abs(fx - s / 2) < 0.9)
    pal = np.array([hexrgb(h) for h in ("#b8623f", "#c47048", "#a8583a", "#c9784e", "#9e5236")])
    bid = (bx * 7 + by * 13 + half * 3) % 211
    tone = pal[r.integers(len(pal), size=211)[bid]] * r.uniform(0.9, 1.06, 211)[bid, None]
    out = tone * (1 + 0.04 * pnoise((n, n), 1.5, r))[..., None]
    return np.where(joint[..., None], hexrgb("#d6c2a4"), out)


def asphalt(n=512):
    r = np.random.default_rng(8)
    v = 1 + 0.06 * pnoise((n, n), 30, r) + 0.08 * pnoise((n, n), 0.8, r)
    return grey(np.clip(v, 0.7, 1.2) / 1.2)


def mix3(n, cols, sigmas, seed, grain=0.06):
    """Blend of colours by smooth periodic noise (soil, grass, jungle floor, sand base)."""
    r = np.random.default_rng(seed)
    pal = [hexrgb(h) for h in cols]
    out = np.broadcast_to(pal[0], (n, n, 3)).copy()
    for c, s in zip(pal[1:], sigmas):
        w = np.clip(pnoise((n, n), s, r) * 0.8 + 0.2, 0, 1)[..., None]
        out = out * (1 - w) + c * w
    return out * (1 + grain * pnoise((n, n), 1.0, r))[..., None]


def grass(n=512):
    r = np.random.default_rng(9)
    out = mix3(n, ["#5d8f34", "#6fa13c", "#4d7d2c", "#7fa846"], [30, 12, 50], 9, 0.0)
    blades = pnoise((n, n), 0.7, r) * 0.08
    return out * (1 + blades)[..., None]


def soil(n=512):
    return mix3(n, ["#8f6a45", "#6f8a3a", "#a5774c", "#5f7f33"], [40, 18, 60], 10)


def jungle_floor(n=512):
    return mix3(n, ["#2f4a22", "#3d5a28", "#4a4a2a", "#25401c"], [25, 10, 45], 11, 0.1)


def sand(n=512):
    r = np.random.default_rng(12)
    out = mix3(n, ["#ead6ab", "#e2cb9c", "#f0dfba"], [35, 12], 12, 0.05)
    yy, xx = np.mgrid[0:n, 0:n] / n
    rip = np.sin(2 * np.pi * (9 * yy + 0.6 * pnoise((n, n), 60, r) / 6))
    return out * (1 + 0.025 * rip)[..., None]


def desert_floor(n=512):
    """Petra's desert floor: pinkish sand and gravel, mottled at two scales, fine grain (no discrete pebbles: from afar
    they read as a dotted grid)."""
    r = np.random.default_rng(31)
    out = mix3(n, ["#d3ad87", "#caa17b", "#d9b690", "#c59c77"], [14, 6, 24], 31, 0.05)   # low contrast: no visible repeat
    return out * (1 + 0.035 * pnoise((n, n), 2.0, r) + 0.03 * pnoise((n, n), 0.7, r))[..., None]


def sandstone(n=512):
    """Petra's rose-red sandstone: wavy strata (rows = height on cliffs, corner_uvs) of rose, salmon, deep red, pale
    ochre and violet-brown, lightly mottled."""
    r = np.random.default_rng(32)
    yy = np.mgrid[0:n, 0:n][0] / n
    v = yy + 0.05 * pnoise((n, n), 50, r) + 0.012 * pnoise((n, n), 8, r)
    cols = np.array([hexrgb(h) for h in ("#b8674a", "#d39a73", "#9a4f3c", "#e3bb92", "#c47a57", "#8c5a4e", "#cf8c64")])
    x = (v * 9) % len(cols)                                    # 9 strata per repeat, colour changes smoothly
    i0 = np.floor(x).astype(int) % len(cols)
    f = x - np.floor(x)
    f = np.clip((f - 0.35) / 0.3, 0, 1)
    f = f * f * (3 - 2 * f)
    out = cols[i0] * (1 - f)[..., None] + cols[(i0 + 1) % len(cols)] * f[..., None]
    fine = (np.sin(2 * np.pi * 40 * v) + 1) / 2                # thin laminae
    out = out * (0.93 + 0.07 * fine)[..., None]
    return out * (1 + 0.07 * pnoise((n, n), 18, r) + 0.05 * pnoise((n, n), 1.0, r))[..., None]


def garden_dry(n=512):
    """A desert yard: earth with patches of dry grass."""
    return mix3(n, ["#a88259", "#b8926a", "#9a9060", "#c4a077"], [35, 14, 55], 33, 0.08)


def cactus(n=512):
    """Columnar cactus skin (greyscale, tinted): 8 ribs around (u 0-1 = the circumference), light areoles on the ridges."""
    r = np.random.default_rng(34)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    ph = (xx / n * 8) % 1.0
    rib = np.abs(np.sin(np.pi * ph)) ** 0.7
    v = 0.55 + 0.45 * rib
    areole = (np.abs(ph - 0.5) < 0.05) & (((yy / n * 30) % 1.0) < 0.12)
    v = np.where(areole, 1.35, v)
    return np.clip(grey(v * (1 + 0.04 * pnoise((n, n), 4, r))) / 1.35, 0, 1)


def leaves(n=512, flowers=None, share=0.0, seed=13, pal=("#3d6e2a", "#4f8a33", "#5f9a3a", "#2f5a24", "#467f2e")):
    """Opaque canopy texture: leaf clusters (Voronoi) lit from above; optional flower clusters."""
    r = np.random.default_rng(seed)
    i, d1, d2 = voronoi(n, 52, r, 0.9)
    pal = np.array([hexrgb(h) for h in pal])
    tone = pal[r.integers(len(pal), size=i.max() + 1)[i]]
    if flowers is not None:
        fp = np.array([hexrgb(h) for h in flowers])
        f = r.uniform(0, 1, i.max() + 1) < share
        tone = np.where(f[i][..., None], fp[r.integers(len(fp), size=i.max() + 1)[i]], tone)
    edge = np.clip((d2 - d1) / 4.0, 0, 1)
    sh = (0.6 + 0.4 * edge ** 0.6) * (1 + 0.18 * pnoise((n, n), 16, r))      # leaf clusters + clumps in light / shade
    return tone * sh[..., None] * (1 + 0.08 * pnoise((n, n), 1.2, r))[..., None]


def wood(n=512):
    r = np.random.default_rng(14)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    g = pnoise((n, n), 1.0, r) * 0.03 + np.sin(xx / 3.0 + 4 * pnoise((n, n), 25, r)) * 0.035
    return grey(np.clip(0.88 + g + 0.03 * pnoise((n, n), 20, r), 0.7, 1.0))


def door(n=512):
    """Vertical planks with grooves and two horizontal battens (any u offset reads right)."""
    r = np.random.default_rng(15)
    base = wood(n)[..., 0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    plank = n / 8
    groove = np.minimum(xx % plank, plank - xx % plank) < 2.0
    v = np.where(groove, base * 0.6, base)
    for y0 in (0.18, 0.62):
        b = np.abs(yy / n - y0) < 0.035
        v = np.where(b, base * 0.95, v)
        v = np.where(np.abs(yy / n - y0 + 0.037) < 0.004, base * 0.6, v)
    return grey(v)


def shutter(n=512):
    """Louvred shutter slats every 5 cm (1 m per repeat)."""
    yy = np.mgrid[0:n, 0:n][0].astype(float)
    f = (yy % (n / 20)) / (n / 20)
    v = 0.62 + 0.38 * f ** 0.7
    return grey(v * (0.97 + 0.03 * wood(n)[..., 0]))


def palm_trunk(n=512):
    """Coconut palm trunk: grey-brown bark with leaf-scar rings (u around the trunk, v along it; 1 m)."""
    r = np.random.default_rng(16)
    yy = np.mgrid[0:n, 0:n][0].astype(float)
    ring = (yy % (n / 9)) / (n / 9)
    v = 0.75 + 0.2 * np.sin(np.pi * ring) ** 0.5 + 0.05 * pnoise((n, n), 3, r)
    return np.clip(v, 0, 1)[..., None] * hexrgb("#8c7a66")[None, None] * (1 + 0.05 * pnoise((n, n), 1, r))[..., None]


# ---------------------------------------------------------------------------------------------- cut-outs
def _rgba(w, h, draw_fn, ss=2):
    """Supersampled RGBA drawing; draw_fn(draw, W, H) draws on an image whose row 0 is the TOP."""
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(im), w * ss, h * ss)
    im = im.resize((w, h), Image.LANCZOS)
    a = np.asarray(im).astype(float) / 255
    rgb = a[..., :3]
    alpha = a[..., 3:]
    # bleed colours into transparent pixels so mip-maps and filtering never pull in black fringes
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(6))).astype(float) / 255
    rgb = np.where(alpha > 0.02, rgb / np.maximum(alpha, 1e-3), blur[..., :3] / np.maximum(blur[..., 3:], 1e-3))
    return np.concatenate([np.clip(rgb, 0, 1), alpha], -1)[::-1]          # back to row 0 = v 0


def palm_frond(w=256, h=1024):
    r = np.random.default_rng(17)
    greens = ["#4f7d2c", "#5d8f33", "#6b9a3a", "#456f28", "#7aa645"]

    def fn(d, W, H):
        cx = W / 2
        for k in range(150):
            t = 0.04 + 0.95 * k / 150
            y = H * (1 - t)
            L = W * 0.49 * np.sin(np.pi * min(1.0, t * 1.05)) ** 0.55 * r.uniform(0.85, 1.05)
            for s in (-1, 1):
                ang = np.radians(r.uniform(52, 66))
                ex, ey = cx + s * L * np.sin(ang), y - L * np.cos(ang)
                wd = W * 0.022
                c = tuple(int(255 * x) for x in hexrgb(greens[r.integers(len(greens))]))
                d.polygon([(cx, y + wd), (cx, y - wd), (ex, ey)], fill=c + (255,))
        d.line([(cx, H), (cx, 0)], fill=(150, 160, 90, 255), width=max(2, W // 40))
    return _rgba(w, h, fn)


def fan_leaf(n=512):
    r = np.random.default_rng(18)

    def fn(d, W, H):
        cx, cy = W / 2, H * 0.98
        for a in np.linspace(np.radians(-100), np.radians(100), 46):
            L = H * 0.93 * r.uniform(0.9, 1.0)
            for da in (-0.028, 0.028):
                pass
            p = [(cx, cy), (cx + L * np.sin(a - 0.03), cy - L * np.cos(a - 0.03)),
                 (cx + L * 0.9 * np.sin(a), cy - L * 0.9 * np.cos(a)), (cx + L * np.sin(a + 0.03), cy - L * np.cos(a + 0.03))]
            g = r.uniform(0.85, 1.1)
            d.polygon(p, fill=(int(84 * g), int(130 * g), int(70 * g), 255))
        d.ellipse([cx - W * 0.05, cy - W * 0.05, cx + W * 0.05, cy + W * 0.05], fill=(110, 120, 60, 255))
    return _rgba(n, n, fn)


def banana_leaf(w=256, h=1024):
    r = np.random.default_rng(19)

    def fn(d, W, H):
        ys = np.linspace(0.02, 0.98, 60)
        half = np.sin(np.pi * ys) ** 0.6 * W * 0.46
        left = [(W / 2 - hw, H * (1 - y)) for y, hw in zip(ys, half)]
        right = [(W / 2 + hw, H * (1 - y)) for y, hw in zip(ys, half)][::-1]
        d.polygon(left + right, fill=(104, 160, 58, 255))
        for _ in range(14):                                       # tears between veins
            y = H * (1 - r.uniform(0.15, 0.9))
            s = r.choice([-1, 1])
            d.line([(W / 2 + s * W * 0.1, y), (W / 2 + s * W * 0.5, y - H * 0.02)], fill=(0, 0, 0, 0), width=max(2, W // 80))
        for k in range(40):                                       # parallel veins
            y = H * (1 - 0.05 - 0.9 * k / 40)
            for s in (-1, 1):
                d.line([(W / 2, y), (W / 2 + s * W * 0.45, y - H * 0.03)], fill=(90, 142, 50, 255), width=1)
        d.line([(W / 2, H), (W / 2, 0)], fill=(185, 210, 120, 255), width=max(3, W // 30))
    return _rgba(w, h, fn)


def flowers(n=512):
    """Tropical understorey card: lance leaves with heliconia / hibiscus colours."""
    r = np.random.default_rng(20)

    def fn(d, W, H):
        for _ in range(18):
            x0 = W * r.uniform(0.2, 0.8)
            a = r.uniform(-0.7, 0.7)
            L = H * r.uniform(0.5, 0.9)
            x1, y1 = x0 + L * np.sin(a), H - L * np.cos(a)
            wd = W * 0.05
            c = tuple(int(x) for x in r.uniform([50, 110, 40], [90, 160, 70]))
            d.polygon([(x0, H), (x0 + (x1 - x0) * 0.5 - wd, H - (H - y1) * 0.5), (x1, y1), (x0 + (x1 - x0) * 0.5 + wd, H - (H - y1) * 0.5)],
                      fill=c + (255,))
        cols = [(216, 48, 47), (242, 194, 48), (230, 70, 120), (245, 130, 40)]
        for _ in range(22):
            x, y = W * r.uniform(0.15, 0.85), H * r.uniform(0.1, 0.55)
            c = cols[r.integers(len(cols))]
            s = W * r.uniform(0.025, 0.05)
            d.polygon([(x, y - s * 1.6), (x + s, y), (x, y + s * 0.6), (x - s, y)], fill=c + (255,))
    return _rgba(n, n, fn)


def balusters(n=512):
    """Turned wooden balusters (Cartagena balconies), 9 per metre, greyscale wood (tinted)."""
    def fn(d, W, H):
        k = 9
        for i in range(k):
            cx = (i + 0.5) * W / k
            prof = [(0.0, 0.32), (0.08, 0.32), (0.1, 0.22), (0.25, 0.3), (0.42, 0.42), (0.5, 0.2), (0.58, 0.42),
                    (0.75, 0.3), (0.9, 0.22), (0.92, 0.32), (1.0, 0.32)]
            hw = W / k * 0.5
            left = [(cx - hw * w_, H * (1 - y)) for y, w_ in prof]
            right = [(cx + hw * w_, H * (1 - y)) for y, w_ in prof][::-1]
            d.polygon(left + right, fill=(225, 225, 225, 255))
            d.line([(cx - hw * 0.12, H), (cx - hw * 0.12, 0)], fill=(190, 190, 190, 255), width=2)
    return _rgba(n, n, fn)


def iron_railing(n=512):
    """Wrought-iron railing: bars every 12 cm with C-scrolls between two bands (1 m x 1 m)."""
    def fn(d, W, H):
        c = (230, 230, 230, 255)
        bw = max(3, W // 90)
        for y in (0.04, 0.16, 0.84, 0.96):
            d.line([(0, H * (1 - y)), (W, H * (1 - y))], fill=c, width=bw)
        k = 8
        for i in range(k):
            x = (i + 0.5) * W / k
            d.line([(x, 0), (x, H)], fill=c, width=bw)
            r_ = W / k * 0.42
            for yc, s in ((0.28, 1), (0.72, -1)):
                cy = H * (1 - yc)
                d.arc([x + W / k * 0.5 - r_, cy - r_, x + W / k * 0.5 + r_, cy + r_], 200 if s > 0 else 20,
                      340 if s > 0 else 160, fill=c, width=bw)
    return _rgba(n, n, fn)


# ---------------------------------------------------------------------------------------------- landmark towers
def marble(n=512):
    """Veined marble (the green columns of the Eagle Belfry): a net of light veins on mottled stone, greyscale (tinted);
    2 m per repeat."""
    r = np.random.default_rng(31)
    a, b = pnoise((n, n), 22, r), pnoise((n, n), 9, r)
    veins = np.exp(-(a / 0.09) ** 2) + 0.6 * np.exp(-(b / 0.07) ** 2)
    v = 0.66 + 0.06 * pnoise((n, n), 50, r) + 0.03 * pnoise((n, n), 3, r) + 0.32 * np.clip(veins, 0, 1)
    return grey(v)


ROMAN = ["XII", "I", "II", "III", "IIII", "V", "VI", "VII", "VIII", "IX", "X", "XI"]


def clock(n=512, dial=(31, 98, 100, 255), ink=(226, 182, 72, 255)):
    """Clock face (used as it is): teal dial, gilt Roman numerals set radially, minute dots, hands at ten past ten
    (dial and ink colours can change: the white dial of the Gothic church)."""
    gold, teal = ink, dial

    def fn(d, W, H):
        c, R = W / 2, W / 2
        d.rectangle([0, 0, W, H], fill=teal)
        for rr, col in ((R * 0.995, gold), (R * 0.95, teal), (R * 0.93, gold), (R * 0.905, teal)):
            d.ellipse([c - rr, c - rr, c + rr, c + rr], fill=col)
        for m in range(60):
            if m % 5:
                a = np.radians(90 - 6 * m)
                x, y = c + 0.86 * R * np.cos(a), c - 0.86 * R * np.sin(a)
                d.ellipse([x - W / 200, y - W / 200, x + W / 200, y + W / 200], fill=gold)
        gh, lw = 0.17 * R, max(3, W // 110)                  # glyph height, stroke width
        for k, txt in enumerate(ROMAN):
            a = np.radians(90 - 30 * k)
            out, tan = np.array([np.cos(a), np.sin(a)]), np.array([np.sin(a), -np.cos(a)])
            base = np.array([c, c]) + out * 0.6 * R                   # glyph foot centre (glyph 'up' = outward)
            ws = [0.5 * gh if ch != "I" else 0.2 * gh for ch in txt]
            x = -sum(ws) / 2
            for ch, w_ in zip(txt, ws):
                strokes = {"I": [((0.5, 0), (0.5, 1))], "V": [((0, 1), (0.5, 0)), ((0.5, 0), (1, 1))],
                           "X": [((0, 0), (1, 1)), ((0, 1), (1, 0))]}[ch]
                for (x0, y0), (x1, y1) in strokes:
                    p = [base + tan * (x + xx * w_) + out * (yy * gh) for xx, yy in ((x0, y0), (x1, y1))]
                    d.line([(p[0][0], H - p[0][1]), (p[1][0], H - p[1][1])], fill=gold, width=lw)
                x += w_
        for ang, L, wd in ((90 - 30 * 10 - 2.5, 0.48, 0.05), (90 - 6 * 10, 0.72, 0.035)):    # hour, minute hands
            a = np.radians(ang)
            e = np.array([np.cos(a), np.sin(a)])
            s = np.array([-e[1], e[0]]) * wd * R
            tip, tail = np.array([c, c]) + e * L * R, np.array([c, c]) - e * 0.12 * R
            poly = [tail + s, tip, tail - s]
            d.polygon([(p[0], H - p[1]) for p in poly], fill=gold)
        d.ellipse([c - 0.06 * R, c - 0.06 * R, c + 0.06 * R, c + 0.06 * R], fill=gold)
    return _rgba(n, n, fn)


def filigree(n=512):
    """Cut-out baroque relief (gables, dome dormers, cartouches): a palmette on a stem, C-scrolls curling out on both
    sides, rosettes and leaves; mirror-symmetric, light grey (tinted by the trim colour). Drawn for a panel whose
    bottom edge is the springing line and whose outline narrows to a point at the top."""
    def fn(d, W, H):
        col = (238, 238, 238, 255)
        lw = max(4, W // 40)
        cx = W / 2
        P = lambda x, y: (cx + x * W, H - y * H)                       # (x from centre, y up) in panel units
        for s in (-1, 1):
            # C-scrolls rising from the stem foot and curling outward, with an inner spiral
            for (ox, oy, r0, a0, a1) in ((0.24, 0.30, 0.17, 200, 470), (0.24, 0.30, 0.09, 230, 520), (0.36, 0.16, 0.08, 120, 420)):
                xc, yc = s * ox, oy
                t = np.radians(np.linspace(a0, a1, 40))
                rr = r0 * np.linspace(1.0, 0.45, 40)
                pts = [P(xc + s * q * np.cos(tt), yc + q * np.sin(tt) * W / H) for q, tt in zip(rr, t)]
                d.line(pts, fill=col, width=lw, joint="curve")
            # acanthus leaves along the stem
            for y0, L, a in ((0.18, 0.2, 35), (0.38, 0.16, 40), (0.56, 0.11, 45)):
                aa = np.radians(a)
                tip = (s * L * np.sin(aa), y0 + L * np.cos(aa) * W / H * 0.7)
                mid = (s * L * 0.55 * np.sin(aa) + s * 0.03, y0 + L * 0.3)
                d.polygon([P(0, y0), P(*mid), P(*tip), P(s * L * 0.2 * np.sin(aa), y0 + L * 0.45)], fill=col)
            # rosettes low on each side
            for (x0, y0, rr) in ((0.37, 0.42, 0.045), (0.16, 0.08, 0.035)):
                for k in range(6):
                    a = 2 * np.pi * k / 6
                    px, py = s * x0 + rr * np.cos(a), y0 + rr * np.sin(a) * W / H
                    q = rr * 0.55
                    d.ellipse([P(px - q, py + q * W / H)[0], P(px - q, py + q * W / H)[1], P(px + q, py - q * W / H)[0],
                               P(px + q, py - q * W / H)[1]], fill=col)
        d.line([P(0, 0.0), P(0, 0.62)], fill=col, width=lw)            # stem
        # palmette: a pointed central leaf and two pairs of petals
        d.polygon([P(0, 0.92), P(0.05, 0.72), P(0.03, 0.6), P(0, 0.56), P(-0.03, 0.6), P(-0.05, 0.72)], fill=col)
        for s in (-1, 1):
            d.polygon([P(0, 0.6), P(s * 0.12, 0.78), P(s * 0.1, 0.66)], fill=col)
            d.polygon([P(0, 0.56), P(s * 0.17, 0.62), P(s * 0.13, 0.54)], fill=col)
        d.line([P(-0.3, 0.02), P(0.3, 0.02)], fill=col, width=lw)
    return _rgba(n, n, fn)


def shell(w=512, h=256):
    """Cut-out scallop shell (the hoods of the Golden Clock Tower): fluted fan from a hinge at the bottom centre,
    scalloped rim; light grey (tinted)."""
    def fn(d, W, H):
        col = (240, 240, 240, 255)
        c = (W / 2, H)
        R = 0.49 * W
        k = 13
        for i in range(k):
            a0, a1 = np.pi * i / k + 0.012, np.pi * (i + 1) / k - 0.012
            t = np.linspace(a0, a1, 10)
            rim = [(c[0] + R * (0.93 + 0.07 * np.sin(np.pi * (tt - a0) / (a1 - a0))) * np.cos(tt),
                    c[1] - R * (0.93 + 0.07 * np.sin(np.pi * (tt - a0) / (a1 - a0))) * np.sin(tt) * 2 * H / W) for tt in t]
            d.polygon([(c[0] + 0.16 * R * np.cos(a0), c[1] - 0.16 * R * np.sin(a0) * 2 * H / W)] + rim
                      + [(c[0] + 0.16 * R * np.cos(a1), c[1] - 0.16 * R * np.sin(a1) * 2 * H / W)], fill=col)
        d.pieslice([c[0] - 0.13 * R, c[1] - 0.13 * R * 2 * H / W, c[0] + 0.13 * R, c[1] + 0.13 * R * 2 * H / W], 180, 360, fill=col)
    return _rgba(w, h, fn)


def scales(n=512):
    """Greyscale fish-scale tiles (tinted: the stone-grey bulbs of the Bahia church)."""
    a = dome_tiles(n)
    v = a.mean(-1)
    v = 0.62 + 0.38 * (v - v.min()) / max(v.max() - v.min(), 1e-6)
    return grey(v)


def window_lit(cw=256, h=512):
    """Window interiors seen through the glass, an atlas of four cells (u 0-0.25, ...): lit room with curtains and a
    lamp; lit with a plant; half-lit; dark. Colours as they are; mapped once per window (UV 0-1 within a cell)."""
    r = np.random.default_rng(41)
    cells = []
    for kind in range(4):
        y, x = np.mgrid[0:h, 0:cw] / np.array([h, cw])[:, None, None]
        glow = [1.0, 0.9, 0.55, 0.0][kind]
        cx, cy = [0.5, 0.4, 0.55, 0.5][kind], 0.62
        d = np.hypot((x - cx) * 1.6, (y - cy) * 1.0)
        room = np.array([0.16, 0.09, 0.05]) + glow * np.array([0.95, 0.62, 0.25]) * np.exp(-(d / 0.42) ** 2)[..., None]
        room = room * (1 - 0.35 * (y < 0.2)[..., None]) if kind < 3 else np.array([0.07, 0.09, 0.1]) + 0.05 * y[..., None]
        if kind == 3:                                              # dark: night-blue glass with a reflection streak
            room = room + 0.12 * np.exp(-((x - 0.8 * y - 0.1) / 0.08) ** 2)[..., None]
        floor = (y < 0.12)[..., None]
        room = np.where(floor, room * 0.6, room)
        cur = np.clip(0.24 - x, 0, 1) + np.clip(x - 0.76, 0, 1)    # curtains at both sides with folds
        fold = 0.8 + 0.2 * np.sin(x * 90)
        ccol = np.array([0.93, 0.86, 0.72]) * (0.55 + 0.45 * glow)
        room = np.where((cur > 0)[..., None] & (y > 0.08)[..., None], ccol * fold[..., None], room)
        if kind == 1:                                              # potted plant on the sill
            pl = (np.hypot((x - 0.42) * 2.2, (y - 0.2) * 1.4) < 0.16)
            room = np.where(pl[..., None], np.array([0.18, 0.38, 0.14]) * (0.7 + 0.3 * glow), room)
        if kind < 3:                                               # lamp shade
            ls = (np.abs(x - cx) < 0.07) & (np.abs(y - cy - 0.04) < 0.05)
            room = np.where(ls[..., None], np.array([1.0, 0.86, 0.55]), room)
        cells.append(np.clip(room, 0, 1))
    return np.concatenate(cells, axis=1)


def window_glass(n=256):
    """Glazing: dark teal glass with a soft sky reflection."""
    y, x = np.mgrid[0:n, 0:n] / n
    base = np.array([0.12, 0.26, 0.29]) + np.array([0.18, 0.24, 0.24]) * y[..., None]
    streak = 0.18 * np.exp(-((x + 0.6 * y - 0.75) / 0.1) ** 2)[..., None]
    return np.clip(base + streak, 0, 1)


TEXTURES = {
    "plaster.png": plaster, "roof_tiles_a.png": roof_tiles, "roof_tiles_b.png": lambda: roof_tiles(aged=True),
    "dome_tiles.png": dome_tiles, "cobble.png": cobble, "pavers.png": pavers, "brick_plaza.png": brick_plaza,
    "asphalt.png": asphalt, "grass.png": grass, "soil.png": soil, "jungle_floor.png": jungle_floor, "sand.png": sand,
    "leaves.png": leaves, "bougainvillea.png": lambda: leaves(flowers=["#d6336c", "#e8559a", "#c22a6a", "#f07ab0"], share=0.6, seed=21),
    "flamboyant.png": lambda: leaves(flowers=["#e2401f", "#f05a28", "#d63a1f", "#f27a30"], share=0.7, seed=22),
    "wood.png": wood, "door.png": door, "shutter.png": shutter, "palm_trunk.png": palm_trunk,
    "palm_frond.png": palm_frond, "fan_leaf.png": fan_leaf, "banana_leaf.png": banana_leaf, "flowers.png": flowers,
    "balusters.png": balusters, "iron_railing.png": iron_railing,
    "marble.png": marble, "clock.png": clock, "filigree.png": filigree, "shell.png": shell,
    "window_lit.png": window_lit, "window_glass.png": window_glass, "scales.png": scales,
    "clock_white.png": lambda: clock(dial=(238, 236, 228, 255), ink=(30, 30, 34, 255)),
    "desert_floor.png": desert_floor, "sandstone.png": sandstone, "garden_dry.png": garden_dry, "cactus.png": cactus,
    "leaves_dry.png": lambda: leaves(seed=35, pal=("#7f7b55", "#918b60", "#a39b70", "#6c6b48", "#8a8a5c")),
}


def ensure(out: Path = OUT, force: bool = False) -> Path:
    """Write every texture once (cached); returns the directory."""
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in TEXTURES.items():
        if force or not (out / name).exists():
            save(fn(), out / name)
    return out


if __name__ == "__main__":
    print(ensure(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT, force=True))
