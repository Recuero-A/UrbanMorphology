"""Milestone 16 rendering: one standard map style for every figure (docs/M16_PROTOCOL.md §8).

Hillshade of the TRUE terrain (light from the north-west, 45 degrees), water in blue, streets in near-black with width by
road class, bridges in red; the 512 m tile (or the domain) at a fixed extent and scale; optional 100 m scale bar.
Everything is drawn from stored graphs and terrain arrays, never regenerated.
"""
from __future__ import annotations

import numpy as np

WIDTH = {0: 1.6, 1: 1.2, 2: 0.8, 3: 0.6}


def hillshade(elev: np.ndarray, res: float = 4.0, vert: float = 2.0) -> np.ndarray:
    from matplotlib.colors import LightSource
    z = np.nan_to_num(elev.astype(float), nan=float(np.nanmedian(elev)) if np.isfinite(elev).any() else 0.0)
    return LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=vert, dx=res, dy=res)


def draw(ax, d: dict | None, g, size: float = 512.0, margin: float = 256.0, title: str = "", scalebar: bool = True,
         note: str = ""):
    """Draw graph ``g`` (tile metres) over terrain dict ``d`` (4 m grid with ``margin``), extent [0, size]^2."""
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ext = (-margin, size + margin, -margin, size + margin)
    if d is not None:
        e = np.where(d["valid"], d["elev"], np.nan)
        flat = not np.isfinite(e).any() or (np.nanmax(e) - np.nanmin(e)) < 0.5
        if not flat:
            ax.imshow(hillshade(e), cmap="gray", vmin=0, vmax=1, extent=ext, origin="upper", alpha=0.55, zorder=0)
        else:
            ax.set_facecolor("#f4f2ee")
        w = np.asarray(d.get("water", np.zeros(1)))
        if w.ndim == 2 and (w > 0).any():
            ax.imshow(np.ma.masked_where(w == 0, w), cmap="Blues", vmin=0, vmax=1.2, extent=ext, origin="upper",
                      alpha=0.8, zorder=1)
    if g is not None and len(g.edges):
        cls = g.classes() if hasattr(g, "classes") else np.full(len(g.edges), 2)
        st = g.structs() if hasattr(g, "structs") else np.zeros(len(g.edges), int)
        for (a, b), c, s in zip(g.edges, cls, st):
            p = g.nodes[[a, b]]
            ax.plot(p[:, 0], p[:, 1], color="#c0392b" if s == 1 else "#151515", lw=WIDTH.get(int(c), 0.8),
                    solid_capstyle="round", zorder=3)
    elif g is not None:
        ax.text(size / 2, size / 2, "empty", ha="center", va="center", fontsize=9, color="#b03a2e", zorder=4)
    if scalebar:
        ax.plot([size * 0.06, size * 0.06 + 100], [size * 0.05] * 2, color="k", lw=2, zorder=5)
        ax.text(size * 0.06 + 50, size * 0.075, "100 m", ha="center", fontsize=6, zorder=5)
    if title:
        ax.set_title(title, fontsize=7)
    if note:
        ax.text(size * 0.98, size * 0.02, note, ha="right", va="bottom", fontsize=5.5, zorder=5,
                bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=1))
