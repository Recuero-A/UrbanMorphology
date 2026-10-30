"""Matplotlib figures for (3, H, W) layout arrays.

Palette was checked with a colour-vision-deficiency validator (all pairs
distinguishable under protan/deutan simulation); identity is never colour-only
because every figure carries a text legend or title.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no GUI needed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#2b2b2b"
INK_MUTED = "#6b6b6b"
LAYER_COLORS = {
    "roads": "#4562a8",
    "buildings": "#cf8530",
    "green_spaces": "#2f7d4a",
}
LAYER_LABELS = {"roads": "Roads (buffered)", "buildings": "Buildings", "green_spaces": "Green spaces"}
# Paint order for the combined map: later layers are drawn on top.
PAINT_ORDER = ("green_spaces", "buildings", "roads")
CHANNEL_INDEX = {"roads": 0, "buildings": 1, "green_spaces": 2}


def _hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i : i + 2], 16) / 255 for i in (1, 3, 5)])


def composite_rgb(layout: np.ndarray, layers=PAINT_ORDER) -> np.ndarray:
    """(3,H,W) binary layout -> (H,W,3) float RGB image."""
    h, w = layout.shape[1:]
    img = np.tile(_hex_to_rgb(SURFACE), (h, w, 1))
    for name in layers:
        mask = layout[CHANNEL_INDEX[name]].astype(bool)
        img[mask] = _hex_to_rgb(LAYER_COLORS[name])
    return img


def _decorate(ax, size_m: float, title: str) -> None:
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=22)
    ax.set_xlabel("metres east", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("metres north", color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    for s in ax.spines.values():
        s.set_color("#d0d0d0")
    ticks = np.arange(0, size_m + 1, 128)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # Scale bar (100 m) and north arrow sit above the map, never on top of data.
    y0 = size_m * 1.035
    x1 = size_m * 0.90
    ax.plot([x1 - 100, x1], [y0, y0], color=INK, lw=2, solid_capstyle="butt", clip_on=False)
    ax.text(x1 - 50, y0 + 4, "100 m", ha="center", va="bottom", fontsize=8, color=INK, clip_on=False)
    ax.annotate("N", xy=(size_m * 0.97, size_m * 1.075), xytext=(size_m * 0.97, size_m * 1.015),
                ha="center", va="center", fontsize=8, color=INK, annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.set_xlim(0, size_m)
    ax.set_ylim(0, size_m)


def _footer(fig, meta: dict) -> None:
    text = (
        f"{meta.get('description', meta['sample_id'])}  ·  {meta['projected_crs']}  ·  "
        f"{meta['resolution_px']}×{meta['resolution_px']} px @ {meta['pixel_size_m']:g} m/px\n"
        "Map data © OpenStreetMap contributors (ODbL)"
    )
    fig.text(0.01, 0.01, text, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")


def _new_fig(size=(6.4, 7.0)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def plot_single_layer(layout: np.ndarray, meta: dict, layer: str, out: Path) -> Path:
    size = meta["physical_size_m"]
    fig, ax = _new_fig()
    img = composite_rgb(layout, layers=(layer,))
    ax.imshow(img, extent=(0, size, 0, size), interpolation="nearest")
    cov = layout[CHANNEL_INDEX[layer]].mean() * 100
    _decorate(ax, size, f"{LAYER_LABELS[layer]} — channel {CHANNEL_INDEX[layer]}  ({cov:.1f}% of pixels)")
    _footer(fig, meta)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_combined(layout: np.ndarray, meta: dict, out: Path) -> Path:
    size = meta["physical_size_m"]
    fig, ax = _new_fig((6.4, 7.4))
    ax.imshow(composite_rgb(layout), extent=(0, size, 0, size), interpolation="nearest")
    _decorate(ax, size, "Urban layout — combined channels")
    handles = [
        Patch(facecolor=LAYER_COLORS[n], edgecolor="none",
              label=f"{LAYER_LABELS[n]}  {layout[CHANNEL_INDEX[n]].mean() * 100:.1f}%")
        for n in ("roads", "buildings", "green_spaces")
    ]
    handles.append(Patch(facecolor=SURFACE, edgecolor="#bdbdbd", label="Unclassified open space"))
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.055), labelcolor=INK)
    _footer(fig, meta)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_vector_vs_raster(sample, meta: dict, out: Path) -> Path:
    """Side-by-side: clipped vector data vs. its 128x128 raster (shows what is lost)."""
    size = meta["physical_size_m"]
    b = sample.bounds
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), facecolor=SURFACE)
    ax_v, ax_r = axes
    for ax in axes:
        ax.set_facecolor(SURFACE)

    def shift(gdf):
        return gdf.translate(-b.xmin, -b.ymin)

    if not sample.green.empty:
        shift(sample.green).plot(ax=ax_v, color=LAYER_COLORS["green_spaces"], linewidth=0)
    if not sample.buildings.empty:
        shift(sample.buildings).plot(ax=ax_v, color=LAYER_COLORS["buildings"], edgecolor=SURFACE, linewidth=0.3)
    if not sample.road_polygons.empty:
        shift(sample.road_polygons).plot(ax=ax_v, color=LAYER_COLORS["roads"], alpha=0.85, linewidth=0)
    if not sample.road_centerlines.empty:
        shift(sample.road_centerlines).plot(ax=ax_v, color=SURFACE, linewidth=0.5)
    ax_v.set_xlim(0, size)
    ax_v.set_ylim(0, size)
    ax_v.set_aspect("equal")
    _decorate(ax_v, size, "Vector (projected, clipped) + centerlines")

    ax_r.imshow(composite_rgb(sample.layout), extent=(0, size, 0, size), interpolation="nearest")
    _decorate(ax_r, size, f"Raster {meta['resolution_px']}×{meta['resolution_px']} (what the model sees)")
    handles = [Patch(facecolor=LAYER_COLORS[n], label=LAYER_LABELS[n]) for n in ("roads", "buildings", "green_spaces")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.05), labelcolor=INK)
    _footer(fig, meta)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out
