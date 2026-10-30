"""Exploratory comparison figures for the Milestone 2 dataset.

Design choices:
* Every tile is shown as a dot (n = 15 per city is too small for box plots
  alone); a short black bar marks each city's median.
* Cities are grouped by country, but each city keeps its own column: the
  unit of comparison is the city, because tiles from one city are spatially
  clustered, not independent draws from a "national" population.
* Colour = country (validated categorical slots 1-3); countries and cities are
  also labelled in text, so colour is never the only cue.
* Hollow markers = tiles flagged ``buildings_incomplete_suspected`` so readers
  can see whether patterns are driven by mapping gaps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # backend is chosen by the calling script, not here
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from urbanmorph.sampling.study_area import load_config
from urbanmorph.visualization.layout_plot import INK, INK_MUTED, SURFACE, composite_rgb

# Validated categorical slots 1-3 (blue, orange, aqua), assigned to countries in
# config order - fixed order, never cycled.
CATEGORICAL_SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")


def label(name: str) -> str:
    """'new_york' -> 'New York', 'united_states' -> 'United States'."""
    return name.replace("_", " ").title()


def _orders_from_config():
    order: dict[str, list[str]] = {}
    for c in load_config()["cities"]:
        order.setdefault(c["country"], []).append(c["city"])
    return tuple(order), {k: tuple(v) for k, v in order.items()}


COUNTRY_ORDER, CITY_ORDER = _orders_from_config()
COUNTRY_COLORS = dict(zip(COUNTRY_ORDER, CATEGORICAL_SLOTS))
COUNTRY_LABELS = {c: label(c) for c in COUNTRY_ORDER}

METRICS = [
    ("buildings_coverage", "Building coverage (fraction of pixels)", False),
    ("roads_coverage", "Road coverage (buffered carriageway)", False),
    ("green_spaces_coverage", "Green-space coverage (mapped only)", False),
    ("intersection_density_per_km2", "Intersections per km²", False),
    ("orientation_order_phi", "Street-grid order φ (1 = grid)", False),
    ("building_area_median_m2", "Median footprint area (m², log)", True),
]


def city_positions() -> dict[tuple[str, str], float]:
    pos, x = {}, 0.0
    for c in COUNTRY_ORDER:
        for city in CITY_ORDER[c]:
            pos[(c, city)] = x
            x += 1
        x += 0.6  # gap between countries
    return pos


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c8c8c8")
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)


def _city_axis(ax, pos):
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([label(city) for (_, city) in pos], rotation=45, ha="right", fontsize=8, color=INK)
    for c in COUNTRY_ORDER:
        xs = [x for (cc, _), x in pos.items() if cc == c]
        ax.text(np.mean(xs), 1.0, COUNTRY_LABELS[c], transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8, color=INK_MUTED)


def metric_strips(df: pd.DataFrame, out: Path) -> Path:
    pos = city_positions()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=SURFACE)
    rng = np.random.default_rng(0)  # deterministic jitter
    for ax, (col, label, logy) in zip(axes.ravel(), METRICS):
        _style(ax)
        for (c, city), x in pos.items():
            sub = df[(df.country == c) & (df.city == city)]
            vals = sub[col].to_numpy(dtype=float)
            ok = ~np.isnan(vals)
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            incomplete = sub["flags_warning"].fillna("").str.contains("buildings_incomplete").to_numpy()
            color = COUNTRY_COLORS[c]
            ax.scatter((x + jitter)[ok & ~incomplete], vals[ok & ~incomplete], s=22, color=color,
                       edgecolor=SURFACE, linewidth=0.6, zorder=3)
            ax.scatter((x + jitter)[ok & incomplete], vals[ok & incomplete], s=22, facecolor="none",
                       edgecolor=color, linewidth=1.1, zorder=3)
            if ok.any():
                ax.plot([x - 0.3, x + 0.3], [np.median(vals[ok])] * 2, color=INK, linewidth=2, zorder=4,
                        solid_capstyle="butt")
        if logy:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=10, color=INK, loc="left", pad=16)
        _city_axis(ax, pos)
    handles = [Line2D([], [], marker="o", linestyle="", color=COUNTRY_COLORS[c], label=COUNTRY_LABELS[c])
               for c in COUNTRY_ORDER]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK_MUTED,
                       label="OSM buildings likely incomplete"),
                Line2D([], [], color=INK, linewidth=2, label="city median")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Urban-morphology metrics per 512 m tile — EXPLORATORY pilot (≈15 tiles per city)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "Tiles are spatially clustered within cities; differences are descriptive, not evidence "
             "about countries. Data: © OpenStreetMap contributors (ODbL), GHS-BUILT-S R2023A (CC BY 4.0).",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out


def orientation_roses(df: pd.DataFrame, hist_by_sample: dict[str, list[float]], out: Path) -> Path:
    """Length-weighted orientation histogram per city (pooled over its tiles), mirrored to 360 deg."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 11.5), subplot_kw={"projection": "polar"}, facecolor=SURFACE)
    for row, c in enumerate(COUNTRY_ORDER):
        for col, city in enumerate(CITY_ORDER[c]):
            ax = axes[row, col]
            sub = df[(df.country == c) & (df.city == city)]
            hs = []
            for _, r in sub.iterrows():
                h = np.asarray(hist_by_sample.get(r.sample_id, []), dtype=float)
                if h.size and h.sum() > 0:
                    hs.append(h * r["street_length_m"])  # weight tiles by street length
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_facecolor(SURFACE)
            ax.set_yticklabels([])
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(["N", "", "E", "", "S", "", "W", ""], fontsize=8, color=INK_MUTED)
            ax.grid(color="#e4e4e4")
            if not hs:
                continue
            p = np.sum(hs, axis=0)
            p = p / p.sum()
            n = len(p)
            centers = np.radians(np.arange(n) * 180.0 / n)
            theta = np.concatenate([centers, centers + np.pi])
            ax.bar(theta, np.concatenate([p, p]), width=np.radians(180.0 / n) * 0.92,
                   color=COUNTRY_COLORS[c], edgecolor=SURFACE, linewidth=0.5)
            nz = p[p > 0]
            H = -(nz * np.log(nz)).sum()
            phi = 1 - ((H - np.log(2)) / (np.log(n) - np.log(2))) ** 2
            ax.set_title(f"{label(city)} ({COUNTRY_LABELS[c]})\npooled φ = {phi:.2f}, n = {len(hs)}",
                         fontsize=9.5, color=INK, pad=12)
    fig.suptitle("Street orientation, length-weighted, pooled per city — EXPLORATORY", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.01, 0.005, "Bearings from UTM grid north, folded to 0–180° (streets are undirected) and mirrored "
             "for display. Pooling mixes differently oriented neighbourhoods.", fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out


def representative_tiles(df: pd.DataFrame, layouts: dict[str, np.ndarray], out: Path) -> Path:
    """Per city: tiles with the lowest, median and highest building coverage (a fixed rule, no cherry-picking)."""
    fig, axes = plt.subplots(9, 3, figsize=(7.2, 22), facecolor=SURFACE)
    r = 0
    for c in COUNTRY_ORDER:
        for city in CITY_ORDER[c]:
            sub = df[(df.country == c) & (df.city == city)].sort_values("buildings_coverage")
            picks = [sub.iloc[0], sub.iloc[len(sub) // 2], sub.iloc[-1]] if len(sub) else []
            for k, name in enumerate(("lowest", "median", "highest")):
                ax = axes[r, k]
                ax.axis("off")
                if k < len(picks):
                    p = picks[k]
                    ax.imshow(composite_rgb(layouts[p.sample_id]), interpolation="nearest")
                    ax.set_title(f"{p.sample_id}\n{name} bldg cov {p.buildings_coverage:.2f} · {p.era}", fontsize=6.5,
                                 color=INK)
            axes[r, 0].text(-0.08, 0.5, f"{label(city)}\n{COUNTRY_LABELS[c]}", transform=axes[r, 0].transAxes,
                            rotation=90, ha="right", va="center", fontsize=9, color=INK)
            r += 1
    fig.suptitle("Representative tiles: lowest / median / highest building coverage per city",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.text(0.02, 0.003, "Roads blue · buildings amber · green spaces green · 512 m × 512 m, 4 m/px. "
             "© OpenStreetMap contributors (ODbL).", fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=(0.03, 0.01, 1, 0.985))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out


def completeness_scatter(df: pd.DataFrame, out: Path) -> Path:
    """OSM building coverage vs. satellite-derived GHSL built-up: a mapping-completeness check."""
    fig, ax = plt.subplots(figsize=(6.6, 6.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(color="#ececec", linewidth=0.8)
    for c in COUNTRY_ORDER:
        sub = df[df.country == c]
        ax.scatter(sub.ghsl_built_2020, sub.building_coverage_vector, s=24, color=COUNTRY_COLORS[c],
                   edgecolor=SURFACE, linewidth=0.6, label=COUNTRY_LABELS[c], zorder=3)
    lim = max(0.8, float(np.nanmax(df[["ghsl_built_2020", "building_coverage_vector"]].to_numpy())) + 0.05)
    ax.plot([0, lim], [0, lim], color=INK_MUTED, linewidth=1, linestyle="--")
    ax.plot([0, lim], [0, 0.5 * lim], color=INK_MUTED, linewidth=1, linestyle=":")
    ax.text(lim * 0.97, lim * 0.93, "OSM = GHSL", ha="right", fontsize=8, color=INK_MUTED)
    ax.text(lim * 0.97, lim * 0.44, "OSM = ½ GHSL (flag threshold)", ha="right", fontsize=8, color=INK_MUTED)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("GHSL 2020 built-up fraction (satellite)", fontsize=9, color=INK)
    ax.set_ylabel("OSM building footprint coverage (vector)", fontsize=9, color=INK)
    ax.set_title("Is OSM building mapping complete? (per tile)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-city medians/IQR and design-weighted means of the main metrics."""
    rows = []
    for c in COUNTRY_ORDER:
        for city in CITY_ORDER[c]:
            sub = df[(df.country == c) & (df.city == city)]
            row = {"country": c, "city": city, "n_tiles": len(sub)}
            for col, _, _ in METRICS:
                v = sub[col].astype(float)
                w = sub["design_weight"].astype(float)
                ok = v.notna()
                row[f"{col}_median"] = v.median()
                row[f"{col}_iqr"] = v.quantile(0.75) - v.quantile(0.25)
                row[f"{col}_weighted_mean"] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
            row["share_buildings_incomplete_flag"] = sub["flags_warning"].fillna("").str.contains("buildings_incomplete").mean()
            row["share_green_empty"] = sub["flags_info"].fillna("").str.contains("green_unmapped").mean()
            rows.append(row)
    return pd.DataFrame(rows)
