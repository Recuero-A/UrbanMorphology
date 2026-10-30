"""Milestone 2, step A: build the sampling plan (no OSM requests).

For every city: candidate 512 m grid cells inside the boundary -> GHSL
built-up fractions -> eligibility -> strata -> stratified, spatially separated
random selection of primaries + reserves. Writes:

    data/processed/m2/sampling/plan.csv            selected tiles (primary + reserve)
    data/processed/m2/sampling/strata_summary.csv  stratum sizes / allocations per city
    outputs/m2/sampling/<CODE>.png                 maps for human review

Usage:
    python scripts/m2_plan_sampling.py --calibrate   # first time: freeze density thresholds in the config
    python scripts/m2_plan_sampling.py               # re-run with frozen thresholds (deterministic)
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer

from urbanmorph.config import M2_DIR, M2_OUTPUTS
from urbanmorph.sampling import strata as st
from urbanmorph.sampling.grid import candidate_tiles, main_component, sample_id
from urbanmorph.sampling.select import city_rng, select_tiles
from urbanmorph.sampling.study_area import DEFAULT_CONFIG, cities, fetch_boundary, load_config, utm_crs

STRATUM_COLORS = {"low": "#c9d3e6", "medium": "#7f95c4", "high": "#34497f"}


def candidates_for(city, cfg):
    boundary = fetch_boundary(city)
    crs = utm_crs(boundary)
    cands = candidate_tiles(boundary, crs, cfg["tile"]["size_m"])
    cands = st.add_ghsl_columns(cands, city.code, boundary.total_bounds, crs, cfg)
    sa = cfg["study_area"]
    cands["above_threshold"] = cands["ghsl_built_2020"] >= sa["min_builtup_fraction"]
    cands["eligible"] = cands["above_threshold"]
    if sa.get("main_agglomeration_only", False):
        idx = cands.index[cands["above_threshold"]]
        main = main_component(cands.loc[idx, "grid_e"], cands.loc[idx, "grid_n"], sa["agglomeration_max_gap_cells"],
                              sa.get("agglomeration_min_component_fraction", 1.0))
        cands.loc[idx[~main], "eligible"] = False
    return boundary, crs, cands


def freeze_thresholds(path: Path, thresholds) -> None:
    text = path.read_text()
    new = re.sub(r"density_thresholds = \[[^\]]*\]", f"density_thresholds = [{thresholds[0]}, {thresholds[1]}]", text, count=1)
    path.write_text(new)


def plot_city(city, boundary, crs, cands, plan, out):
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#fcfcfb")
    boundary.to_crs(crs).boundary.plot(ax=ax, color="#9a9a9a", linewidth=0.6)
    el = cands[cands["eligible"]]
    det = cands[cands["above_threshold"] & ~cands["eligible"]]
    if len(det):
        det.plot(ax=ax, color="#e3e3e3", linewidth=0, label=f"detached, excluded ({len(det)})")
    for d, color in STRATUM_COLORS.items():
        sub = el[el["density"] == d]
        if len(sub):
            sub.plot(ax=ax, color=color, linewidth=0, label=f"eligible, {d} density ({len(sub)})")
    prim = plan[plan["role"] == "primary"]
    res = plan[plan["role"] == "reserve"]
    ax.scatter((prim.xmin + prim.xmax) / 2, (prim.ymin + prim.ymax) / 2, s=46, marker="s",
               facecolor="#cf8530", edgecolor="#2b2b2b", linewidth=0.8, zorder=5, label=f"selected ({len(prim)})")
    ax.scatter((res.xmin + res.xmax) / 2, (res.ymin + res.ymax) / 2, s=40, marker="o",
               facecolor="none", edgecolor="#2b2b2b", linewidth=1.0, zorder=5, label=f"reserve ({len(res)})")
    for _, r in prim.iterrows():
        ax.annotate(r["era"].replace("_", "–"), ((r.xmin + r.xmax) / 2, r.ymax), fontsize=6,
                    color="#2b2b2b", ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
    xs = pd.concat([el.xmin, el.xmax]); ys = pd.concat([el.ymin, el.ymax])
    pad = 2000
    ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{city.city.replace('_', ' ').title()} ({city.country.replace('_', ' ').title()}) — sampling plan, {crs}",
                 loc="left", fontsize=11)
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9)
    fig.text(0.01, 0.01, "Eligible = GHSL 2020 built-up ≥ threshold, tile fully inside boundary. "
             "Labels: era of urbanisation.\nBoundary © OpenStreetMap contributors (ODbL); "
             "built-up: GHS-BUILT-S R2023A (JRC, CC BY 4.0)", fontsize=6.5, color="#6b6b6b")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="compute and freeze density thresholds")
    args = ap.parse_args()

    cfg = load_config()
    data = {c.code: (c, *candidates_for(c, cfg)) for c in cities(cfg)}

    if args.calibrate:
        th = st.calibrate_density_thresholds({k: v[3][v[3]["eligible"]] for k, v in data.items()}, "ghsl_built_2020")
        freeze_thresholds(DEFAULT_CONFIG, th)
        print(f"Froze density thresholds {th} in {DEFAULT_CONFIG}")
        cfg = load_config()
    th = cfg["strata"]["density_thresholds"]
    if th[1] <= th[0]:
        print("Density thresholds not calibrated; run with --calibrate first")
        return 1

    plans, summary = [], []
    (M2_DIR / "sampling").mkdir(parents=True, exist_ok=True)
    (M2_OUTPUTS / "sampling").mkdir(parents=True, exist_ok=True)
    for code, (city, boundary, crs, cands) in data.items():
        cands["density"] = st.density_class(cands["ghsl_built_2020"], th)
        cands["era"] = st.era_class(cands, cfg["study_area"]["ghsl_epochs"], cfg["strata"]["era_builtup_fraction"])
        cands["stratum"] = cands["density"] + "|" + cands["era"]
        el = cands[cands["eligible"]].reset_index(drop=True)
        sel = cfg["selection"]
        plan = select_tiles(el, sel["per_city"], sel["reserves_per_city"], sel["min_center_distance_m"],
                            city_rng(cfg["seed"], code))
        to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(((plan.xmin + plan.xmax) / 2).to_numpy(), ((plan.ymin + plan.ymax) / 2).to_numpy())
        plan = plan.drop(columns="geometry").assign(
            sample_id=[sample_id(code, e, n) for e, n in zip(plan.grid_e, plan.grid_n)],
            country=city.country, city=city.city, city_code=code, crs=crs, center_lat=lat, center_lon=lon,
        )
        plans.append(plan)
        for h, size in sorted(plan.attrs["stratum_sizes"].items()):
            summary.append({"city_code": code, "stratum": h, "n_candidates": size,
                            "allocated": plan.attrs["allocation"].get(h, 0),
                            "selected": int(((plan.stratum == h) & (plan.role == "primary")).sum()),
                            "shortfall": plan.attrs["shortfall"].get(h, 0)})
        plot_city(city, boundary, crs, cands, plan, M2_OUTPUTS / "sampling" / f"{code}.png")
        print(f"{code}: {len(cands)} cells in boundary, {int(cands.above_threshold.sum())} built-up, {len(el)} eligible, "
              f"{(plan.role == 'primary').sum()} selected, {(plan.role == 'reserve').sum()} reserves")

    cols = ["sample_id", "country", "city", "city_code", "role", "rank", "stratum", "density", "era",
            "p_inclusion", "grid_e", "grid_n", "crs", "xmin", "ymin", "xmax", "ymax", "center_lat", "center_lon",
            "ghsl_built_1975", "ghsl_built_2000", "ghsl_built_2020", "ghsl_nres_share_2020"]
    pd.concat(plans)[cols].to_csv(M2_DIR / "sampling" / "plan.csv", index=False)
    pd.DataFrame(summary).to_csv(M2_DIR / "sampling" / "strata_summary.csv", index=False)
    print(f"Thresholds {th}; plan written to {M2_DIR / 'sampling' / 'plan.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
