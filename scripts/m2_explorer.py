"""Interactive dataset explorer (Matplotlib only, runs locally).

    python scripts/m2_explorer.py                          # interactive window
    python scripts/m2_explorer.py --a ID --b ID --save out.png   # headless comparison image

Controls (interactive):
    * Country / City radio buttons choose the city for panel A.
    * "< prev" / "next >" step through that city's tiles.
    * "Pin to B" copies panel A into panel B, so two tiles can be compared
      side by side (e.g. pin a Lima tile, then browse Seoul).
Each panel shows the three-channel raster and its morphology metrics,
quality flags, stratum and split.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import Button, RadioButtons

from urbanmorph.data.m2_dataset import load_index, load_m2_sample
from urbanmorph.visualization.compare import CITY_ORDER, COUNTRY_LABELS, COUNTRY_ORDER, label
from urbanmorph.visualization.layout_plot import INK, INK_MUTED, LAYER_COLORS, LAYER_LABELS, SURFACE, composite_rgb


def metrics_text(meta: dict) -> str:
    m, s = meta["metrics"], meta["sampling"]
    flags = [f"{f['severity']}: {f['code']}" for f in meta["quality_flags"]] or ["none"]
    return "\n".join([
        f"{label(meta['city'])}, {COUNTRY_LABELS.get(meta['country'], meta['country'])}  ·  split: {meta.get('split')}",
        f"stratum: {s.get('stratum')}  (GHSL built 2020 {s.get('ghsl_built_2020', float('nan')):.2f})",
        f"building coverage   {m['buildings_coverage']:.3f}   (vector {m['building_coverage_vector']:.3f})",
        f"road coverage       {m['roads_coverage']:.3f}",
        f"green coverage      {m['green_spaces_coverage']:.3f}",
        f"buildings           {m['building_count']}  (median {m['building_area_median_m2']:.0f} m²)",
        f"intersections/km²   {m['intersection_density_per_km2']:.0f}",
        f"orientation φ       {m['orientation_order_phi']:.2f}   grid at {m['grid_orientation_deg']:.0f}°",
        "flags: " + "; ".join(flags),
    ])


def draw_panel(ax_img, ax_txt, sample_id: str | None, index):
    ax_img.clear()
    ax_txt.clear()
    ax_img.axis("off")
    ax_txt.axis("off")
    if sample_id is None:
        ax_img.text(0.5, 0.5, "Panel B empty\n(press 'Pin to B')", ha="center", va="center", color=INK_MUTED)
        return
    layout, meta = load_m2_sample(sample_id, index)
    ax_img.imshow(composite_rgb(layout), interpolation="nearest")
    ax_img.set_title(sample_id, fontsize=10, color=INK)
    ax_txt.text(0, 1, metrics_text(meta), va="top", ha="left", family="monospace", fontsize=8, color=INK)


def build_figure():
    fig = plt.figure(figsize=(13, 8.2), facecolor=SURFACE)
    axes = {
        "a_img": fig.add_axes([0.22, 0.30, 0.37, 0.62]),
        "a_txt": fig.add_axes([0.22, 0.02, 0.37, 0.25]),
        "b_img": fig.add_axes([0.62, 0.30, 0.37, 0.62]),
        "b_txt": fig.add_axes([0.62, 0.02, 0.37, 0.25]),
    }
    handles = [Patch(color=LAYER_COLORS[k], label=LAYER_LABELS[k]) for k in ("roads", "buildings", "green_spaces")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.6, 0.995))
    return fig, axes


def save_comparison(a: str, b: str | None, out: Path) -> Path:
    index = load_index()
    fig, ax = build_figure()
    draw_panel(ax["a_img"], ax["a_txt"], a, index)
    draw_panel(ax["b_img"], ax["b_txt"], b, index)
    fig.text(0.01, 0.97, "UrbanMorph dataset explorer", fontsize=12, color=INK)
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out


def interactive() -> None:
    index = load_index()
    df = index[index["in_dataset"]]
    state = {"country": COUNTRY_ORDER[0], "city": CITY_ORDER[COUNTRY_ORDER[0]][0], "i": 0, "b": None}
    fig, ax = build_figure()
    fig.text(0.01, 0.97, "UrbanMorph dataset explorer", fontsize=12, color=INK)

    def tiles():
        return sorted(df[(df.country == state["country"]) & (df.city == state["city"])]["sample_id"])

    def current():
        t = tiles()
        return t[state["i"] % len(t)] if t else None

    def redraw():
        draw_panel(ax["a_img"], ax["a_txt"], current(), index)
        draw_panel(ax["b_img"], ax["b_txt"], state["b"], index)
        n = len(tiles())
        counter.set_text(f"tile {state['i'] % max(n, 1) + 1} / {n}")
        fig.canvas.draw_idle()

    ax_country = fig.add_axes([0.01, 0.72, 0.18, 0.18], facecolor=SURFACE)
    ax_country.set_title("Country", fontsize=9, loc="left", color=INK)
    country_radio = RadioButtons(ax_country, [COUNTRY_LABELS[c] for c in COUNTRY_ORDER])
    ax_city = fig.add_axes([0.01, 0.48, 0.18, 0.18], facecolor=SURFACE)
    ax_city.set_title("City", fontsize=9, loc="left", color=INK)
    widgets = {"city": RadioButtons(ax_city, [label(c) for c in CITY_ORDER[state["country"]]])}

    def on_city(text):
        state["city"] = {label(c): c for c in CITY_ORDER[state["country"]]}[text]
        state["i"] = 0
        redraw()

    def on_country(text):
        state["country"] = {v: k for k, v in COUNTRY_LABELS.items()}[text]
        state["city"], state["i"] = CITY_ORDER[state["country"]][0], 0
        ax_city.clear()
        ax_city.set_title("City", fontsize=9, loc="left", color=INK)
        widgets["city"] = RadioButtons(ax_city, [label(c) for c in CITY_ORDER[state["country"]]])
        widgets["city"].on_clicked(on_city)
        redraw()

    country_radio.on_clicked(on_country)
    widgets["city"].on_clicked(on_city)
    prev_b = Button(fig.add_axes([0.01, 0.38, 0.085, 0.05]), "< prev")
    next_b = Button(fig.add_axes([0.105, 0.38, 0.085, 0.05]), "next >")
    pin_b = Button(fig.add_axes([0.01, 0.30, 0.18, 0.05]), "Pin to B")
    counter = fig.text(0.01, 0.44, "", fontsize=9, color=INK_MUTED)

    def step(d):
        state["i"] += d
        redraw()

    prev_b.on_clicked(lambda _: step(-1))
    next_b.on_clicked(lambda _: step(+1))

    def pin(_):
        state["b"] = current()
        redraw()

    pin_b.on_clicked(pin)
    fig._widgets = (country_radio, widgets, prev_b, next_b, pin_b)  # keep references alive
    redraw()
    plt.show()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", help="sample ID for panel A (headless mode)")
    ap.add_argument("--b", help="sample ID for panel B (headless mode)")
    ap.add_argument("--save", type=Path, help="write a comparison image instead of opening a window")
    args = ap.parse_args()
    if args.save:
        if not args.a:
            ap.error("--save needs --a")
        print(f"Wrote {save_comparison(args.a, args.b, args.save)}")
        return 0
    interactive()
    return 0


if __name__ == "__main__":
    sys.exit(main())
