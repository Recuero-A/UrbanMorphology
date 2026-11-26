"""Milestone 16 blinded style-sorting kit (docs/M16_PROTOCOL.md §5): maps for raters, answer key kept apart.

Maps: the 16 real `sort` reference windows and generated networks from predefined seeds (301, 302) on the flat and
8 % incline confirmatory sites: for every style, M* and NA-fixed with proposals + topology, plus NA-fixed rerank and
style-off decoys. Every map is drawn identically (black streets on white, 512 m, no terrain, no title, no scale cue
other than a fixed 100 m bar), coded K01... in a seeded random order. Rater options: grid / organic / radial /
cul-de-sac / mixed / uncertain. The answer key (source, style requested or documented, run id) is written to
outputs/m16/sorting_kit/key.json and must not be shown to raters.

    python scripts/m16_sorting_kit.py   ->  outputs/m16/sorting_kit/{maps/K*.png, items.json, key.json}
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from m16_figures import graph_of
from m16_plans import CONF_SEEDS, STYLES, proposals
from urbanmorph.visualization.m16_render import draw

OUT = ROOT / "outputs" / "m16" / "sorting_kit"
FZ = json.loads((ROOT / "configs" / "m16_frozen.json").read_text())["alpha"]


def items():
    refs = pickle.loads((ROOT / "data" / "processed" / "m16" / "style_refs" / "graphs.pkl").read_bytes())
    out = [{"source": "real", "id": i, "style": s, "graph": g} for i, s, sub, g in refs if sub == "sort"]
    base = {"sites": "syn_conf", "backbone": "TN_s0", "start": "centre", "seeds": CONF_SEEDS}
    for st in STYLES:
        for m in ("M*", "NA-fixed"):
            spec = {**base, "method": m, "style": st, "proposals": proposals("style_topo", FZ[m][st]["style_topo"]["prop_alpha"])}
            for site in ("conf_flat", "conf_incline8"):
                g, row, rid = graph_of(spec, site, 301 if m == "M*" else 302)
                out.append({"source": "generated", "id": f"{rid}|{site}", "style": st, "mode": "proposals+topology",
                            "method": m, "graph": g})
        g, row, rid = graph_of({**base, "method": "NA-fixed", "style": st}, "conf_flat", 301)
        out.append({"source": "generated", "id": f"{rid}|conf_flat", "style": st, "mode": "rerank", "method": "NA-fixed",
                    "graph": g})
    for site in ("conf_flat", "conf_incline8"):
        g, row, rid = graph_of({**base, "method": "M*", "style": "off"}, site, 302)
        out.append({"source": "generated", "id": f"{rid}|{site}", "style": "none (style off)", "mode": "off",
                    "method": "M*", "graph": g})
    return out


def main() -> int:
    its = [x for x in items() if x["graph"] is not None]
    rng = np.random.default_rng(2026_11_21)
    order = rng.permutation(len(its))
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    key, pub = [], []
    for k, i in enumerate(order, 1):
        x = its[i]
        code = f"K{k:02d}"
        fig, ax = plt.subplots(figsize=(4, 4))
        draw(ax, None, x["graph"], scalebar=True)
        ax.set_facecolor("white")
        fig.savefig(OUT / "maps" / f"{code}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        key.append({"code": code, **{k2: v for k2, v in x.items() if k2 != "graph"}})
        pub.append({"code": code, "image": f"maps/{code}.png"})
    (OUT / "items.json").write_text(json.dumps({"options": ["grid", "organic", "radial", "cul-de-sac", "mixed", "uncertain"],
                                                "items": pub}, indent=1))
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    print(len(pub), "maps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
