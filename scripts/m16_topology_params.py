"""Milestone 16: topology-rule parameters from the `calib` real style references (docs/M16_PROTOCOL.md §4).

Fixed before the development sweep, from real data only:
* grid lattice spacing and radial ring spacing = sqrt(median interior block area) of the grid references, rounded to 8 m;
* cul-de-sac branch step = median terminal length of the cul-de-sac references / term_depth (3), clipped to [25, 60] m.

    python scripts/m16_topology_params.py     ->  configs/m16_topology.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from urbanmorph.graphs.network_outcomes import terminal
from urbanmorph.graphs.style_metrics import block_areas

REFS = ROOT / "data" / "processed" / "m16" / "style_refs" / "graphs.pkl"
TERM_DEPTH = 3


def main() -> int:
    refs = pickle.loads(REFS.read_bytes())
    calib = [(i, s, g) for i, s, sub, g in refs if sub == "calib"]
    areas = np.concatenate([block_areas(g) for _, s, g in calib if s == "grid"])
    spacing = float(8 * round(np.sqrt(np.median(areas)) / 8))
    term = [terminal(g)["term_len_med"] for _, s, g in calib if s == "culdesac"]
    term = [t for t in term if np.isfinite(t)]
    branch = float(np.clip(np.median(term) / TERM_DEPTH, 25.0, 60.0))
    out = {"source": "configs/m16_style_refs (calib subset)", "grid_blocks": int(len(areas)),
           "grid_median_block_m2": float(np.median(areas)), "grid_spacing_m": spacing, "ring_spacing_m": spacing,
           "culdesac_term_len_med_m": float(np.median(term)), "branch_len_m": round(branch, 1), "term_depth": TERM_DEPTH,
           "n_spokes": 8, "topo_continue": 2.0}
    (ROOT / "configs" / "m16_topology.json").write_text(json.dumps(out, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
