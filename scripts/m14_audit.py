"""Milestone 14 phase 1: is connectivity / coverage actually a problem of the M13 generators? (stored runs only)

    python scripts/m14_audit.py     # -> outputs/m14/audit/{audit.json, audit.md}

Uses the stored M13 runs (TN backbone, 3 training seeds, inference seeds 1-4) of A (standard) and M* (frozen
selection) on DEV-B and GEN, and the real graphs of the same tiles. Computes the M14 closed-window metrics
(graphs/m14_metrics) and density adaptation (does generated density follow the real density of the tile?).
Development data only; no generation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr

import m13_experiments as ex  # noqa: E402
from m13_scoreboard import SEL, find, graph  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.m14_metrics import tile_metrics_m14
from urbanmorph.graphs.tc_potentials import TerrainField
from urbanmorph.graphs.terrain_generate import terrain_of

OUT = PROJECT_ROOT / "outputs" / "m14" / "audit"
KEYS = ["closed_giant", "islands", "island_share", "od_unreach", "od_unreach_land", "od_unreach_water", "od_detour",
        "coverage_60", "street_m_per_conn"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res, md = {}, ["# M14 phase-1 audit: connectivity, coverage, density adaptation (stored M13 runs)\n"]
    for tiles in ("devB", "gen"):
        T = ex.tile_set(tiles)
        tfs = [TerrainField(terrain_of(t["d"])) for t in T]
        rng = np.random.default_rng(0)
        real = [tile_metrics_m14(t["graph"], tf, rng) for t, tf in zip(T, tfs)]
        dens_real = np.array([r["m12_street_km_per_km2"] for r in ex.real_rows(T, tiles)])
        out = {"real": {k: float(np.nanmean([r[k] for r in real])) for k in KEYS}}
        for lab, pre, ov in (("A", "standard", {}), ("M*", SEL["preset"], SEL["overrides"])):
            per_seed, dens = [], []
            for b in ("TN_s0", "TN_s1", "TN_s2"):
                run = find(tiles, b, pre, ov)
                rows = []
                for s, v in run["seeds"].items():
                    rng = np.random.default_rng(int(s))
                    rows.append([tile_metrics_m14(graph(G), tf, rng) for G, tf in zip(v["graphs"], tfs)])
                    dens.append([r["m12_street_km_per_km2"] for r in v["rows"]])
                per_seed.append({k: float(np.nanmean([r[k] for rr in rows for r in rr])) for k in KEYS})
            d = np.mean(dens, 0)
            rho = spearmanr(d, dens_real).statistic
            slope = float(np.polyfit(dens_real, d, 1)[0])
            out[lab] = {k: [p[k] for p in per_seed] for k in KEYS} | {"density_spearman": float(rho),
                                                                   "density_slope": slope}
        res[tiles] = out
        md += [f"\n## {tiles}\n", "| metric | real | A (s0, s1, s2) | M* (s0, s1, s2) |", "|---|---|---|---|"]
        for k in KEYS:
            f = lambda v: ", ".join(f"{x:.3g}" for x in v)
            md.append(f"| {k} | {out['real'][k]:.3g} | {f(out['A'][k])} | {f(out['M*'][k])} |")
        md.append(f"| density Spearman (gen vs real, per tile) | 1 | {out['A']['density_spearman']:.2f} | "
                  f"{out['M*']['density_spearman']:.2f} |")
        md.append(f"| density slope (gen on real) | 1 | {out['A']['density_slope']:.2f} | {out['M*']['density_slope']:.2f} |")
    (OUT / "audit.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "audit.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
