"""Milestone 18, D7 (c): the GLO-30 backbone TNg on the 95 GEN2 tiles, against TN, with the M14 estimators.

Runs: configs/m18_plans/gen2_tng.json (TNg s0-s2; M*, M* + repair, NA-SMC; seeds 1-3) and the M14 confirmatory TN runs
(m14_report.get). Per backbone family and method: same-land OD unreachability, islands per tile, street density, S_T
(terrain gap) and S10 (network-statistics gap) against the real GEN2 networks (m12_evaluate.score, W1/IQR, as M12-M14),
and the paired block contrast NA-SMC - M* on OD unreachability (m14_analyze.block_contrast, 3 backbones).

    python scripts/m18_tng_gen2.py  ->  outputs/m18/glo30/gen2_tng.{md,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from urbanmorph.experiments.jsonio import atomic_json

OUT = ROOT / "outputs" / "m18" / "glo30"
LABELS = ("M*", "M*+repair", "NA-SMC")


def main() -> int:
    import m14_analyze as ma
    from m12_evaluate import score
    from m14_report import get
    from urbanmorph.graphs.terrain_metrics import S_T_KEYS
    real = ma.real("gen2")
    plain = lambda r: {k[4:] if k.startswith("m12_") else k: v for k, v in r.items()}
    ref = [plain(r) for r in real]
    s10_keys = [k for k in ref[0] if k.startswith("s10_")]
    res, runs = {}, {}
    for fam in ("TN", "TNg"):
        for lab in LABELS:
            rs = [get("gen2", f"{fam}_s{b}", lab) for b in range(3)]
            rs = [r for r in rs if r is not None]
            runs[(fam, lab)] = rs
            if not rs:
                continue
            vals = {"od_unreach_land": [], "islands": [], "street_km_per_km2": [], "S_T": [], "S10": []}
            for r in rs:
                for sd in r["seeds"].values():
                    rows = [plain(x) for x in sd["rows"]]
                    vals["od_unreach_land"].append(np.nanmean([x["od_unreach_land"] for x in rows]))
                    vals["islands"].append(np.nanmean([x["islands"] for x in rows]))
                    vals["street_km_per_km2"].append(np.nanmean([x["street_km_per_km2"] for x in rows]))
                    vals["S_T"].append(score(rows, ref, S_T_KEYS))
                    vals["S10"].append(score(rows, ref, s10_keys))
            res[f"{fam}|{lab}"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)} for k, v in vals.items()}
    contrasts = {}
    for fam in ("TN", "TNg"):
        a, b = runs.get((fam, "M*"), []), runs.get((fam, "NA-SMC"), [])
        if len(a) == len(b) == 3:
            contrasts[fam] = ma.block_contrast(list(zip(a, b)), "od_unreach_land")
    real_v = {"od_unreach_land": float(np.nanmean([r["od_unreach_land"] for r in real])),
              "street_km_per_km2": float(np.nanmean([r["street_km_per_km2"] for r in real]))}
    atomic_json(OUT / "gen2_tng.json", {"summary": res, "NA-SMC_minus_Mstar": contrasts, "real": real_v})
    md = ["# M18: the GLO-30 backbone TNg vs TN on GEN2 (95 tiles, 3 backbones × 3 seeds)", "",
          f"Real GEN2: OD unreachability {real_v['od_unreach_land']:.3f}, {real_v['street_km_per_km2']:.1f} km/km². "
          "S_T and S10: W1/IQR gap to the real networks (lower = closer). Script `scripts/m18_tng_gen2.py`.", "",
          "| Backbone | Method | OD unreach. | islands / tile | km/km² | S_T | S10 |", "|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        fam, lab = k.split("|")
        md.append(f"| {fam} | {lab} | {v['od_unreach_land']['mean']:.3f} | {v['islands']['mean']:.2f} | "
                  f"{v['street_km_per_km2']['mean']:.1f} | {v['S_T']['mean']:.3f} | {v['S10']['mean']:.3f} |")
    md += ["", "Paired block contrast NA-SMC − M\\* on OD unreachability (m14_analyze.block_contrast):", ""]
    for fam, c in contrasts.items():
        md.append(f"* {fam}: Δ {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}], sign-flip p {c['p_signflip']:.4g}, "
                  f"per backbone {', '.join(f'{x:+.4f}' for x in c['per_seed'])}")
    (OUT / "gen2_tng.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
