"""Milestone 12, EXPLORATORY: sensitivity of conditional generation to the sampling temperature T.

    python scripts/m12_temperature_sweep.py   # -> outputs/m12/exploratory/temperature/{sweep.json,sweep.png,examples.png}

The pre-registered value stays T = 0.5 (protocol §2); nothing here changes a reported result or selects a model.
Development tiles only (regime A: the 52 val + val_t tiles with a DEM), 4 samples per tile, training seed 0, the
same sampling seed as regime A. Scores as regime A (S_T, conditional S10), plus street density, premature
termination (< 50% of the real tile's street length), bridges and unbridged water per tile.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import m12_evaluate as ev  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.levels import BRIDGE
from urbanmorph.graphs.terrain_metrics import S_T_KEYS

OUT = PROJECT_ROOT / "outputs" / "m12" / "exploratory" / "temperature"
TEMPS = (0.3, 0.5, 0.7, 0.9, 1.0)
SPECS = ("N0", "TN", "TNlp", "TN+HR", "TNlp+HR")


def summarise(rows, ref):
    term = np.array([max(0.0, 1 - r["street_km_per_km2"] / max(q["street_km_per_km2"], 1e-9)) for r, q in zip(rows, ref)])
    return {"S_T": ev.score(rows, ref, S_T_KEYS), "S10_cond": ev.score(rows, ref, ev.KEYS),
            "street_km_per_km2": float(np.nanmean([r["street_km_per_km2"] for r in rows])),
            "premature_share": float((term > 0.5).mean()),
            "bridges_per_tile": float(np.nanmean([r["bridges"] for r in rows])),
            "unbridged_water_m_per_tile": float(np.nanmean([r["unbridged_water_m"] for r in rows])),
            "dead_end_ratio": float(np.nanmean([r["dead_end_ratio"] for r in rows]))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    real = ev.real_graphs()
    tiles = ev.dev_tiles_with_dem(real)
    ref = ev.per_tile([g for _, _, g, _ in tiles], [d for *_, d in tiles])
    pool = ev.density_pool(real)
    ds = [d for *_, d in tiles for _ in range(ev.N_SAMP)]
    f = OUT / "sweep.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for spec in SPECS:
        for T in TEMPS:
            key = f"{spec}@{T}"
            if key in res:
                continue
            ev.TEMP = T
            t0 = time.time()
            gs, _ = ev.generate_for(spec, 0, ds, device, 1000, pool)
            rows = ev.per_tile(gs, ds)
            avg = [{k: float(np.nanmean([rows[i * ev.N_SAMP + j].get(k, np.nan) for j in range(ev.N_SAMP)]))
                    for k in rows[0]} for i in range(len(tiles))]
            res[key] = summarise(avg, ref)
            f.write_text(json.dumps(res, indent=1))
            print(key, {k: round(v, 3) for k, v in res[key].items()}, f"{time.time() - t0:.0f}s", flush=True)
    # line panels: one panel per measure, one line per model
    meas = [("S_T", "S_T (lower is better)"), ("S10_cond", "S10, conditional (lower is better)"),
            ("street_km_per_km2", "street km per km² (real ≈ 13.6)"), ("premature_share", "premature share (lower is better)"),
            ("bridges_per_tile", "bridges per tile (real ≈ 1.6)")]
    fig, ax = plt.subplots(1, len(meas), figsize=(4.0 * len(meas), 3.6))
    for a, (m, title) in zip(ax, meas):
        for spec in SPECS:
            a.plot(TEMPS, [res[f"{spec}@{T}"][m] for T in TEMPS], marker="o", ms=4, lw=1.6, label=spec)
        a.axvline(0.5, color="#999999", ls="--", lw=1)
        a.set_title(title, fontsize=9, loc="left")
        a.set_xlabel("temperature T")
        a.grid(color="#e6e6e6")
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, "Exploratory. Regime A development tiles (n = 52), training seed 0, 4 samples per tile. "
             "Dashed: the pre-registered T = 0.5.", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "sweep.png", dpi=140)
    # one held-out San Diego tile at every temperature, TN-lp + HR
    sid = "USA_SAN_e958_n7125"
    d = next(d for s, _, _, d in tiles if s == sid)
    fig, ax = plt.subplots(1, len(TEMPS), figsize=(3.2 * len(TEMPS), 3.4))
    z = d["elev"][64:192, 64:192]
    gy, gx = np.gradient(z, 4.0)
    shade = np.clip(0.6 + 0.5 * (gx - gy) / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
    for a, T in zip(ax, TEMPS):
        ev.TEMP = T
        g = ev.generate_for("TNlp+HR", 0, [d], device, 1000, pool)[0][0]
        a.imshow(shade, cmap="gray", extent=(0, 512, 0, 512), vmin=0, vmax=1.2)
        for s, b in zip(g.segments(), g.structs()):
            a.plot([s[0], s[2]], [s[1], s[3]], color="#c0392b" if b == BRIDGE else "#1f1f1f", lw=2.2 if b == BRIDGE else 1.0)
        a.set_xticks([]), a.set_yticks([])
        a.set_xlim(0, 512), a.set_ylim(0, 512)
        a.set_title(f"T = {T}", fontsize=10)
    fig.suptitle(f"TN-lp + HR on held-out tile {sid} (exploratory)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=120)
    print("written", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
