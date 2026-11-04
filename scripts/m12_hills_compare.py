"""Milestone 12 showcase: the same streets model on gentle and on high hills (3 x 3 landscapes, TN_s0 + HR).

    python scripts/m12_hills_compare.py   # -> outputs/m12/figures/hills_comparison.png (+ printed statistics)

Plan of each landscape: hillshade, 5 m contours, water, streets coloured by grade. Statistics per landscape,
length-weighted over ground streets: grade quantiles and how streets sit against the slope (|cos| of the angle between
street and terrain gradient on land steeper than 5%: 1 = straight up the slope, 0 = along the contour; a street
network that ignored the terrain would average about 0.64, the mean |cos| of a uniformly random direction).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.terrain.synthesis import generate, scenario

CITIES = PROJECT_ROOT / "outputs" / "m12" / "cities"
FIG = PROJECT_ROOT / "outputs" / "m12" / "figures"
RUNS = [("rolling_hills", "Rolling hills (original)"), ("high_hills", "High hills (4x relief)")]
L, RES = 1536.0, 4.0


def street_stats(S, T):
    """Length-weighted grade and slope alignment of ground streets, from the terrain sampled every 2 m."""
    g = S[S[:, 5] == 0]
    grades, cosines, w_g, w_c = [], [], [], []
    for x0, y0, x1, y1 in g[:, :4]:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / 2.0) + 1)
        t = np.linspace(0, 1, n)
        x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        z = T.sample(x, y)
        d = np.hypot(np.diff(x), np.diff(y))
        ok = np.isfinite(z[:-1]) & np.isfinite(z[1:]) & (d > 0)
        if not ok.any():
            continue
        grades.append(np.abs(np.diff(z))[ok] / d[ok])
        w_g.append(d[ok])
        e = 4.0
        gx = (T.sample(x + e, y) - T.sample(x - e, y)) / (2 * e)
        gy = (T.sample(x, y + e) - T.sample(x, y - e)) / (2 * e)
        s = np.hypot(gx, gy)
        u = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)
        m = np.isfinite(s) & (s > 0.05)
        if m.any():
            cosines.append(np.abs(gx[m] * u[0] + gy[m] * u[1]) / s[m])
            w_c.append(np.full(m.sum(), 1.0))
    G, WG = np.concatenate(grades), np.concatenate(w_g)
    o = np.argsort(G)
    cum = np.cumsum(WG[o]) / WG.sum()
    q = lambda p: float(G[o][np.searchsorted(cum, p)])
    C = np.concatenate(cosines) if cosines else np.array([np.nan])
    return {"km": float(WG.sum() / 1000), "grade_p50": q(0.5), "grade_p90": q(0.9), "share_gt10": float(WG[G > 0.10].sum() / WG.sum()),
            "contour_cos": float(np.mean(C)), "n_on_slope": int(len(C))}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8.4))
    for a, (sc, title) in zip(ax, RUNS):
        T = generate(scenario(sc, 4), -256, -256, L + 256, L + 256)
        z = T.elev[64:-64, 64:-64]
        gy, gx = np.gradient(z, RES)
        lv = np.array([-1.0, 1.0, 1.4]) / np.linalg.norm([-1.0, 1.0, 1.4])
        shade = (-gx * lv[0] + gy * lv[1] + lv[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1)
        ext = (0, L, 0, L)
        a.imshow(shade, cmap="gray", vmin=0.2, vmax=1.05, extent=ext)
        a.contour(np.flipud(z), levels=np.arange(np.floor(z.min() / 5) * 5, z.max(), 5.0), colors="#a0785a",
                  linewidths=0.35, extent=ext, origin="lower")
        w = T.water_kind[64:-64, 64:-64]
        a.imshow(np.ma.masked_where(w == 0, w), cmap="Blues", vmin=0, vmax=2.5, extent=ext, alpha=0.9)
        S = np.load(CITIES / f"big_TN_s0_HR_{sc}4_3x3_s2050" / "streets.npy")
        st = street_stats(S, T)
        seg = S[:, :4].reshape(-1, 2, 2)
        gr = np.array([abs(T.sample(np.array([s[1, 0]]), np.array([s[1, 1]]))[0] - T.sample(np.array([s[0, 0]]), np.array([s[0, 1]]))[0])
                       / max(np.hypot(*(s[1] - s[0])), 1e-6) for s in seg])
        br = S[:, 5] == 1
        a.add_collection(LineCollection(seg, colors="#f7f4ee", linewidths=3.4))      # casing: readable on dark slopes
        lc = LineCollection(seg[~br], array=np.clip(gr[~br] * 100, 0, 20), cmap="plasma", linewidths=1.8)
        lc.set_clim(0, 20)
        a.add_collection(lc)
        a.add_collection(LineCollection(seg[br], colors="#d62728", linewidths=3.2))
        a.set_xlim(0, L), a.set_ylim(0, L)
        a.set_xticks([]), a.set_yticks([])
        a.set_title(f"{title}: relief {z.max() - z.min():.0f} m, terrain median slope {100 * np.median(np.hypot(gx, gy)):.0f}%\n"
                    f"streets {st['km']:.1f} km; street grade median {100 * st['grade_p50']:.1f}%, p90 {100 * st['grade_p90']:.1f}%; "
                    f"alignment |cos| {st['contour_cos']:.2f}", fontsize=10)
        print(sc, st)
    cb = fig.colorbar(lc, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("street grade (%)")
    fig.text(0.01, 0.01, "TN_s0 + heuristic replanning, T = 0.5, street seed 2050, terrain seed 4. 5 m contours; water blue; "
             "bridges red. Alignment |cos|: 1 = straight up the slope, 0 = along the contour, ~0.64 = random.",
             fontsize=9, color="#555555")
    fig.savefig(FIG / "hills_comparison.png", dpi=110, bbox_inches="tight")
    print("written", FIG / "hills_comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
