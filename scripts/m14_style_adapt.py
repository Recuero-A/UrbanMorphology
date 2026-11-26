"""M14 EXPLORATORY (not pre-registered): does a generator adapt its street-pattern style to the place it is asked to fill?

Style statistics per tile (already stored for every real and generated tile by the M13/M14 runs):
  * phi     Boeing (2019) orientation order: 0 = orientations uniform (organic), 1 = a perfect two-axis grid;
  * deg4    share of 4-way junctions among junctions (grids high, organic and suburban low);
  * angdev  mean deviation of junction angles from 90 degrees (grids low);
  * sinuos  length-weighted mean sinuosity of links (junction-to-junction path length / straight distance, links with a
            chord >= 30 m); computed here from the stored graphs (real and generated are both resampled to ~26 m steps);
  * turn    median absolute heading change at interior (degree-2) vertices, degrees (straight streets ~0).
Adaptation measures per method, over the tiles of a set (inference seeds 1-3 kept separate, then averaged):
  * rho     Spearman correlation between the real and the generated statistic of the SAME tile;
  * slope   OLS slope of generated on real (1 = follows the real style fully, 0 = one style everywhere);
  * mae     mean |generated - real|.
References: "city oracle" predicts every tile by its city's real mean (how much style is explained by city alone);
P is a style-blind procedural generator (no adaptation expected). Block-bootstrap 95% CIs (geographic blocks, 2000 draws).
The backbone sees only elevation, slope and water, so any adaptation must come through the terrain input; TNlp / TNse
(street imprint removed from the DEM) show how much of it is the street-bed leakage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import m14_experiments as mx  # noqa: E402
from m13_scoreboard import graph  # noqa: E402
from m14_figures import FIG  # noqa: E402
from m14_report import get  # noqa: E402
from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m14" / "style"
STATS = {"phi": "s10_orientation_phi", "deg4": "s10_share_deg4_of_junctions", "angdev": "s10_junction_angle_dev_deg",
         "sinuos": "sinuos", "turn": "turn"}
LABELS = ["P", "A", "M*", "M*+repair", "M13-SMC", "NA-myopic", "NA-fixed", "NA-SMC"]
LEAK = ["A", "M*", "NA-myopic"]          # the methods with TNlp / TNse runs on GEN2
CITY = {"ALY": "Alcoi", "TGN": "Tarragona", "SAC": "Sacramento", "SBA": "Santa Barbara", "GRO": "Girona",
        "MRS": "Manresa", "SFO": "San Francisco"}
B = 2000


def curve_stats(g) -> dict:
    out = {"sinuos": np.nan, "turn": np.nan}
    if g is None or len(g.edges) == 0:
        return out
    adj = [[] for _ in range(g.n)]
    for a, b in g.edges:
        adj[a].append(b), adj[b].append(a)
    deg = np.array([len(x) for x in adj])
    P = g.nodes
    turns = []
    for v in np.where(deg == 2)[0]:
        d1, d2 = P[v] - P[adj[v][0]], P[adj[v][1]] - P[v]
        c = np.dot(d1, d2) / max(np.linalg.norm(d1) * np.linalg.norm(d2), 1e-9)
        turns.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    seen, num, den = set(), 0.0, 0.0
    for u in np.where(deg != 2)[0]:
        for w in adj[u]:
            if (u, w) in seen:
                continue
            path, prev, cur = [u, w], u, w
            seen.add((u, w))
            while deg[cur] == 2:
                prev, cur = cur, (adj[cur][0] if adj[cur][0] != prev else adj[cur][1])
                path.append(cur)
            seen.add((cur, prev))
            pts = P[path]
            length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            chord = float(np.linalg.norm(pts[-1] - pts[0]))
            if chord >= 30.0:
                num, den = num + length * length / chord, den + length
    if den:
        out["sinuos"] = num / den
    if turns:
        out["turn"] = float(np.median(turns))
    return out


_REAL, _RUN = {}, {}


def real(tiles):
    if tiles not in _REAL:
        R = json.loads((PROJECT_ROOT / "outputs" / "m14" / "real" / f"{tiles}.json").read_text())
        for r, t in zip(R, mx.tile_set(tiles)):
            assert r["sid"] == t["sid"]
            r.update(curve_stats(t["graph"]))
        _REAL[tiles] = R
    return _REAL[tiles]


def run_of(tiles, bb, lab):
    """The stored run with the curvature statistics added to every row (cached)."""
    if (tiles, bb, lab) not in _RUN:
        run = get(tiles, bb, lab)
        if run is not None:
            for s in run["seeds"].values():
                for row, G in zip(s["rows"], s["graphs"]):
                    row.update(curve_stats(graph(G)))
        _RUN[tiles, bb, lab] = run
    return _RUN[tiles, bb, lab]


def measures(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.std(x) == 0:
        return np.nan, np.nan, np.nan
    return stats.spearmanr(x, y)[0], np.polyfit(x, y, 1)[0], float(np.mean(np.abs(y - x)))


def per_seed(R, run, key, idx=None):
    """[(rho, slope, mae)] per inference seed; idx restricts to a tile subset (bootstrap)."""
    x = np.array([r[key] for r in R], float)
    out = []
    for s in sorted(run["seeds"], key=int):
        y = np.array([r[key] for r in run["seeds"][s]["rows"]], float)
        out.append(measures(x[idx], y[idx]) if idx is not None else measures(x, y))
    return np.nanmean(np.array(out, float), axis=0)


def boot_idx(R, rng):
    blocks = sorted({r["block"] for r in R})
    by = {b: [i for i, r in enumerate(R) if r["block"] == b] for b in blocks}
    for _ in range(B):
        yield np.concatenate([by[b] for b in rng.choice(blocks, len(blocks))])


def analyse(tiles, bb="TN_s0", labels=LABELS):
    R, rng = real(tiles), np.random.default_rng(20261120)
    res = {}
    for lab in labels:
        run = run_of(tiles, bb, lab)
        if run is None:
            continue
        res[lab] = {}
        for k, key in STATS.items():
            pt = per_seed(R, run, key)
            bs = np.array([per_seed(R, run, key, i) for i in boot_idx(R, rng)])
            res[lab][k] = {"rho": pt[0], "slope": pt[1], "mae": pt[2],
                           "rho_ci": np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist(),
                           "slope_ci": np.nanpercentile(bs[:, 1], [2.5, 97.5]).tolist()}
    # city oracle: every tile predicted by its city's real mean
    city = np.array([r["sid"].split("_")[1] for r in R])
    res["city oracle"] = {}
    for k, key in STATS.items():
        x = np.array([r[key] for r in R], float)
        y = np.array([np.nanmean(x[city == c]) for c in city])
        rho, slope, mae = measures(x, y)
        res["city oracle"][k] = {"rho": rho, "slope": slope, "mae": mae}
    return res


def per_city(tiles, bb="TN_s0", labels=LABELS):
    R = real(tiles)
    city = np.array([r["sid"].split("_")[1] for r in R])
    out = {}
    for c in dict.fromkeys(city):
        m = city == c
        row = {"n": int(m.sum())}
        for k, key in STATS.items():
            row[f"real_{k}"] = float(np.nanmean([R[i][key] for i in np.where(m)[0]]))
        for lab in labels:
            run = run_of(tiles, bb, lab)
            if run is None:
                continue
            for k, key in STATS.items():
                row[f"{lab}_{k}"] = float(np.nanmean([run["seeds"][s]["rows"][i][key] for s in run["seeds"] for i in np.where(m)[0]]))
        out[CITY.get(c, c)] = row
    return out


def fig(tiles="gen2", bb="TN_s0"):
    R = real(tiles)
    x = np.array([r[STATS["phi"]] for r in R], float)
    city = [CITY.get(r["sid"].split("_")[1]) for r in R]
    cc = {c: col for c, col in zip(dict.fromkeys(city), ("#2a6f97", "#b23a48", "#e09f3e", "#6c9a8b"))}
    show = ["P", "M*", "NA-fixed", "NA-SMC"]
    f, ax = plt.subplots(1, len(show), figsize=(4.0 * len(show), 4.2), layout="constrained", sharey=True)
    for a, lab in zip(ax, show):
        run = run_of(tiles, bb, lab)
        y = np.nanmean([[r[STATS["phi"]] for r in run["seeds"][s]["rows"]] for s in run["seeds"]], axis=0)
        for c in cc:
            m = np.array(city) == c
            a.scatter(x[m], y[m], s=18, color=cc[c], label=c, alpha=0.85, edgecolor="white", linewidth=0.5)
        a.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
        rho = analyse.cache[lab]["phi"]["rho"]
        a.set_title(f"{lab}   ρ = {rho:.2f}", fontsize=10)
        a.set_xlim(0, 1), a.set_ylim(0, 1), a.set_xlabel("real φ of the tile")
        a.grid(alpha=0.25)
    ax[0].set_ylabel("generated φ (mean of 3 samples)")
    ax[0].legend(fontsize=8, loc="upper left")
    f.suptitle(f"{tiles.upper()}: orientation order φ (0 organic … 1 grid), real vs generated, same tile (TN_s0). "
               "Dashed = perfect adaptation", fontsize=10)
    f.savefig(FIG / f"fig_style_adapt_{tiles}.png", dpi=120)


def fmt(v, ci=None):
    if not np.isfinite(v):
        return "—"
    return f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    md, js = ["# M14 exploratory: street-pattern style adaptation", "", "Generated by `scripts/m14_style_adapt.py` "
              "(exploratory, not pre-registered; development data only). See the script docstring for definitions.", ""], {}
    for tiles in ("gen2", "gen", "devB"):
        res = analyse(tiles)
        js[tiles] = {"adapt": res, "per_city": per_city(tiles)}
        if tiles == "gen2":
            analyse.cache = res
        md += [f"## {tiles.upper()} (TN_s0)", "", "| Method | φ ρ [95% CI] | φ slope [95% CI] | φ MAE | 4-way ρ | 4-way slope | "
               "angle-dev ρ | sinuosity ρ | sinuosity slope | turn ρ |", "|---|---|---|---|---|---|---|---|---|---|"]
        for lab, r in res.items():
            md.append(f"| {lab} | {fmt(r['phi']['rho'], r['phi'].get('rho_ci'))} | {fmt(r['phi']['slope'], r['phi'].get('slope_ci'))} | "
                      f"{fmt(r['phi']['mae'])} | {fmt(r['deg4']['rho'])} | {fmt(r['deg4']['slope'])} | {fmt(r['angdev']['rho'])} | "
                      f"{fmt(r['sinuos']['rho'])} | {fmt(r['sinuos']['slope'])} | {fmt(r['turn']['rho'])} |")
        pc = js[tiles]["per_city"]
        for k, what, f in (("phi", "orientation order φ", "{:.2f}"), ("sinuos", "sinuosity", "{:.3f}"),
                           ("turn", "median turn at bends (°)", "{:.1f}")):
            md += ["", f"Per city, mean {what} (real vs generated):", "", "| City | n | real | " + " | ".join(LABELS) + " |",
                   "|---|---|---|" + "---|" * len(LABELS)]
            for c, row in pc.items():
                md.append(f"| {c} | {row['n']} | {f.format(row[f'real_{k}'])} | " +
                          " | ".join(f.format(row[f'{l}_{k}']) if f"{l}_{k}" in row else "—" for l in LABELS) + " |")
        md.append("")
    md += ["## Leakage check: GEN2, backbones TN vs TNlp (DEM low-passed) vs TNse (streets erased from the DEM)", "",
           "| Method | Backbone | φ ρ [95% CI] | φ slope | 4-way ρ |", "|---|---|---|---|---|"]
    js["leak"] = {}
    for bb in ("TN_s0", "TNlp_s0", "TNse_s0"):
        res = analyse("gen2", bb, LEAK)
        js["leak"][bb] = res
        for lab in LEAK:
            if lab in res:
                r = res[lab]
                md.append(f"| {lab} | {bb} | {fmt(r['phi']['rho'], r['phi']['rho_ci'])} | {fmt(r['phi']['slope'])} | {fmt(r['deg4']['rho'])} |")
    (OUT / "style_adapt.json").write_text(json.dumps(js, indent=1, default=float))
    (OUT / "STYLE_ADAPT.md").write_text("\n".join(md) + "\n")
    fig("gen2")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
