"""M* vs A scoreboard and a random map gallery, from the STORED Milestone 13 runs (no new generation).

    python scripts/m13_scoreboard.py   # -> outputs/m13/scoreboard/{scoreboard.md, scoreboard.json, gallery.png}

Development data only (DEV-B: 27 tiles; GEN: 100 tiles of unseen cities); TN backbone, 3 training seeds x 4 inference
seeds. Metrics are the protocol's (m13_analyze.summarize), compared with the real streets of the same tiles. Paired
block-bootstrap CIs (m13_analyze.contrast) for S_T and S10_cond per training seed. Exploratory summary: the
pre-registered verdicts stay those of outputs/m13/FINAL_REPORT.md. Gallery: 12 GEN tiles drawn at random (seed 7)
before looking at any map; training seed 0, inference seed 1 (first sample).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import m13_analyze as an  # noqa: E402
import m13_experiments as ex  # noqa: E402
from m13_elevation_maps import contour_step  # noqa: E402
from m13_three_tiles import draw  # noqa: E402
from urbanmorph.experiments.graph_io import graph_from_json
from urbanmorph.graphs.road_graph import RoadGraph

OUT = ex.OUT / "scoreboard"
SEL = json.loads((ex.OUT / "selection.json").read_text())["M_star"]
# (key, label, target): "low" = lower is better, "one" = closer to 1, "real" = closer to the real streets, "high"
ROWS = [("S_T", "S_T terrain score", "low"), ("S10_cond", "S10 street-network realism", "low"),
        ("energy_distance", "joint realism: energy distance", "low"), ("mmd", "joint realism: MMD", "low"),
        ("density_ratio", "street density / real", "one"), ("giant_share", "largest connected piece", "real"),
        ("intersections_per_km2", "intersections per km²", "real"), ("dead_end_ratio_m13", "dead-end ratio", "real"),
        ("contour_align", "contour alignment", "real"), ("bridges", "bridges per tile", "real"),
        ("river_crossings_unbridged", "unbridged river crossings per tile", "low"),
        ("jaccard_diversity", "diversity between samples", "high")]


def find(tiles, backbone, preset, ov):
    for f in sorted(ex.RUNS.glob(f"{tiles}__{backbone}__{preset}__*.json")):
        sp = json.loads(f.read_text())["spec"]
        if sp.get("overrides", {}) == ov and not sp.get("repair") and sp.get("pot_version") == "v3" \
                and sp.get("potentials", "TN") == "TN" and sp["seeds"] == [1, 2, 3, 4]:
            return json.loads(f.read_text())
    raise FileNotFoundError(tiles, backbone, preset, ov)


def real_value(s, key, real):
    if key == "giant_share":
        return float(np.nanmean([r["giant_share"] for r in real]))
    if key == "density_ratio":
        return 1.0
    return s.get(key + "_real", np.nan)


def winner(key, target, m, a, real):
    if not (np.isfinite(m) and np.isfinite(a)):
        return "—"
    dm, da = {"low": (m, a), "high": (-m, -a), "one": (abs(m - 1), abs(a - 1)),
              "real": (abs(m - real), abs(a - real))}[target]
    return "tie" if abs(dm - da) < 1e-9 else ("M*" if dm < da else "A")


def scoreboard():
    out, md = {}, ["# M* vs A scoreboard (development data, TN backbone; exploratory)\n",
                   "Mean over 3 training seeds; [min, max] across seeds. Winner = better on the mean (closer to real "
                   "where that is the target). Not significance: see the CI section.\n"]
    for tiles, name in (("devB", "DEV-B (27 tiles, confirmation set)"), ("gen", "GEN (100 tiles: San Francisco, "
                                                                                "Girona, Manresa; never trained on)")):
        real = an.real_rows(tiles)
        per = {"M*": [], "A": []}
        cis = {"S_T": [], "S10_cond": []}
        for b in ("TN_s0", "TN_s1", "TN_s2"):
            rm, ra = find(tiles, b, SEL["preset"], SEL["overrides"]), find(tiles, b, "standard", {})
            per["M*"].append(an.summarize(rm))
            per["A"].append(an.summarize(ra))
            for k in cis:
                cis[k].append(an.contrast(ra, rm, k, n_boot=1000))
        md += [f"\n## {name}\n", "| Metric | target | real | M* | A | better |", "|---|---|---|---|---|---|"]
        wins = {"M*": 0, "A": 0, "tie": 0}
        rows = []
        for key, lab, tgt in ROWS:
            vm = np.array([s.get(key, np.nan) for s in per["M*"]], float)
            va = np.array([s.get(key, np.nan) for s in per["A"]], float)
            rv = real_value(per["A"][0], key, real)
            w = winner(key, tgt, np.nanmean(vm), np.nanmean(va), rv)
            wins[w] = wins.get(w, 0) + (w in wins)
            f = lambda v: f"{np.nanmean(v):.3g} [{np.nanmin(v):.3g}, {np.nanmax(v):.3g}]"
            tg = {"low": "lower", "high": "higher", "one": "1.0", "real": "≈ real"}[tgt]
            md.append(f"| {lab} | {tg} | {rv:.3g} | {f(vm)} | {f(va)} | **{w}** |" if np.isfinite(rv) else
                      f"| {lab} | {tg} | — | {f(vm)} | {f(va)} | **{w}** |")
            rows.append({"key": key, "real": rv, "M*": vm.tolist(), "A": va.tolist(), "better": w})
        md.append(f"\nCount of metrics: M* better {wins['M*']}, A better {wins['A']}, tie {wins.get('tie', 0)}.\n")
        md.append("Paired block-bootstrap 95 % CI of M* − A, per training seed (negative = M* better):\n")
        for k, v in cis.items():
            md.append(f"* {k}: " + "; ".join(f"s{i}: {c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"
                                            f"{' (significant)' if c['hi'] < 0 or c['lo'] > 0 else ''}"
                                            for i, c in enumerate(v)))
        out[tiles] = {"rows": rows, "wins": wins, "ci": cis}
    (OUT / "scoreboard.json").write_text(json.dumps(out, indent=1, default=float))
    (OUT / "scoreboard.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


graph = graph_from_json   # re-exported: the stored-run graph format (urbanmorph.experiments.graph_io)


def gallery():
    tiles = ex.tile_set("gen")
    rm, ra = find("gen", "TN_s0", SEL["preset"], SEL["overrides"]), find("gen", "TN_s0", "standard", {})
    pick = sorted(np.random.default_rng(7).choice(len(tiles), 12, replace=False))
    fig, ax = plt.subplots(6, 6, figsize=(18, 18.6), layout="constrained")
    for n, i in enumerate(pick):
        t = tiles[i]
        d = t["d"]
        z, w = d["elev"][64:192, 64:192], d["water"][64:192, 64:192]
        land = z[w == 0] if (w == 0).any() else z
        lo, hi = float(np.floor(land.min())), float(np.ceil(land.max())) + 1
        r, c0 = divmod(n, 2)
        panels = [("Real", t["graph"], ex.real_rows([], "gen")[i]["street_km_per_km2"])]
        for lab, res in (("M*", rm), ("A", ra)):
            panels.append((lab, graph(res["seeds"]["1"]["graphs"][i]), res["seeds"]["1"]["rows"][i]["street_km_per_km2"]))
        for k, (lab, g, dens) in enumerate(panels):
            a = ax[r, 3 * c0 + k]
            draw(a, d, g, lo, hi, contour_step(lo, hi))
            a.set_title(f"{t['sid'] if k == 0 else ''}\n{lab}: {dens:.1f} km/km²", fontsize=8)
    fig.suptitle("12 random GEN tiles (seed 7): real | M* | A, first sample, TN_s0. Terrain colour: low green → high "
                 "beige; water blue; bridges red", fontsize=11)
    fig.savefig(OUT / "gallery.png", dpi=85)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gallery()
    scoreboard()
    return 0


if __name__ == "__main__":
    sys.exit(main())
