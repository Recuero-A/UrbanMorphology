"""Milestone 16 (exploratory): general realism of the KAIST campus cities against a pool of real neighbourhoods.

Defined 2026-11-21, before computing, at the owner's request (rank by realism, not by distance to KAIST):
* real pool: 156 real 512 m windows = the 120 M15 pilot windows (Spain, Korea, Peru; data/processed/m15/real) + the
  36 M16 style references (configs/m16_style_refs; grid / organic / radial / cul-de-sac cities);
* features (none optimised by any method): street km, 3-way and 4-way shares, dead ends per km, log(1 + loops per
  km2), median and SD of log block area, median block shape factor, rectangular-block share, right-angle junction
  share, stroke junction continuity, dead-end-tree length share, circuity 200-350 m; each standardised by the real
  pool's mean and SD; a missing value is imputed with the real median (counted);
* per window scores: TYPICALITY = mean distance to the 5 nearest real windows / the same for real windows (leave one
  out; 1 = as typical as a real place), PRECISION and DENSITY (Naeem et al. 2020, k = 5, against the real pool),
  IN-RANGE = every feature inside the real 2.5-97.5 % range;
* every method: 3 seeds x 9 windows = 27 windows; the real KAIST windows (vehicular roads) are scored the same way.

    python scripts/m16_realism.py   ->  outputs/m16/kaist/realism.{md,json}, ranked figure sheets
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from m16_kaist_all import CACHE, ORDER, key_of, window_rows
from m16_kaist_campus import OUT, domain

FEATS = ["street_km", "share_3way", "share_4way", "deadend_per_km", "log_loops", "block_logA_med", "block_logA_sd",
         "block_phi_med", "block_rect_share", "jangle_right_share", "stroke_xjunc_lw", "tree_len_share", "circuity_200_350"]
POOL_CACHE = OUT / "realism_pool.pkl"
K = 5


def vec(r: dict) -> list:
    r = dict(r)
    lp = r.get("loops_per_km2")
    r["log_loops"] = np.log1p(lp) if lp is not None and np.isfinite(lp) else np.nan
    return [float(r[k]) if r.get(k) is not None and np.isfinite(r[k]) else np.nan for k in FEATS]


def real_pool():
    if POOL_CACHE.exists():
        return pickle.loads(POOL_CACHE.read_bytes())
    from urbanmorph.graphs.network_outcomes import outcome_row
    from urbanmorph.graphs.style_metrics import morphology_row
    rows, src = [], []
    for sid, metro, g, _ in pickle.loads((ROOT / "data" / "processed" / "m15" / "real" / "graphs.pkl").read_bytes()):
        if len(g.edges) == 0:
            continue
        rng = np.random.default_rng(0)
        rows.append({**morphology_row(g, None, centre=(256.0, 256.0), rng=rng), **outcome_row(g)})
        src.append(f"M15 {metro}")
    for _, r in pd.read_csv(ROOT / "configs" / "m16_style_refs" / "metrics.csv").iterrows():
        rows.append(r.to_dict())
        src.append(f"ref {r['style']}")
    X = np.array([vec(r) for r in rows], float)
    POOL_CACHE.write_bytes(pickle.dumps((X, src)))
    return X, src


def main() -> int:
    X, src = real_pool()
    med = np.nanmedian(X, 0)
    Xi = np.where(np.isfinite(X), X, med)
    mu, sd = Xi.mean(0), Xi.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (Xi - mu) / sd
    lo, hi = np.percentile(Xi, 2.5, 0), np.percentile(Xi, 97.5, 0)
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    kth = np.sort(D, 1)[:, K - 1]                                  # real radius (k-th neighbour, leave one out)
    real_typ = float(np.mean(np.sort(D, 1)[:, :K].mean(1)))

    def scores(rows):
        V = np.array([vec(r) for r in rows if r is not None], float)
        miss = int((~np.isfinite(V)).sum())
        Vi = np.where(np.isfinite(V), V, med)
        G = (Vi - mu) / sd
        d = np.sqrt(((G[:, None] - Z[None]) ** 2).sum(-1))
        typ = np.sort(d, 1)[:, :K].mean(1) / real_typ
        prec = (d <= kth[None]).any(1)
        dens = (d <= kth[None]).sum(1) / K
        inr = ((Vi >= lo) & (Vi <= hi)).all(1)
        return typ, prec, dens, inr, miss, len(V)

    big, _, _ = domain()
    allg = pickle.loads((OUT / "all_graphs.pkl").read_bytes())
    res = []
    for lab in ORDER:
        per = []
        for seed in (1, 2, 3):
            f = CACHE / f"{key_of(lab)}_s{seed}.pkl"
            if f.exists():
                per += window_rows(pickle.loads(f.read_bytes())["graph"], big, np.random.default_rng([seed, 16]))
        typ, prec, dens, inr, miss, n = scores(per)
        res.append({"method": lab, "typicality": float(np.mean(typ)), "typ_p90": float(np.percentile(typ, 90)),
                    "precision": float(prec.mean()), "density": float(dens.mean()), "in_range": float(inr.mean()),
                    "windows": n, "imputed_values": miss})
    rr = window_rows(allg["real"], big, np.random.default_rng(0))
    typ, prec, dens, inr, miss, n = scores(rr)
    res.append({"method": "real KAIST (vehicular)", "typicality": float(np.mean(typ)), "typ_p90": float(np.percentile(typ, 90)),
                "precision": float(prec.mean()), "density": float(dens.mean()), "in_range": float(inr.mean()),
                "windows": n, "imputed_values": miss})
    kd = pd.DataFrame(json.load(open(OUT / "all.json"))["rows"]).groupby("method")["kaist_distance"].mean()
    df = pd.DataFrame(res)
    df["kaist_distance"] = df["method"].map(kd)
    gen = df[df.method != "real KAIST (vehicular)"].sort_values("typicality").reset_index(drop=True)
    gen["rank"] = np.arange(1, len(gen) + 1)
    real = df[df.method == "real KAIST (vehicular)"]
    md = ["# Realism of the KAIST campus cities against 156 real neighbourhoods (EXPLORATORY)", "",
          "Real pool: 120 M15 pilot windows (Spain, Korea, Peru) + 36 style references. 13 features, none optimised by "
          "any method. Typicality = mean distance to the 5 nearest real windows / the same for real windows "
          "(1 = as typical as a real place; lower is better; ranking key). Precision / density (k = 5) and in-range "
          "(all features within the real 2.5–97.5 % range) against the same pool. 27 windows per method (3 seeds x 9).",
          "", "| Rank | Method | Typicality (mean) | Typicality (worst 10 %) | Precision | Density | In range | KAIST distance |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in gen.iterrows():
        md.append(f"| {r['rank']} | {r['method']} | {r['typicality']:.2f} | {r['typ_p90']:.2f} | {r['precision']:.2f} | "
                  f"{r['density']:.2f} | {r['in_range']:.2f} | {r['kaist_distance']:.2f} |")
    for _, r in real.iterrows():
        md.append(f"| ref | {r['method']} | {r['typicality']:.2f} | {r['typ_p90']:.2f} | {r['precision']:.2f} | "
                  f"{r['density']:.2f} | {r['in_range']:.2f} | 0.00 |")
    md += ["", "Real pool windows score typicality 1.00 by construction (leave one out). The real KAIST campus is scored "
               "against the same pool: a university campus is not a typical urban neighbourhood, so its typicality is high."]
    (OUT / "realism.md").write_text("\n".join(md) + "\n")
    json.dump({"rows": gen.to_dict("records") + real.to_dict("records"), "features": FEATS, "k": K,
               "pool": {"n": int(len(X)), "sources": pd.Series(src).str.split().str[0].value_counts().to_dict()}},
              open(OUT / "realism.json", "w"), indent=1, default=float)
    print("\n".join(md))
    # ranked figure sheets
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as im
    import matplotlib.pyplot as plt
    cities = ROOT / "outputs" / "m16" / "cities"
    items = [("real", f"Real KAIST — reference (typicality {float(real['typicality'].iloc[0]):.2f})")] + \
            [(key_of(r["method"]), f"{r['rank']}. {r['method']} (typicality {r['typicality']:.2f})") for _, r in gen.iterrows()]
    for src_, view, name in (("godot", "godot_oblique", "fig_kaist_all_godot_oblique"), ("godot", "godot_aerial", "fig_kaist_all_godot_aerial"),
                             ("godot", "godot_street", "fig_kaist_all_godot_street"), ("renders", "oblique", "fig_kaist_all_blender_oblique"),
                             ("renders", "aerial", "fig_kaist_all_blender_aerial")):
        fig, ax = plt.subplots(4, 4, figsize=(24, 14.5 if "aerial" not in view else 24))
        for a, (k, t) in zip(ax.ravel(), items):
            a.imshow(im.imread(cities / f"kaist_all_{k}_s1" / src_ / f"{view}.png"))
            a.set_title(t, fontsize=13)
            a.axis("off")
        fig.suptitle(f"Whole KAIST campus (1536 m, seed 1), ranked by realism (typicality vs 156 real neighbourhoods; lower = more typical) — "
                     f"{'Godot 4' if src_ == 'godot' else 'Blender'} {view.replace('godot_', '')}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(ROOT / "outputs" / "m16" / "figures" / f"{name}.png", dpi=60 if "aerial" not in view else 80)
        plt.close(fig)
    out = ROOT / "outputs" / "m16" / "figures" / "kaist_aerial"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    for j, (k, t) in enumerate(items):
        for src_, view, tag in (("renders", "aerial", "blender"), ("godot", "godot_aerial", "godot")):
            img = im.imread(cities / f"kaist_all_{k}_s1" / src_ / f"{view}.png")
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 100, h / 100 + 0.6), dpi=100)
            ax = fig.add_axes([0, 0, 1, h / (h + 60)])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.5, 1 - 30 / (h + 60), f"{t} — whole KAIST campus 1536 m, seed 1, {tag} top view", ha="center",
                     va="center", fontsize=18)
            fig.savefig(out / f"{j:02d}_{k}_{tag}.png", dpi=100)
            plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
