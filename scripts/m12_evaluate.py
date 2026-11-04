"""Milestone 12: pre-registered evaluation (docs/M12_EVALUATION_PROTOCOL.md). Development tiles only.

    python scripts/m12_evaluate.py --regime A     # conditional real terrain: S_T, water, network, failures
    python scripts/m12_evaluate.py --regime B     # unconditional S10 (256 tiles, sampling seed 123, as M10)
    python scripts/m12_evaluate.py --regime C     # procedural stress tests (10 scenarios x 5 seeds)
    python scripts/m12_evaluate.py --regime H5    # structure-head AUC (teacher forcing, validation steps)
    python scripts/m12_evaluate.py --regime stats # bootstrap tests of H1-H4 from the saved regime A / B files

Outputs: outputs/m12/evaluations/<regime>/*.json. Temperature 0.5 for every learned model (fixed in advance).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m10_street_diagnosis import KEYS  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.procedural_terrain import sample_P
from urbanmorph.graphs.road_graph import graph_metrics
from urbanmorph.graphs.terrain_generate import TerrainPolicy, terrain_of
from urbanmorph.graphs.terrain_metrics import S_T_KEYS, tile_metrics
from urbanmorph.graphs.terrain_sequence import channel_list
from urbanmorph.models.terrain_graph_turtle import TerrainGraphTurtle
from urbanmorph.terrain.road_feasibility import Thresholds
from urbanmorph.terrain.synthesis import SCENARIOS, generate, scenario

CK = PROJECT_ROOT / "checkpoints" / "m12"
OUT = PROJECT_ROOT / "outputs" / "m12" / "evaluations"
GRAPHS3D = PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"
TERRAIN = PROJECT_ROOT / "data" / "processed" / "m12" / "terrain"
TH = Thresholds.load(PROJECT_ROOT / "configs" / "m12_feasibility.json")
TEMP, N_SAMP, N_BOOT = 0.5, 4, 2000
SPECS = {"N0": ("N0", "none"), "R": ("N0", "R"), "TN": ("TN", "none"), "TN+R": ("TN", "R"), "TN+HR": ("TN", "HR"), "TN+HRV": ("TN", "HRV"),
         "TNelev": ("TNelev", "none"), "TNslope": ("TNslope", "none"), "TNwater": ("TNwater", "none"),
         "TNelevslope": ("TNelevslope", "none"), "P": (None, "P")}


def load_model(run, device):
    ck = torch.load(CK / run / "best.pt", map_location=device, weights_only=False)
    c = ck["config"]
    m = TerrainGraphTurtle(c["model"]["hidden"], c["model"]["n_mix"], c["model"]["n_classes"],
                           n_terrain=len(channel_list(c["terrain"]))).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, c["terrain"], ck["norm"]


def seeds_of(run):
    """Seeds whose training FINISHED (summary.json); a best.pt alone may be mid-training."""
    return [s for s in (0, 1) if (PROJECT_ROOT / "outputs" / "m12" / "training" / f"{run}_s{s}" / "summary.json").exists()]


def generate_for(spec, seed, tiles, device, sample_seed, density_pool):
    """Graphs and policy stats for ``tiles`` (list of terrain dicts)."""
    run, mode = SPECS[spec]
    if mode == "P":
        rng = np.random.default_rng(sample_seed)
        return [sample_P(terrain_of(d), rng, float(rng.choice(density_pool)), TH) for d in tiles], {}
    model, names, norm = load_model(f"{run}_s{seed}", device)
    pol = TerrainPolicy(tiles, names, norm, mode=mode, th=TH)
    gs = sample_graphs(model, len(tiles), device=device, seed=sample_seed, temperature=TEMP, terrain=pol)
    return gs, pol.summary()


def real_graphs():
    g = pickle.loads(GRAPHS3D.read_bytes())
    return {sid: (sp, gr) for sp in g for sid, _, gr in g[sp]}


def dev_tiles_with_dem(real):
    out = []
    for sid, (sp, g) in sorted(real.items()):
        if sp in ("val", "val_t"):
            d = dict(np.load(TERRAIN / f"{sid}.npz"))
            if d["valid"].all():
                out.append((sid, sp, g, d))
    return out


def density_pool(real):
    return [graph_metrics(g)["intersections_per_km2"] for sid, (sp, g) in real.items() if sp == "train"]


def per_tile(gs, ds):
    rows = []
    for g, d in zip(gs, ds):
        m = tile_metrics(g, terrain_of(d))
        m.update({k: graph_metrics(g)[k] for k in KEYS})
        rows.append(m)
    return rows


def regime_A(device, specs):
    real = real_graphs()
    tiles = dev_tiles_with_dem(real)
    out = OUT / "A"
    out.mkdir(parents=True, exist_ok=True)
    ref = per_tile([g for _, _, g, _ in tiles], [d for *_, d in tiles])
    (out / "real.json").write_text(json.dumps({"tiles": [(s, sp) for s, sp, _, _ in tiles], "rows": ref}, default=float))
    pool = density_pool(real)
    for spec in specs:
        run = SPECS[spec][0]
        for seed in ([0] if run is None else seeds_of(run)):
            f = out / f"{spec}_s{seed}.json"
            if f.exists():
                continue
            ds = [d for *_, d in tiles for _ in range(N_SAMP)]
            t0 = time.time()
            gs, stats = generate_for(spec, seed, ds, device, 1000 + seed, pool)
            rows = per_tile(gs, ds)
            sec = time.time() - t0
            keys = sorted({k for r in rows for k in r})
            vals = lambda i, k: np.array([rows[i * N_SAMP + j].get(k, np.nan) for j in range(N_SAMP)], float)
            avg = [{k: float(np.nanmean(vals(i, k))) if np.isfinite(vals(i, k)).any() else np.nan for k in keys}
                   for i in range(len(tiles))]
            f.write_text(json.dumps({"spec": spec, "seed": seed, "seconds": sec, "policy": stats, "rows": avg}, default=float))
            print(spec, seed, f"{sec:.0f}s", flush=True)


def regime_B(device, specs):
    """Unconditional S10: 256 tiles at sampling seed 123; terrain models get terrain from random TRAINING tiles."""
    real = real_graphs()
    out = OUT / "B"
    out.mkdir(parents=True, exist_ok=True)
    train_ids = sorted(s for s, (sp, _) in real.items() if sp == "train")
    rng = np.random.default_rng(123)
    pick = rng.choice(train_ids, 256)
    ds = [dict(np.load(TERRAIN / f"{s}.npz")) for s in pick]
    pool = density_pool(real)
    for spec in specs:
        run = SPECS[spec][0]
        for seed in ([0] if run is None else seeds_of(run)):
            f = out / f"{spec}_s{seed}.json"
            if f.exists():
                continue
            gs, stats = generate_for(spec, seed, ds, device, 123, pool)
            rows = [{k: graph_metrics(g)[k] for k in KEYS} for g in gs]
            f.write_text(json.dumps({"spec": spec, "seed": seed, "policy": stats, "rows": rows}, default=float))
            print("B", spec, seed, flush=True)
    for sp in ("val", "val_t"):
        rows = [{k: graph_metrics(g)[k] for k in KEYS} for s, (x, g) in real.items() if x == sp]
        (out / f"real_{sp}.json").write_text(json.dumps(rows, default=float))


def regime_C(device, specs, n_seeds=5):
    out = OUT / "C"
    out.mkdir(parents=True, exist_ok=True)
    real = real_graphs()
    pool = density_pool(real)
    ds, meta = [], []
    for name in sorted(SCENARIOS):
        for sd in range(n_seeds):
            t = generate(scenario(name, sd), -256, -256, 768, 768)
            ds.append({"elev": t.elev, "valid": np.ones(t.elev.shape, bool), "water": t.water_kind})
            meta.append((name, sd))
    for spec in specs:
        run = SPECS[spec][0]
        for seed in ([0] if run is None else seeds_of(run)[:1]):
            f = out / f"{spec}_s{seed}.json"
            if f.exists():
                continue
            gs, stats = generate_for(spec, seed, ds, device, 77, pool)
            rows = []
            for (name, sd), g, d in zip(meta, gs, ds):
                t = terrain_of(d)
                m = tile_metrics(g, t)
                from urbanmorph.terrain.road_feasibility import evaluate
                seg, cls, st = g.segments(), g.classes(), g.structs()
                viol = sum(1 for k in range(len(seg)) if st[k] == 0 and evaluate(t, seg[k, :2], seg[k, 2:], int(cls[k]), TH).kind
                           in ("excessive_grade", "impossible_crossing", "water_obstruction", "ends_in_water"))
                m.update(scenario=name, seed=sd, edges=int(len(seg)), violating_edges=int(viol),
                         components=graph_metrics(g)["components"])
                rows.append(m)
            f.write_text(json.dumps({"spec": spec, "seed": seed, "policy": stats, "rows": rows}, default=float))
            print("C", spec, seed, flush=True)


def regime_H5(device):
    from m12_train import samples_for  # noqa: E402
    real = real_graphs()
    out = OUT / "H5"
    out.mkdir(parents=True, exist_ok=True)
    terr = {}
    items = [(s, g) for s, (sp, g) in sorted(real.items()) if sp in ("val", "val_t")]
    for s, _ in items:
        terr[s] = dict(np.load(TERRAIN / f"{s}.npz"))
    res = {}
    for run in ("N0", "TN"):
        for seed in seeds_of(run):
            model, names, norm = load_model(f"{run}_s{seed}", device)
            pops, _ = samples_for(items, terr, np.random.default_rng(5), names, norm, [0] * len(items))
            probs, labels = [], []
            with torch.no_grad():
                for i in range(0, len(pops["n_targets"]), 1024):
                    b = {k: torch.from_numpy(v[i:i + 1024]).to(device) for k, v in pops.items()}
                    model.pop_nll(b)
                    p = torch.softmax(model.last_struct_logit[:, :-1], -1)[..., 1].cpu().numpy()
                    lab = b["target_struct"].cpu().numpy()
                    m = lab >= 0
                    probs.append(p[m])
                    labels.append(lab[m] == 1)
            p, y = np.concatenate(probs), np.concatenate(labels)
            r = np.argsort(np.argsort(p)) + 1
            auc = (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / max(y.sum() * (~y).sum(), 1)
            res[f"{run}_s{seed}"] = {"auc_bridge": float(auc), "n_bridge_steps": int(y.sum()), "n_steps": int(len(y)),
                                     "recall_at_0.5": float((p[y] > 0.5).mean()) if y.any() else None}
            print(run, seed, res[f"{run}_s{seed}"], flush=True)
    (out / "auc.json").write_text(json.dumps(res, indent=1))


def w1(a, b):
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def score(gen, ref, keys, idx=None):
    """Mean W1/IQR over ``keys`` between per-tile generated and real values (tiles ``idx``)."""
    vals = []
    for k in keys:
        a = np.array([r[k] for r in gen], float)
        b = np.array([r[k] for r in ref], float)
        if idx is not None:
            a, b = a[idx], b[idx]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        iqr = np.subtract(*np.percentile(b, [75, 25])) if len(b) > 3 else np.nan
        if len(a) and len(b) and iqr > 0:
            vals.append(w1(a, b) / iqr)
    return float(np.mean(vals)) if vals else np.nan


def per_key(gen, ref, keys):
    return {k: score(gen, ref, [k]) for k in keys}


def boot_delta(fn, n, rng):
    d = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d.append(fn(idx))
    d = np.array(d)
    d = d[np.isfinite(d)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [np.nan, np.nan]


def stats():
    A = OUT / "A"
    real = json.loads((A / "real.json").read_text())
    ref, tiles = real["rows"], real["tiles"]
    n = len(ref)
    load = lambda spec, seed: json.loads((A / f"{spec}_s{seed}.json").read_text())["rows"] if (A / f"{spec}_s{seed}.json").exists() else None
    rng = np.random.default_rng(2026)
    res = {"n_tiles": n, "tiles_by_split": {sp: sum(1 for _, s in tiles if s == sp) for sp in ("val", "val_t")}, "A": {}, "tests": {}}
    term = lambda rows: np.array([max(0.0, 1 - r["street_km_per_km2"] / max(q["street_km_per_km2"], 1e-9)) for r, q in zip(rows, ref)])
    for f in sorted(A.glob("*_s*.json")):
        spec, seed = f.stem.rsplit("_s", 1)
        rows = load(spec, int(seed))
        pol = json.loads(f.read_text())["policy"]
        res["A"][f.stem] = {"S_T": score(rows, ref, S_T_KEYS), "S_T_per_metric": per_key(rows, ref, S_T_KEYS),
                            "S10_cond": score(rows, ref, KEYS),
                            "unbridged_water_m_per_tile": float(np.nanmean([r["unbridged_water_m"] for r in rows])),
                            "bridges_per_tile": float(np.nanmean([r["bridges"] for r in rows])),
                            "termination_deficit": float(term(rows).mean()),
                            "premature_share": float((term(rows) > 0.5).mean()),
                            "empty_share": float(np.nanmean([r["empty_share"] for r in rows])), "policy": pol}
    res["A"]["real"] = {"unbridged_water_m_per_tile": float(np.nanmean([r["unbridged_water_m"] for r in ref])),
                        "bridges_per_tile": float(np.nanmean([r["bridges"] for r in ref])),
                        "empty_share": float(np.nanmean([r["empty_share"] for r in ref]))}
    pairs = {"H1": ("TN", "N0", S_T_KEYS), "H3": ("TN+R", "R", S_T_KEYS)}
    for h, (a, b, keys) in pairs.items():
        for seed in (0, 1):
            ra, rb = load(a, seed), load(b, seed)
            if ra is None or rb is None:
                continue
            sa, sb = score(ra, ref, keys), score(rb, ref, keys)
            ci = boot_delta(lambda idx: score(ra, ref, keys, idx) - score(rb, ref, keys, idx), n, rng)
            need = 0.2 if h == "H1" else 0.1
            res["tests"][f"{h}_s{seed}"] = {"S_T_a": sa, "S_T_b": sb, "rel_change": (sa - sb) / sb, "delta_ci95": ci,
                                            "pass": bool((sa - sb) / sb <= -need and ci[1] < 0)}
    for seed in (0, 1):
        ra, rb = load("TN+HR", seed), load("TN+R", seed)
        if ra is None or rb is None:
            continue
        da, db = term(ra), term(rb)
        ci = boot_delta(lambda idx: da[idx].mean() - db[idx].mean(), n, rng)
        dst = score(ra, ref, S_T_KEYS) - score(rb, ref, S_T_KEYS)
        res["tests"][f"H4_s{seed}"] = {"deficit_HR": float(da.mean()), "deficit_R": float(db.mean()), "delta_ci95": ci,
                                       "delta_S_T": dst, "pass": bool(ci[1] < 0 and dst <= 0.05)}
    B = OUT / "B"
    if (B / "real_val.json").exists():
        refB = json.loads((B / "real_val.json").read_text())
        res["B"] = {}
        for f in sorted(B.glob("*_s*.json")):
            rows = json.loads(f.read_text())["rows"]
            res["B"][f.stem] = {"S10_vs_val": score(rows, refB, KEYS)}
        for seed in (0, 1):
            a, b = res["B"].get(f"TN_s{seed}"), res["B"].get(f"N0_s{seed}")
            if a and b:
                res["tests"][f"H2_s{seed}"] = {"S10_TN": a["S10_vs_val"], "S10_N0": b["S10_vs_val"],
                                               "pass": bool(a["S10_vs_val"] <= b["S10_vs_val"] + 0.10)}
    # by terrain class of the real tile (landform p90, from the Phase 2 audit)
    import pandas as pd
    aud = pd.read_csv(PROJECT_ROOT / "outputs" / "m12" / "audits" / "terrain_audit.csv").set_index("sample_id")
    cls = np.array([aud.terrain_class_landform.get(s, "nan") for s, _ in tiles])
    res["by_class"] = {}
    for c in ("flat", "moderate", "steep"):
        idx = np.flatnonzero(cls == c)
        if len(idx) < 4:
            continue
        res["by_class"][c] = {"n": int(len(idx))}
        for f in sorted(A.glob("*_s*.json")):
            res["by_class"][c][f.stem] = score(load(*f.stem.rsplit("_s", 1)[:1], int(f.stem.rsplit("_s", 1)[1])), ref, S_T_KEYS, idx)
    (OUT / "stats.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["tests"], indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True, choices=["A", "B", "C", "H5", "stats"])
    ap.add_argument("--specs", default=",".join(SPECS))
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    specs = [s for s in a.specs.split(",") if s]
    specs = [s for s in specs if SPECS[s][0] is None or seeds_of(SPECS[s][0])]
    if a.regime == "A":
        regime_A(device, specs)
    elif a.regime == "B":
        regime_B(device, specs)
    elif a.regime == "C":
        regime_C(device, specs)
    elif a.regime == "H5":
        regime_H5(device)
    else:
        return stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
