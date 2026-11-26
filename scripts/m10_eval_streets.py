"""Milestone 10: evaluate the street generators exactly as pre-registered (docs/M10_EVALUATION_PROTOCOL.md).

    python scripts/m10_eval_streets.py --sample A        # one model, both seeds -> outputs/m10/streets/samples/
    python scripts/m10_eval_streets.py --sample C
    python scripts/m10_eval_streets.py --report          # all sampled models -> outputs/m10/streets/eval.json + figures

Models: A (M6 roads v2), E (procedural), B (city), N (noise), C (stage 1 -> stage 2), CN (C with noise),
C_oracle (stage 2 on the REAL validation major + mid networks: conditional, exploratory).
"_w" models (A_w, B_w, C_w, C_oracle_w) were trained with learning-rate warmup after the first B / C runs collapsed
(protocol §6.1); B_w and C_w must pass P1-P4 against both A and A_w. CN_w (C with N's noise, warmup) is
EXPLORATORY: N failed P3, so the pre-registered CN was not run; its criteria are computed but never a pass.
256 tiles per seed, sampling seed 123, CPU. Reference: the 19 validation tiles. Test tiles are never read.
"""
import argparse
import json
import pickle
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

from m6_build_roads import load_graphs  # noqa: E402
from m10_street_diagnosis import KEYS, junctions  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.data.m2_dataset import load_index_with_expansion
from urbanmorph.graphs.generate import sample_graphs
from urbanmorph.graphs.procedural import sample_procedural
from urbanmorph.graphs.road_graph import TILE_M, graph_metrics, subgraph
from urbanmorph.models.graph_gen import GraphTurtle
from urbanmorph.models.vae import count_parameters
from urbanmorph.visualization.layout_plot import INK, SURFACE

OUT = PROJECT_ROOT / "outputs" / "m10" / "streets"
SAMPLES = OUT / "samples"
CK = PROJECT_ROOT / "checkpoints"
N_TILES, SAMPLE_SEED, N_BOOT = 256, 123, 1000
MODELS = ["A", "E", "B", "N", "C", "CN", "C_oracle", "A_w", "B_w", "C_w", "C_oracle_w", "CN_w"]
CANDIDATES = ["B", "N", "C", "CN", "B_w", "C_w"]


def load_model(path: Path) -> GraphTurtle:
    ck = torch.load(path / "model.pt", map_location="cpu")
    mc = ck["config"]["model"]
    cities = ck.get("cities")
    m = GraphTurtle(mc["hidden"], mc["n_mix"], turtle=mc["turtle"], n_classes=mc["n_classes"],
                    n_cond=len(cities) if cities else 0)
    m.load_state_dict(ck["model"])
    m.cities = cities
    return m.eval()


def city_info():
    idx = load_index_with_expansion()
    return dict(zip(idx.sample_id, idx.city))


def sample_model(name: str, seed: int, data, city_of, temp: float = 1.0) -> dict:
    t0 = time.time()
    states, cities, params = [], None, None
    w = "_w" if name.endswith("_w") else ""
    name = name.removesuffix("_w")
    if name == "E":
        links = {}
        for sid, _, g in data["train"]:
            v = graph_metrics(g)["mean_link_length_m"]
            if np.isfinite(v):
                links.setdefault(city_of[sid], []).append(v)
        gs, cities = sample_procedural(N_TILES, links, seed=SAMPLE_SEED + seed)
    elif name == "A":
        m = load_model(CK / "m10" / f"A_w_s{seed}" if w else CK / "m6" / f"roads_v2_s{seed}")
        params = count_parameters(m)
        gs = sample_graphs(m, N_TILES, seed=SAMPLE_SEED, states_out=states, temperature=temp)
    elif name in ("B", "N"):
        m = load_model(CK / "m10" / f"{ {'B': 'B_city', 'N': 'N_noise'}[name]}{w}_s{seed}")
        params = count_parameters(m)
        cond = None
        if name == "B":
            train_c = [city_of[sid] for sid, _, _ in data["train"]]
            rng = np.random.default_rng(SAMPLE_SEED)
            cities = [train_c[i] for i in rng.integers(len(train_c), size=N_TILES)]   # training proportions
            cond = np.eye(len(m.cities), dtype=np.float32)[[m.cities.index(c) for c in cities]]
        gs = sample_graphs(m, N_TILES, seed=SAMPLE_SEED, states_out=states, cond=cond, temperature=temp)
    else:
        sfx = "_noise" if name == "CN" else ""
        m2 = load_model(CK / "m10" / f"C2_local{sfx}{w}_s{seed}")
        if name == "C_oracle":
            val = [subgraph(g, g.edge_class <= 1) for _, _, g in data["val"]]
            given = [val[i % len(val)] for i in range(N_TILES)]
            params = count_parameters(m2)
        else:
            m1 = load_model(CK / "m10" / f"C1_major{sfx}{w}_s{seed}")
            params = count_parameters(m1) + count_parameters(m2)
            given = sample_graphs(m1, N_TILES, seed=SAMPLE_SEED, temperature=temp)
        gs = sample_graphs(m2, N_TILES, seed=SAMPLE_SEED + 1, states_out=states, given=given, temperature=temp)
    J = [j for s in states for j in junctions(s)]
    mech = {k: {"share": float(np.mean([x == k for x, _, _ in J])), "median_dev_deg": float(np.median([d for x, d, _ in J if x == k]))}
            for k in sorted({x for x, _, _ in J})} if J else None
    rules = {k: float(np.mean([s.rules[k] for s in states])) for k in states[0].rules} if states else None
    return {"graphs": gs, "cities": cities, "seconds": time.time() - t0, "params": params, "mechanisms": mech,
            "rules_per_tile": rules, "hit_max_nodes": int(sum(s.n >= 1000 for s in states))}


def spearman(a, b):
    r = lambda x: np.argsort(np.argsort(x)).astype(float)
    return float(np.corrcoef(r(np.asarray(a)), r(np.asarray(b)))[0, 1])


def w1(a, b):
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def metric_table(graphs):
    rows = [graph_metrics(g) for g in graphs]
    return {k: np.array([r[k] for r in rows], float) for k in KEYS}


def w1_iqr_all(gen, ref, iqr):
    out = {}
    for k in KEYS:
        a, b = gen[k][np.isfinite(gen[k])], ref[k][np.isfinite(ref[k])]
        out[k] = w1(a, b) / iqr[k] if len(a) and len(b) else float("nan")
    return out


def draw(ax, g, title=None):
    lw = {0: 2.2, 1: 1.6, 2: 0.9, 3: 0.6}
    col = {0: "#1f3a5f", 1: "#3d6fb6", 2: "#555555", 3: "#999999"}
    for (a, b), c in zip(g.edges, g.classes()):
        ax.plot(*g.nodes[[a, b]].T, color=col[int(c)], lw=lw[int(c)], solid_capstyle="round")
    ax.set_xlim(0, TILE_M)
    ax.set_ylim(0, TILE_M)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, color=INK, loc="left")


def report(data, city_of):
    val = [g for _, _, g in data["val"]]
    ref = metric_table(val)
    iqr = {k: (np.subtract(*np.nanpercentile(ref[k], [75, 25])) or 1.0) for k in KEYS}
    train_city_phi = {}
    for sid, _, g in data["train"]:
        train_city_phi.setdefault(city_of[sid], []).append(graph_metrics(g)["orientation_phi"])
    rng = np.random.default_rng(0)
    boot = rng.integers(len(val), size=(N_BOOT, len(val)))
    res = {"reference": {"n_val_tiles": len(val), "val_median": {k: float(np.nanmedian(ref[k])) for k in KEYS},
                         "val_iqr": iqr}, "models": {}}
    tables = {}
    temps = sorted({f.stem.rsplit("_s", 1)[0] for f in SAMPLES.glob("*_T*_s*.pkl")})
    for name in MODELS + temps:
        for seed in (0, 1):
            f = SAMPLES / f"{name}_s{seed}.pkl"
            if not f.exists():
                continue
            s = pickle.loads(f.read_bytes())
            tab = metric_table(s["graphs"])
            tables[(name, seed)] = tab
            wi = w1_iqr_all(tab, ref, iqr)
            r = {"S10": float(np.nanmean(list(wi.values()))), "w1_iqr": wi,
                 "median": {k: float(np.nanmedian(tab[k])) for k in KEYS},
                 "validity": {"share_nonempty": float(np.mean([len(g.edges) > 0 for g in s["graphs"]])),
                              "hit_max_nodes": s["hit_max_nodes"]},
                 "diversity_iqr_ratio": {k: float(np.subtract(*np.nanpercentile(tab[k], [75, 25])) / iqr[k])
                                         for k in ("orientation_phi", "intersections_per_km2")},
                 "junction_mechanisms": s["mechanisms"], "rules_per_tile": s["rules_per_tile"],
                 "params": s["params"], "sampling_s_per_tile": s["seconds"] / N_TILES}
            if s["cities"] is not None:
                gc = {}
                for c, v in zip(s["cities"], tab["orientation_phi"]):
                    gc.setdefault(c, []).append(v)
                cs = sorted(gc)
                rho = spearman([np.nanmedian(gc[c]) for c in cs], [np.nanmedian(train_city_phi[c]) for c in cs])
                r["city_phi_spearman_vs_train"] = float(rho)
                r["city_phi_median"] = {c: float(np.nanmedian(gc[c])) for c in cs}
            res["models"][f"{name}_s{seed}"] = r
    # pre-registered criteria: paired bootstrap of S10 against A (and A_w for "_w" candidates) of the same seed
    def criteria(name, seed, base):
        d = []
        for I in boot:
            rb = {k: ref[k][I] for k in KEYS}
            sc = np.nanmean(list(w1_iqr_all(tables[(name, seed)], rb, iqr).values()))
            sa = np.nanmean(list(w1_iqr_all(tables[(base, seed)], rb, iqr).values()))
            d.append(sc - sa)
        r, ra = res["models"][f"{name}_s{seed}"], res["models"][f"{base}_s{seed}"]
        ci = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        crit = {"P1_delta_ci_below_0": ci[1] < 0, "P2_S10_le_0.8A": r["S10"] <= 0.8 * ra["S10"],
                "P3_junction_dev_le_8": r["median"]["junction_angle_dev_deg"] <= 8.0,
                "P4_no_metric_worse_0.25": all(r["w1_iqr"][k] <= ra["w1_iqr"][k] + 0.25 for k in KEYS)}
        if name.startswith("B"):
            crit["B_city_spearman_ge_0.7"] = r.get("city_phi_spearman_vs_train", -1) >= 0.7
        return {"point": r["S10"] - ra["S10"], "ci95": ci}, {k: bool(v) for k, v in crit.items()}

    for name in CANDIDATES + ["E", "C_oracle", "A_w", "C_oracle_w", "CN_w"]:
        for seed in (0, 1):
            if (name, seed) not in tables or ("A", seed) not in tables:
                continue
            r = res["models"][f"{name}_s{seed}"]
            r["delta_S10_vs_A"], r["criteria"] = criteria(name, seed, "A")
            ok = all(r["criteria"].values())
            if name.endswith("_w") and name != "A_w" and ("A_w", seed) in tables:
                r["delta_S10_vs_A_w"], r["criteria_vs_A_w"] = criteria(name, seed, "A_w")
                ok = ok and all(r["criteria_vs_A_w"].values())
            r["pass"] = bool(ok) if name in CANDIDATES else None
    (OUT / "eval.json").write_text(json.dumps(res, indent=1))

    # figures: 4 samples per model (seed 0) + 4 real validation tiles
    rows = [("real val", val[:4])] + [(n, pickle.loads((SAMPLES / f"{n}_s0.pkl").read_bytes())["graphs"][:4])
                                      for n in MODELS if (SAMPLES / f"{n}_s0.pkl").exists() and n not in ("B", "C", "C_oracle")]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9, 2.3 * len(rows)), facecolor=SURFACE)
    for (lab, gs), ax_row in zip(rows, np.atleast_2d(axes)):
        for k, (g, ax) in enumerate(zip(gs, ax_row)):
            draw(ax, g, lab if k == 0 else None)
    fig.tight_layout()
    fig.savefig(OUT / "samples_grid.png", dpi=120, facecolor=SURFACE)
    names = [k for k in res["models"]]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(names) + 1), facecolor=SURFACE)
    ax.barh(names, [res["models"][k]["S10"] for k in names], color="#3d6fb6")
    ax.axvline(0.8 * res["models"]["A_s0"]["S10"], color=INK, ls="--", lw=0.8)
    ax.set_xlabel("S10 = mean W1/IQR vs real validation (lower is better); dashed = 0.8 x A", fontsize=8)
    ax.invert_yaxis()
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "s10.png", dpi=130, facecolor=SURFACE)
    for k in names:
        r = res["models"][k]
        print(f"{k:12s} S10 {r['S10']:.3f}  jdev {r['median']['junction_angle_dev_deg']:5.1f}  "
              f"int/km2 {r['median']['intersections_per_km2']:6.1f}  phi {r['median']['orientation_phi']:.2f}  "
              f"{'' if 'criteria' not in r else r['criteria']}"
              f"{'' if 'criteria_vs_A_w' not in r else '  vs A_w ' + str(r['criteria_vs_A_w'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=MODELS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--temp", type=float, default=1.0, help="exploratory: sampling temperature (saved as <model>_T<t>)")
    a = ap.parse_args()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    data, city_of = load_graphs(), city_info()
    torch.set_num_threads(2)
    if a.sample:
        for seed in (0, 1):
            s = sample_model(a.sample, seed, data, city_of, a.temp)
            tag = a.sample if a.temp == 1.0 else f"{a.sample}_T{a.temp:g}".replace(".", "")
            (SAMPLES / f"{tag}_s{seed}.pkl").write_bytes(pickle.dumps(s))
            print(a.sample, seed, f"{s['seconds']:.0f}s", flush=True)
    if a.report:
        report(data, city_of)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
