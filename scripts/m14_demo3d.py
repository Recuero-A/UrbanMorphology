"""Milestone 14 3D demonstration cities (downstream illustration; not part of any road-generation result).

Every method of the confirmatory set on IDENTICAL difficult synthetic terrain (mountain ridges, deep river valley,
mountain pass): same backbone (TN_s0), same sampling seed, first sample (no cherry-picking). Streets are passed to the
M12 city builder (BF parcels and buildings, M13 downstream fixes), rendered with Blender, and the oblique renders are
assembled into one contact sheet. Synthetic terrain has no real street network, so there is no "real" column.

    python scripts/m14_demo3d.py [--scenarios mountain_ridges:0 deep_valley:0 mountain_pass:0] [--seed 1] [--temperature 0.5]

--temperature is the displacement temperature of the base model (SamplerConfig.temperature; every confirmatory run used
the M12 reference 0.5). It does not affect P. Non-default temperatures are tagged _T<value> in every output name.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import m13_experiments as ex  # noqa: E402
from m14_plans import methods  # noqa: E402
from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.m13_metrics import tile_metrics_m13
from urbanmorph.graphs.m14_metrics import tile_metrics_m14
from urbanmorph.graphs.m14_sampler import NET_PRESETS, net_preset, sample_net
from urbanmorph.graphs.tc_potentials import Potentials, TerrainField
from urbanmorph.graphs.tc_sampler import preset, sample_tc
from urbanmorph.graphs.terrain_generate import terrain_of
from urbanmorph.terrain.synthesis import generate, scenario

OUT = PROJECT_ROOT / "outputs" / "m14" / "cities"
FZ = json.loads((PROJECT_ROOT / "configs" / "m14_frozen.json").read_text())
SHOW = ["P", "A", "M*", "M*+repair", "M13-SMC", "NA-myopic", "NA-fixed", "NA-SMC"]
SAFE = {"M*": "Mstar", "M*+repair": "Mstar-repair"}
PY = PROJECT_ROOT / ".venv" / "bin" / "python"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def terrain(name, seed):
    t = generate(scenario(name, seed), -256, -256, 768, 768)
    return {"elev": t.elev, "valid": np.ones(t.elev.shape, bool), "water": t.water_kind, "water_surface": t.water_surface}


def sample_all(d, seed, temperature, device="cpu"):
    from m12_evaluate import TH, density_pool, real_graphs
    from urbanmorph.graphs.procedural_terrain import sample_P
    from urbanmorph.graphs.repair import repair
    model, names, norm, _ = ex.load_backbone("TN_s0", device)
    pots = Potentials.load(PROJECT_ROOT / "configs" / "m13_potentials" / "TN.json")
    spec = {lab: (pre, ov, extra) for lab, pre, ov, extra in methods(FZ)}
    out = {}
    for lab in SHOW:
        pre, ov, extra = spec[lab]
        ov = {**ov, "temperature": temperature}
        if pre == "P":
            out[lab] = sample_P(terrain_of(d), np.random.default_rng(seed), float(np.median(density_pool(real_graphs()))), TH)
            continue
        if pre in NET_PRESETS:
            gs, _, _ = sample_net(model, [d], names, norm, pots, net_preset(pre, **ov), seed=seed, device=device)
        else:
            gs, _, _ = sample_tc(model, [d], names, norm, pots, preset(pre, **ov), seed=seed, device=device)
        out[lab] = repair(gs[0], terrain_of(d), TH)[0] if extra.get("repair") else gs[0]
    return out


def metrics(g, d, seed):
    tf, rng = TerrainField(terrain_of(d)), np.random.default_rng(1000 + seed)
    m = {**tile_metrics_m13(g, tf, rng), **tile_metrics_m14(g, tf, rng)}
    return {"edges": int(len(g.edges)), "bridges": int((np.asarray(g.structs()) == 1).sum()),
            "od_unreach_land": m["od_unreach_land"], "islands": m["islands"], "dead_end_ratio": m["dead_end_ratio_m13"]}


def build(name, sseed, lab, pkl, seed, sfx):
    key = SAFE.get(lab, lab) + sfx
    r = subprocess.run([str(PY), "scripts/m12_build_city3d.py", "--scenario", name, "--scenario-seed", str(sseed),
                        "--model", "TN_s0", "--seed", str(seed), "--graph", str(pkl), key, key, "--m13-fixes",
                        "--out", str(OUT)], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    dd = Path(json.loads(r.stdout.strip().splitlines()[-1])["dir"])
    with open(dd / "blender.log", "w") as f:
        b = subprocess.run([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python",
                            "blender/m11_scene.py", "--", "--tile", str(dd), "--res", "1400"], cwd=PROJECT_ROOT,
                           stdout=f, stderr=subprocess.STDOUT)
    if b.returncode:
        raise RuntimeError(f"blender failed for {dd}")
    return dd


def sheet(rows, view, seed, temperature, sfx, tag):
    fig, ax = plt.subplots(len(rows), len(SHOW), figsize=(3.6 * len(SHOW), 2.35 * len(rows)), layout="constrained")
    ax = np.atleast_2d(ax)
    for r, (name, cells) in enumerate(rows):
        for c, lab in enumerate(SHOW):
            dd, m = cells[lab]
            a = ax[r, c]
            a.imshow(plt.imread(dd / "renders" / f"{view}.png"))
            a.set_xticks([]), a.set_yticks([])
            a.set_title(f"{lab}\nOD {m['od_unreach_land']:.2f} · isl {m['islands']} · dead {m['dead_end_ratio']:.2f} · "
                        f"bridges {m['bridges']}", fontsize=8, fontweight="bold" if lab == "NA-SMC" else "normal")
            if c == 0:
                a.set_ylabel(name.replace("_", " "), fontsize=10)
    fig.suptitle(f"M14 methods on identical synthetic terrain (TN_s0, sampling seed {seed}, temperature {temperature}, first sample, {view} view). "
                 "OD = same-land unreachable pairs, isl = isolated components, dead = dead-end ratio", fontsize=11)
    p = OUT / f"sheet_{view}_{tag}_s{seed}{sfx}.png"
    fig.savefig(p, dpi=100)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["mountain_ridges:0", "deep_valley:0", "mountain_pass:0"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.5)
    a = ap.parse_args()
    sfx = "" if a.temperature == 0.5 else f"_T{a.temperature:g}"
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {}
    for sc in a.scenarios:
        name, sseed = sc.split(":")
        d = terrain(name, int(sseed))
        gs = sample_all(d, a.seed, a.temperature)
        pkl = OUT / f"graphs_{name}{sseed}_s{a.seed}{sfx}.pkl"
        pkl.write_bytes(pickle.dumps({SAFE.get(l, l) + sfx: g for l, g in gs.items()}))
        cells = {}
        for lab, g in gs.items():
            m = metrics(g, d, a.seed)
            cells[lab] = (build(name, int(sseed), lab, pkl, a.seed, sfx), m)
            summary[f"{name}{sseed}/{lab}"] = {**m, "dir": str(cells[lab][0].relative_to(PROJECT_ROOT))}
            print(name, lab, m, flush=True)
        rows.append((name, cells))
    tag = "-".join(sc.replace(":", "") for sc in a.scenarios)   # one sheet per scenario set, never overwritten
    (OUT / f"summary_{tag}_s{a.seed}{sfx}.json").write_text(json.dumps(summary, indent=1, default=float))
    for view in ("oblique", "aerial"):
        print(sheet(rows, view, a.seed, a.temperature, sfx, tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
