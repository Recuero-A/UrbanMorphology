"""Milestone 18, decision D7 (c): train the TNg backbone (M12 TN on Copernicus GLO-30 terrain inputs).

    python scripts/m18_train_tng.py --seed 0 --resume latest     # start, or continue after any interruption
    python scripts/m18_train_tng.py --seed 0 --smoke 1 --smoke-dir DIR   # pilot: 1 epoch, 64 train / 16 val tiles, in DIR

This runs scripts/m12_train.py's own ``main()`` unchanged (same model, recipe, splits, seeds, checkpointing and
resume). The wrapper only points it at configs/m18_tng.toml, whose [model] / [train] / [sample] must equal
configs/m6_roads.toml (checked), and replaces its data loader so that:
* tiles with LiDAR terrain in M12 (Spain, US) read data/processed/m18/terrain_glo30/<sid>.npz (GLO-30 elev / valid;
  water and structure labels checked identical to M12);
* tiles without a DEM (Japan, valid = 0) read their unchanged M12 arrays, exactly as TN did;
* the channel normalisation is data/processed/m18/terrain_glo30/norm.json (the M12 recipe on the GLO-30 training
  tiles; scripts/m18_build_glo30_terrain.py). It is stored in every checkpoint, as for TN.
The run config records ``dem_variant = "glo30"`` (part of the checkpoint's config hash: a resume with other inputs is
refused). Checkpoints: checkpoints/m12/TNg_s<seed>/ (new; an existing directory is only ever resumed, never
restarted); logs: outputs/m18/glo30/training/TNg_s<seed>/.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import m12_train as T  # noqa: E402

CFG = ROOT / "configs" / "m18_tng.toml"
BASE = ROOT / "configs" / "m6_roads.toml"


def glo30_loader(tc: dict):
    terrain_dir, fallback, graphs_p = ROOT / tc["terrain_dir"], ROOT / tc["fallback_dir"], ROOT / tc["graphs"]

    def load_data(dem_variant: str = ""):
        assert dem_variant == tc["dem_variant"], dem_variant
        graphs = pickle.loads(graphs_p.read_bytes())
        assert set(graphs) <= {"train", "val", "val_t"}, set(graphs)        # development splits only
        terr, n_glo, n_none = {}, 0, 0
        for sp in graphs:
            for sid, _, _ in graphs[sp]:
                lid = dict(np.load(fallback / f"{sid}.npz"))
                f = terrain_dir / f"{sid}.npz"
                if lid["valid"].any():
                    if not f.exists():
                        raise FileNotFoundError(f"{f}: LiDAR terrain in M12 but no GLO-30 array "
                                                "(run scripts/m18_build_glo30_terrain.py)")
                    d = dict(np.load(f))
                    assert set(d) == set(lid), (sid, set(d), set(lid))
                    assert np.array_equal(d["water"], lid["water"]) and np.array_equal(d["struct_label"], lid["struct_label"])
                    terr[sid] = d
                    n_glo += 1
                else:
                    assert not f.exists(), f
                    terr[sid] = lid
                    n_none += 1
        print(f"[TNg] terrain: {n_glo} tiles from GLO-30 ({terrain_dir.relative_to(ROOT)}), {n_none} tiles without "
              f"a DEM (unchanged M12 arrays)", flush=True)
        return graphs, terr, json.loads((terrain_dir / "norm.json").read_text())

    return load_data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--resume", metavar="latest|PATH", default=None)
    ap.add_argument("--ckpt-interval", type=float, default=None)
    ap.add_argument("--smoke", type=int, default=0, help="pilot epochs on 64 train / 16 val tiles (writes to --smoke-dir)")
    ap.add_argument("--smoke-dir", default=None)
    a = ap.parse_args()
    cfg, base = tomllib.loads(CFG.read_text()), tomllib.loads(BASE.read_text())
    for k in ("model", "train", "sample"):
        assert cfg[k] == base[k], f"[{k}] of {CFG.name} differs from {BASE.name}: TNg must keep the TN recipe"
    tc = cfg["tng"]
    assert a.seed in tc["seeds"], a.seed
    name = tc["name"].format(seed=a.seed)
    assert name.startswith("TNg_"), name
    T.CFG = CFG
    T.load_data = glo30_loader(tc)
    if a.smoke:
        assert a.smoke_dir, "--smoke needs --smoke-dir (pilot checkpoints never go to checkpoints/)"
        T.CKPT, T.RUNS = Path(a.smoke_dir) / "checkpoints", Path(a.smoke_dir) / "runs"
    else:
        T.CKPT, T.RUNS = ROOT / tc["checkpoints"], ROOT / tc["runs"]
        if (T.CKPT / name).exists() and not a.resume:
            sys.exit(f"{T.CKPT / name} exists: continue it with --resume latest (it is never restarted)")
    argv = ["m12_train.py", "--name", name, "--seed", str(a.seed), "--terrain", tc["terrain"],
            "--warmup", str(tc["warmup"]), "--dem-variant", tc["dem_variant"]]
    if tc["lowpass_m"]:
        argv += ["--lowpass", str(tc["lowpass_m"])]
    if a.resume:
        argv += ["--resume", a.resume]
    if a.ckpt_interval:
        argv += ["--ckpt-interval", str(a.ckpt_interval)]
    if a.smoke:
        argv += ["--pilot", str(a.smoke)]
    print("[TNg]", " ".join(argv[1:]), "| ckpt", T.CKPT / name, "| runs", T.RUNS / name, flush=True)
    sys.argv = argv
    return T.main()


if __name__ == "__main__":
    sys.exit(main())
