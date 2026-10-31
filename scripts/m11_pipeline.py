"""Milestone 11: reproduce every 3D output with one command (build -> Blender -> Godot -> evaluation).

    python scripts/m11_pipeline.py                 # the six M11 tiles
    python scripts/m11_pipeline.py --no-godot      # skip the Godot check / benchmark (opens a window)

Order: m11_height_audit -> m11_height_priors -> m11_calibrate_heights must have run once (outputs/m11/audit/).
Blender: BLENDER env var or /Applications/Blender.app. Test tiles are refused by m11_build_city.py.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
TILES = [("real", "ESP_BCN_e837_n8952"), ("real", "USA_LAX_e758_n7370"), ("real", "USA_NYC_e1149_n8784"),
         ("real", "JPN_TYO_e764_n7739"), ("gen", "2026 0 spain"), ("gen", "2026 1 united_states")]


def sh(cmd, log=None):
    t = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if log:
        Path(log).write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    return round(time.time() - t, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-godot", action="store_true")
    ap.add_argument("--res", default="1600")
    a = ap.parse_args()
    out = ROOT / "outputs" / "m11"
    timing, names = {}, []
    for kind, spec in TILES:
        if kind == "real":
            name, args = spec, ["--real", spec]
        else:
            seed, idx, country = spec.split()
            name, args = f"gen_N_T05_s{seed}_{int(idx):02d}_{country}", ["--generated", seed, "--index", idx, "--country", country]
        names.append(name)
        t_build = sh([PY, "-W", "ignore", "scripts/m11_build_city.py", *args], out / f"build_{name}.log")
        t_bl = sh([BLENDER, "-b", "--factory-startup", "--python-exit-code", "1", "--python", "blender/m11_scene.py", "--",
                   "--tile", str(out / name), "--res", a.res], out / name / "blender.log")
        timing[name] = {"build_s": t_build, "blender_s": t_bl}
        print(name, timing[name], flush=True)
    if not a.no_godot:
        timing["godot_s"] = sh([PY, "scripts/m11_godot.py", "--tiles", *names, "--bench", "20"], out / "godot.log")
    timing["evaluate_s"] = sh([PY, "-W", "ignore", "scripts/m11_evaluate.py"], out / "evaluate.log")
    (out / "pipeline_timing.json").write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
