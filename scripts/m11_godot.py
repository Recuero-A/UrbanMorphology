"""Milestone 11: Godot 4 import / collision check (headless) and frame-rate benchmark (windowed) per tile.

    python scripts/m11_godot.py --tiles ESP_BCN_e837_n8952 gen_N_T05_s2026_00_spain [--bench 20]

Writes outputs/m11/<tile>/godot_check.json and godot_fps.json. GODOT env var overrides the binary path.
The benchmark opens a window for --bench seconds (a fixed aerial orbit, then a street-level glide).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanmorph.config import PROJECT_ROOT

GODOT = os.environ.get("GODOT", str(Path.home() / "Downloads" / "Godot.app" / "Contents" / "MacOS" / "Godot"))
VIEWER = PROJECT_ROOT / "godot" / "viewer"
OUT = PROJECT_ROOT / "outputs" / "m11"


def run(tile: str, mode: list, out: Path, headless: bool, timeout: int) -> dict:
    d = OUT / tile
    cmd = [GODOT] + (["--headless"] if headless else []) + ["--path", str(VIEWER), "--", "--glb", str(d / "scene.glb"),
                                                             "--viewer", str(d / "viewer.json"), *mode, "--out", str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"godot failed for {tile}: {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", required=True)
    ap.add_argument("--bench", type=float, default=20.0)
    ap.add_argument("--no-bench", action="store_true")
    a = ap.parse_args()
    subprocess.run([GODOT, "--headless", "--path", str(VIEWER), "--import"], capture_output=True, timeout=300)
    for t in a.tiles:
        chk = run(t, ["--check"], OUT / t / "godot_check.json", True, 300)
        print(t, "check", json.dumps(chk), flush=True)
        if not a.no_bench:
            fps = run(t, ["--benchmark", str(a.bench)], OUT / t / "godot_fps.json", False, int(a.bench) + 180)
            print(t, "fps", json.dumps(fps), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
