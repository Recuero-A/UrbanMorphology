"""Milestone 12: run the training plan, two jobs at a time, each resumable (``--resume latest``).

    python scripts/m12_train_all.py          # re-running continues unfinished runs from their latest checkpoint

Plan (docs/M12_EVALUATION_PROTOCOL.md §2): N0 and TN with 2 seeds; terrain-channel ablations with 1 seed. A run
with outputs/m12/training/<name>/summary.json is complete and skipped.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
JOBS = [("N0_s0", 0, ""), ("TN_s0", 0, "elev,slope,water"), ("N0_s1", 1, ""), ("TN_s1", 1, "elev,slope,water"),
        ("TNelev_s0", 0, "elev"), ("TNslope_s0", 0, "slope"), ("TNwater_s0", 0, "water"), ("TNelevslope_s0", 0, "elev,slope")]
PARALLEL = 2


def main() -> int:
    todo = [j for j in JOBS if not (ROOT / "outputs" / "m12" / "training" / j[0] / "summary.json").exists()]
    running = []
    while todo or running:
        while todo and len(running) < PARALLEL:
            name, seed, terr = todo.pop(0)
            log = open(ROOT / "outputs" / "m12" / "training" / f"{name}.out", "a")
            cmd = [PY, str(ROOT / "scripts" / "m12_train.py"), "--name", name, "--seed", str(seed), "--resume", "latest"]
            if terr:
                cmd += ["--terrain", terr]
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
            print(time.strftime("%H:%M"), "start", name, flush=True)
        for name, p in list(running):
            if p.poll() is not None:
                print(time.strftime("%H:%M"), "done", name, "exit", p.returncode, flush=True)
                running.remove((name, p))
        time.sleep(20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
