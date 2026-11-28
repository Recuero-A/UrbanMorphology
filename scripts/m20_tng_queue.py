"""Milestone 20, D1: run the remaining backbone trainings on MPS, two at a time (docs/M20_LOG.md D1-impl).

    nohup python -u scripts/m20_tng_queue.py > logs/m20_tng_queue.log 2>&1 &

Each job is scripts/m20_train_tng.py --variant V --seed S --resume latest, logging to logs/m20_tng_TNg<V>_s<S>.log;
a job whose outputs/m20/tng/training/<name>/summary.json exists is done and skipped; a job already running (its
checkpoint lock / a live process) is waited for. TNg-R jobs start only once data/processed/m20/tng/towns_R.ready
exists (written after the TNg-R windows are built and logged); until then the next other job runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "m20" / "tng" / "training"
READY_R = ROOT / "data" / "processed" / "m20" / "tng" / "towns_R.ready"
JOBS = [("32", 1), ("H", 1), ("32", 2), ("H", 2), ("R", 0), ("R", 1), ("R", 2), ("H25", 0), ("H50", 0)]
SLOTS = 2


def name(v, s):
    return f"TNg{v}_s{s}"


def running_elsewhere():
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return {name(l.split("--variant ")[1].split()[0], int(l.split("--seed ")[1].split()[0]))
            for l in out.splitlines() if "m20_train_tng.py" in l and "--variant" in l and "--smoke" not in l}


def main() -> int:
    procs = {}
    todo = list(JOBS)
    while todo or procs:
        for n, p in list(procs.items()):
            if p.poll() is not None:
                print(time.strftime("%H:%M:%S"), n, "exited", p.returncode, flush=True)
                del procs[n]
        todo = [j for j in todo if not (RUNS / name(*j) / "summary.json").exists()]
        busy = running_elsewhere()
        free = SLOTS - len(busy)
        for j in list(todo):
            if free <= 0:
                break
            n = name(*j)
            if n in busy or n in procs:
                continue
            if j[0] == "R" and not READY_R.exists():
                continue
            log = open(ROOT / "logs" / f"m20_tng_{n}.log", "a")
            procs[n] = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts" / "m20_train_tng.py"), "--variant", j[0],
                                         "--seed", str(j[1]), "--resume", "latest"], cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "2"})
            print(time.strftime("%H:%M:%S"), n, "started pid", procs[n].pid, flush=True)
            todo.remove(j)
            free -= 1
        if not todo and not procs:
            break
        time.sleep(30)
    print("queue done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
