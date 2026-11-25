"""Milestone 16 experiment runner (CLI over urbanmorph.experiments.runner; docs/M16_PROTOCOL.md).

    python scripts/m16_run.py --plan configs/m16_plans/<name>.json [--worker k/n] [--threads 2]

Output: outputs/m16/runs/<run_id>.json, rewritten atomically after every seed (resumable; the id hashes the spec).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.experiments.runner import run

RUNS = PROJECT_ROOT / "outputs" / "m16" / "runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--worker", default="0/1")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    k, n = map(int, a.worker.split("/"))
    plan = json.loads(Path(a.plan).read_text())
    for j, spec in enumerate(plan["runs"]):
        if j % n == k:
            run(spec, RUNS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
