"""Milestone 3b: data-size control for the spatial VAE (B trained on the ORIGINAL 97 training tiles).

Separates the architecture effect from the data effect: baseline (global, 97) vs A (global, 565)
vs B_orig97 (spatial, 97) vs B (spatial, 565). Same config as B except include_expansion_m3b = false.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m3b_experiments import merged_configs  # noqa: E402
from urbanmorph.models.train import CHECKPOINT_DIR, train  # noqa: E402

cfg = {c["experiment"]["name"]: c for c in merged_configs()}["B_spatial16_exp"]
cfg["data"]["include_expansion_m3b"] = False
cfg["experiment"]["name"] = "B_spatial16_orig97"
cfg["experiment"]["description"] = "Control: Experiment B architecture/loss on the original 97 training tiles"
if not (CHECKPOINT_DIR / cfg["experiment"]["name"] / "best.pt").exists():
    s = train(cfg, quiet=True)
    print(f"B_spatial16_orig97: best epoch {s['best_epoch']}/{s['epochs_run']}, {s['train_seconds']} s, {s['n_train']} train tiles", flush=True)
sys.exit(subprocess.call([sys.executable, str(Path(__file__).parent / "m3b_evaluate.py"), "B_spatial16_orig97"]))
