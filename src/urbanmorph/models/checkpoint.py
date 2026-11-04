"""Resumable training state for Milestone 12 (and any later training script).

A training script keeps everything that determines its future in one ``state`` dict and hands it to a
``Checkpointer``:

    ck = Checkpointer(CKPT / name, config=cfg, interval_s=300)
    state = ck.resume(a.resume, model, opt) or fresh_state()      # --resume latest | PATH | (none)
    ...
    for each batch:
        ... optimiser step ...
        if ck.due():                     # 5 minutes passed, or Ctrl+C / SIGTERM received
            ck.save(state, model, opt)
            if ck.stop_requested: return
    ck.save(state, model, opt, tag="epoch")                      # end of every epoch

What is saved: model and optimiser state, an optional LR scheduler and gradient scaler, the user ``state``
dict (epoch, global step, position inside the epoch, best score, history, ...), the RNG states (Python, NumPy
global, torch CPU and MPS / CUDA, and any ``numpy.random.Generator`` passed as ``rngs``), the configuration
and its SHA-256.

Crash safety: a checkpoint is written to a temporary file in the same directory, flushed and fsynced, then
moved into place with ``os.replace`` (atomic on POSIX). The previous ``latest.pt`` is first renamed to
``previous.pt``, so a crash at any moment leaves at least one complete checkpoint. ``resume`` tries
``latest.pt`` and falls back to ``previous.pt`` if the first one cannot be read or is incomplete.

Exact resumption: a script must be able to re-enter its loop at the saved position. The M10 road loop
regenerates each chunk's training traces from the NumPy generator, so a mid-epoch state records the
generator state *at the start of the current chunk* plus the batch index; on resume the chunk is
regenerated identically and the loop skips the batches already done (tests/test_checkpoint.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

FORMAT = 1


def config_hash(config) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def rng_state(rngs: dict | None = None) -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        s["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    s["generators"] = {k: g.bit_generator.state for k, g in (rngs or {}).items()}
    return s


def set_rng_state(s: dict, rngs: dict | None = None) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "mps" in s and torch.backends.mps.is_available():
        torch.mps.set_rng_state(s["mps"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    for k, g in (rngs or {}).items():
        g.bit_generator.state = s["generators"][k]


def atomic_save(obj, path: Path) -> None:
    """torch.save to ``path`` atomically; the file previously at ``path`` becomes ``previous.pt``."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        os.replace(path, path.with_name("previous.pt"))
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)                      # persist the directory entries too
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Checkpointer:
    def __init__(self, directory, config=None, interval_s: float = 300.0, rngs: dict | None = None,
                 handle_signals: bool = True):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.interval_s = interval_s
        self.rngs = rngs or {}
        self.last_save = time.monotonic()
        self.stop_requested = False
        self._old = {}
        if handle_signals:
            self.install_signal_handlers()

    # ------------------------------------------------------------------ signals
    def install_signal_handlers(self) -> None:
        """First Ctrl+C / SIGTERM: finish the current step, save, stop. A second one aborts immediately."""
        def handler(signum, frame):
            if self.stop_requested:
                self.restore_signal_handlers()
                raise KeyboardInterrupt
            self.stop_requested = True
            print(f"\n[checkpoint] signal {signum}: saving at the next safe point (repeat to abort)", flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._old[sig] = signal.signal(sig, handler)

    def restore_signal_handlers(self) -> None:
        for sig, h in self._old.items():
            signal.signal(sig, h)
        self._old = {}

    # ------------------------------------------------------------------ save / load
    def due(self) -> bool:
        return self.stop_requested or time.monotonic() - self.last_save >= self.interval_s

    def save(self, state: dict, model, optimizer=None, scheduler=None, scaler=None, tag: str = "time") -> Path:
        payload = {"format": FORMAT, "complete": True, "tag": tag, "saved_at": time.time(),
                   "config": self.config, "config_sha256": config_hash(self.config),
                   "model": model.state_dict(),
                   "optimizer": optimizer.state_dict() if optimizer is not None else None,
                   "scheduler": scheduler.state_dict() if scheduler is not None else None,
                   "scaler": scaler.state_dict() if scaler is not None else None,
                   "rng": rng_state(self.rngs), "state": state}
        path = self.dir / "latest.pt"
        atomic_save(payload, path)
        self.last_save = time.monotonic()
        return path

    def _read(self, path: Path):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:                                   # truncated or corrupt file
            print(f"[checkpoint] cannot read {path.name}: {type(e).__name__}", flush=True)
            return None
        if not (isinstance(ck, dict) and ck.get("complete") and ck.get("format") == FORMAT):
            print(f"[checkpoint] {path.name} is incomplete", flush=True)
            return None
        return ck

    def resume(self, which, model, optimizer=None, scheduler=None, scaler=None, allow_config_change=False):
        """``which``: None (fresh start), "latest", or a checkpoint path. Returns the saved ``state`` or None."""
        if not which:
            return None
        paths = [self.dir / "latest.pt", self.dir / "previous.pt"] if which == "latest" else [Path(which)]
        ck = next((c for c in (self._read(p) for p in paths if p.exists()) if c is not None), None)
        if ck is None:
            if which == "latest":
                print(f"[checkpoint] no valid checkpoint in {self.dir}: starting fresh", flush=True)
                return None
            raise FileNotFoundError(which)
        if ck["config_sha256"] != config_hash(self.config) and not allow_config_change:
            raise ValueError("checkpoint was written with a different configuration; refusing to resume")
        model.load_state_dict(ck["model"])
        for obj, key in ((optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler")):
            if obj is not None and ck[key] is not None:
                obj.load_state_dict(ck[key])
        set_rng_state(ck["rng"], self.rngs)
        self.last_save = time.monotonic()
        print(f"[checkpoint] resumed from {ck['tag']} checkpoint, step {ck['state'].get('step')}", flush=True)
        return ck["state"]


class RunLog:
    """Append-only JSON-lines log: resuming adds a ``resume`` event and continues, nothing is overwritten."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **row) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"t": time.time(), **row}, default=float) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()] if self.path.exists() else []


def add_resume_args(ap) -> None:
    ap.add_argument("--resume", metavar="latest|PATH", default=None,
                    help="continue from the most recent valid checkpoint ('latest') or from a checkpoint file")
    ap.add_argument("--ckpt-interval", type=float, default=300.0, help="seconds between time-based checkpoints")
