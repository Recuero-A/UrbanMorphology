"""Checkpoint / resume (src/urbanmorph/models/checkpoint.py): interrupted + resumed training must equal
uninterrupted training. The toy loop has the structure of the M10 road loop: per-epoch order, chunks whose
data are regenerated from a NumPy generator, batches, cosine LR with warmup, gradient clipping, dropout."""
import math
import os
import signal

import numpy as np
import pytest
import torch
from torch import nn

from urbanmorph.models.checkpoint import Checkpointer, RunLog

EPOCHS, TILES, CHUNK, ROWS, BATCH = 3, 8, 4, 32, 8
CFG = {"lr": 1e-2, "lr_min": 1e-3, "warmup": 4}


def run(d, stop_at=None, resume=None, cfg=CFG):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(0)
    ck = Checkpointer(d, config=cfg, interval_s=1e9, rngs={"data": rng}, handle_signals=False)
    st = ck.resume(resume, model, opt) or {"epoch": 1, "chunk": 0, "batch": 0, "step": 0, "chunk_rng": None, "order": None}
    log = RunLog(d / "log.jsonl")
    log.write(event="resume" if resume else "start", step=st["step"])
    for ep in range(st["epoch"], EPOCHS + 1):
        lr_ep = cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * (ep - 1) / EPOCHS))
        if st["order"] is None:
            st["order"] = rng.permutation(TILES).tolist()
        for c in range(st["chunk"], TILES, CHUNK):
            if st["chunk_rng"] is None:
                st["chunk_rng"] = rng.bit_generator.state
            else:
                rng.bit_generator.state = st["chunk_rng"]          # regenerate this chunk exactly
            x = torch.tensor(rng.normal(size=(ROWS, 4)), dtype=torch.float32)
            y = (x ** 2).sum(1, keepdim=True)
            perm = rng.permutation(ROWS)
            for b in range(st["batch"], ROWS, BATCH):
                for g in opt.param_groups:
                    g["lr"] = lr_ep * min(1.0, (st["step"] + 1) / cfg["warmup"])
                bi = perm[b:b + BATCH]
                loss = ((model(x[bi]) - y[bi]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                st["step"] += 1
                st["batch"] = b + BATCH
                if st["step"] == stop_at:
                    ck.stop_requested = True                      # what Ctrl+C / SIGTERM does
                if ck.due():
                    ck.save(st, model, opt)
                    if ck.stop_requested:
                        log.write(event="interrupted", step=st["step"])
                        return model, False
            st.update(batch=0, chunk=c + CHUNK, chunk_rng=None)
        st.update(epoch=ep + 1, chunk=0, order=None)
        ck.save(st, model, opt, tag="epoch")
        log.write(epoch=ep, step=st["step"])
    return model, True


def params(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


@pytest.mark.parametrize("stop_at", [1, 5, 8, 13, 23])          # mid-chunk, chunk end, epoch end, ...
def test_resume_equals_uninterrupted(tmp_path, stop_at):
    ref, done = run(tmp_path / "ref")
    assert done
    _, done = run(tmp_path / "cut", stop_at=stop_at)
    assert not done
    res, done = run(tmp_path / "cut", resume="latest")
    assert done
    assert torch.allclose(params(ref), params(res), rtol=0, atol=1e-7)


def test_twice_interrupted_and_log_appends(tmp_path):
    ref, _ = run(tmp_path / "ref")
    run(tmp_path / "cut", stop_at=3)
    run(tmp_path / "cut", stop_at=17, resume="latest")
    res, done = run(tmp_path / "cut", resume="latest")
    assert done and torch.allclose(params(ref), params(res), rtol=0, atol=1e-7)
    events = [r.get("event") for r in RunLog(tmp_path / "cut" / "log.jsonl").read()]
    assert events.count("interrupted") == 2 and events.count("resume") == 2 and events[0] == "start"


def test_corrupt_latest_falls_back_to_previous(tmp_path):
    run(tmp_path / "cut", stop_at=20)                              # epoch-end checkpoints exist, then a stop save
    (tmp_path / "cut" / "latest.pt").write_bytes(b"truncated")     # simulate a crash mid-write of a non-atomic file
    m = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
    ck = Checkpointer(tmp_path / "cut", config=CFG, rngs={"data": np.random.default_rng(0)}, handle_signals=False)
    st = ck.resume("latest", m, torch.optim.Adam(m.parameters()))
    assert st is not None and st["step"] < 20                     # the previous valid checkpoint


def test_config_change_refused(tmp_path):
    run(tmp_path / "cut", stop_at=5)
    with pytest.raises(ValueError):
        run(tmp_path / "cut", resume="latest", cfg={**CFG, "lr": 5e-3})


def test_no_checkpoint_starts_fresh(tmp_path):
    ref, _ = run(tmp_path / "ref")
    res, done = run(tmp_path / "empty", resume="latest")
    assert done and torch.equal(params(ref), params(res))


def test_signal_sets_stop_flag(tmp_path):
    ck = Checkpointer(tmp_path, config=CFG)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        assert ck.stop_requested and ck.due()
    finally:
        ck.restore_signal_handlers()
