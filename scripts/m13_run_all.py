"""Milestone 13 unattended experiment runner (independent of any interactive session).

    python scripts/m13_run_all.py start  --config configs/m13_full.toml [--caffeinate]   # detached; returns at once
    python scripts/m13_run_all.py status                                                  # prints outputs/m13/STATUS.md
    python scripts/m13_run_all.py stop                                                    # graceful: SIGTERM -> checkpoint
    python scripts/m13_run_all.py resume --config configs/m13_full.toml [--caffeinate]   # = start; completed jobs skipped
    python scripts/m13_run_all.py run    --config configs/m13_full.toml                   # foreground worker (what start runs)

Guarantees
* Jobs run SEQUENTIALLY (never two large MPS jobs at once). Each job has a unique id, a command and a ``done`` marker
  (a file the job writes on success, e.g. a training summary.json). A job whose marker exists is skipped on every run.
* Training jobs use scripts/m12_train.py with ``--resume latest``: the trainer checkpoints every 5 min, at epoch ends
  and on SIGTERM (atomic writes, previous checkpoint kept, config hash checked), so an interrupted job resumes where it
  stopped instead of restarting.
* State: outputs/m13/runner/state.json (atomic write). Per job: config, git commit, start / end, exit status, attempts,
  log path, status in {completed, running, interrupted, failed, failed_oom, skipped, blocked}. Failed jobs are never
  removed from the record. Logs are appended (outputs/m13/runner/logs/<id>.log), never overwritten.
* Retries: at most ``max_attempts`` per job; an out-of-memory failure is not retried with the same configuration.
  A job whose ``after`` dependencies did not complete is ``blocked``; independent jobs continue.
* Budget: cumulative active compute is tracked; when ``budget_hours`` is reached the current job is stopped at a safe
  checkpoint (SIGTERM), a partial report is written and the runner exits with status 3.
* Resources: refuses to start a job with less than ``min_free_gb`` of free disk; records free memory per job.
* Sleep: training cannot run while the Mac sleeps. ``--caffeinate`` starts ``caffeinate -i -w <runner pid>``, which
  prevents IDLE sleep while the runner lives; it does NOT keep a closed-lid MacBook awake. Closing the lid suspends
  training; on wake the trainer simply continues; after a reboot run ``resume``.
Exit status: 0 all jobs completed; 2 some failed / blocked; 3 budget reached; 130 stopped.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "m13" / "runner"
STATE = RUN / "state.json"
PIDF = RUN / "runner.pid"
STATUS = ROOT / "outputs" / "m13" / "STATUS.md"
PY = str(ROOT / ".venv" / "bin" / "python")
_child = None
_stop = False


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"jobs": {}, "active_seconds": 0.0, "runs": []}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def free_gb():
    return shutil.disk_usage(ROOT).free / 2 ** 30


def free_mem_gb():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = 16384
        free = sum(int(l.split()[-1].rstrip(".")) for l in out.splitlines() if l.startswith(("Pages free", "Pages inactive")))
        return free * page / 2 ** 30
    except Exception:
        return float("nan")


def write_status(cfg, st, current=None):
    jobs = cfg["job"]
    lines = [f"# Milestone 13 runner status", "", f"Updated {now()} (UTC). Commit {git_commit()}.", ""]
    act = st.get("active_seconds", 0.0)
    budget = cfg["runner"].get("budget_hours", 36) * 3600
    lines.append(f"Active compute: {act / 3600:.2f} h of the {budget / 3600:.0f} h budget.")
    done = [j for j in jobs if st["jobs"].get(j["id"], {}).get("status") in ("completed", "skipped")]
    lines.append(f"Jobs: {len(done)} of {len(jobs)} completed.")
    if current:
        rec = st["jobs"].get(current, {})
        lines.append(f"Current: **{current}** (since {rec.get('start')}).")
        ck = latest_checkpoint(current, cfg)
        if ck:
            lines.append(f"Last checkpoint: {ck}")
    rem = [j for j in jobs if j not in done]
    est = sum(float(j.get("est_hours", 0.5)) for j in rem)
    lines.append(f"Estimated remaining: {est:.1f} h (from per-job estimates).")
    lines += ["", "| Job | Status | Attempts | Start | End | Exit |", "|---|---|---|---|---|---|"]
    for j in jobs:
        r = st["jobs"].get(j["id"], {})
        lines.append(f"| {j['id']} | {r.get('status', 'pending')} | {r.get('attempts', 0)} | {r.get('start', '')} | "
                     f"{r.get('end', '')} | {r.get('exit', '')} |")
    best = ROOT / "outputs" / "m13" / "best_dev_result.md"
    if best.exists():
        lines += ["", "## Current best development result", "", best.read_text()]
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text("\n".join(lines) + "\n")


def latest_checkpoint(job_id, cfg):
    j = next((x for x in cfg["job"] if x["id"] == job_id), None)
    if not j or "checkpoint_dir" not in j:
        return None
    p = ROOT / j["checkpoint_dir"] / "latest.pt"
    return f"{p.relative_to(ROOT)} ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%H:%M:%S')})" if p.exists() else None


def on_term(sig, frame):
    global _stop
    _stop = True
    if _child is not None and _child.poll() is None:
        _child.send_signal(signal.SIGTERM)                 # the trainer checkpoints on SIGTERM


def preflight(cfg, log):
    r = cfg["runner"]
    problems = []
    for mod in r.get("python_modules", ["torch", "numpy", "scipy", "shapely", "skimage"]):
        if subprocess.run([PY, "-c", f"import {mod}"], capture_output=True).returncode:
            problems.append(f"missing python module {mod}")
    for f in r.get("required_paths", []):
        if not (ROOT / f).exists():
            problems.append(f"missing {f}")
    if free_gb() < r.get("min_free_gb", 20):
        problems.append(f"only {free_gb():.1f} GB free")
    if r.get("run_tests", True):
        t = subprocess.run([PY, "-m", "pytest", "-q", "tests/"], cwd=ROOT, capture_output=True, text=True)
        log.write(f"[{now()}] tests: {t.stdout.strip().splitlines()[-1] if t.stdout.strip() else t.returncode}\n")
        if t.returncode:
            problems.append("unit tests failed")
    return problems


def run_worker(cfg_path: Path):
    global _child
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    cfg = tomllib.loads(cfg_path.read_text())
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    PIDF.write_text(str(os.getpid()))
    st = load_state()
    st["runs"].append({"start": now(), "pid": os.getpid(), "config": str(cfg_path), "commit": git_commit()})
    atomic_json(STATE, st)
    rlog = open(RUN / "runner.log", "a")
    rlog.write(f"[{now()}] runner start pid {os.getpid()} config {cfg_path}\n")
    problems = preflight(cfg, rlog)
    if problems:
        rlog.write(f"[{now()}] preflight failed: {problems}\n")
        st["runs"][-1]["preflight"] = problems
        atomic_json(STATE, st)
        write_status(cfg, st)
        return 2
    budget = cfg["runner"].get("budget_hours", 36) * 3600
    max_att = cfg["runner"].get("max_attempts", 2)
    code = 0
    for j in cfg["job"]:
        if _stop:
            code = 130
            break
        jid = j["id"]
        rec = st["jobs"].setdefault(jid, {"attempts": 0})
        if (ROOT / j["done"]).exists():
            if rec.get("status") != "completed":
                rec.update(status="completed" if rec.get("attempts") else "skipped", note="done marker present")
            atomic_json(STATE, st)
            continue
        deps = [d for d in j.get("after", []) if st["jobs"].get(d, {}).get("status") not in ("completed", "skipped")]
        if deps:
            rec.update(status="blocked", note=f"waiting for {deps}")
            atomic_json(STATE, st)
            code = 2
            continue
        if rec.get("status") == "failed_oom" or rec.get("failures", 0) >= max_att:   # interruptions do not count
            rec.setdefault("status", "failed")
            code = 2
            continue
        if st["active_seconds"] >= budget:
            rlog.write(f"[{now()}] budget reached before {jid}\n")
            code = 3
            break
        if free_gb() < cfg["runner"].get("min_free_gb", 20):
            rec.update(status="blocked", note="insufficient disk")
            code = 2
            continue
        rec.update(status="running", start=now(), attempts=rec.get("attempts", 0) + 1, commit=git_commit(),
                   cmd=j["cmd"], free_mem_gb=round(free_mem_gb(), 1), log=f"outputs/m13/runner/logs/{jid}.log",
                   seeds=j.get("seeds"), config=j)
        atomic_json(STATE, st)
        write_status(cfg, st, jid)
        with open(RUN / "logs" / f"{jid}.log", "a") as lf:
            lf.write(f"\n===== {now()} attempt {rec['attempts']} commit {rec['commit']}\n")
            lf.flush()
            t0 = time.time()
            _child = subprocess.Popen([PY if c == "{python}" else c for c in j["cmd"]], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
            last = time.time()
            while _child.poll() is None:
                time.sleep(5)
                if time.time() - last > 60:
                    st["active_seconds"] += time.time() - last
                    last = time.time()
                    atomic_json(STATE, st)
                    write_status(cfg, st, jid)
                    if st["active_seconds"] >= budget and not _stop:
                        rlog.write(f"[{now()}] budget reached during {jid}: stopping at a checkpoint\n")
                        on_term(None, None)
                        code = 3
            st["active_seconds"] += time.time() - last
            rc = _child.returncode
            _child = None
        tail = (RUN / "logs" / f"{jid}.log").read_text()[-4000:]
        oom = ("out of memory" in tail.lower()) or rc in (-9, 137)
        rec.update(end=now(), exit=rc, seconds=round(time.time() - t0, 1))
        if rc == 0 and (ROOT / j["done"]).exists():
            rec["status"] = "completed"
        elif _stop:
            rec["status"] = "interrupted"
        else:
            rec["status"] = "failed_oom" if oom else "failed"
            rec["failures"] = rec.get("failures", 0) + 1
            code = 2
        rlog.write(f"[{now()}] {jid} -> {rec['status']} (exit {rc}, {rec['seconds']} s)\n")
        rlog.flush()
        atomic_json(STATE, st)
        write_status(cfg, st)
        if _stop:
            code = 3 if code == 3 else 130
            break
    write_status(cfg, st)
    rlog.write(f"[{now()}] runner exit {code}\n")
    PIDF.unlink(missing_ok=True)
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "resume", "run", "status", "stop"])
    ap.add_argument("--config", default="configs/m13_full.toml")
    ap.add_argument("--caffeinate", action="store_true")
    a = ap.parse_args()
    if a.action == "status":
        print(STATUS.read_text() if STATUS.exists() else "no status yet")
        if PIDF.exists():
            pid = int(PIDF.read_text())
            alive = subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0
            print(f"runner pid {pid}: {'running' if alive else 'not running (stale pid file)'}")
        return 0
    if a.action == "stop":
        if not PIDF.exists():
            print("no runner")
            return 0
        os.kill(int(PIDF.read_text()), signal.SIGTERM)
        print("SIGTERM sent: the current job checkpoints and the runner exits")
        return 0
    if a.action == "run":
        return run_worker(ROOT / a.config)
    if PIDF.exists() and subprocess.run(["ps", "-p", PIDF.read_text().strip()], capture_output=True).returncode == 0:
        print("a runner is already running:", PIDF.read_text())
        return 1
    RUN.mkdir(parents=True, exist_ok=True)
    out = open(RUN / "runner_stdout.log", "a")
    p = subprocess.Popen([PY, str(Path(__file__).resolve()), "run", "--config", a.config], cwd=ROOT, stdout=out,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"runner started, pid {p.pid}")
    if a.caffeinate:
        subprocess.Popen(["caffeinate", "-i", "-w", str(p.pid)], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("caffeinate -i active while the runner lives (prevents idle sleep; a closed lid still sleeps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
