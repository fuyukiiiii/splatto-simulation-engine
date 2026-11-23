"""
RL training as a separate local process, one at a time.

`train_runner.py` runs Gym2's own ExperimentManager in its own process group, with
W&B offline. The web service only starts it, reads its status file and log, and
can stop it: no request ever waits on training.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .gym2 import GYM2_ROOT, LAB_ROOT, RUNS_DIR

TRAIN_DIR = RUNS_DIR / "training"
RUNNER = LAB_ROOT / "service" / "train_runner.py"
PROJECT = "SplattoLab"
_procs: dict[str, subprocess.Popen] = {}

_TIMESTEPS = re.compile(r"\|\s*total_timesteps\s*\|\s*(\d+)")
_EVAL = re.compile(r"Eval num_timesteps=(\d+), episode_reward=([-\d.eE+]+)")


class TrainingError(ValueError):
    pass


def _job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]{4,64}", job_id):
        raise TrainingError("Bad job id.")
    return TRAIN_DIR / job_id


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def status(job_id: str) -> dict:
    d = _job_dir(job_id)
    if not d.exists():
        raise TrainingError("No such training job.")
    job = _read(d / "job.json")
    st = _read(d / "status.json")
    log = (d / "log.txt").read_text(errors="replace") if (d / "log.txt").exists() else ""
    proc = _procs.get(job_id)
    running = proc.poll() is None if proc else _alive(st.get("pid"))
    state = st.get("state", "starting")
    if state in ("starting", "training") and not running:
        state = "failed" if proc is None or proc.returncode not in (0, None) else state
        if state == "failed" and not st.get("error"):
            st["error"] = "The training process stopped without reporting (see the log)."
    steps = [int(m) for m in _TIMESTEPS.findall(log)]
    evals = [{"timesteps": int(a), "episode_reward": float(b)} for a, b in _EVAL.findall(log)]
    total = job.get("total_timesteps") or 1
    done = max(steps) if steps else 0
    return {
        "id": job_id,
        **job,
        "state": state,
        "running": running,
        "timesteps_done": done,
        "progress": min(1.0, done / total),
        "evaluations": evals,
        "run_name": st.get("run_name"),
        "error": st.get("error"),
        "log_tail": log[-3000:],
        "configured_defaults": st.get("configured_defaults"),
        "updated_at": st.get("updated_at"),
    }


def list_jobs() -> list[dict]:
    if not TRAIN_DIR.exists():
        return []
    jobs = []
    for d in sorted(TRAIN_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "job.json").exists():
            try:
                jobs.append(status(d.name))
            except TrainingError:
                pass
    return jobs


def start(algorithm: str, total_timesteps: int, n_envs: int | None, seed: int) -> dict:
    if algorithm not in ("ppo", "sac", "recurrentppo"):
        raise TrainingError("Algorithm must be ppo, sac or recurrentppo (Gym2's configs/agents).")
    if not (1_000 <= total_timesteps <= 50_000_000):
        raise TrainingError("Total timesteps must be between 1,000 and 50,000,000.")
    if n_envs is not None and not (1 <= n_envs <= 16):
        raise TrainingError("Parallel environments must be between 1 and 16.")
    if any(j["running"] for j in list_jobs()):
        raise TrainingError("A training run is already going. Stop it or wait for it to finish.")
    try:
        import stable_baselines3  # noqa: F401
        import wandb  # noqa: F401
    except ImportError as exc:
        raise TrainingError(f"Training dependencies are not installed ({exc.name}). Run setup.sh with --train.")

    job_id = f"lab-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    d = _job_dir(job_id)
    d.mkdir(parents=True)
    job = {"algorithm": algorithm, "total_timesteps": int(total_timesteps), "n_envs": n_envs, "seed": int(seed),
           "project": PROJECT, "started_at": time.time()}
    (d / "job.json").write_text(json.dumps(job))
    env = {
        **os.environ,
        "WANDB_MODE": "offline", "WANDB_SILENT": "true", "WANDB_NAME": job_id, "WANDB_DIR": str(d),
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [sys.executable, str(RUNNER), "--algorithm", algorithm, "--total_timesteps", str(int(total_timesteps)),
           "--seed", str(int(seed)), "--project", PROJECT, "--group", "lab", "--status_file", str(d / "status.json")]
    if n_envs:
        cmd += ["--n_envs", str(int(n_envs))]
    log = open(d / "log.txt", "w")
    _procs[job_id] = subprocess.Popen(cmd, cwd=str(GYM2_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return status(job_id)


def cancel(job_id: str) -> dict:
    st = status(job_id)
    proc = _procs.get(job_id)
    pid = proc.pid if proc else _read(_job_dir(job_id) / "status.json").get("pid")
    if pid and st["running"]:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            pass
        path = _job_dir(job_id) / "status.json"
        data = _read(path)
        data.update(state="cancelled", updated_at=time.time())
        path.write_text(json.dumps(data))
    return status(job_id)
