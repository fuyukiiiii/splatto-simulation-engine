"""
Trained policies: found where Gym2's training writes them, evaluated as Gym2 does.

Gym2's ExperimentManager saves `train_data/<project>/<algorithm>/<det|stoch>/models/<run>/best_model.zip`
and the matching observation normalisation in `.../envs/<run>/best_vecnormalize.pkl`. Gym2's
experiments/evaluate_rl.py loads both, sets VecNormalize to evaluation mode (`training=False`,
`norm_reward=False`) and predicts deterministically on a FlattenObservation-wrapped env. The
Lab does the same, but steps the environment itself so that every state can be recorded: the
saved VecNormalize is only used to normalise each observation, exactly as it would inside the
vectorised env.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .gym2 import GYM2_ROOT

ALGORITHMS = ("ppo", "sac", "recurrentppo")


def _roots() -> list[Path]:
    roots = [GYM2_ROOT / "train_data"]
    extra = os.environ.get("SIM_LAB_POLICY_DIRS")
    if extra:
        roots += [Path(p) for p in extra.split(os.pathsep) if p]
    return roots


def discover() -> list[dict]:
    out = []
    for root in _roots():
        if not root.exists():
            continue
        for model in sorted(root.glob("*/*/*/models/*/*_model.zip")):
            run_dir = model.parent
            mode_dir = run_dir.parent.parent
            algorithm = mode_dir.parent.name.lower()
            if algorithm not in ALGORITHMS:
                continue
            which = model.stem.replace("_model", "")  # best | last
            vecnorm = mode_dir / "envs" / run_dir.name / f"{which}_vecnormalize.pkl"
            out.append({
                "id": str(model.relative_to(root.parent)) if root.parent in model.parents else str(model),
                "root": str(root),
                "project": mode_dir.parent.parent.name,
                "algorithm": algorithm,
                "mode": mode_dir.name,
                "run": run_dir.name,
                "checkpoint": which,
                "model_path": str(model),
                "vecnormalize_path": str(vecnorm) if vecnorm.exists() else None,
                "modified": model.stat().st_mtime,
            })
    return out


class LoadedPolicy:
    def __init__(self, meta: dict):
        from stable_baselines3 import PPO, SAC
        from sb3_contrib import RecurrentPPO

        cls = {"ppo": PPO, "sac": SAC, "recurrentppo": RecurrentPPO}[meta["algorithm"]]
        self.meta = meta
        self.model = cls.load(meta["model_path"], device="cpu")
        self.obs_rms = None
        self.clip_obs = 10.0
        self.epsilon = 1e-8
        if meta["vecnormalize_path"]:
            import pickle

            with open(meta["vecnormalize_path"], "rb") as f:
                vn = pickle.load(f)
            if getattr(vn, "norm_obs", True):
                self.obs_rms = vn.obs_rms
            self.clip_obs = vn.clip_obs
            self.epsilon = vn.epsilon
        self.recurrent = meta["algorithm"] == "recurrentppo"
        self.reset()

    def reset(self):
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def check(self, env):
        from gymnasium import spaces

        expected = self.model.observation_space.shape
        got = spaces.flatdim(env.observation_space)
        if expected != (got,):
            raise ValueError(
                f"This policy expects {expected[0]} observations; the environment gives {got}. "
                "It was trained with different observation modules and cannot be evaluated here."
            )
        if self.model.action_space.shape != env.action_space.shape:
            raise ValueError("This policy was trained for a different set of controlled inputs.")

    def act(self, flat_obs: np.ndarray) -> np.ndarray:
        o = np.asarray(flat_obs, dtype=np.float32)[None, :]
        if self.obs_rms is not None:  # VecNormalize.normalize_obs
            o = np.clip((o - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs).astype(np.float32)
        action, self.state = self.model.predict(o, state=self.state, episode_start=self.episode_start, deterministic=True)
        self.episode_start = np.zeros((1,), dtype=bool)
        return np.asarray(action[0], dtype=np.float32)

    def describe(self) -> dict:
        return {k: self.meta[k] for k in ("id", "algorithm", "run", "checkpoint", "project", "mode")} | {
            "normalised_observations": self.obs_rms is not None,
        }


_cache: dict[str, LoadedPolicy] = {}


def load(policy_id: str) -> LoadedPolicy | None:
    for meta in discover():
        if meta["id"] == policy_id:
            key = f"{meta['model_path']}:{meta['modified']}"
            if key not in _cache:
                _cache.clear()
                _cache[key] = LoadedPolicy(meta)
            return _cache[key]
    return None
