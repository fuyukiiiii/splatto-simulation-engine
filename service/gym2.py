"""
Where GreenLight-Gym2 lives and how the Lab builds its environment.

Everything here reads Gym2's own files: the registered environment's kwargs
(`gl_gym/__init__.py` via gymnasium's registry), the environment config YAML,
the rule-based controller's YAML, and the bundled weather folder. Nothing is
re-declared, so a newer Gym2 checkout shows up as it is.
"""
from __future__ import annotations

import copy
import os
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import yaml

import gl_gym  # noqa: F401  (registers gl_gym/GreenLightTomato-v0)
from gl_gym.environments.greenlight_env import GreenLightEnv

ENV_ID = "gl_gym/GreenLightTomato-v0"
LAB_ROOT = Path(__file__).resolve().parent.parent
GYM2_ROOT = Path(os.environ.get("GYM2_ROOT", LAB_ROOT / "vendor" / "GreenLight-Gym2")).resolve()
PACKAGE_ROOT = Path(gl_gym.__file__).resolve().parent
WEATHER_DIR = PACKAGE_ROOT / "data" / "weather"
RUNS_DIR = LAB_ROOT / "runs"

ACTUATORS = [
    # name, index, what the model does with the value (from aux_states.py / ode.py)
    {"key": "uBoil", "index": 0, "label": "Boiler heating",
     "meaning": "Fraction of the boiler's capacity put into the heating pipes: hBoilPipe = uBoil · pBoil / aFlr [W m⁻²]."},
    {"key": "uCO2", "index": 1, "label": "CO₂ dosing",
     "meaning": "Fraction of the external CO₂ source's capacity: mcExtAir = uCO2 · phiExtCo2 / aFlr [mg m⁻² s⁻¹]."},
    {"key": "uThScr", "index": 2, "label": "Thermal screen",
     "meaning": "Thermal screen closure, 0 = open, 1 = closed."},
    {"key": "uVent", "index": 3, "label": "Roof ventilation",
     "meaning": "Roof vent opening as a fraction of the maximum roof vent area aRoof."},
    {"key": "uLamp", "index": 4, "label": "Supplemental lighting",
     "meaning": "Fraction of the lamps' maximum electrical input: qLampIn = thetaLampMax · uLamp [W m⁻²]."},
    {"key": "uBlScr", "index": 5, "label": "Blackout screen",
     "meaning": "Blackout screen closure, 0 = open, 1 = closed."},
]


@lru_cache(maxsize=1)
def registered_kwargs() -> dict:
    """The kwargs `gl_gym/GreenLightTomato-v0` is registered with."""
    return copy.deepcopy(gym.spec(ENV_ID).kwargs)


@lru_cache(maxsize=1)
def rule_based_defaults() -> dict:
    with open(GYM2_ROOT / "configs" / "agents" / "rule_based.yml") as f:
        return dict(yaml.safe_load(f)["GreenLightEnv"])


@lru_cache(maxsize=1)
def version_info() -> dict:
    commit = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(GYM2_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except Exception:
        commit = None
    packages = {}
    for name in ("gl-gym", "casadi", "numpy", "gymnasium", "pandas", "scipy", "stable-baselines3", "sb3-contrib", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "env_id": ENV_ID,
        "gl_gym_version": packages["gl-gym"],
        "gym2_commit": commit,
        "gym2_root": str(GYM2_ROOT),
        "packages": packages,
        "license": "GreenLight-Gym2 is AGPL-3.0-or-later",
    }


def weather_catalog() -> list[dict]:
    """Every location folder and year file Gym2 can load, as it ships (`<dir>/<location>/<year>.csv`)."""
    out = []
    if not WEATHER_DIR.exists():
        return out
    for loc in sorted(p for p in WEATHER_DIR.iterdir() if p.is_dir()):
        years = sorted(int(f.stem) for f in loc.glob("*.csv") if f.stem.isdigit())
        if years:
            out.append({"location": loc.name, "years": years})
    return out


def make_env(**overrides) -> GreenLightEnv:
    """The registered environment, with Gym2-supported kwargs overridden."""
    kwargs = registered_kwargs()
    kwargs.update(overrides)
    return gym.make(ENV_ID, **kwargs).unwrapped
