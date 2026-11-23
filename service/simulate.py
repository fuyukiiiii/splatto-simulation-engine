"""
One GreenLight-Gym2 episode, run and measured.

The environment is `gl_gym/GreenLightTomato-v0` with only Gym2-supported kwargs
changed (season length, weather scenario, constraints, reward prices, the
parameter provider). Controls come from Gym2's `RuleBasedController` (raw
control space, `normalize_actions=False`, as Gym2's evaluate_baseline.py does)
or from a trained Stable-Baselines3 policy (Gym2's normalised Δu space, as
evaluate_rl.py does).

Every number reported is either a Gym2 output (states, observations, the
reward's info dict) or a sum of one over the episode. Energy uses the same
formulas Gym2's reward and the GreenLight equations use, integrated exactly:
controls are held constant over each 900 s solver step.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from gymnasium import spaces

from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext

from . import params as P
from .gym2 import ACTUATORS, WEATHER_DIR, make_env, registered_kwargs, rule_based_defaults, version_info

ACTUATOR_KEYS = [a["key"] for a in ACTUATORS]
PRICE_KEYS = ["fruit_price", "heating_price", "elec_price", "co2_price", "dmfm", "pen_lamp"]
CONSTRAINT_KEYS = ["co2_min", "co2_max", "temp_min", "temp_max", "rh_min", "rh_max"]
MAX_SEASON_DAYS = 366


class SimulationError(ValueError):
    pass


class SimulationTimeout(SimulationError):
    """The run reached the wall-clock limit (SIM_TIME_LIMIT_S) and was stopped."""


# A run stops after this many seconds, so a request the caller has given up on (the site waits
# 55 s) does not keep the engine busy. Checked every simulated day.
TIME_LIMIT_S = float(os.environ.get("SIM_TIME_LIMIT_S", "45"))

# Rule-based settings with a physical meaning get a plausible range; the rest only a generous bound.
SETTING_RANGES = {
    "temp_setpoint_day": (5, 35), "temp_setpoint_night": (5, 35), "co2_day": (0, 2000),
    "lamps_on": (-1, 25), "lamps_off": (-1, 25), "lamps_day_start": (-1, 367), "lamps_day_stop": (-1, 367),
    "lamps_off_sun": (0, 2000), "lamp_rad_sum_limit": (0, 100), "rh_max": (0, 100), "rhMax": (0, 100),
    "useBlScr": (0, 1), "heat_deadzone": (0, 30), "t_vent_off": (-30, 30), "thScrSpDay": (-30, 40), "thScrSpNight": (-30, 40),
}


@dataclass
class RunConfig:
    location: str
    growth_year: int
    start_day: int
    season_length: int
    seed: int = 0
    prices: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, float] = field(default_factory=dict)
    controller: dict[str, Any] = field(default_factory=lambda: {"type": "rule_based"})
    actuator_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def defaults() -> dict:
    kw = registered_kwargs()
    scen = kw["weather_scenario_sampler_kwargs"]
    return {
        "scenario": {"location": scen["location"], "growth_year": scen["growth_year"], "start_day": scen["start_day"]},
        "season_length": kw["season_length"],
        "seed": 0,
        "prices": {k: kw["reward_kwargs"][k] for k in PRICE_KEYS},
        "constraints": dict(kw["constraints"]),
        "dt": kw["dt"],
        "delta_u_max": kw["delta_u_max"],
        "pred_horizon": kw["pred_horizon"],
        "observation_modules": list(kw["observation_modules"]),
        "rule_based": rule_based_defaults(),
    }


# ---------------------------------------------------------------- validation

def _finite(name: str, value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise SimulationError(f"{name} must be a number, got {value!r}.")
    if not math.isfinite(v):
        raise SimulationError(f"{name} must be finite, got {value!r}.")
    return v


def _validate(cfg: RunConfig) -> tuple[dict, dict, dict]:
    d = defaults()
    available = {p.name: sorted(int(f.stem) for f in p.glob("*.csv") if f.stem.isdigit()) for p in WEATHER_DIR.iterdir() if p.is_dir()}
    if cfg.location not in available:
        raise SimulationError(f"No weather for '{cfg.location}'. Gym2 has: {', '.join(sorted(available)) or 'none'}.")
    years = available[cfg.location]
    if cfg.growth_year not in years:
        raise SimulationError(f"No {cfg.location} weather for {cfg.growth_year}. Available years: {', '.join(map(str, years))}.")
    if not (1 <= cfg.season_length <= MAX_SEASON_DAYS):
        raise SimulationError(f"Season length must be 1–{MAX_SEASON_DAYS} days, got {cfg.season_length}.")
    days_in_year = 366 if calendar.isleap(cfg.growth_year) else 365
    if not (0 <= cfg.start_day < days_in_year):
        raise SimulationError(f"Start day must be 0–{days_in_year - 1} for {cfg.growth_year}, got {cfg.start_day}.")
    # The loader reads the season plus the forecast horizon, and the next year's file when it runs past 31 Dec.
    needed_days = cfg.start_day + cfg.season_length + d["pred_horizon"] + 1 / 96
    if needed_days > days_in_year and (cfg.growth_year + 1) not in years:
        raise SimulationError(
            f"Day {cfg.start_day} + {cfg.season_length} days runs past 31 Dec {cfg.growth_year}, and Gym2 has no "
            f"{cfg.location} {cfg.growth_year + 1} file to continue with. Start earlier or shorten the season."
        )

    prices = dict(d["prices"])
    for k, v in (cfg.prices or {}).items():
        if k not in PRICE_KEYS:
            raise SimulationError(f"Unknown price '{k}'. Supported: {', '.join(PRICE_KEYS)}.")
        prices[k] = _finite(k, v)
        if prices[k] < 0:
            raise SimulationError(f"{k} cannot be negative ({prices[k]:g}).")
    if not (0 < prices["dmfm"] <= 1):
        raise SimulationError(f"dmfm (dry-matter fraction) must be in (0, 1], got {prices['dmfm']:g}.")

    constraints = dict(d["constraints"])
    for k, v in (cfg.constraints or {}).items():
        if k not in CONSTRAINT_KEYS:
            raise SimulationError(f"Unknown constraint '{k}'. Supported: {', '.join(CONSTRAINT_KEYS)}.")
        constraints[k] = _finite(k, v)
    for q in ("co2", "temp", "rh"):
        if constraints[f"{q}_min"] >= constraints[f"{q}_max"]:
            raise SimulationError(f"{q}_min must be below {q}_max.")

    overrides = {}
    for k, o in (cfg.actuator_overrides or {}).items():
        if k not in ACTUATOR_KEYS:
            raise SimulationError(f"Unknown actuator '{k}'. Gym2's are {', '.join(ACTUATOR_KEYS)}.")
        mode = (o or {}).get("mode", "controller")
        if mode == "controller":
            continue
        if mode != "fixed":
            raise SimulationError(f"Actuator {k}: mode must be 'controller' or 'fixed', got {mode!r}.")
        v = _finite(f"{k} value", o.get("value"))
        if not (0 <= v <= 1):
            raise SimulationError(f"Actuator {k} fixed value must be within Gym2's control bounds [0, 1], got {v:g}. Not clamped.")
        overrides[k] = v
    return prices, constraints, overrides


# ---------------------------------------------------------------- controllers

RULE_BASED_HELP = {
    # From the comments and expressions in gl_gym/components/rule_based.py.
    "temp_setpoint_day": ("Heating setpoint when it is day inside (sun or lamps)", "°C"),
    "temp_setpoint_night": ("Heating setpoint at night", "°C"),
    "co2_day": ("CO₂ setpoint during the day (0 at night)", "ppm"),
    "lamps_on": ("Hour the lamps' light period starts", "h"),
    "lamps_off": ("Hour the lamps' light period ends", "h"),
    "lamps_off_sun": ("Lamps stay off above this outdoor global radiation", "W m⁻²"),
    "lamp_rad_sum_limit": ("Lamps stay off when the day's outdoor radiation sum exceeds this", "MJ m⁻² day⁻¹"),
    "heat_deadzone": ("Ventilation for heat starts this far above the heating setpoint", "°C"),
    "rh_max": ("Ventilation for humidity starts above this RH", "%"),
    "useBlScr": ("Close the blackout screen at night while lamps are on (1) or never (0)", "0/1"),
    "thScrSpDay": ("Thermal screen closes below this outdoor temperature by day", "°C"),
    "thScrSpNight": ("Thermal screen closes below this outdoor temperature at night", "°C"),
    "lampExtraHeat": ("Lamps switch off this far above the ventilation temperature", "°C"),
    "t_vent_off": ("Vents close this far below the heating setpoint", "°C"),
}


def _rule_based(settings: dict | None) -> tuple[RuleBasedController, dict]:
    base = rule_based_defaults()
    merged = dict(base)
    for k, v in (settings or {}).items():
        if k not in base:
            raise SimulationError(f"Unknown rule-based setting '{k}'. Supported: {', '.join(base)}.")
        merged[k] = _finite(k, v)
        lo, hi = SETTING_RANGES.get(k, (-1e4, 1e4))
        if not (lo <= merged[k] <= hi):
            raise SimulationError(f"Rule-based setting {k} = {merged[k]:g} is outside {lo:g} to {hi:g}. Not clamped: change the value.")
        if "band" in k.lower() and merged[k] == 0:
            raise SimulationError(f"Rule-based setting {k} cannot be 0 (Gym2 divides by it).")
    return RuleBasedController(**merged), merged


# ---------------------------------------------------------------- the episode

def _ctx(env) -> StepContext:
    return StepContext(
        t=env.timestep, dt=env.dt, Np=env.Np, x_prev=env.x_prev, x=env.x, u=env.u, p=env.p,
        d=env.weather_data, hour_of_day=env.hour_of_day, day_of_year=env.day_of_year,
    )


def run(cfg: RunConfig, policy_loader: Callable[[str], Any] | None = None) -> dict:
    prices, constraints, fixed = _validate(cfg)
    try:
        vector, param_report = P.build_vector(cfg.parameters)
    except P.ParameterError as exc:
        raise SimulationError(str(exc))

    ctype = (cfg.controller or {}).get("type", "rule_based")
    rl = ctype == "rl"
    if ctype not in ("rule_based", "rl"):
        raise SimulationError(f"Unknown controller '{ctype}'. Supported: rule_based, rl.")
    if rl and fixed:
        raise SimulationError("Fixed actuators apply to the rule-based controller only: a trained policy acts on all six controls.")

    scenario = {"location": cfg.location, "growth_year": int(cfg.growth_year), "start_day": int(cfg.start_day)}
    env = make_env(
        season_length=int(cfg.season_length),
        weather_scenario_sampler="fixed",
        weather_scenario_sampler_kwargs=scenario,
        reward_kwargs=prices,
        constraints=constraints,
        normalize_actions=rl,  # policies act on Δu in [-1, 1]; the rule-based controller on positions in [0, 1]
        parameter_provider="set",
        parameter_provider_kwargs={"parameter_sets": [vector]},
    )
    try:
        obs, _ = env.reset(seed=int(cfg.seed), options={"scenario": scenario})
    except FileNotFoundError as exc:
        raise SimulationError(f"Weather file missing: {exc}")

    controller_info: dict[str, Any]
    if rl:
        policy = policy_loader(cfg.controller.get("policy_id", "")) if policy_loader else None
        if policy is None:
            raise SimulationError("No such trained policy.")
        policy.check(env)
        policy.reset()
        controller_info = {"type": "rl", **policy.describe()}

        def act(o):
            flat = spaces.flatten(env.observation_space, o)
            return policy.act(flat)
    else:
        controller, settings = _rule_based(cfg.controller.get("settings"))
        controller_info = {"type": "rule_based", "settings": settings, "fixed_actuators": fixed}
        fixed_idx = [(ACTUATOR_KEYS.index(k), v) for k, v in fixed.items()]

        def act(_o):
            u = controller.predict(_ctx(env))
            for i, v in fixed_idx:
                u[i] = v
            return u.astype(np.float32)

    N = env.N
    nx = len(env.x)
    X = np.zeros((N + 1, nx))
    U = np.zeros((N, 6))
    IND = np.zeros((N + 1, 4))
    WEA = np.zeros((N + 1, 5))
    INFO_KEYS = ["reward", "profit", "revenue", "heat_cost", "co2_cost", "elec_cost", "fruit_growth_dm", "temp_penalty", "co2_penalty", "rh_penalty", "penalty", "lamp_penalty"]
    INFO = np.zeros((N, len(INFO_KEYS)))
    X[0] = env.x
    IND[0] = obs["IndoorClimateObservations"]
    WEA[0] = obs["WeatherObservations"]
    failed_at = None
    steps = 0
    t0 = time.time()
    while steps < N:
        action = act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        U[steps] = env.u
        X[steps + 1] = env.x
        IND[steps + 1] = obs["IndoorClimateObservations"]
        WEA[steps + 1] = obs["WeatherObservations"]
        INFO[steps] = [float(info.get(k, 0.0)) for k in INFO_KEYS]
        steps += 1
        if info.get("solver_failed"):
            failed_at = steps
            break
        if terminated or truncated:
            break
        if steps % 96 == 0 and time.time() - t0 > TIME_LIMIT_S:
            raise SimulationTimeout(
                f"The simulation took longer than {TIME_LIMIT_S:g} s and was stopped after {steps * env.dt / 86400:.0f} of "
                f"{cfg.season_length} days. Try a shorter season."
            )
    wall = time.time() - t0

    return _summarise(env, cfg, scenario, prices, constraints, vector, param_report, controller_info,
                      X[: steps + 1], U[:steps], IND[: steps + 1], WEA[: steps + 1], INFO[:steps], INFO_KEYS,
                      steps, N, failed_at, wall)


def _summarise(env, cfg, scenario, prices, constraints, p, param_report, controller_info,
               X, U, IND, WEA, INFO, INFO_KEYS, steps, N, failed_at, wall) -> dict:
    dt = env.dt
    hours = dt / 3600.0
    info = {k: INFO[:, i] for i, k in enumerate(INFO_KEYS)}

    # Energy, per m² floor, with the same expressions as GreenhouseReward and aux_states (a[220], a[37], a[222]).
    heat_w_m2 = U[:, 0] * p[108] / p[46]
    lamp_w_m2 = U[:, 4] * p[172]
    co2_mg_m2_s = U[:, 1] * p[109] / p[46]
    heating_kwh = float(np.sum(heat_w_m2) * dt / 3.6e6)
    lamp_kwh = float(np.sum(lamp_w_m2) * dt / 3.6e6)
    co2_kg = float(np.sum(co2_mg_m2_s) * dt * 1e-6)
    gym2_heat = float(info["heat_cost"].sum())
    gym2_elec = float(info["elec_cost"].sum())
    gym2_co2 = float(info["co2_cost"].sum())

    def check(ours, theirs):
        return bool(math.isclose(ours, theirs, rel_tol=1e-6, abs_tol=1e-9))

    cross_check = {
        "heating": check(heating_kwh * prices["heating_price"], gym2_heat),
        "lighting": check(lamp_kwh * prices["elec_price"], gym2_elec),
        "co2": check(co2_kg * prices["co2_price"], gym2_co2),
    }

    # Climate constraint violations, as GreenhouseReward._output_violations computes them, in physical units.
    co2, temp, rh = IND[1:, 0], IND[1:, 1], IND[1:, 2]

    def band(values, lo, hi, unit):
        below = np.clip(lo - values, 0, None)
        above = np.clip(values - hi, 0, None)
        return {
            "unit": unit, "min": lo, "max": hi,
            "hours_below": float((below > 0).sum() * hours), "hours_above": float((above > 0).sum() * hours),
            "unit_hours_below": float(below.sum() * hours), "unit_hours_above": float(above.sum() * hours),
            "worst_below": float(below.max(initial=0)), "worst_above": float(above.max(initial=0)),
        }

    def inside(values, lo, hi):
        return (values >= lo) & (values <= hi)

    t_ok = inside(temp, constraints["temp_min"], constraints["temp_max"])
    rh_ok = inside(rh, constraints["rh_min"], constraints["rh_max"])
    co2_ok = inside(co2, constraints["co2_min"], constraints["co2_max"])
    n = max(1, len(temp))
    violations = {
        "share_of_time_within": {
            "all": float((t_ok & rh_ok & co2_ok).sum() / n), "temperature": float(t_ok.sum() / n),
            "rh": float(rh_ok.sum() / n), "co2": float(co2_ok.sum() / n),
        },
        "temperature": band(temp, constraints["temp_min"], constraints["temp_max"], "°C"),
        "rh": band(rh, constraints["rh_min"], constraints["rh_max"], "%"),
        "co2": band(co2, constraints["co2_min"], constraints["co2_max"], "ppm"),
        "gym2_scaled_penalty": {
            "temperature": float(info["temp_penalty"].sum()), "rh": float(info["rh_penalty"].sum()),
            "co2": float(info["co2_penalty"].sum()), "total": float(info["penalty"].sum()),
        },
    }

    sla = p[142]
    fruit_dw_kg = float(info["fruit_growth_dm"].sum() * 1e-6)
    crop = {
        "initial": {"cFruit": X[0, 25], "cLeaf": X[0, 23], "cStem": X[0, 24], "cBuf": X[0, 22], "tCanSum": X[0, 26], "lai": sla * X[0, 23]},
        "final": {"cFruit": X[-1, 25], "cLeaf": X[-1, 23], "cStem": X[-1, 24], "cBuf": X[-1, 22], "tCanSum": X[-1, 26], "lai": sla * X[-1, 23]},
        "fruit_dw_increase_kg_m2": fruit_dw_kg,
        "fruit_fw_equivalent_kg_m2": fruit_dw_kg / prices["dmfm"],
        "dmfm_used": prices["dmfm"],
        "mean_tcan": float(X[1:, 4].mean()) if len(X) > 1 else float(X[0, 4]),
        "tEndSum": float(p[163]),
        "cFruitMax": float(p[145]),
        "harvest_active": bool(X[:, 25].max() > 0.8 * p[145]),
    }
    crop = json.loads(json.dumps(crop, default=float))

    economics = {
        "revenue": float(info["revenue"].sum()),
        "heating_cost": gym2_heat, "lighting_cost": gym2_elec, "co2_cost": gym2_co2,
        "operating_cost": gym2_heat + gym2_elec + gym2_co2,
        "gross_margin": float(info["profit"].sum()),
        "currency": "EUR", "per": "m² floor",
        "prices": prices,
        "note": "Revenue is Gym2's fruit fresh-weight increase × fruit price; costs are energy × price. No labour, depreciation, other inputs or market effects.",
    }

    # Hourly series for the charts: states at each hour, controls as the hour's mean.
    k = max(1, int(round(3600 / dt)))
    idx = np.arange(0, len(X), k)
    n_hours = steps // k
    U_h = U[: n_hours * k].reshape(n_hours, k, 6).mean(axis=1) if n_hours else np.zeros((0, 6))
    viol_step = (
        np.clip(constraints["temp_min"] - temp, 0, None) + np.clip(temp - constraints["temp_max"], 0, None) > 0
    ).astype(float)
    series = {
        "t_hours": (idx * hours).round(4).tolist(),
        "day_of_year": (scenario["start_day"] + idx * dt / 86400).round(4).tolist(),
        "indoor": {
            "temp_air": IND[idx, 1].round(3).tolist(), "rh_air": IND[idx, 2].round(2).tolist(),
            "co2_ppm": IND[idx, 0].round(1).tolist(), "pipe_temp": IND[idx, 3].round(2).tolist(),
            "tCan": X[idx, 4].round(3).tolist(), "tCan24": X[idx, 21].round(3).tolist(),
        },
        "outdoor": {
            "radiation": WEA[idx, 0].round(1).tolist(), "temp": WEA[idx, 1].round(2).tolist(),
            "rh": WEA[idx, 2].round(1).tolist(), "co2_ppm": WEA[idx, 3].round(1).tolist(), "wind": WEA[idx, 4].round(2).tolist(),
        },
        "controls_t_hours": ((np.arange(n_hours) + 0.5) * k * hours).round(4).tolist(),
        "controls": {key: U_h[:, i].round(4).tolist() for i, key in enumerate(ACTUATOR_KEYS)},
        "crop": {
            "cFruit": X[idx, 25].round(0).tolist(), "cLeaf": X[idx, 23].round(0).tolist(), "cStem": X[idx, 24].round(0).tolist(),
            "cBuf": X[idx, 22].round(0).tolist(), "tCanSum": X[idx, 26].round(2).tolist(), "lai": (sla * X[idx, 23]).round(4).tolist(),
        },
        "cumulative": {
            "heating_kwh": np.concatenate([[0], np.cumsum(heat_w_m2) * dt / 3.6e6])[idx].round(4).tolist(),
            "lamp_kwh": np.concatenate([[0], np.cumsum(lamp_w_m2) * dt / 3.6e6])[idx].round(4).tolist(),
            "co2_kg": np.concatenate([[0], np.cumsum(co2_mg_m2_s) * dt * 1e-6])[idx].round(5).tolist(),
            "revenue": np.concatenate([[0], np.cumsum(info["revenue"])])[idx].round(4).tolist(),
            "cost": np.concatenate([[0], np.cumsum(info["heat_cost"] + info["elec_cost"] + info["co2_cost"])])[idx].round(4).tolist(),
        },
        "temp_violation_hours": np.concatenate([[0], np.cumsum(viol_step) * hours])[idx].round(3).tolist(),
    }

    config_record = {
        "scenario": scenario, "season_length": int(cfg.season_length), "seed": int(cfg.seed),
        "prices": prices, "constraints": constraints, "parameters": cfg.parameters or {},
        "controller": cfg.controller, "actuator_overrides": cfg.actuator_overrides or {},
    }
    digest = hashlib.sha256(json.dumps(config_record, sort_keys=True, default=str).encode()).hexdigest()[:16]

    return {
        "config": config_record,
        "config_hash": digest,
        "parameter_vector_hash": hashlib.sha256(np.asarray(p, dtype=np.float64).tobytes()).hexdigest()[:16],
        "version": version_info(),
        "controller": controller_info,
        "parameters": param_report,
        "run": {
            "steps": steps, "expected_steps": N, "dt_s": dt, "complete": steps == N and failed_at is None,
            "solver_failed_at_step": failed_at, "wall_time_s": round(wall, 2),
            "action_space": "Δu = action × 0.1 per step, in [-1, 1] (normalize_actions=True)" if controller_info["type"] == "rl"
            else "Raw control positions in [0, 1] (normalize_actions=False)",
        },
        "metrics": {
            "energy": {
                "heating_kwh_m2": heating_kwh, "lighting_kwh_m2": lamp_kwh, "co2_kg_m2": co2_kg,
                "total_energy_kwh_m2": heating_kwh + lamp_kwh,
                "method": "Σ u·capacity·Δt over 900 s steps; controls are constant within a step, so the sum is exact for the model.",
                "matches_gym2_reward_costs": cross_check,
            },
            "economics": economics,
            "climate": {
                "temp_air": _stats(IND[1:, 1]), "rh_air": _stats(IND[1:, 2]), "co2_ppm": _stats(IND[1:, 0]), "pipe_temp": _stats(IND[1:, 3]),
            },
            "violations": violations,
            "crop": crop,
            "reward": {"total": float(info["reward"].sum()), "note": "Gym2's scaled reward: min-max scaled profit minus scaled climate penalties. Unitless."},
        },
        "series": series,
        "assumptions": _assumptions(env, p),
    }


def _stats(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"mean": None, "min": None, "max": None}
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def _assumptions(env, p) -> list[str]:
    return [
        f"Model: GreenLight (Katzin/Vanthoor) tomato crop–climate model as implemented in GreenLight-Gym2, CasADi CVODES, {env.dt:.0f} s steps.",
        f"All results are per m² of floor. Gym2's reference house is {p[46]:.0f} m² (aFlr) with {p[108] / p[46]:.0f} W m⁻² boiler capacity, "
        f"{p[109] / p[46]:.1f} mg m⁻² s⁻¹ CO₂ supply and {p[172]:.0f} W m⁻² LED lamps unless changed.",
        "The crop starts from Gym2's fixed initial state (an established tomato crop, temperature sum 3098 °C·day); the start date does not plant a new crop.",
        "Outdoor CO₂ is fixed at 400 ppm by Gym2's weather loader; weather is the recorded year, so the run is deterministic.",
        "Fruit revenue counts the growth of fruit dry matter (cFruit) converted with dmfm; GreenLight's harvest flow above cFruitMax is not counted as revenue.",
        "These are model outputs for comparing scenarios, not validated engineering or commercial yield predictions.",
    ]
