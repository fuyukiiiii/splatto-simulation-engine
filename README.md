# Splatto Simulation Engine

The engine behind the Simulation in [Splatto Design](https://app.splattoworks.com): a small FastAPI
service over [GreenLight-Gym2](https://github.com/BartvLaatum/GreenLight-Gym2), the GreenLight
tomato crop–climate model as a Gymnasium environment. GreenLight-Gym2 is used unmodified, pinned
at commit `da8b22c`.

This repository is published to meet GreenLight-Gym2's licence (AGPL-3.0-or-later): it is the
complete source of the engine that Splatto Design runs for its users.

## What it does

| Endpoint | |
| --- | --- |
| `GET /` | This page's summary and the source links |
| `GET /health` | Versions (gl-gym, GreenLight-Gym2 commit, packages) |
| `GET /catalog` | Defaults, weather years, actuators, controller settings |
| `GET /parameters` | All 208 GreenLight parameters with name, description, unit, default, Gym2 bounds |
| `POST /simulate` | One season: climate, energy, crop, controls, violations, model economics |
| `GET /rl/policies`, `/rl/jobs`, `POST /rl/train` | Trained policies and local training (Gym2's own pipeline) |

When `SIM_LAB_KEY` is set (as in production), every endpoint except `/` and `/health` needs the
`x-simulation-key` header.

## Run it

```bash
PYTHON=python3.12 ./setup.sh            # add --train for PyTorch / Stable-Baselines3
.venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8100
.venv/bin/python -m pytest tests
```

Or as a container (this is how it runs on Google Cloud Run):

```bash
docker build -t splatto-simulation-engine .
docker run -p 7860:7860 -e SIM_LAB_KEY=change-me splatto-simulation-engine
```

## Audit of GreenLight-Gym2 (gl-gym 0.3.2, commit da8b22c, 14 Nov 2026)

### Directly supported (used as Gym2 provides it)

| Feature | Gym2 source | In the Lab |
| --- | --- | --- |
| `gl_gym/GreenLightTomato-v0`, 900 s CasADi/CVODES steps, 28 states | `gl_gym/__init__.py`, `environments/greenlight_env.py` | Every run builds the registered env; only supported kwargs change |
| Indoor climate: CO₂ ppm, air T, RH, pipe T | `IndoorClimateObservations` | Climate tab |
| Outdoor weather: radiation, T, RH, CO₂, wind | `WeatherObservations` | Climate tab |
| Crop states: tCan24, cFruit, tCanSum; full state (cBuf, cLeaf, cStem, tCan) | `BasicCropObservations`, env state `x` | Crop tab |
| Six actuators uBoil, uCO2, uThScr, uVent, uLamp, uBlScr | `components/actions.py` | Control tab (hourly means) |
| Action semantics | `NamedControlActionScheme` | Rule-based: positions in [0, 1] (`normalize_actions=False`). Policies: Δu = action × 0.1 per step (`normalize_actions=True`). Shown on the Control tab |
| Rule-based baseline controller + its 29 settings | `components/rule_based.py`, `configs/agents/rule_based.yml` | Default controller; settings editable |
| Profit reward: revenue, heat/elec/CO₂ costs, scaled penalties | `components/rewards.py` (`GreenhouseReward`) | Energy/economics; prices via `reward_kwargs` |
| Climate constraints | env `constraints` | Editable; violations reported in °C/%/ppm-hours and as Gym2's scaled penalty |
| Weather scenarios (location, year, start day, season length) | `components/weather.py`, `data/weather/Amsterdam/2008–2012.csv` | Weather section; list read from the folder |
| Parameter registry (5 named, with bounds) | `configs/greenlight_parameters.py` | Parameter explorer; out-of-bound values rejected, never clamped |
| "set" parameter provider | `components/parameters.py` | How every run receives its parameter vector |
| RL training (PPO, SAC, RecurrentPPO) with VecNormalize, eval callbacks | `RL/experiment_manager.py`, `RL/utils.py`, `configs/agents/*.yml` | AI control tab, separate process |
| RL evaluation | `experiments/evaluate_rl.py` | Same normalisation and deterministic prediction |

### Small adapters (written here, documented in code)

| Adapter | Why | Where |
| --- | --- | --- |
| All 208 parameters with name, description, unit, section | Gym2's registry covers 5; the rest are only comments in `default_params.py` | `service/params.py` parses the file; re-evaluates its assignments so derived values (capAir, pBoil, …) follow overrides; startup check proves the re-evaluation equals `init_default_params` exactly |
| Fixed actuators | Gym2 has no "hold this actuator" switch; with raw controls the controller's output can be overridden per step | `service/simulate.py` (rule-based only) |
| Energy in kWh/kg | Gym2 reports costs, not quantities | Same expressions as the reward and aux states, integrated per step; cross-checked against Gym2's costs on every run |
| Recording every state | `evaluate_rl.py` only keeps info dicts | `service/policies.py` applies the saved VecNormalize statistics itself and steps the env |
| macOS training | Gym2's `SubprocVecEnv` workers hit a circular import under macOS's forkserver | `service/train_runner.py` preloads `gl_gym.environments` in the forkserver; also overrides only `total_timesteps`/`n_envs` and names the W&B run (offline) |

### Not available in this Gym2 version

- Weather other than Amsterdam 2008–2012. `processing/fetch_weather_nasa.py` can download NASA POWER data into
  the weather folder; the Lab lists whatever is there, but does not fetch.
- Trained policies: none ship with the repo. Train one in the Lab or drop a Gym2 `train_data` run in.
- Crops other than tomato, planting dates (the crop always starts from the same established state), harvest
  yield (GreenLight's harvest flow is not counted by the reward), inter-lighting, grow pipes, shading screen and
  side vents (parameters exist; Gym2's controls fix them off).
- Time-varying prices: `price_model.py` has hourly/daily classes, but `GreenhouseReward` always builds fixed prices.

## Verified

- `tests/test_service.py`: a default run equals Gym2's README rule-based loop exactly (reward, costs, final fruit);
  energy integration equals Gym2's reward costs; runs are reproducible; bounds are enforced without clamping;
  derived parameters follow overrides; missing weather years are refused.
- A policy trained through the Lab evaluates to the same reward and margin as `experiments/evaluate_rl.py`
  (−645.2708 / −18.5261 € m⁻² on an 8k-step PPO test run).


## Limits

- Model outputs for comparing scenarios, not validated engineering or commercial yield predictions.
- RL training is a local developer feature; the hosted engine has no training dependencies.

## Licence

AGPL-3.0-or-later (see `LICENSE`), as GreenLight-Gym2. GreenLight-Gym2 is © its authors; cite their
AgriControl 2025 paper when using it (see its repository).
