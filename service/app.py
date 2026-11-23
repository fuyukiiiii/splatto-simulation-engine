"""
Splatto Simulation Lab: a minimal local API over GreenLight-Gym2.

Run (from greenhouse-cpq/simulation-lab):
    .venv/bin/python -m uvicorn service.app:app --host 127.0.0.1 --port 8100

Endpoints are plain `def`, so FastAPI runs each in its threadpool: a 60-day
simulation (a few seconds) never holds up another request, and training runs in
its own process (training.py).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import hmac
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import params as P
from . import policies, simulate, training
from .gym2 import ACTUATORS, version_info, weather_catalog

app = FastAPI(title="Splatto Simulation Lab", version="0.1.0")

# The live Design app's page, open in a browser on this computer, calls the engine
# directly; so may the local app. No other site may.
ALLOWED_ORIGINS = ["https://app.splattoworks.com"] + [f"http://localhost:{port}" for port in range(3000, 3011)]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


# Hosted (SIM_LAB_KEY set, as on the Hugging Face Space): only Splatto's server, which sends the
# key, may run anything; the landing page and the health check stay open.
OPEN_PATHS = {"/", "/health"}


@app.middleware("http")
async def require_key(request: Request, call_next):
    key = os.environ.get("SIM_LAB_KEY")
    if key and request.method != "OPTIONS" and request.url.path not in OPEN_PATHS:
        if not hmac.compare_digest(request.headers.get("x-simulation-key", ""), key):
            return JSONResponse({"detail": "Not allowed."}, status_code=401)
    return await call_next(request)


@app.middleware("http")
async def allow_private_network(request: Request, call_next):
    """Chrome asks before a public page reaches localhost; say yes to the allowed origins only."""
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true" and request.headers.get("origin") in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

P.self_check()  # refuse to start if the parameter re-evaluation ever drifts from Gym2's defaults


class Scenario(BaseModel):
    location: str
    growth_year: int
    start_day: int


class Controller(BaseModel):
    type: Literal["rule_based", "rl"] = "rule_based"
    settings: dict[str, float] = Field(default_factory=dict)
    policy_id: Optional[str] = None


class ActuatorOverride(BaseModel):
    mode: Literal["controller", "fixed"] = "controller"
    value: Optional[float] = None


class SimulationRequest(BaseModel):
    scenario: Scenario
    season_length: int
    seed: int = 0
    prices: dict[str, float] = Field(default_factory=dict)
    constraints: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, float] = Field(default_factory=dict)
    controller: Controller = Field(default_factory=Controller)
    actuator_overrides: dict[str, ActuatorOverride] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    algorithm: Literal["ppo", "sac", "recurrentppo"] = "ppo"
    total_timesteps: int
    n_envs: Optional[int] = None
    seed: int = 666


def _rl_available() -> dict:
    try:
        import sb3_contrib  # noqa: F401
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        import wandb  # noqa: F401
        return {"available": True}
    except ImportError as exc:
        return {"available": False, "reason": f"{exc.name} is not installed (setup.sh --train)."}


@app.get("/", response_class=HTMLResponse)
def landing() -> str:
    v = version_info()
    return (
        "<!doctype html><meta charset=utf-8><title>Splatto Simulation Engine</title>"
        "<body style='font:15px/1.6 system-ui;margin:48px;max-width:640px;color:#0c121a'>"
        "<h1 style='font-size:20px'>Splatto Simulation Engine</h1>"
        "<p>The engine behind the Simulation in Splatto Design: GreenLight-Gym2 "
        f"{v['gl_gym_version']} (commit {(v['gym2_commit'] or '')[:7]}), unmodified, AGPL-3.0-or-later.</p>"
        "<p>Source: <a href='https://github.com/fuyukiiiii/splatto-simulation-engine'>this engine</a> and "
        "<a href='https://github.com/BartvLaatum/GreenLight-Gym2'>GreenLight-Gym2</a>.</p></body>"
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, **version_info()}


@app.get("/catalog")
def catalog() -> dict:
    d = simulate.defaults()
    rb_help = {k: {"label": v[0], "unit": v[1]} for k, v in simulate.RULE_BASED_HELP.items()}
    return {
        "version": version_info(),
        "defaults": d,
        "max_season_days": simulate.MAX_SEASON_DAYS,
        "weather": weather_catalog(),
        "actuators": ACTUATORS,
        "rule_based_help": rb_help,
        "rl": _rl_available(),
        "parameter_count": len(P.catalog()),
    }


@app.get("/parameters")
def parameters() -> list[dict]:
    return P.catalog()


@app.post("/simulate")
def run_simulation(req: SimulationRequest) -> dict:
    cfg = simulate.RunConfig(
        location=req.scenario.location,
        growth_year=req.scenario.growth_year,
        start_day=req.scenario.start_day,
        season_length=req.season_length,
        seed=req.seed,
        prices=req.prices,
        constraints=req.constraints,
        parameters=req.parameters,
        controller=req.controller.model_dump(),
        actuator_overrides={k: v.model_dump() for k, v in req.actuator_overrides.items()},
    )
    try:
        return simulate.run(cfg, policy_loader=policies.load)
    except (simulate.SimulationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/rl/policies")
def rl_policies() -> list[dict]:
    return policies.discover()


@app.get("/rl/jobs")
def rl_jobs() -> list[dict]:
    return training.list_jobs()


@app.post("/rl/train")
def rl_train(req: TrainRequest) -> dict[str, Any]:
    try:
        return training.start(req.algorithm, req.total_timesteps, req.n_envs, req.seed)
    except training.TrainingError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/rl/jobs/{job_id}")
def rl_job(job_id: str) -> dict:
    try:
        return training.status(job_id)
    except training.TrainingError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/rl/jobs/{job_id}/cancel")
def rl_cancel(job_id: str) -> dict:
    try:
        return training.cancel(job_id)
    except training.TrainingError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
