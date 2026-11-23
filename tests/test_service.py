"""
Checks that the Lab reports GreenLight-Gym2 as Gym2 itself runs it.

The reference loop below is the README's "Using the rule-based controller
programmatically" example, unchanged apart from collecting results.
"""
import gymnasium as gym
import numpy as np
import pytest
from fastapi.testclient import TestClient

import gl_gym  # noqa: F401
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext

from service import app as A
from service import params as P
from service import simulate as S
from service.app import app
from service.gym2 import rule_based_defaults

client = TestClient(app)
DEFAULT = {"scenario": {"location": "Amsterdam", "growth_year": 2010, "start_day": 59}, "season_length": 60}


def reference_episode():
    env = gym.make("gl_gym/GreenLightTomato-v0", normalize_actions=False)
    env.reset(seed=0)
    controller = RuleBasedController(**rule_based_defaults())
    raw = env.unwrapped
    total, costs = 0.0, 0.0
    while True:
        ctx = StepContext(t=raw.timestep, dt=raw.dt, Np=raw.Np, x_prev=raw.x_prev, x=raw.x, u=raw.u, p=raw.p,
                          d=raw.weather_data, hour_of_day=raw.hour_of_day, day_of_year=raw.day_of_year)
        _, r, te, tr, info = env.step(controller.predict(ctx).astype(np.float32))
        total += r
        costs += info["variable_costs"]
        if te or tr:
            return total, costs, raw.x.copy()


@pytest.fixture(autouse=True)
def roomy_limits(monkeypatch):
    # The suite runs many simulations from one client; the limit itself has its own test.
    monkeypatch.setattr(A, "LIMITER", A.RateLimiter(per_minute=1000, per_hour=1000, per_day=1000))


@pytest.fixture(scope="module")
def baseline():
    res = client.post("/simulate", json=DEFAULT)
    assert res.status_code == 200, res.text
    return res.json()


def test_parameter_reevaluation_matches_gym2():
    P.self_check()
    assert len(P.catalog()) == 208


def test_baseline_matches_gym2_reference(baseline):
    total, costs, x_final = reference_episode()
    m = baseline["metrics"]
    assert baseline["run"]["complete"]
    assert m["reward"]["total"] == pytest.approx(total, rel=1e-9)
    assert m["economics"]["operating_cost"] == pytest.approx(costs, rel=1e-9)
    assert m["crop"]["final"]["cFruit"] == pytest.approx(x_final[25], rel=1e-9)


def test_energy_integration_matches_gym2_costs(baseline):
    assert all(baseline["metrics"]["energy"]["matches_gym2_reward_costs"].values())
    e = baseline["metrics"]["energy"]
    assert e["heating_kwh_m2"] > 0 and e["lighting_kwh_m2"] > 0 and e["co2_kg_m2"] > 0


def test_runs_are_reproducible(baseline):
    again = client.post("/simulate", json=DEFAULT).json()
    assert again["config_hash"] == baseline["config_hash"]
    assert again["metrics"] == baseline["metrics"]


def test_out_of_range_registry_value_is_rejected_not_clamped():
    res = client.post("/simulate", json={**DEFAULT, "parameters": {"lamp_power": 500}})
    assert res.status_code == 422 and "Not clamped" in res.json()["detail"]


def test_unknown_parameter_is_rejected():
    res = client.post("/simulate", json={**DEFAULT, "parameters": {"not_a_parameter": 1}})
    assert res.status_code == 422


def test_weather_outside_the_bundled_years_is_rejected():
    res = client.post("/simulate", json={**DEFAULT, "scenario": {"location": "Amsterdam", "growth_year": 2012, "start_day": 340}})
    assert res.status_code == 422 and "2013" in res.json()["detail"]
    res = client.post("/simulate", json={**DEFAULT, "scenario": {"location": "Madrid", "growth_year": 2010, "start_day": 59}})
    assert res.status_code == 422


def test_fixed_actuator_is_held_and_bounded():
    res = client.post("/simulate", json={**DEFAULT, "season_length": 7, "actuator_overrides": {"uLamp": {"mode": "fixed", "value": 0}}})
    assert res.status_code == 200
    assert res.json()["metrics"]["energy"]["lighting_kwh_m2"] == 0
    res = client.post("/simulate", json={**DEFAULT, "actuator_overrides": {"uLamp": {"mode": "fixed", "value": 1.5}}})
    assert res.status_code == 422


def test_derived_parameters_follow_overrides():
    res = client.post("/simulate", json={**DEFAULT, "season_length": 7, "parameters": {"hAir": 4.5}})
    assert res.status_code == 200
    moved = {d["name"] for d in res.json()["parameters"]["derived_changes"]}
    assert {"capAir", "capCo2Air", "capTop", "capCo2Top"} <= moved


def test_only_the_design_app_may_call_the_engine_from_a_browser():
    preflight = {"access-control-request-method": "POST", "access-control-request-headers": "content-type",
                 "access-control-request-private-network": "true"}
    ok = client.options("/simulate", headers={**preflight, "origin": "https://app.splattoworks.com"})
    assert ok.status_code == 200
    assert ok.headers.get("access-control-allow-origin") == "https://app.splattoworks.com"
    assert ok.headers.get("access-control-allow-private-network") == "true"
    other = client.options("/simulate", headers={**preflight, "origin": "https://example.com"})
    assert other.headers.get("access-control-allow-origin") is None
    assert other.headers.get("access-control-allow-private-network") is None


def test_hosted_engine_runs_only_for_the_key(monkeypatch):
    monkeypatch.setenv("SIM_LAB_KEY", "test-key")
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/catalog").status_code == 401
    assert client.post("/simulate", json={**DEFAULT, "season_length": 2}).status_code == 401
    assert client.post("/simulate", json={**DEFAULT, "season_length": 2}, headers={"x-simulation-key": "wrong"}).status_code == 401
    assert client.post("/simulate", json={**DEFAULT, "season_length": 2}, headers={"x-simulation-key": "test-key"}).status_code == 200


def test_rate_limit_per_caller_and_per_day(monkeypatch):
    monkeypatch.setattr(A, "LIMITER", A.RateLimiter(per_minute=2, per_hour=60, per_day=3))
    short = {**DEFAULT, "season_length": 1}
    assert client.post("/simulate", json=short, headers={"x-simulation-client": "a"}).status_code == 200
    assert client.post("/simulate", json=short, headers={"x-simulation-client": "a"}).status_code == 200
    third = client.post("/simulate", json=short, headers={"x-simulation-client": "a"})
    assert third.status_code == 429 and "Wait a minute" in third.json()["detail"]
    assert client.post("/simulate", json=short, headers={"x-simulation-client": "b"}).status_code == 200
    daily = client.post("/simulate", json=short, headers={"x-simulation-client": "c"})
    assert daily.status_code == 429 and "daily limit" in daily.json()["detail"]


def test_busy_engine_answers_429(monkeypatch):
    import threading
    monkeypatch.setattr(A, "RUNNING", threading.BoundedSemaphore(1))
    A.RUNNING.acquire()
    monkeypatch.setattr(A.RUNNING, "acquire", lambda timeout=None: False)
    res = client.post("/simulate", json={**DEFAULT, "season_length": 1})
    assert res.status_code == 429 and "busy" in res.json()["detail"]


def test_long_run_stops_at_the_time_limit(monkeypatch):
    monkeypatch.setattr(S, "TIME_LIMIT_S", 0.0)
    res = client.post("/simulate", json={**DEFAULT, "season_length": 3})
    assert res.status_code == 504 and "shorter season" in res.json()["detail"]


def test_unbounded_parameter_gets_a_safety_range():
    # cLeaf has no Gym2 bound; a value 1000x its default is refused, not clamped
    unbounded = next(p for p in P.catalog() if p["low"] is None and p["high"] is None and p["default"] > 0)
    res = client.post("/simulate", json={**DEFAULT, "season_length": 1, "parameters": {unbounded["name"]: unbounded["default"] * 1000}})
    assert res.status_code == 422 and "safe range" in res.json()["detail"]


def test_controller_settings_are_bounded():
    res = client.post("/simulate", json={**DEFAULT, "season_length": 1, "controller": {"type": "rule_based", "settings": {"temp_setpoint_day": 80}}})
    assert res.status_code == 422 and "Not clamped" in res.json()["detail"]
    res = client.post("/simulate", json={**DEFAULT, "season_length": 1, "controller": {"type": "rule_based", "settings": {"vent_heat_Pband": 0}}})
    assert res.status_code == 422


def test_seed_and_season_are_validated():
    assert client.post("/simulate", json={**DEFAULT, "seed": -1}).status_code == 422
    assert client.post("/simulate", json={**DEFAULT, "season_length": 0}).status_code == 422
    assert client.post("/simulate", json={**DEFAULT, "season_length": 400}).status_code == 422


def test_no_training_when_switched_off(monkeypatch):
    monkeypatch.setattr(A, "TRAINING", False)
    assert client.post("/rl/train", json={"algorithm": "ppo", "total_timesteps": 1000}).status_code == 403


def test_hosted_engine_refuses_to_start_without_its_key(monkeypatch):
    import importlib
    monkeypatch.setenv("K_SERVICE", "splatto-simulation-engine")
    monkeypatch.delenv("SIM_LAB_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SIM_LAB_KEY"):
        importlib.reload(A)
    monkeypatch.delenv("K_SERVICE")
    importlib.reload(A)
