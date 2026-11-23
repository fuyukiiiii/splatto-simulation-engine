"""
GreenLight's parameter vector, read from Gym2's own source.

Gym2 has two parameter sources, and the Lab shows both as they are:

* `TOMATO_PARAMETER_REGISTRY` (gl_gym/configs/greenlight_parameters.py): the
  five named parameters Gym2 itself lets you override, with the bounds Gym2
  declares. Out-of-bounds values are rejected here, never clamped.
* `init_default_params` (gl_gym/configs/default_params.py): all 208 nominal
  values, each line commented `# name; description [unit]` under a section
  heading. Some values are derived from others (`pBoil = 130 * aFlr`,
  `capAir = hAir * rhoAir * cPAir`, ...).

To change any of the 208, the Lab re-evaluates that file's assignments in order
with the overrides in place, so derived values follow their own formulas (a new
`hAir` gives a new `capAir`), exactly as if the file had been edited. Values
that are overridden directly are never recomputed. The resulting vector goes to
Gym2 through its own `"set"` parameter provider. Gym2 declares no range for
parameters outside its registry, so none is invented: such values only have to
be finite numbers, and the run reports which were changed and which derived
values moved with them.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

import gl_gym.configs.default_params as default_params_module
from gl_gym.configs.default_params import init_default_params
from gl_gym.configs.greenlight_parameters import TOMATO_PARAMETER_REGISTRY

from .gym2 import PACKAGE_ROOT

NUM_PARAMS = 208

_ASSIGN = re.compile(r"^\s*params\[(\d+)\]\s*=\s*(.+?)\s*;?\s*(?:#\s*(.*))?$")
_SECTION = re.compile(r"^\s*#\s*(.+?)\s*$")
_REF = re.compile(r"params\[(\d+)\]")
_UNIT = re.compile(r"\[([^\[\]]*)\]\s*$")


class ParameterError(ValueError):
    pass


@lru_cache(maxsize=1)
def _source_lines() -> list[str]:
    return Path(default_params_module.__file__).read_text().splitlines()


@lru_cache(maxsize=1)
def assignments() -> list[dict]:
    """Every `params[i] = expr` in default_params.py, in order, with its comment and section."""
    out = []
    section = "Other"
    pending_comment = None
    for line in _source_lines():
        stripped = line.strip()
        m = _ASSIGN.match(line)
        if m:
            index, expr, comment = int(m.group(1)), m.group(2).rstrip(";").strip(), m.group(3)
            if comment is None and pending_comment:
                comment = pending_comment
            pending_comment = None
            out.append({"index": index, "expr": expr, "comment": comment or "", "section": section})
            continue
        s = _SECTION.match(line)
        if s and stripped.startswith("#"):
            text = s.group(1)
            # A heading is an all-caps line ("# SOIL PARAMETERS") or the file's first heading;
            # anything else is the comment of the assignment that follows (e.g. capPipe).
            if text.isupper() or text.lower().startswith("physical constants"):
                section = text.title() if text.isupper() else text[0].upper() + text[1:]
                pending_comment = None
            else:
                pending_comment = text
    return out


def _parse_comment(comment: str) -> tuple[str, str, str]:
    """`name; description [unit]` -> (name, description, unit). Unit is '' where the source gives none."""
    comment = comment.strip()
    if ";" in comment:
        name, rest = comment.split(";", 1)
    else:
        parts = comment.split(None, 1)
        name, rest = (parts[0], parts[1] if len(parts) > 1 else "")
    name = name.strip().rstrip(";")
    rest = rest.strip()
    unit = ""
    m = _UNIT.search(rest)
    if m:
        unit = m.group(1).strip()
        rest = rest[: m.start()].strip()
    return name, rest, unit


@lru_cache(maxsize=1)
def _model_references() -> set[int]:
    """Parameter indices the model equations and the reward read (comment lines skipped)."""
    refs: set[int] = set()
    files = [
        PACKAGE_ROOT / "models" / "GreenLight" / "aux_states.py",
        PACKAGE_ROOT / "models" / "GreenLight" / "ode.py",
        PACKAGE_ROOT / "components" / "rewards.py",
    ]
    for f in files:
        for line in f.read_text().splitlines():
            code = line.split("#", 1)[0]
            refs.update(int(i) for i in re.findall(r"\bp\[(\d+)\]", code))
    return refs


def _evaluate(overrides: dict[int, float]) -> np.ndarray:
    """default_params.py's assignments in order, overridden indices held at their override."""
    params = np.zeros(NUM_PARAMS, dtype=np.float32)  # the same dtype init_default_params uses
    namespace = {"params": params, "np": np, "pow": pow}
    for a in assignments():
        i = a["index"]
        if i in overrides:
            params[i] = overrides[i]
        else:
            params[i] = eval(a["expr"], {"__builtins__": {}}, namespace)  # noqa: S307 - Gym2's own source
    for i, v in overrides.items():
        params[i] = v
    return params.astype(np.float64)


@lru_cache(maxsize=1)
def catalog() -> list[dict]:
    defaults = np.asarray(init_default_params(NUM_PARAMS), dtype=np.float64)
    used = _model_references()
    registry = {p.index: p for p in TOMATO_PARAMETER_REGISTRY.params}
    by_index = {a["index"]: a for a in assignments()}
    # Indices a model-used parameter is derived from count as used too.
    derived_from: dict[int, set[int]] = {i: set(int(r) for r in _REF.findall(a["expr"])) for i, a in by_index.items()}
    effective = set(used)
    changed = True
    while changed:
        changed = False
        for i in list(effective):
            for src in derived_from.get(i, ()):
                if src not in effective:
                    effective.add(src)
                    changed = True

    out = []
    for i in range(NUM_PARAMS):
        a = by_index.get(i)
        name, description, unit = _parse_comment(a["comment"]) if a else (f"p{i}", "", "")
        reg = registry.get(i)
        sources = sorted(derived_from.get(i, ()))
        out.append({
            "index": i,
            "name": name,
            "registry_name": reg.name if reg else None,
            "description": description,
            "unit": (reg.unit if reg and reg.unit else unit),
            "category": a["section"] if a else "Other",
            "default": float(defaults[i]),
            "low": reg.low if reg else None,
            "high": reg.high if reg else None,
            "formula": a["expr"] if sources else None,
            "derived_from": sources,
            "used_by_model": i in effective,
            "used_directly": i in used,
        })
    return out


def _by_name() -> dict[str, dict]:
    m = {}
    for p in catalog():
        m[p["name"]] = p
        if p["registry_name"]:
            m[p["registry_name"]] = p
    return m


def _outside_safety_range(default: float, v: float) -> str | None:
    """
    Safety range, not physics: within a factor of 100 of GreenLight's default, same sign, so no
    request can push the solver into extremes that take minutes. Gym2's own bounds come first.
    Returns the allowed range as text when `v` is outside it.
    """
    if default > 0:
        lo, hi = default / 100, default * 100
    elif default < 0:
        lo, hi = default * 100, default / 100
    else:
        lo, hi = -1000.0, 1000.0
    return None if lo <= v <= hi else f"{lo:g} to {hi:g}"


def resolve_overrides(raw: dict[str, float] | None) -> tuple[dict[int, float], list[dict]]:
    """Validate overrides by name (GreenLight name or Gym2 registry name). Raises on anything unsupported."""
    raw = raw or {}
    names = _by_name()
    resolved: dict[int, float] = {}
    report = []
    for key, value in raw.items():
        p = names.get(key)
        if p is None and key.isdigit() and int(key) < NUM_PARAMS:
            p = catalog()[int(key)]
        if p is None:
            raise ParameterError(f"Unknown parameter '{key}'. Use a name from the parameter explorer.")
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ParameterError(f"Parameter '{key}' must be a number, got {value!r}.")
        if not math.isfinite(v):
            raise ParameterError(f"Parameter '{key}' must be finite, got {value!r}.")
        if p["low"] is not None and v < p["low"]:
            raise ParameterError(f"{p['registry_name'] or p['name']} = {v:g} is below Gym2's bound {p['low']:g} {p['unit']}. Not clamped: change the value.")
        if p["high"] is not None and v > p["high"]:
            raise ParameterError(f"{p['registry_name'] or p['name']} = {v:g} is above Gym2's bound {p['high']:g} {p['unit']}. Not clamped: change the value.")
        if p["low"] is None and p["high"] is None:
            bad = _outside_safety_range(p["default"], v)
            if bad:
                raise ParameterError(f"{p['name']} = {v:g} is outside the safe range {bad} {p['unit']}. Not clamped: change the value.")
        if p["index"] in resolved and resolved[p["index"]] != v:
            raise ParameterError(f"Parameter '{key}' is given twice with different values.")
        resolved[p["index"]] = v
        report.append({"index": p["index"], "name": p["name"], "registry_name": p["registry_name"], "value": v, "default": p["default"], "unit": p["unit"], "bounded_by_gym2": p["low"] is not None or p["high"] is not None})
    return resolved, report


def build_vector(raw_overrides: dict[str, float] | None) -> tuple[np.ndarray, dict]:
    """The full 208-vector for these overrides, and what changed."""
    resolved, report = resolve_overrides(raw_overrides)
    vector = _evaluate(resolved)
    defaults = np.asarray(init_default_params(NUM_PARAMS), dtype=np.float64)
    cat = catalog()
    derived_changes = [
        {"index": i, "name": cat[i]["name"], "default": float(defaults[i]), "value": float(vector[i]), "unit": cat[i]["unit"]}
        for i in range(NUM_PARAMS)
        if i not in resolved and not np.isclose(vector[i], defaults[i], rtol=0, atol=0)
    ]
    if not np.all(np.isfinite(vector)):
        bad = [cat[i]["name"] for i in range(NUM_PARAMS) if not np.isfinite(vector[i])]
        raise ParameterError(f"These parameters became non-finite from the overrides: {', '.join(bad)}.")
    return vector, {"overrides": report, "derived_changes": derived_changes}


def self_check() -> None:
    """Re-evaluating default_params.py with no overrides must reproduce init_default_params exactly."""
    ours = _evaluate({})
    theirs = np.asarray(init_default_params(NUM_PARAMS), dtype=np.float64)
    if not np.array_equal(ours, theirs):
        diff = [i for i in range(NUM_PARAMS) if ours[i] != theirs[i]]
        raise RuntimeError(f"Parameter re-evaluation differs from Gym2 at indices {diff}")
