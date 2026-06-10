from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class Float:
    name: str
    low: float
    high: float
    step: float | None = None
    log: bool = False


@dataclass
class Int:
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False


@dataclass
class Categorical:
    name: str
    choices: list


ParamDef = Float | Int | Categorical


class SearchSpaceManager:
    def __init__(self, space: list[ParamDef]) -> None:
        self._space = space

    @property
    def n_dims(self) -> int:
        return len(self._space)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self._space]

    def encode(self, params: dict[str, Any]) -> list[float]:
        return [_encode(params[p.name], p) for p in self._space]

    def decode(self, normalized: list[float]) -> dict[str, Any]:
        return {p.name: _decode(nv, p) for p, nv in zip(self._space, normalized)}

    def to_list(self) -> list[dict]:
        result = []
        for p in self._space:
            d = {"name": p.name, "type": type(p).__name__}
            d.update({k: v for k, v in vars(p).items() if k != "name"})
            result.append(d)
        return result

    @classmethod
    def from_list(cls, dicts: list[dict]) -> SearchSpaceManager:
        params: list[ParamDef] = []
        for raw in dicts:
            d = dict(raw)
            t = d.pop("type")
            name = d.pop("name")
            if t == "Float":
                params.append(Float(name, **d))
            elif t == "Int":
                params.append(Int(name, **d))
            elif t == "Categorical":
                params.append(Categorical(name, **d))
            else:
                raise ValueError(f"unknown param type: {t}")
        return cls(params)


# --- encode/decode helpers ---

def _encode(value: Any, defn: ParamDef) -> float:
    if isinstance(defn, Float):
        if defn.log:
            lo, hi = math.log(defn.low), math.log(defn.high)
            return float((math.log(float(value)) - lo) / (hi - lo))
        return float((float(value) - defn.low) / (defn.high - defn.low))

    if isinstance(defn, Int):
        span = defn.high - defn.low
        if span == 0:
            return 0.5
        if defn.log:
            lo, hi = math.log(defn.low), math.log(defn.high)
            return float((math.log(float(value)) - lo) / (hi - lo))
        return float((int(value) - defn.low) / span)

    if isinstance(defn, Categorical):
        n = len(defn.choices)
        if n <= 1:
            return 0.5
        return float(defn.choices.index(value) / (n - 1))

    return 0.5


def _decode(nv: float, defn: ParamDef) -> Any:
    nv = max(0.0, min(1.0, float(nv)))

    if isinstance(defn, Float):
        if defn.log:
            lo, hi = math.log(defn.low), math.log(defn.high)
            raw = math.exp(lo + nv * (hi - lo))
        else:
            raw = defn.low + nv * (defn.high - defn.low)
        if defn.step is not None:
            raw = round(raw / defn.step) * defn.step
        return float(max(defn.low, min(defn.high, raw)))

    if isinstance(defn, Int):
        if defn.log:
            lo, hi = math.log(defn.low), math.log(defn.high)
            raw = int(round(math.exp(lo + nv * (hi - lo))))
        else:
            raw = int(round(defn.low + nv * (defn.high - defn.low)))
        raw = max(defn.low, min(defn.high, raw))
        if defn.step > 1:
            raw = int(round((raw - defn.low) / defn.step)) * defn.step + defn.low
            raw = max(defn.low, min(defn.high, raw))
        return int(raw)

    if isinstance(defn, Categorical):
        n = len(defn.choices)
        idx = max(0, min(n - 1, round(nv * (n - 1))))
        return defn.choices[idx]

    return nv
