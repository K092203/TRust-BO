from __future__ import annotations

import json
from typing import Any

import optuna
from optuna import Study
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler, RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial, TrialState

from .._lib import Engine as _RustEngine
from ..engine import _derive_seed
from ..space import _decode, _encode, Categorical, Float, Int


def _dist_to_def(name: str, dist: BaseDistribution):
    if isinstance(dist, FloatDistribution):
        return Float(name, dist.low, dist.high, step=dist.step, log=dist.log)
    if isinstance(dist, IntDistribution):
        return Int(name, dist.low, dist.high, step=dist.step, log=dist.log)
    if isinstance(dist, CategoricalDistribution):
        return Categorical(name, list(dist.choices))
    raise ValueError(f"unsupported distribution: {type(dist)}")


class TrustBoOptunaSampler(BaseSampler):
    """Optuna sampler backed by trust-bo (ask/tell を Optuna に橋渡しする外部層)。"""

    def __init__(self, seed: int = 42, max_history: int = 500, config: dict | None = None) -> None:
        self._seed = seed
        self._max_history = max_history
        self._config = config or {}
        self._rust = _RustEngine()
        self._fallback = RandomSampler(seed=seed)
        self._ss_builder = IntersectionSearchSpace()
        self._ask_count = 0
        self._tr_states: list = []

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial) -> dict[str, BaseDistribution]:
        return self._ss_builder.calculate(study)

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, float]:
        if not search_space:
            return {}

        names = sorted(search_space.keys())
        defs = [_dist_to_def(n, search_space[n]) for n in names]
        n_dims = len(names)

        past: list[FrozenTrial] = study.get_trials(states=(TrialState.COMPLETE,))[-self._max_history:]
        params_list, values_list = [], []
        sign = -1.0 if study.direction == optuna.study.StudyDirection.MINIMIZE else 1.0

        for t in past:
            if t.value is None or not all(n in t.params for n in names):
                continue
            params_list.append([_encode(t.params[n], d) for n, d in zip(names, defs)])
            values_list.append(sign * t.value)

        n_init = max(10, min(2 * (n_dims + 1), 50))
        adaptive_l_init = min(1.0, max(0.3, (5.0 / n_init) ** (1.0 / max(n_dims, 1))))
        config = {
            "n_dims": n_dims, "batch_size": 1, "n_init": n_init,
            "ensemble_size": 5, "epochs": 500, "learning_rate": 5e-4,
            "n_cem_samples": 512, "n_cem_iters": 25,
            "elite_fraction": 0.1, "beta": 2.0, "acquisition": "ei",
            "tau_succ": 3, "tau_fail": 10,
            "l_init": adaptive_l_init, "l_max": 1.0, "l_min": 0.0078125,
            **self._config,
        }

        seed = _derive_seed(self._seed, self._ask_count)
        self._ask_count += 1

        result = json.loads(self._rust.propose(
            params_list, values_list, [True] * len(params_list),
            [], json.dumps(self._tr_states), json.dumps(config), seed,
        ))

        self._tr_states = result["tr_states"]
        normalized = result["candidates"][0]
        return {n: _decode(nv, d) for n, d, nv in zip(names, defs, normalized)}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._fallback.sample_independent(study, trial, param_name, param_distribution)
