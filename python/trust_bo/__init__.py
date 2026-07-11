from .engine import TRustBOEngine
from .multifidelity import CascadeMFEngine
from .multiobjective import MultiObjectiveEngine, hypervolume_2d
from .rolling_engine import JobEvaluator, MockEvaluator, RollingTRustBOEngine, SlurmEvaluator
from .space import Categorical, Float, Int

__all__ = [
    "TRustBOEngine",
    "CascadeMFEngine",
    "MultiObjectiveEngine", "hypervolume_2d",
    "Float", "Int", "Categorical",
    "JobEvaluator", "MockEvaluator", "RollingTRustBOEngine", "SlurmEvaluator",
]

# Deprecated sklearn/scipy-based engines (legacy-tandem extra). Import lazily so the
# core package works without scipy/sklearn installed; only exposed if the extra is present.
try:  # pragma: no cover
    from .tandem import TandemEngine, TandemEngineV2
    __all__ += ["TandemEngine", "TandemEngineV2"]
except ImportError:
    pass
