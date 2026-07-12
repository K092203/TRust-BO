from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SU2_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "su2"
sys.path.insert(0, str(SU2_DIR))

from airfoil_mesh import (  # noqa: E402
    cst_geometry_metrics,
    naca00xx_weights,
    validate_cst_geometry,
)
from su2_runner import SU2Settings, run_cst  # noqa: E402


def test_naca0012_geometry_passes() -> None:
    wu, wl = naca00xx_weights()
    valid, metrics, reason = validate_cst_geometry(wu, wl)
    assert valid and reason is None
    assert metrics["max_thickness"] > 0.11
    assert metrics["area"] > 0.08


def test_equal_and_reversed_surfaces_fail() -> None:
    weights = np.full(8, 0.1)
    valid, _, reason = validate_cst_geometry(weights, weights)
    assert not valid and reason == "surface_crossing"
    valid, _, reason = validate_cst_geometry(-weights, weights)
    assert not valid and reason == "surface_crossing"


def test_thickness_and_area_thresholds_are_independent() -> None:
    wu, wl = naca00xx_weights()
    metrics = cst_geometry_metrics(wu, wl)
    valid, _, reason = validate_cst_geometry(
        wu, wl, min_max_thickness=metrics["max_thickness"] + 1e-6, min_area=0.0
    )
    assert not valid and reason == "max_thickness"
    valid, _, reason = validate_cst_geometry(
        wu, wl, min_max_thickness=0.0, min_area=metrics["area"] + 1e-6
    )
    assert not valid and reason == "section_area"


def test_geometry_metrics_are_deterministic() -> None:
    wu, wl = naca00xx_weights()
    assert cst_geometry_metrics(wu, wl) == cst_geometry_metrics(wu, wl)


def test_nonfinite_weights_fail() -> None:
    wu, wl = naca00xx_weights()
    wu[2] = np.nan
    valid, metrics, reason = validate_cst_geometry(wu, wl)
    assert not valid and not metrics and reason == "nonfinite_geometry"


def test_invalid_geometry_fails_before_workdir_or_solver(tmp_path: Path) -> None:
    weights = np.full(8, 0.1)
    settings = SU2Settings(workroot=str(tmp_path), su2_run="/does/not/exist")
    cl, cd, feasible, info = run_cst(weights, weights, settings=settings)
    assert (cl, cd, feasible) == (0.0, 0.0, False)
    assert info["error"] == "geometry_surface_crossing"
    assert info["workdir"] is None
    assert list(tmp_path.iterdir()) == []


def test_ultrathin_positive_geometry_fails() -> None:
    lower = np.full(8, 0.05)
    upper = lower + 1e-6
    valid, _, reason = validate_cst_geometry(upper, lower)
    assert not valid and reason in {"surface_crossing", "max_thickness"}


def test_removed_tandem_api_is_not_exported() -> None:
    import trust_bo

    assert not hasattr(trust_bo, "TandemEngine")
    assert not hasattr(trust_bo, "TandemEngineV2")
