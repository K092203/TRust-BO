import sys
from pathlib import Path

import numpy as np
import pytest

SU2_DIR = Path(__file__).parents[1] / "benchmarks" / "su2"
sys.path.insert(0, str(SU2_DIR))

from wing3d_mesh import baseline_parameters  # noqa: E402
from wing3d_runner import (  # noqa: E402
    HistoryStats,
    Wing3DSettings,
    _is_converged,
    _read_history,
    run_wing3d,
)


def test_invalid_geometry_returns_before_workdir_or_su2(tmp_path):
    values = baseline_parameters()
    values[5] = -0.01
    objective, cd, feasible, info = run_wing3d(
        values, Wing3DSettings(workroot=str(tmp_path / "must_not_exist")),
    )
    assert (objective, cd, feasible) == (0.0, 0.0, False)
    assert info["workdir"] is None
    assert info["error"] == "geometry_nonpositive_thickness_coefficient"
    assert not (tmp_path / "must_not_exist").exists()


def test_iteration_limit_is_not_mistaken_for_early_convergence(tmp_path):
    settings = Wing3DSettings(max_iter=800, convergence_window=2500)
    stats = HistoryStats(-1.0, 0.1, 800, 799, {}, window_rows=800)
    converged, reason = _is_converged(stats, settings)
    assert not converged
    assert reason == "insufficient_window"


def test_su2_cauchy_accepts_short_history(tmp_path):
    history = tmp_path / "history.csv"
    run_log = tmp_path / "run.log"
    _write_history(history, [-1.0] * 20, [0.1] * 20)
    run_log.write_text(
        "All convergence criteria satisfied.\n"
        "| Cauchy[CL] | 9e-6 | < 1e-5 | Yes |\n"
        "| Cauchy[CD] | 8e-6 | < 1e-5 | Yes |\n"
    )
    settings = Wing3DSettings(convergence_window=100)

    stats = _read_history(str(history), settings.convergence_window, str(run_log))
    converged, reason = _is_converged(stats, settings)

    assert stats.window_rows == 20
    assert stats.su2_cauchy
    assert converged
    assert reason == "su2_cauchy"


def test_sigterm_false_cauchy_is_rejected(tmp_path):
    history = tmp_path / "history.csv"
    run_log = tmp_path / "run.log"
    _write_history(history, [-0.67678] * 23, [0.48515] * 23)
    run_log.write_text(
        "Interrupt signal (15) received, saving files and exiting.\n"
        "All convergence criteria satisfied.\n"
        "| Cauchy[CL] | 0.00727609 | > 1e-5 | No |\n"
        "| Cauchy[CD] | 0.0360563 | > 1e-5 | No |\n"
    )
    settings = Wing3DSettings(convergence_window=100)
    stats = _read_history(str(history), settings.convergence_window, str(run_log),
                          settings.cauchy_elems)
    assert not stats.su2_cauchy
    assert _is_converged(stats, settings) == (False, "insufficient_window")


def test_cauchy_average_uses_configured_tail(tmp_path):
    history = tmp_path / "history.csv"
    run_log = tmp_path / "run.log"
    _write_history(history, [-10.0] * 10 + [-1.0] * 3, [1.0] * 10 + [0.1] * 3)
    run_log.write_text(
        "| Cauchy[CL] | 9e-6 | < 1e-5 | Yes |\n"
        "| Cauchy[CD] | 8e-6 | < 1e-5 | Yes |\n"
    )
    stats = _read_history(str(history), 100, str(run_log), cauchy_elems=3)
    assert stats.su2_cauchy
    assert stats.window_rows == 3
    assert stats.cl == -1.0
    assert stats.cd == pytest.approx(0.1)


def _write_history(path, cl_values, cd_values):
    lines = ['"Inner_Iter","CL","CD","rms[U]"']
    lines.extend(
        f"{i},{cl},{cd},-4.0" for i, (cl, cd) in enumerate(zip(cl_values, cd_values))
    )
    path.write_text("\n".join(lines) + "\n")


def test_windowed_average_accepts_bounded_oscillation(tmp_path):
    history = tmp_path / "history.csv"
    phase = np.linspace(0.0, 4.0 * np.pi, 100)
    _write_history(history, -1.0 + 0.05 * np.sin(phase), 0.1 + 0.005 * np.sin(phase))
    settings = Wing3DSettings(convergence_window=100, max_cv=0.15, max_mean_drift=0.10)
    stats = _read_history(str(history), settings.convergence_window)
    converged, reason = _is_converged(stats, settings)
    assert converged
    assert reason == "window_stable"
    assert abs(stats.cl + 1.0) < 1e-12
    assert abs(stats.cd - 0.1) < 1e-12


def test_large_oscillation_is_infeasible(tmp_path):
    history = tmp_path / "history.csv"
    phase = np.linspace(0.0, 4.0 * np.pi, 100)
    _write_history(history, -1.0 + 0.4 * np.sin(phase), 0.1 + 0.04 * np.sin(phase))
    settings = Wing3DSettings(convergence_window=100, max_cv=0.15)
    converged, reason = _is_converged(
        _read_history(str(history), settings.convergence_window), settings,
    )
    assert not converged
    assert reason == "excessive_variation"


def test_monotonic_drift_is_infeasible(tmp_path):
    history = tmp_path / "history.csv"
    _write_history(history, np.linspace(-0.8, -1.2, 100), np.linspace(0.08, 0.12, 100))
    settings = Wing3DSettings(convergence_window=100, max_cv=0.20, max_mean_drift=0.10)
    converged, reason = _is_converged(
        _read_history(str(history), settings.convergence_window), settings,
    )
    assert not converged
    assert reason == "window_drift"
