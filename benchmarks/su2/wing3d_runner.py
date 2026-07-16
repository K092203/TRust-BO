"""SU2 wrapper for the 65-D Formula Student front-wing benchmark.

The Rust optimizer maximizes objective values.  This wrapper therefore returns
the positive downforce-to-drag ratio ``-CL / CD`` (a front wing has ``CL < 0``
at the intended incidence), rather than SU2's signed lift coefficient.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

try:
    from .wing3d_mesh import (
        baseline_parameters,
        generate_wing_mesh,
        mesh_quality,
        render_su2_config,
        validate_mesh_topology,
        validate_wing_geometry,
        write_su2,
    )
except ImportError:  # direct execution from benchmarks/su2
    from wing3d_mesh import (
        baseline_parameters,
        generate_wing_mesh,
        mesh_quality,
        render_su2_config,
        validate_mesh_topology,
        validate_wing_geometry,
        write_su2,
    )


# Unlike the legacy 2-D wrapper, the executable path exists on this host and
# the default work root is writable even when the SU2 installation is read-only.
SU2_RUN_DEFAULT = os.environ.get("SU2_RUN", "/home/kotaro/su2/bin")
SU2_WORK_DEFAULT = os.environ.get("SU2_WORK", "/tmp/trust_bo_su2_wing3d")


@dataclass
class Wing3DSettings:
    velocity: tuple[float, float, float] = (11.0, 0.0, 0.0)
    density: float = 1.225
    mu: float = 1.81e-5
    max_iter: int = 6000
    n_threads: int = 4
    timeout_s: float = 1800.0
    keep_workdir: bool = False
    workroot: str = field(default_factory=lambda: SU2_WORK_DEFAULT)
    su2_run: str = SU2_RUN_DEFAULT
    # Average over approximately one observed lift/drag oscillation.  A full
    # window is required unless SU2 itself satisfies its Cauchy criteria.
    convergence_window: int = 2500
    max_cv: float = 0.15
    # Relative difference between the first- and second-half window means.
    # This rejects sustained drift while allowing bounded periodic motion.
    max_mean_drift: float = 0.10
    cauchy_elems: int = 300
    min_cd: float = 1.0e-3


@dataclass(frozen=True)
class HistoryStats:
    cl: float | None
    cd: float | None
    n_rows: int
    inner_iter: int | None
    residuals: dict[str, float]
    window_rows: int = 0
    cl_std: float | None = None
    cd_std: float | None = None
    cl_cv: float | None = None
    cd_cv: float | None = None
    cl_mean_drift: float | None = None
    cd_mean_drift: float | None = None
    su2_cauchy: bool = False


def _clean_header(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def _log_has_valid_cauchy_table(run_log: str | None) -> bool:
    """Trust only the final CL/CD Cauchy table, never SU2's prose message."""
    if run_log is None:
        return False
    try:
        with open(run_log, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    if "interrupt signal" in text.lower():
        return False
    statuses: dict[str, bool] = {}
    pattern = re.compile(r"\|\s*Cauchy\[(CL|CD)\]\s*\|.*?\|\s*(Yes|No)\s*\|", re.IGNORECASE)
    for field, status in pattern.findall(text):
        statuses[field.upper()] = status.lower() == "yes"
    return statuses == {"CL": True, "CD": True}


def _read_history(history_csv: str, window: int = 2500,
                  run_log: str | None = None, cauchy_elems: int = 300) -> HistoryStats:
    """Read window-averaged CL/CD and coefficient stability statistics.

    SU2's CSV spelling has varied across versions, so coefficient matching is
    deliberately case-insensitive and accepts both CL/CD and LIFT/DRAG names.
    """
    su2_cauchy = _log_has_valid_cauchy_table(run_log)

    try:
        with open(history_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return HistoryStats(None, None, 0, None, {}, su2_cauchy=su2_cauchy)
    if not rows:
        return HistoryStats(None, None, 0, None, {}, su2_cauchy=su2_cauchy)

    last = {str(k).strip().strip('"'): str(v).strip() for k, v in rows[-1].items()
            if k is not None and v is not None}
    cl_key = cd_key = None
    inner_iter = None
    residuals: dict[str, float] = {}
    for key, raw in last.items():
        normalized = _clean_header(key)
        try:
            value = float(raw)
        except ValueError:
            continue
        if normalized in {"CL", "LIFT", "LIFTCOEFF", "LIFTCOEFFICIENT"}:
            cl_key = key
        elif normalized in {"CD", "DRAG", "DRAGCOEFF", "DRAGCOEFFICIENT"}:
            cd_key = key
        elif normalized in {"INNERITER", "ITER", "ITERATION"}:
            inner_iter = int(value)
        elif normalized.startswith("RMS"):
            residuals[key] = value
    if cl_key is None or cd_key is None:
        return HistoryStats(
            None, None, len(rows), inner_iter, residuals, su2_cauchy=su2_cauchy,
        )

    coefficients: list[tuple[float, float]] = []
    averaging_window = cauchy_elems if su2_cauchy else window
    if averaging_window <= 0:
        return HistoryStats(None, None, len(rows), inner_iter, residuals,
                            su2_cauchy=su2_cauchy)
    for row in rows[-averaging_window:]:
        cleaned = {str(k).strip().strip('"'): v for k, v in row.items() if k is not None}
        try:
            pair = (float(cleaned[cl_key]), float(cleaned[cd_key]))
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(pair)):
            coefficients.append(pair)
    if not coefficients:
        return HistoryStats(
            None, None, len(rows), inner_iter, residuals, su2_cauchy=su2_cauchy,
        )

    array = np.asarray(coefficients, dtype=float)
    means = np.mean(array, axis=0)
    stds = np.std(array, axis=0)
    cvs = np.divide(stds, np.abs(means), out=np.full(2, np.inf), where=means != 0.0)
    midpoint = len(array) // 2
    if midpoint:
        first = np.mean(array[:midpoint], axis=0)
        second = np.mean(array[midpoint:], axis=0)
        drifts = np.divide(
            np.abs(second - first), np.abs(means), out=np.full(2, np.inf),
            where=means != 0.0,
        )
    else:
        drifts = np.asarray([np.inf, np.inf])
    return HistoryStats(
        float(means[0]), float(means[1]), len(rows), inner_iter, residuals,
        len(array), float(stds[0]), float(stds[1]), float(cvs[0]), float(cvs[1]),
        float(drifts[0]), float(drifts[1]), su2_cauchy,
    )


def _parse_clcd(history_csv: str) -> tuple[float | None, float | None, int]:
    """Compatibility-sized CL/CD parser, analogous to ``su2_runner.py``."""
    stats = _read_history(history_csv)
    return stats.cl, stats.cd, stats.n_rows


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_trajectory(history_csv: str) -> tuple[list[dict], dict | None]:
    try:
        with open(history_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return [], None
    if not rows:
        return [], None

    def compact(row: dict) -> dict:
        result = {}
        for key, raw in row.items():
            if key is None or raw is None:
                continue
            clean_key = key.strip().strip('"')
            normalized = _clean_header(clean_key)
            if normalized in {"INNERITER", "CL", "CD"} or normalized.startswith("RMS"):
                try:
                    result[clean_key] = (int(float(raw)) if normalized == "INNERITER"
                                         else float(raw))
                except ValueError:
                    result[clean_key] = None
        return result

    snapshots = []
    checkpoints = {0, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500}
    for row in rows:
        try:
            iteration = int(float(row.get("Inner_Iter", "")))
        except (TypeError, ValueError):
            continue
        if iteration in checkpoints:
            snapshots.append(compact(row))
    final = compact(rows[-1])
    if not snapshots or snapshots[-1].get("Inner_Iter") != final.get("Inner_Iter"):
        snapshots.append(final)
    return snapshots, final


def _append_trajectory(path: str, values: np.ndarray, info: dict, feasible: bool,
                       history_csv: str | None = None) -> None:
    snapshots, final_history = _read_trajectory(history_csv) if history_csv else ([], None)
    record = _json_safe({
        "values": np.asarray(values).tolist(),
        "su2_started": bool(history_csv and os.path.exists(history_csv)),
        "feasible": bool(feasible),
        "info": info,
        "snapshots": snapshots,
        "final_history": final_history,
    })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")


def _try_append_trajectory(path: str, values: np.ndarray, info: dict, feasible: bool,
                           history_csv: str | None = None) -> None:
    """Best-effort logging: telemetry I/O must never change CFD validity."""
    try:
        _append_trajectory(path, values, info, feasible, history_csv)
    except Exception as exc:  # noqa: BLE001 - isolated measurement output
        info.setdefault("trajectory_error", f"{type(exc).__name__}: {exc}")


def _is_converged(stats: HistoryStats, settings: Wing3DSettings) -> tuple[bool, str]:
    """Accept bounded coefficient oscillation, but reject short or drifting runs."""
    if stats.su2_cauchy:
        return True, "su2_cauchy"
    if stats.window_rows < settings.convergence_window:
        return False, "insufficient_window"
    metrics = (stats.cl_cv, stats.cd_cv, stats.cl_mean_drift, stats.cd_mean_drift)
    if any(value is None or not np.isfinite(value) for value in metrics):
        return False, "invalid_window_statistics"
    if max(stats.cl_cv, stats.cd_cv) > settings.max_cv:
        return False, "excessive_variation"
    if max(stats.cl_mean_drift, stats.cd_mean_drift) > settings.max_mean_drift:
        return False, "window_drift"
    return True, "window_stable"


def run_wing3d(values: np.ndarray, settings: Wing3DSettings | None = None,
               save_trajectory_path: str | None = None) -> tuple[float, float, bool, dict]:
    """Run a 65-D front wing in SU2.

    Returns ``(objective, cd, feasible, info)``.  ``objective`` is the value
    to maximize, namely downforce-to-drag ``-CL / CD``; it is not CL itself.
    """
    s = settings or Wing3DSettings()
    values = np.asarray(values, dtype=float)
    t0 = time.perf_counter()
    valid_geometry, geometry, geometry_error = validate_wing_geometry(values)
    info: dict = {"workdir": None, "geometry": geometry}
    feasible = False
    workdir: str | None = None
    if not valid_geometry:
        info["error"] = f"geometry_{geometry_error}"
        info["elapsed_s"] = time.perf_counter() - t0
        if save_trajectory_path is not None:
            _try_append_trajectory(save_trajectory_path, values, info, feasible)
        return 0.0, 0.0, False, info

    try:
        os.makedirs(s.workroot, exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="su2_wing3d_", dir=s.workroot)
    except OSError as exc:
        info["error"] = f"workdir_create: {exc}"
        info["elapsed_s"] = time.perf_counter() - t0
        if save_trajectory_path is not None:
            _try_append_trajectory(save_trajectory_path, values, info, feasible)
        return 0.0, 0.0, False, info
    info["workdir"] = workdir
    history_csv = os.path.join(workdir, "history.csv")
    try:
        try:
            mesh = generate_wing_mesh(values)
        except Exception as exc:  # noqa: BLE001 - geometry/mesh failures are infeasible
            info["error"] = f"mesh_gen: {exc}"
            return 0.0, 0.0, False, info
        info["mesh"] = mesh_quality(mesh)
        topology_ok, topology, topology_error = validate_mesh_topology(mesh)
        info["mesh_topology"] = topology
        if not topology_ok:
            info["error"] = f"mesh_topology_{topology_error}"
            return 0.0, 0.0, False, info

        mesh_path = os.path.join(workdir, "mesh.su2")
        write_su2(mesh, mesh_path)
        history_base = os.path.join(workdir, "history")
        cfg_path = os.path.join(workdir, "case.cfg")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(render_su2_config(
                mesh_path, history_base, values, iterations=s.max_iter,
                density=s.density, mu=s.mu, velocity=s.velocity,
                cauchy_elems=s.cauchy_elems,
            ))

        env = dict(os.environ)
        env["PATH"] = s.su2_run + ":" + env.get("PATH", "")
        env["SU2_RUN"] = s.su2_run
        env["OMP_NUM_THREADS"] = str(s.n_threads)
        log_path = os.path.join(workdir, "run.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            try:
                rc = subprocess.run(
                    [os.path.join(s.su2_run, "SU2_CFD"), cfg_path],
                    cwd=workdir, env=env, stdout=logf, stderr=subprocess.STDOUT,
                    timeout=s.timeout_s,
                ).returncode
            except subprocess.TimeoutExpired:
                info["error"] = "timeout"
                return 0.0, 0.0, False, info
        info["returncode"] = rc
        if rc != 0:
            info["error"] = "su2_returncode"
            return 0.0, 0.0, False, info

        stats = _read_history(history_csv, s.convergence_window, log_path, s.cauchy_elems)
        cl, cd = stats.cl, stats.cd
        info.update({
            "cl": cl, "cd": cd, "iters": stats.n_rows, "inner_iter": stats.inner_iter,
            "final_rms": stats.residuals, "window_rows": stats.window_rows,
            "cl_std": stats.cl_std, "cd_std": stats.cd_std,
            "cl_cv": stats.cl_cv, "cd_cv": stats.cd_cv,
            "cl_mean_drift": stats.cl_mean_drift, "cd_mean_drift": stats.cd_mean_drift,
            "su2_cauchy": stats.su2_cauchy,
        })
        if cl is None or cd is None:
            info["error"] = "no_clcd"
            return 0.0, 0.0, False, info
        if not (np.isfinite(cl) and np.isfinite(cd)) or cd <= s.min_cd:
            info["error"] = "nonphysical"
            return 0.0, cd, False, info
        converged, convergence_reason = _is_converged(stats, s)
        info["converged"] = converged
        info["convergence_reason"] = convergence_reason
        if not converged:
            info["error"] = "not_converged"
            return 0.0, cd, False, info

        # A valid front-wing solution must produce downforce in this coordinate
        # system.  Treat positive CL as an orientation/configuration error, not
        # something to be hidden by flipping the objective sign.
        if cl >= 0.0:
            info["error"] = "unexpected_nonnegative_cl"
            return 0.0, cd, False, info
        objective = -cl / cd
        if not np.isfinite(objective):
            info["error"] = "nonfinite_objective"
            return 0.0, cd, False, info
        feasible = True
        return objective, cd, True, info
    finally:
        info["elapsed_s"] = time.perf_counter() - t0
        if save_trajectory_path is not None:
            _try_append_trajectory(save_trajectory_path, values, info, feasible, history_csv)
        if workdir is not None and not s.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    settings = Wing3DSettings(
        max_iter=int(os.environ.get("ITER", "6000")),
        n_threads=int(os.environ.get("NTHREAD", "4")),
        timeout_s=float(os.environ.get("TIMEOUT", "1800")),
        keep_workdir=True,
    )
    objective, cd, feasible, info = run_wing3d(baseline_parameters(), settings)
    print(f"feasible={feasible} objective={objective:.5f} CD={cd:.6f} "
          f"CL={info.get('cl')} iters={info.get('inner_iter')} "
          f"convergence={info.get('convergence_reason')} error={info.get('error')}")
    print(f"workdir={info.get('workdir')} elapsed={info.get('elapsed_s', 0):.1f}s")
