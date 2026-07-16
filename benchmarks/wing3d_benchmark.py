"""Resumable TRust-BO run harness for the 65-D SU2 front-wing problem.

This is intentionally a single-method harness: TRust-BO with its residual-GP
Phase 2 enabled.  It is not an A/B benchmark.  Every completed batch is
written to JSONL and the corresponding ``TRustBOEngine.save`` checkpoint is
updated, so an interrupted CFD run can continue without re-running completed
SU2 cases.

Environment variables:
  BUDGET=250 WORKERS=2 NTHREAD=10 ITER=6000 SEED=0
  RESULTS=benchmarks/wing3d_results.jsonl
  ENGINE_STATE=<RESULTS stem>.engine.zip RUN_STATE=<RESULTS stem>.state.json
  SMOKE=1  # budget=8, ITER=800 (use explicit SU2_RUN and SU2_WORK)

Example:
  SU2_RUN=/home/kotaro/su2/bin SU2_WORK=/home/kotaro/su2/work \\
    SMOKE=1 .venv/bin/python benchmarks/wing3d_benchmark.py
"""

from __future__ import annotations

import json
import hashlib
import inspect
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR / "su2"))
sys.path.insert(0, str(BENCHMARK_DIR.parent / "python"))

import wing3d_mesh  # noqa: E402
from wing3d_mesh import baseline_parameters, validate_wing_geometry  # noqa: E402
from wing3d_runner import Wing3DSettings, run_wing3d  # noqa: E402


DIM = 65
BATCH_SIZE = 8
BUDGET = int(os.environ.get("BUDGET", "250"))
WORKERS = int(os.environ.get("WORKERS", "2"))
NTHREAD = int(os.environ.get("NTHREAD", "10"))
ITER = int(os.environ.get("ITER", "6000"))
SEED = int(os.environ.get("SEED", "0"))

_smoke = os.environ.get("SMOKE", "").strip() not in ("", "0", "false", "False")
if _smoke:
    BUDGET = min(BUDGET, 8)
    ITER = min(ITER, 800)

_default_results = BENCHMARK_DIR / (
    "wing3d_results_smoke.jsonl" if _smoke else "wing3d_results.jsonl"
)
RESULTS_PATH = Path(os.environ.get("RESULTS", _default_results))
ENGINE_STATE_PATH = Path(os.environ.get(
    "ENGINE_STATE", RESULTS_PATH.with_suffix(".engine.zip"),
))
RUN_STATE_PATH = Path(os.environ.get(
    "RUN_STATE", RESULTS_PATH.with_suffix(".state.json"),
))

# A fully independent +/-0.02 and 0.65--1.35 box produces only about 20%
# valid shapes: the dense validator correctly rejects abrupt spanwise changes.
# These still vary every physical coefficient independently, but use the
# widest box that gave approximately 80% validity in a 200-point preflight.
CAMBER_DELTA = 0.005
THICKNESS_SCALE = (0.85, 1.15)
PLANFORM_BOUNDS = (
    (0.55, 0.75),  # half_span
    (0.20, 0.35),  # root_chord
    (0.65, 1.00),  # taper_ratio
    (-0.05, 0.08), # le_sweep_offset
    (-5.0, 15.0),  # aoa_deg
)

CONFIG = {
    "batch_size": BATCH_SIZE,
    "enable_phase2": True,
    "acquisition": "ei",
    "phase2_early_frac": 0.25,
}


def _make_space():
    """Return 65 Float definitions in ``parse_wing_parameters`` order."""
    from trust_bo import Float

    base = baseline_parameters()
    space = []
    for station in range(6):
        offset = station * 10
        for coefficient in range(5):
            value = float(base[offset + coefficient])
            space.append(Float(
                f"camber_s{station}_c{coefficient}",
                value - CAMBER_DELTA, value + CAMBER_DELTA,
            ))
        for coefficient in range(5):
            value = float(base[offset + 5 + coefficient])
            space.append(Float(
                f"thickness_s{station}_c{coefficient}",
                value * THICKNESS_SCALE[0], value * THICKNESS_SCALE[1],
            ))
    for name, (low, high) in zip(
        ("half_span", "root_chord", "taper_ratio", "le_sweep_offset", "aoa_deg"),
        PLANFORM_BOUNDS,
        strict=True,
    ):
        space.append(Float(name, low, high))
    assert len(space) == DIM
    return space


def candidate_to_values(candidate: dict[str, float]) -> np.ndarray:
    """Invert ``_make_space`` without relying on dictionary insertion order."""
    values = []
    for station in range(6):
        values.extend(candidate[f"camber_s{station}_c{i}"] for i in range(5))
        values.extend(candidate[f"thickness_s{station}_c{i}"] for i in range(5))
    values.extend(candidate[name] for name in (
        "half_span", "root_chord", "taper_ratio", "le_sweep_offset", "aoa_deg",
    ))
    return np.asarray(values, dtype=float)


def _json_safe(value: Any) -> Any:
    """Make SU2/numpy telemetry safe for durable JSONL output."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(_json_safe(value), f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def _atomic_engine_save(engine, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    engine.save(temporary)
    os.replace(temporary, path)


def _jsonl_append(record: dict) -> None:
    """Atomically rewrite JSONL, dropping only a torn malformed tail record."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    if RESULTS_PATH.exists():
        lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
        _read_jsonl_records(lines)
        existing = [line for line in lines if line.strip()]
        if existing:
            try:
                json.loads(existing[-1])
            except json.JSONDecodeError:
                existing.pop()
    existing.append(json.dumps(_json_safe(record), allow_nan=False))
    temporary = RESULTS_PATH.with_name(RESULTS_PATH.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        f.write("\n".join(existing) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, RESULTS_PATH)


def _load_run_state() -> dict | None:
    if not RUN_STATE_PATH.exists():
        return None
    with open(RUN_STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl_records(lines: list[str]) -> list[dict]:
    """Parse JSONL strictly, tolerating only a malformed final non-empty line."""
    nonempty = [(line_number, line) for line_number, line in enumerate(lines, start=1)
                if line.strip()]
    records = []
    for index, (line_number, line) in enumerate(nonempty):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index != len(nonempty) - 1:
                raise RuntimeError(
                    f"malformed non-tail JSONL line {line_number}"
                ) from exc
            print(f"[resume] discarding torn JSONL tail line {line_number}")
    return records


def _load_batch_record(iteration: int) -> dict | None:
    """Return a durable completed result for a still-pending batch, if any."""
    if not RESULTS_PATH.exists():
        return None
    lines = RESULTS_PATH.read_text(encoding="utf-8").splitlines()
    for record in _read_jsonl_records(lines):
        if record.get("iteration") == iteration and record.get("kind") == "batch":
            return record
    return None


def _same_candidates(left: list[dict], right: list[dict]) -> bool:
    return left == right


def _history_has_batch(engine, candidates: list[dict]) -> bool:
    history = engine.history()
    if len(history) < len(candidates):
        return False
    return [trial.parameters for trial in history[-len(candidates):]] == candidates


def _settings() -> Wing3DSettings:
    # Keep the production 2,500-row criterion, while allowing the requested
    # 800-iteration smoke run to validate a complete (shorter) final window.
    return Wing3DSettings(
        max_iter=ITER,
        n_threads=NTHREAD,
        convergence_window=min(Wing3DSettings().convergence_window, ITER),
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _su2_binary_sha256(settings: Wing3DSettings) -> str:
    executable = os.path.join(settings.su2_run, "SU2_CFD")
    try:
        return _sha256_file(executable)
    except OSError as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def _function_defaults(function) -> dict:
    return {
        name: parameter.default
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _protocol_fingerprint() -> tuple[str, dict]:
    settings = _settings()
    baseline = baseline_parameters()
    rendered_config = wing3d_mesh.render_su2_config(
        "<mesh>", "<history>", baseline, iterations=settings.max_iter,
        density=settings.density, mu=settings.mu, velocity=settings.velocity,
        cauchy_elems=settings.cauchy_elems,
    )
    protocol = {
        "iter": ITER,
        "nthread": NTHREAD,
        "workers": WORKERS,
        "settings": {
            "velocity": settings.velocity, "density": settings.density, "mu": settings.mu,
            "timeout_s": settings.timeout_s, "convergence_window": settings.convergence_window,
            "max_cv": settings.max_cv, "max_mean_drift": settings.max_mean_drift,
            "cauchy_elems": settings.cauchy_elems, "min_cd": settings.min_cd,
        },
        "space": {
            "dim": DIM, "baseline": baseline.tolist(),
            "camber_delta": CAMBER_DELTA, "thickness_scale": THICKNESS_SCALE,
            "planform_bounds": PLANFORM_BOUNDS,
        },
        "su2_binary_sha256": _su2_binary_sha256(settings),
        "su2_config": rendered_config,
        "mesh": {
            "generate_defaults": _function_defaults(wing3d_mesh.generate_wing_mesh),
            "validator_defaults": _function_defaults(wing3d_mesh.validate_wing_geometry),
            "wing_height": wing3d_mesh.WING_HEIGHT,
            "ground_z": wing3d_mesh.GROUND_Z,
            "endplate_bounds": wing3d_mesh.ENDPLATE_BOUNDS,
            "implementation_sha256": _sha256_file(wing3d_mesh.__file__),
            "airfoil_mesh_sha256": _sha256_file(BENCHMARK_DIR / "su2" / "airfoil_mesh.py"),
        },
        "runner_implementation_sha256": _sha256_file(BENCHMARK_DIR / "su2" / "wing3d_runner.py"),
        "benchmark_implementation_sha256": _sha256_file(Path(__file__)),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), protocol


def evaluate_one(candidate: dict[str, float]) -> dict:
    values = candidate_to_values(candidate)
    objective, cd, feasible, info = run_wing3d(values, settings=_settings())
    return {
        "parameters": candidate,
        "values": values,
        "objective": float(objective),
        "cd": float(cd),
        "feasible": bool(feasible),
        "info": info,
    }


def evaluate_batch(candidates: list[dict[str, float]]) -> list[dict]:
    """Run one optimizer batch with the validated 2 x 10-thread layout."""
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        return list(executor.map(evaluate_one, candidates))


def _state_template() -> dict:
    fingerprint, protocol = _protocol_fingerprint()
    return {
        "version": 2,
        "budget": BUDGET,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "protocol_fingerprint": fingerprint,
        "protocol": protocol,
        "next_iteration": 0,
        "pending": None,
    }


def _validate_resume_state(state: dict) -> None:
    expected = {"budget": BUDGET, "batch_size": BATCH_SIZE, "seed": SEED}
    mismatch = {key: (state.get(key), value) for key, value in expected.items()
                if state.get(key) != value}
    if mismatch:
        raise ValueError(
            "checkpoint settings differ from this invocation; choose fresh RESULTS "
            f"paths or restore the original settings: {mismatch}"
        )
    fingerprint, _ = _protocol_fingerprint()
    if state.get("protocol_fingerprint") != fingerprint:
        raise ValueError(
            "evaluation protocol fingerprint differs from this checkpoint; use a new "
            "RESULTS path or restore the original settings "
            f"(stored={state.get('protocol_fingerprint')}, current={fingerprint})"
        )


def _record_to_tell(record: dict) -> list[dict]:
    return [
        {"value": result["objective"], "feasible": result["feasible"]}
        for result in record["results"]
    ]


def _finish_pending(engine, state: dict) -> None:
    """Evaluate/replay a pending batch and make its engine checkpoint durable.

    The pending descriptor is written before the post-ask engine checkpoint.
    Therefore a restart can detect the one possible torn-write ordering and
    reproduce the deterministic ask before it either evaluates or replays the
    durable JSONL batch result.
    """
    pending = state["pending"]
    assert pending is not None
    candidates = pending["candidates"]
    iteration = int(pending["iteration"])
    expected_ask_count = int(pending["expected_ask_count"])

    if engine._ask_count < expected_ask_count:  # checkpoint was before ask
        recreated = engine.ask(batch_size=len(candidates))
        if not _same_candidates(recreated, candidates):
            raise RuntimeError("pending candidates do not match deterministic ask replay")
        _atomic_engine_save(engine, ENGINE_STATE_PATH)
    elif engine._ask_count > expected_ask_count:
        raise RuntimeError("engine checkpoint is ahead of its pending batch")

    batch_record = _load_batch_record(iteration)
    if batch_record is None:
        print(f"  iter={iteration:03d}: evaluating {len(candidates)} candidate(s)", flush=True)
        started = time.perf_counter()
        results = evaluate_batch(candidates)
        batch_record = {
            "kind": "batch",
            "iteration": iteration,
            "evaluated_before": len(engine.history()),
            "n_candidates": len(candidates),
            "elapsed_seconds": time.perf_counter() - started,
            "results": results,
        }
        # Results precede tell/checkpoint.  If power is lost here, restart
        # replays these exact results and does not spend another SU2 batch.
        _jsonl_append(batch_record)
    else:
        print(f"  [resume] replaying durable results for iter={iteration:03d}", flush=True)

    if not _history_has_batch(engine, candidates):
        if [result["parameters"] for result in batch_record["results"]] != candidates:
            raise RuntimeError("JSONL batch candidates do not match pending state")
        engine.tell(candidates, _record_to_tell(batch_record))
        _atomic_engine_save(engine, ENGINE_STATE_PATH)

    state["next_iteration"] = iteration + 1
    state["pending"] = None
    _atomic_json_write(RUN_STATE_PATH, state)
    n_feasible = sum(result["feasible"] for result in batch_record["results"])
    best = engine.best()
    best_value = best["objective_values"][0] if best else 0.0
    print(
        f"    feasible={n_feasible}/{len(candidates)} total={len(engine.history())}/{BUDGET} "
        f"best={best_value:.5f}", flush=True,
    )


def _preflight(space) -> None:
    """Check the center and deterministic cold-start points before SU2 starts."""
    center_ok, _, center_reason = validate_wing_geometry(baseline_parameters())
    if not center_ok:
        raise RuntimeError(f"baseline geometry is invalid: {center_reason}")
    from trust_bo import TRustBOEngine

    probe = TRustBOEngine(space=space, direction="maximize", seed=SEED, config=CONFIG)
    cold = probe.ask(batch_size=min(BATCH_SIZE, BUDGET))
    n_valid = sum(validate_wing_geometry(candidate_to_values(candidate))[0] for candidate in cold)
    print(f"  preflight: baseline=feasible, initial Halton geometry={n_valid}/{len(cold)}")


def main() -> None:
    from trust_bo import TRustBOEngine

    if BUDGET <= 0 or WORKERS <= 0 or NTHREAD <= 0:
        raise ValueError("BUDGET, WORKERS, and NTHREAD must be positive")
    space = _make_space()
    print("=" * 72)
    print("  3-D front-wing: TRust-BO + Phase 2 (single-method run)")
    print(f"  dim={DIM} budget={BUDGET} batch={BATCH_SIZE} workers={WORKERS} "
          f"threads/job={NTHREAD} iter={ITER} seed={SEED}")
    print(f"  output={RESULTS_PATH}")
    print("=" * 72)
    _preflight(space)

    state = _load_run_state()
    if state is None:
        if ENGINE_STATE_PATH.exists():
            raise RuntimeError("engine checkpoint exists but run-state JSON is missing")
        state = _state_template()
        engine = TRustBOEngine(space=space, direction="maximize", seed=SEED, config=CONFIG)
        _atomic_engine_save(engine, ENGINE_STATE_PATH)
        _atomic_json_write(RUN_STATE_PATH, state)
    else:
        _validate_resume_state(state)
        if not ENGINE_STATE_PATH.exists():
            raise RuntimeError("run-state JSON exists but engine checkpoint is missing")
        engine = TRustBOEngine.load(ENGINE_STATE_PATH)
        print(f"  [resume] evaluated={len(engine.history())} next_iter={state['next_iteration']}")

    # This is intentionally logged from the instantiated engine: n_init is
    # not configured above and should be the engine's 65-D adaptive value (50).
    print(f"  effective n_init={engine._config['n_init']}")
    while len(engine.history()) < BUDGET:
        if state["pending"] is None:
            batch = min(BATCH_SIZE, BUDGET - len(engine.history()))
            candidates = engine.ask(batch_size=batch)
            state["pending"] = {
                "iteration": state["next_iteration"],
                "candidates": candidates,
                "expected_ask_count": engine._ask_count,
            }
            # Write pending first.  A restart can reproduce this ask if the
            # following engine checkpoint was not yet atomically replaced.
            _atomic_json_write(RUN_STATE_PATH, state)
            _atomic_engine_save(engine, ENGINE_STATE_PATH)
        _finish_pending(engine, state)

    best = engine.best()
    n_feasible = sum(trial.status == "complete" for trial in engine.history())
    print("=" * 72)
    print(f"  complete: feasible={n_feasible}/{len(engine.history())} "
          f"best={best['objective_values'][0] if best else 0.0:.5f}")
    print(f"  JSONL={RESULTS_PATH}")


if __name__ == "__main__":
    main()
