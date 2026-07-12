"""SU2 coarse-mesh (LF) -> SU2 normal-mesh (HF) paired-fidelity correlation gate for MF-2'.

MF-2 (docs/BENCHMARK.md §19) found NeuralFoil<->SU2 fails the correlation gate
(R2=0.284 < 0.75). MF-2' tests the same-physics, lower-resolution alternative:
LF = SU2 RANS on a coarse mesh (n_half=60, nj=45, max_iter=800), HF = SU2 RANS
at the normal benchmark resolution. Both fidelities are expensive, so only
Sobol points are evaluated (no LF-selected top-K supplement -- see
docs/ROADMAP.md MF-2').

SMOKE=1 uses 8 Sobol points at ITER=800. Full mode uses 48 points.
Results are resumable per config hash and candidate id.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import qmc, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SU2_DIR = Path(__file__).resolve().parent / "su2"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(SU2_DIR))

from airfoil_mesh import validate_cst_geometry  # noqa: E402
from su2_evaluator import make_space  # noqa: E402
from su2_runner import SU2Settings, run_cst  # noqa: E402

N_DIMS = 16
AOA = 2.0
RE = 3e6
BOUNDS = np.array([[p.low, p.high] for p in make_space()], dtype=float)
SMOKE = bool(os.environ.get("SMOKE"))
N_SOBOL = int(os.environ.get("N_SOBOL", 8 if SMOKE else 48))
ITER = int(os.environ.get("ITER", 800 if SMOKE else 4000))
LF_ITER = int(os.environ.get("LF_ITER", 800))
WORKERS = int(os.environ.get("CONCURRENT", 4 if SMOKE else 8))
THREADS = int(os.environ.get("NTHREAD", 2))
SU2_RUN = os.environ.get("SU2_RUN", "/home/kotaro/su2/bin")
SU2_WORK = os.environ.get("SU2_WORK", "/tmp/trust_bo_su2_coarse")
CSV_PATH = Path(os.environ.get("CSV", "mf_su2_coarse_correlation_smoke.csv" if SMOKE
                                else "mf_su2_coarse_correlation.csv"))
FIELDNAMES = ["config", "candidate_id", *[f"x{i}" for i in range(N_DIMS)],
              "lf_value", "lf_feasible", "lf_error", "lf_elapsed_s",
              "hf_value", "hf_feasible", "hf_error", "hf_elapsed_s"]

CONFIG = hashlib.sha256(json.dumps({
    "bounds": BOUNDS.tolist(), "aoa": AOA, "re": RE, "n_sobol": N_SOBOL,
    "iter": ITER, "lf_iter": LF_ITER, "n_half_lf": 60, "nj_lf": 45,
    "min_max_thickness": 0.06, "min_area": 0.05,
}, sort_keys=True).encode()).hexdigest()[:16]


def to_candidate(x: np.ndarray) -> dict[str, float]:
    return {f"u{i}": float(x[i]) for i in range(8)} | {
        f"l{i}": float(x[8 + i]) for i in range(8)
    }


def to_array(candidate: dict[str, float]) -> np.ndarray:
    return np.array([candidate[f"u{i}"] for i in range(8)] +
                    [candidate[f"l{i}"] for i in range(8)], dtype=float)


def evaluate_lf(candidate: dict[str, float]) -> tuple[float, bool, str, float]:
    x = to_array(candidate)
    valid, _, geom_error = validate_cst_geometry(x[:8], x[8:])
    if not valid:
        return 0.0, False, f"geometry_{geom_error}", 0.0
    settings = SU2Settings(
        aoa=AOA, max_iter=LF_ITER, n_half=60, nj=45,
        su2_run=SU2_RUN, workroot=SU2_WORK, n_threads=THREADS,
    )
    cl, cd, feasible, info = run_cst(x[:8], x[8:], settings=settings)
    value = cl / cd if feasible and cd > 1e-6 else 0.0
    return float(value), bool(feasible), str(info.get("error", "")), float(info.get("elapsed_s", 0.0))


def evaluate_hf(candidate: dict[str, float]) -> tuple[float, bool, str, float]:
    x = to_array(candidate)
    valid, _, geom_error = validate_cst_geometry(x[:8], x[8:])
    if not valid:
        return 0.0, False, f"geometry_{geom_error}", 0.0
    settings = SU2Settings(
        aoa=AOA, max_iter=ITER, su2_run=SU2_RUN,
        workroot=SU2_WORK, n_threads=THREADS,
    )
    cl, cd, feasible, info = run_cst(x[:8], x[8:], settings=settings)
    value = cl / cd if feasible and cd > 1e-6 else 0.0
    return float(value), bool(feasible), str(info.get("error", "")), float(info.get("elapsed_s", 0.0))


def sobol_candidates() -> list[tuple[str, dict[str, float]]]:
    unit = qmc.Sobol(N_DIMS, scramble=True, seed=0).random(N_SOBOL)
    points = qmc.scale(unit, BOUNDS[:, 0], BOUNDS[:, 1])
    return [(f"sobol_{i:03d}", to_candidate(point)) for i, point in enumerate(points)]


def completed_ids() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="") as handle:
        return {row["candidate_id"] for row in csv.DictReader(handle)
                if row.get("config") == CONFIG
                and row.get("hf_error", "") not in {"timeout", "no_clcd"}
                and not row.get("hf_error", "").startswith("infrastructure:")
                and row.get("lf_error", "") not in {"timeout", "no_clcd"}
                and not row.get("lf_error", "").startswith("infrastructure:")}


def append_row(candidate_id, candidate, lf_result, hf_result) -> None:
    lf_value, lf_feasible, lf_error, lf_elapsed = lf_result
    hf_value, hf_feasible, hf_error, hf_elapsed = hf_result
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        x = to_array(candidate)
        writer.writerow({
            "config": CONFIG, "candidate_id": candidate_id,
            **{f"x{i}": f"{x[i]:.17g}" for i in range(N_DIMS)},
            "lf_value": f"{lf_value:.17g}", "lf_feasible": int(lf_feasible),
            "lf_error": lf_error, "lf_elapsed_s": f"{lf_elapsed:.3f}",
            "hf_value": f"{hf_value:.17g}", "hf_feasible": int(hf_feasible),
            "hf_error": hf_error, "hf_elapsed_s": f"{hf_elapsed:.3f}",
        })


def evaluate_pair(item: tuple[str, dict[str, float]]):
    candidate_id, candidate = item
    lf_result = evaluate_lf(candidate)
    hf_result = evaluate_hf(candidate)
    return candidate_id, candidate, lf_result, hf_result


def summarize() -> bool:
    with CSV_PATH.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["config"] == CONFIG]
    lf_feas = np.array([int(row["lf_feasible"]) for row in rows], dtype=bool)
    hf_feas = np.array([int(row["hf_feasible"]) for row in rows], dtype=bool)
    paired = lf_feas & hf_feas
    n_paired = int(np.sum(paired))
    feas_agreement = float(np.mean(lf_feas == hf_feas)) if rows else 0.0
    lf_mean_s = float(np.mean([float(row["lf_elapsed_s"]) for row in rows])) if rows else 0.0
    hf_mean_s = float(np.mean([float(row["hf_elapsed_s"]) for row in rows])) if rows else 0.0
    print(f"n={len(rows)} feas_agreement={feas_agreement:.3f} "
          f"lf_mean_s={lf_mean_s:.1f} hf_mean_s={hf_mean_s:.1f}")
    if n_paired < 3:
        print(f"config={CONFIG} n={len(rows)} paired={n_paired}; insufficient for correlation")
        print("gate=FAIL")
        return False
    lf = np.array([float(row["lf_value"]) for row in rows])[paired]
    hf = np.array([float(row["hf_value"]) for row in rows])[paired]
    pearson = float(np.corrcoef(lf, hf)[0, 1])
    r2 = pearson ** 2
    rho = float(spearmanr(lf, hf).statistic)

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(lf), len(lf))
        value = float(spearmanr(lf[idx], hf[idx]).statistic)
        if np.isfinite(value):
            boots.append(value)
    rho_lower = float(np.quantile(boots, 0.025)) if boots else -1.0

    gate = r2 > 0.75 and rho >= 0.75
    print(f"config={CONFIG} n={len(rows)} paired={n_paired} "
          f"R2={r2:.3f} rho={rho:.3f} rho_p2.5={rho_lower:.3f} "
          f"feas_agreement={feas_agreement:.3f}")
    print(f"gate={'PASS' if gate else 'FAIL'} (R2>0.75 and rho>=0.75)")
    return gate


def main() -> int:
    items = sobol_candidates()
    done = completed_ids()
    pending = [item for item in items if item[0] not in done]
    print(f"config={CONFIG} total={len(items)} pending={len(pending)} workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(evaluate_pair, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                candidate_id, candidate, lf_result, hf_result = future.result()
            except Exception as exc:
                candidate_id, candidate = item
                err = f"infrastructure:{type(exc).__name__}"
                lf_result = (0.0, False, err, 0.0)
                hf_result = (0.0, False, err, 0.0)
            append_row(candidate_id, candidate, lf_result, hf_result)
            print(candidate_id, "lf=", lf_result, "hf=", hf_result)
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
