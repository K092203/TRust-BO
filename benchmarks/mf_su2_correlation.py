"""NeuralFoil xlarge -> SU2 paired-fidelity correlation gate for MF-2.

SMOKE=1 uses 8 Sobol points plus 4 LF-selected points. Full mode uses
64 + 32 points. Results are resumable per config hash and candidate id.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
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
N_SOBOL = int(os.environ.get("N_SOBOL", 8 if SMOKE else 64))
LF_BUDGET = int(os.environ.get("LF_BUDGET", 32 if SMOKE else 300))
TOP_K = int(os.environ.get("TOP_K", 4 if SMOKE else 32))
MAX_ITER = int(os.environ.get("ITER", 1500 if SMOKE else 4000))
WORKERS = int(os.environ.get("CONCURRENT", 4 if SMOKE else 8))
THREADS = int(os.environ.get("NTHREAD", 2))
SU2_RUN = os.environ.get("SU2_RUN", "/home/kotaro/su2/bin")
SU2_WORK = os.environ.get("SU2_WORK", "/tmp/trust_bo_su2")
CSV_PATH = Path(os.environ.get("CSV", "mf_su2_correlation_smoke.csv" if SMOKE else "mf_su2_correlation.csv"))
FIELDNAMES = ["config", "candidate_id", "source", *[f"x{i}" for i in range(N_DIMS)],
              "lf_value", "lf_feasible", "hf_value", "hf_feasible", "hf_error"]

CONFIG = hashlib.sha256(json.dumps({
    "bounds": BOUNDS.tolist(), "aoa": AOA, "re": RE, "n_sobol": N_SOBOL,
    "lf_budget": LF_BUDGET, "top_k": TOP_K, "iter": MAX_ITER,
    "min_max_thickness": 0.06, "min_area": 0.05,
}, sort_keys=True).encode()).hexdigest()[:16]


def to_candidate(x: np.ndarray) -> dict[str, float]:
    return {f"u{i}": float(x[i]) for i in range(8)} | {
        f"l{i}": float(x[8 + i]) for i in range(8)
    }


def to_array(candidate: dict[str, float]) -> np.ndarray:
    return np.array([candidate[f"u{i}"] for i in range(8)] +
                    [candidate[f"l{i}"] for i in range(8)], dtype=float)


def evaluate_lf(candidate: dict[str, float]) -> tuple[float, bool]:
    import aerosandbox as asb
    import neuralfoil as nf

    x = to_array(candidate)
    valid, _, _ = validate_cst_geometry(x[:8], x[8:])
    if not valid:
        return 0.0, False
    try:
        airfoil = asb.KulfanAirfoil(
            upper_weights=x[:8], lower_weights=x[8:],
            leading_edge_weight=0.0, TE_thickness=0.0,
        )
        result = nf.get_aero_from_airfoil(
            airfoil=airfoil, alpha=AOA, Re=RE, model_size="xlarge"
        )
        cl = float(result["CL"].item())
        cd = float(result["CD"].item())
        confidence = float(result["analysis_confidence"].item())
        feasible = confidence >= 0.5 and cl > 0.0 and cd > 1e-6
        return (cl / cd if feasible else 0.0), feasible
    except Exception:
        return 0.0, False


def sobol_candidates() -> list[tuple[str, str, dict[str, float], float, bool]]:
    unit = qmc.Sobol(N_DIMS, scramble=True, seed=20260712).random(N_SOBOL)
    points = qmc.scale(unit, BOUNDS[:, 0], BOUNDS[:, 1])
    out = []
    for i, point in enumerate(points):
        candidate = to_candidate(point)
        value, feasible = evaluate_lf(candidate)
        out.append((f"sobol_{i:03d}", "sobol", candidate, value, feasible))
    return out


def lf_selected_candidates() -> list[tuple[str, str, dict[str, float], float, bool]]:
    from trust_bo import TRustBOEngine

    engine = TRustBOEngine(
        space=make_space(), direction="maximize", seed=20260712,
        config={"acquisition": "ei", "enable_phase2": True,
                "phase2_early_frac": 0.25},
    )
    evaluated = 0
    while evaluated < LF_BUDGET:
        candidates = engine.ask(batch_size=min(4, LF_BUDGET - evaluated))
        results = []
        for candidate in candidates:
            value, feasible = evaluate_lf(candidate)
            results.append({"value": value, "feasible": feasible})
        engine.tell(candidates, results)
        evaluated += len(candidates)

    ranked = sorted(
        (t for t in engine.history() if t.status == "complete"),
        key=lambda t: float(t.objective_values[0]), reverse=True,
    )
    selected: list = []
    encoded: list[np.ndarray] = []
    skipped: list = []
    for trial in ranked:
        point = (to_array(trial.parameters) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
        if all(np.max(np.abs(point - prior)) > 0.05 for prior in encoded):
            selected.append(trial)
            encoded.append(point)
        else:
            skipped.append(trial)
        if len(selected) == TOP_K:
            break
    selected.extend(skipped[:max(0, TOP_K - len(selected))])
    return [(f"lf_top_{i:03d}", "lf_top", dict(t.parameters),
             float(t.objective_values[0]), True) for i, t in enumerate(selected)]


def evaluate_hf(candidate: dict[str, float]) -> tuple[float, bool, str]:
    x = to_array(candidate)
    settings = SU2Settings(
        aoa=AOA, max_iter=MAX_ITER, su2_run=SU2_RUN,
        workroot=SU2_WORK, n_threads=THREADS,
    )
    cl, cd, feasible, info = run_cst(x[:8], x[8:], settings=settings)
    value = cl / cd if feasible and cd > 1e-6 else 0.0
    return float(value), bool(feasible), str(info.get("error", ""))


def completed_ids() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with CSV_PATH.open(newline="") as handle:
        return {row["candidate_id"] for row in csv.DictReader(handle)
                if row.get("config") == CONFIG
                and row.get("hf_error", "") not in {"timeout", "no_clcd"}
                and not row.get("hf_error", "").startswith("infrastructure:")}


def append_row(item, hf_result) -> None:
    candidate_id, source, candidate, lf_value, lf_feasible = item
    hf_value, hf_feasible, hf_error = hf_result
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        x = to_array(candidate)
        writer.writerow({
            "config": CONFIG, "candidate_id": candidate_id, "source": source,
            **{f"x{i}": f"{x[i]:.17g}" for i in range(N_DIMS)},
            "lf_value": f"{lf_value:.17g}", "lf_feasible": int(lf_feasible),
            "hf_value": f"{hf_value:.17g}", "hf_feasible": int(hf_feasible),
            "hf_error": hf_error,
        })


def summarize() -> bool:
    with CSV_PATH.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["config"] == CONFIG]
    lf_feas = np.array([int(row["lf_feasible"]) for row in rows], dtype=bool)
    hf_feas = np.array([int(row["hf_feasible"]) for row in rows], dtype=bool)
    paired = lf_feas & hf_feas
    precision = float(np.sum(paired) / max(1, np.sum(lf_feas)))
    if np.sum(paired) < 3:
        print(f"paired={np.sum(paired)} precision={precision:.3f}; insufficient")
        return False
    lf = np.array([float(row["lf_value"]) for row in rows])[paired]
    hf = np.array([float(row["hf_value"]) for row in rows])[paired]
    pearson = float(np.corrcoef(lf, hf)[0, 1])
    rho = float(spearmanr(lf, hf).statistic)
    slope = float(np.polyfit(lf, hf, 1)[0])
    q = max(1, math.ceil(len(lf) / 4))
    recall = len(set(np.argsort(lf)[-q:]) & set(np.argsort(hf)[-q:])) / q
    rng = np.random.default_rng(20260712)
    boots = []
    for _ in range(500):
        idx = rng.integers(0, len(lf), len(lf))
        value = float(spearmanr(lf[idx], hf[idx]).statistic)
        if np.isfinite(value):
            boots.append(value)
    lower = float(np.quantile(boots, 0.05)) if boots else -1.0
    gate = (len(lf) >= 40 and pearson**2 > 0.75 and slope > 0.0 and rho >= 0.75
            and lower > 0.60 and recall >= 0.50 and precision >= 0.75)
    print(f"paired={len(lf)} R2={pearson**2:.3f} slope={slope:.3g} rho={rho:.3f} "
          f"rho_p05={lower:.3f} topQ_recall={recall:.3f} precision={precision:.3f} "
          f"gate={'PASS' if gate else 'FAIL'}")
    return gate


def main() -> int:
    items = sobol_candidates() + lf_selected_candidates()
    done = completed_ids()
    pending = [item for item in items if item[0] not in done]
    print(f"config={CONFIG} total={len(items)} pending={len(pending)} workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(evaluate_hf, item[2]): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = (0.0, False, f"infrastructure:{type(exc).__name__}")
            append_row(item, result)
            print(item[0], result)
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
