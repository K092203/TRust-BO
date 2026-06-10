"""naca_optimize.py — TRust-BO × NACA翼型最適化

目的: Cl/Cd の最大化（揚抗比最大化）
変数: camber (m), camber_pos (p), thickness (t)
CFD: mock_cfd.run_openfoam_case()（実OpenFOAM差し替え可）

Usage:
    python naca_optimize.py [--budget 30] [--aoa 5.0] [--mock/--real]
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np
from trust_bo import TRustBOEngine, Float
from mock_cfd import run_openfoam_case

CSV_PATH = Path("naca_results.csv")
ERROR_LOG = Path("openfoam_errors.txt")


def csv_init():
    with open(CSV_PATH, "w", newline="", buffering=1) as f:
        csv.writer(f).writerow(
            ["trial", "camber", "pos", "thickness", "aoa",
             "Cl", "Cd", "Cl_Cd", "feasible", "time_s"])


def csv_append(row: dict):
    with open(CSV_PATH, "a", newline="", buffering=1) as f:
        csv.writer(f).writerow([
            row["trial"], row["camber"], row["pos"], row["thickness"],
            row["aoa"], row.get("Cl", ""), row.get("Cd", ""),
            row.get("Cl_Cd", ""), row["feasible"], row["time_s"],
        ])


def run(budget: int = 30, aoa: float = 5.0):
    space = [
        Float("camber",    0.0,  0.09),
        Float("pos",       0.2,  0.8),
        Float("thickness", 0.08, 0.35),
    ]
    engine = TRustBOEngine(
        space=space,
        direction="maximize",
        seed=42,
        config={"n_init": 6, "batch_size": 1},
    )

    csv_init()
    trial = 0
    best_cl_cd = -float("inf")
    best_params = None

    print(f"NACA翼型最適化: budget={budget}, aoa={aoa}°, mock_cfd=True")
    print(f"{'Trial':>5} {'camber':>8} {'pos':>6} {'thick':>7} "
          f"{'Cl':>7} {'Cd':>8} {'Cl/Cd':>7}")
    print("-" * 60)

    evaluated = 0
    while evaluated < budget:
        cands = engine.ask(batch_size=1)
        c = cands[0]

        t0 = time.perf_counter()
        try:
            result = run_openfoam_case(
                c["camber"], c["pos"], c["thickness"], aoa_deg=aoa)
            elapsed = time.perf_counter() - t0
            trial += 1

            if result is None:
                feasible = False
                csv_append({"trial": trial, "camber": c["camber"],
                            "pos": c["pos"], "thickness": c["thickness"],
                            "aoa": aoa, "feasible": False,
                            "time_s": f"{elapsed:.2f}"})
                engine.tell(cands, [{"value": 0.0, "feasible": False}])
            else:
                feasible = True
                cl_cd = result["Cl_Cd"]
                csv_append({"trial": trial, "camber": c["camber"],
                            "pos": c["pos"], "thickness": c["thickness"],
                            "aoa": aoa, "Cl": result["Cl"],
                            "Cd": result["Cd"], "Cl_Cd": cl_cd,
                            "feasible": True, "time_s": f"{elapsed:.2f}"})
                engine.tell(cands, [{"value": cl_cd, "feasible": True}])

                if cl_cd > best_cl_cd:
                    best_cl_cd = cl_cd
                    best_params = dict(c)

                print(f"{trial:5d} {c['camber']:8.4f} {c['pos']:6.3f} "
                      f"{c['thickness']:7.4f} {result['Cl']:7.4f} "
                      f"{result['Cd']:8.6f} {cl_cd:7.2f}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            with open(ERROR_LOG, "a") as ef:
                ef.write(f"trial={trial} error={e}\n")
            engine.tell(cands, [{"value": 0.0, "feasible": False}])

        evaluated += 1

    print("-" * 60)
    print(f"\n最良 Cl/Cd = {best_cl_cd:.2f}")
    print(f"最良パラメータ: {best_params}")

    # Random baseline 推定（同条件で20点）
    rng = np.random.default_rng(0)
    random_cl_cds = []
    for _ in range(20):
        m  = float(rng.uniform(0.0, 0.09))
        p  = float(rng.uniform(0.2, 0.8))
        t  = float(rng.uniform(0.08, 0.35))
        r = run_openfoam_case(m, p, t, aoa_deg=aoa)
        if r:
            random_cl_cds.append(r["Cl_Cd"])

    random_best = max(random_cl_cds) if random_cl_cds else float("nan")
    print(f"Random best (20点): {random_best:.2f}")
    print(f"TRust-BO vs Random: {(best_cl_cd - random_best):.2f} 改善")

    return {"best_Cl_Cd": best_cl_cd, "best_params": best_params,
            "random_best": random_best}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--aoa", type=float, default=5.0)
    args = parser.parse_args()
    result = run(budget=args.budget, aoa=args.aoa)
    print(f"\nCSV saved to: {CSV_PATH.resolve()}")
