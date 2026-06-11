"""Mock airfoil optimization with TRust-BO.

This example optimizes a 3-variable NACA-style airfoil to maximize lift-to-drag
ratio (Cl/Cd). The CFD evaluation here is a *mock* analytic function — replace
`mock_cfd()` with a real OpenFOAM or SU2 call to use it on a real problem.

Run:

    python examples/mock_airfoil/run_optimization.py
"""
import math

from trust_bo import Float, TRustBOEngine

BUDGET = 40
BATCH_SIZE = 4


def mock_cfd(params: dict) -> dict:
    """Mock CFD evaluation. Replace with your real OpenFOAM/SU2 call.

    A real implementation would: write the geometry, run the solver, parse the
    force coefficients, and return them. It should return {"Cl": ..., "Cd": ...}
    and may raise on a failed simulation (handled as infeasible below).

    Args:
        params: {"camber": float, "camber_pos": float, "thickness": float}
    Returns:
        {"Cl": float, "Cd": float}
    """
    m = params["camber"]        # max camber, % of chord (NACA first digit-ish)
    p = params["camber_pos"]    # position of max camber, fraction of chord
    t = params["thickness"]     # max thickness, fraction of chord

    # Smooth synthetic surrogate with a single sweet spot near (4, 0.4, 0.12).
    cl = 0.9 + 0.12 * m - 0.6 * (p - 0.4) ** 2
    cd = 0.008 + 0.02 * (t - 0.12) ** 2 + 0.0015 * m
    return {"Cl": cl, "Cd": cd}


def evaluate(params: dict) -> dict:
    """Wrap mock_cfd into a TRust-BO result dict.

    We maximize Cl/Cd. The engine minimizes, so we return the negated ratio.
    A failed simulation is reported as infeasible so the optimizer avoids it.
    """
    try:
        out = mock_cfd(params)
        cd = out["Cd"]
        if cd <= 0 or not math.isfinite(out["Cl"]) or not math.isfinite(cd):
            return {"value": 0.0, "feasible": False}
        ratio = out["Cl"] / cd
        return {"value": -ratio, "feasible": True}
    except Exception:
        # Simulation failed (mesh error, solver divergence, ...) -> infeasible
        return {"value": 0.0, "feasible": False}


def main() -> None:
    space = [
        Float("camber", 0.0, 9.0),         # %
        Float("camber_pos", 0.1, 0.9),     # fraction of chord
        Float("thickness", 0.06, 0.20),    # fraction of chord
    ]
    engine = TRustBOEngine(space=space, direction="minimize", seed=0)

    for _ in range(BUDGET):
        candidates = engine.ask(batch_size=BATCH_SIZE)
        engine.tell(candidates, [evaluate(c) for c in candidates])

    best = engine.best()
    best_ratio = -best["objective_values"][0]
    print(f"evaluations  : {BUDGET * BATCH_SIZE}")
    print(f"best Cl/Cd   : {best_ratio:.2f}")
    print(f"best geometry: { {k: round(v, 3) for k, v in best['parameters'].items()} }")


if __name__ == "__main__":
    main()
