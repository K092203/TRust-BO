"""TRust-BO for CFD-style optimization.

Shows the recommended setup for expensive simulation workflows:
  - a small parameter space (airfoil-like: camber, position, thickness)
  - native Phase 2 enabled (endgame refinement with a residual micro-GP)
  - feasible/infeasible handling (diverged or invalid simulations)

Run:  python examples/cfd_basic.py
"""
from trust_bo import Float, TRustBOEngine

# Airfoil-like parameters (NACA 4-digit style)
space = [
    Float("camber", 0.0, 0.09),       # max camber (fraction of chord)
    Float("camber_pos", 0.2, 0.8),    # position of max camber
    Float("thickness", 0.06, 0.18),   # max thickness (fraction of chord)
]

engine = TRustBOEngine(
    space=space,
    direction="maximize",             # e.g. maximize Cl/Cd
    seed=42,
    config={
        "enable_phase2": True,        # native Rust residual-GP endgame (recommended)
        "init_warmup": 2,             # protect the trust region early on
    },
)


def run_cfd(params: dict) -> dict:
    """Replace this with your real CFD evaluation.

    Typical structure:
      1. generate geometry/mesh from `params`
      2. run the solver (e.g. OpenFOAM simpleFoam via subprocess)
      3. parse Cl, Cd from the output

    Return {"value": <objective>, "feasible": <bool>}.
    Mark a run infeasible if the solver diverged or the geometry is invalid —
    TRust-BO learns to avoid those regions via a feasibility surrogate.
    """
    # --- mock physics, replace from here ---------------------------------
    camber, pos, thick = params["camber"], params["camber_pos"], params["thickness"]
    if thick < 0.07 and camber > 0.08:
        return {"value": 0.0, "feasible": False}   # e.g. mesh generation failed
    cl = 6.0 * camber + 0.4 * pos
    cd = 0.006 + 0.05 * thick**2 + 0.5 * camber**2
    return {"value": cl / cd, "feasible": True}
    # --- to here ----------------------------------------------------------


budget = 40        # total CFD runs you can afford
batch = 4          # runs you can do in parallel
done = 0
while done < budget:
    candidates = engine.ask(batch_size=min(batch, budget - done))
    engine.tell(candidates, [run_cfd(c) for c in candidates])
    done += len(candidates)

best = engine.best()
print(f"best Cl/Cd : {best['objective_values'][0]:.2f}")
print(f"best params: { {k: round(v, 4) for k, v in best['parameters'].items()} }")
