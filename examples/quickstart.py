"""TRust-BO quickstart: minimize a 5D sphere function in 20 rounds.

Run:  python examples/quickstart.py
"""
from trust_bo import Float, TRustBOEngine


def objective(params: dict) -> float:
    """Your expensive function goes here (CFD run, experiment, ...)."""
    return sum(v**2 for v in params.values())


# 1. Define the search space
space = [Float(f"x{i}", -5.0, 5.0) for i in range(5)]

# 2. Create the engine
engine = TRustBOEngine(space=space, direction="minimize", seed=42)

# 3. Ask -> evaluate -> tell loop
for round_idx in range(20):
    candidates = engine.ask(batch_size=4)
    results = [{"value": objective(c), "feasible": True} for c in candidates]
    engine.tell(candidates, results)

# 4. Get the best result
best = engine.best()
print(f"best value : {best['objective_values'][0]:.6f}")
print(f"best params: { {k: round(v, 4) for k, v in best['parameters'].items()} }")
