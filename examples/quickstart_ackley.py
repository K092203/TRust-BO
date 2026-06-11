"""TRust-BO quickstart: minimize the 10D Ackley function.

This is a complete, self-contained example. After installing the package
(`pip install .` from the repo root, or `maturin develop --release`), run:

    python examples/quickstart_ackley.py

Ackley has many local minima and a single global minimum of 0 at the origin,
making it a standard stress test for black-box optimizers.
"""
import math

from trust_bo import Float, TRustBOEngine

DIM = 10
BUDGET = 50       # number of ask/tell rounds
BATCH_SIZE = 4    # candidates evaluated per round


def ackley(params: dict) -> float:
    """Standard Ackley function. Global minimum: 0.0 at x = (0, ..., 0)."""
    x = list(params.values())
    n = len(x)
    sum_sq = sum(v * v for v in x)
    sum_cos = sum(math.cos(2.0 * math.pi * v) for v in x)
    return (
        -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / n))
        - math.exp(sum_cos / n)
        + 20.0
        + math.e
    )


def main() -> None:
    # 1. Define the search space: 10 continuous variables in [-5, 5]
    space = [Float(f"x{i}", -5.0, 5.0) for i in range(DIM)]

    # 2. Create the engine
    engine = TRustBOEngine(space=space, direction="minimize", seed=42)

    # 3. Ask -> evaluate -> tell loop
    for _ in range(BUDGET):
        candidates = engine.ask(batch_size=BATCH_SIZE)
        results = [{"value": ackley(c), "feasible": True} for c in candidates]
        engine.tell(candidates, results)

    # 4. Report the best result found
    best = engine.best()
    print(f"evaluations : {BUDGET * BATCH_SIZE}")
    print(f"best value  : {best['objective_values'][0]:.6f}   (global min = 0.0)")
    print(f"best params : { {k: round(v, 3) for k, v in best['parameters'].items()} }")


if __name__ == "__main__":
    main()
