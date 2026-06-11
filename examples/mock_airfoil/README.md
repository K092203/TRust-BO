# Mock Airfoil Optimization Example

This example uses a mock CFD function. To use it with real CFD, replace
`mock_cfd()` with your OpenFOAM or SU2 evaluation call.

## How to run

```bash
pip install .
python examples/mock_airfoil/run_optimization.py
```

## How to connect to real CFD

1. Replace `mock_cfd()` with your solver call.
2. Return `{"Cl": float, "Cd": float}` from your function.
3. Set `feasible=False` if the simulation fails.
