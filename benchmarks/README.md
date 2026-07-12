# Benchmarks

Scripts used to produce the numbers in the main [README](../README.md) and
[docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md). All scripts write their CSV
output to the current directory (gitignored).

| Script | Compares |
|---|---|
| `benchmark.py` / `benchmark_v2.py` | TRust-BO vs BoTorch (TuRBO-1) vs HEBO vs Random |
| `benchmark_50d.py` | 50D Ackley: TRust-BO vs Random vs GP |
| `benchmark_multi_tr.py` / `_v2.py` | Multi Trust Region (TuRBO-M, experimental) |

Comparison baselines require extra dependencies not used by TRust-BO itself
(`botorch`, `hebo`, `scikit-learn`); install them as needed per script.
