# Contributing to TRust-BO

Thank you for your interest! This is a one-person project so far — any help is genuinely appreciated.

## Bug reports & feature requests

Please open an [Issue](https://github.com/K092203/TRust-BO/issues). Include your OS, Python version, and a minimal reproduction if possible.

## Pull requests

PRs are welcome. Start small — a typo fix or a single bug fix is a great first PR.

1. Fork and create a branch
2. Make your change
3. **Add tests** — `pytest` for Python-facing behavior, `cargo test` for Rust internals
4. Make sure everything passes:
   ```bash
   cargo test --release
   maturin develop --release
   pytest tests/ -q
   ```

## Code style

- Rust: `cargo fmt` (rustfmt defaults)
- Python: [Black](https://github.com/psf/black) defaults

## Benchmark results wanted!

If you use TRust-BO on a **real CFD problem** (or any real engineering optimization), please share your results in an Issue — good or bad. Real-world feedback is the most valuable contribution at this stage.
