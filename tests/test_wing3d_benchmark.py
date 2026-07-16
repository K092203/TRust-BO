import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "wing3d_benchmark.py"
SPEC = importlib.util.spec_from_file_location("wing3d_benchmark_test", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_atomic_jsonl_recovers_torn_tail(tmp_path, monkeypatch):
    results = tmp_path / "results.jsonl"
    results.write_text('{"kind":"batch","iteration":0}\n{"kind":"batch"')
    monkeypatch.setattr(benchmark, "RESULTS_PATH", results)

    benchmark._jsonl_append({"kind": "batch", "iteration": 1})

    records = [json.loads(line) for line in results.read_text().splitlines()]
    assert [record["iteration"] for record in records] == [0, 1]
    assert not results.with_name(results.name + ".tmp").exists()


def test_load_batch_record_rejects_malformed_middle_line(tmp_path, monkeypatch):
    results = tmp_path / "results.jsonl"
    results.write_text(
        '{"kind":"batch","iteration":0}\n'
        '{"kind":"batch"\n'
        '{"kind":"batch","iteration":1}\n'
    )
    monkeypatch.setattr(benchmark, "RESULTS_PATH", results)

    with pytest.raises(RuntimeError, match="malformed non-tail JSONL line 2"):
        benchmark._load_batch_record(1)


def test_resume_rejects_su2_binary_hash_change(tmp_path, monkeypatch):
    executable = tmp_path / "SU2_CFD"
    executable.write_bytes(b"first SU2 binary")
    settings = replace(benchmark._settings(), su2_run=str(tmp_path))
    monkeypatch.setattr(benchmark, "_settings", lambda: settings)
    state = benchmark._state_template()
    executable.write_bytes(b"replacement SU2 binary")

    with pytest.raises(ValueError, match="fingerprint.*new RESULTS path"):
        benchmark._validate_resume_state(state)


def test_protocol_fingerprint_changes_with_workers(monkeypatch):
    before, _ = benchmark._protocol_fingerprint()
    monkeypatch.setattr(benchmark, "WORKERS", benchmark.WORKERS + 1)
    after, _ = benchmark._protocol_fingerprint()
    assert after != before


def test_protocol_fingerprint_changes_with_mesh_and_config(monkeypatch):
    before, _ = benchmark._protocol_fingerprint()
    monkeypatch.setattr(benchmark.wing3d_mesh, "WING_HEIGHT",
                        benchmark.wing3d_mesh.WING_HEIGHT + 0.001)
    height_changed, _ = benchmark._protocol_fingerprint()
    assert height_changed != before

    monkeypatch.undo()
    template = benchmark.wing3d_mesh.SU2_CONFIG_TEMPLATE
    monkeypatch.setattr(benchmark.wing3d_mesh, "SU2_CONFIG_TEMPLATE",
                        template.replace("CFL_NUMBER= 1.0", "CFL_NUMBER= 1.1"))
    config_changed, _ = benchmark._protocol_fingerprint()
    assert config_changed != before
