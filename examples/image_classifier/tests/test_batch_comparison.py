"""Failure-retention and retirement fixtures for the bounded comparison."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


directory = Path(__file__).resolve().parents[1]
if str(directory) not in sys.path:
    sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("batch_comparison_fixture", directory / "compare_batching.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@dataclass
class FixtureResult:
    value: object
    status: str = "completed"
    usable: bool = True


@pytest.fixture
def synthetic_comparison(monkeypatch):
    instances = []
    monkeypatch.setattr(comparison, "BATCHES", (1,))
    monkeypatch.setattr(comparison, "BLOCKS", 1)
    monkeypatch.setattr(comparison, "WARMUPS", 0)
    monkeypatch.setattr(comparison, "SAMPLES", 1)
    monkeypatch.setattr(comparison, "source_identities", lambda: {"synthetic": "unchanged"})
    monkeypatch.setattr(comparison.onnx, "load_model_from_string", lambda data: None)
    monkeypatch.setattr(comparison, "ReferenceEvaluator", lambda graph: SimpleNamespace(run=lambda *args: [np.zeros((1,10), np.float32)]))
    monkeypatch.setattr(comparison, "CurrentProcessMemory", lambda: SimpleNamespace(read=lambda: {"synthetic_process_counter": 1}))

    class Execution:
        invalid_output = False
        fail_close = False

        def __init__(self, model, *, execution_mode, **options):
            self.mode = execution_mode
            self.outstanding = 0
            self.closed = False
            instances.append(self)

        @property
        def execution_provenance(self):
            return {"source_model_sha256": "source", "execution_model_sha256": self.mode,
                    "execution_mode": self.mode}

        def submit_batch(self, dispatch, generation, batch):
            self.batch = batch
            self.outstanding = 1

        def wait(self, generation, timeout):
            self.outstanding = 0
            values = np.zeros((len(self.batch),10), np.float32)
            if self.invalid_output:
                values.fill(np.nan)
            return [FixtureResult(values)]

        def observations(self):
            return []

        def close(self):
            self.closed = True
            if self.fail_close:
                raise RuntimeError("synthetic close failure")

        def reconcile(self, generation):
            return []

    monkeypatch.setattr(comparison, "InferenceExecutor", Execution)
    return Execution, instances


def test_bad_outputs_remain_recorded_and_owner_is_closed(synthetic_comparison):
    execution, instances = synthetic_comparison
    execution.invalid_output = True
    report = {}
    with pytest.raises(AssertionError):
        comparison.collect(SimpleNamespace(data=b"synthetic", sha256="source"), report=report)
    assert report["status"] == "incomplete"
    failed, = [row for row in report["rows"] if row["record"] == "request"]
    assert not failed["validation_passed"]
    assert failed["result"]["value"][0][0] == {"nonfinite_float": "nan"}
    assert instances[0].closed and instances[0].outstanding == 0
    assert any(row["record"] == "close" and row["owner_retired"] for row in report["rows"])


def test_close_failure_cannot_be_reported_complete(synthetic_comparison):
    execution, instances = synthetic_comparison
    execution.fail_close = True
    report = {}
    with pytest.raises(RuntimeError, match="synthetic close"):
        comparison.collect(SimpleNamespace(data=b"synthetic", sha256="source"), report=report)
    assert report["status"] == "incomplete"
    assert instances[0].closed
    failed, = [row for row in report["rows"] if row["record"] == "close_failure"]
    assert not failed["owner_retired"]


def test_elapsed_setup_budget_closes_created_owner(synthetic_comparison, monkeypatch):
    execution, instances = synthetic_comparison
    clock = [0.0]
    original_init = execution.__init__
    def initialize(*args, **kwargs):
        original_init(*args, **kwargs)
        clock[0] = 2.0
    monkeypatch.setattr(execution, "__init__", initialize)
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: clock[0])
    report = {}
    with pytest.raises(TimeoutError, match="session setup"):
        comparison.collect(SimpleNamespace(data=b"synthetic", sha256="source"), max_seconds=1, report=report)
    assert instances[0].closed
    assert not any(row["record"] == "request" for row in report["rows"])
    assert report["status"] == "incomplete"


def test_each_batch_observes_both_mode_orders(synthetic_comparison, monkeypatch):
    _, instances = synthetic_comparison
    monkeypatch.setattr(comparison, "BATCHES", (1, 2, 8))
    monkeypatch.setattr(comparison, "BLOCKS", 3)
    report = comparison.collect(SimpleNamespace(data=b"synthetic", sha256="source"))
    orders = [row for row in report["rows"] if row["record"] == "run_order"]
    for count in (1, 2, 8):
        first_modes = {row["mode_order"][0] for row in orders if row["batch"] == count}
        assert first_modes == {"serial", "batched"}
    assert len(instances) == 18 and all(instance.closed for instance in instances)
    assert report["completed_requests"] == 36 and report["status"] == "completed"
