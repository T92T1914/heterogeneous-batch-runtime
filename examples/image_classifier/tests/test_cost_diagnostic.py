"""Synthetic late-result and trace-failure controls for host diagnostics."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

directory = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cost_diagnostic_fixture", directory / "diagnose_request_cost.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@dataclass
class Result:
    value: object
    status: str = "completed"
    usable: bool = True


@pytest.fixture
def synthetic_consumer(monkeypatch):
    instances = []
    monkeypatch.setattr(diagnostic, "source_identities", lambda: {"synthetic": "unchanged"})
    monkeypatch.setattr(diagnostic, "CurrentProcessMemory", lambda: SimpleNamespace(read=lambda: {}))
    monkeypatch.setattr(diagnostic, "fixtures", lambda: (np.zeros((8,28,28), np.uint8),))
    monkeypatch.setattr(diagnostic, "oracle", lambda *args: [np.zeros((8,10), np.float32)])
    class Session:
        def run(self, batch):
            return np.zeros((8,10), np.float32)
    class Execution:
        timeout = False
        trace_path = ""
        def __init__(self, *args, **kwargs):
            self.outstanding = 0
            self.session = Session()
            self.closed = False
            self.execution_provenance = {"execution_mode": "batched"}
            self.profile_path = self.trace_path
            instances.append(self)
        def submit_batch(self, dispatch, generation, batch):
            self.batch = batch
            self.outstanding = 1
        def wait(self, generation, timeout):
            if self.timeout:
                raise TimeoutError("synthetic late completion")
            self.outstanding = 0
            return [Result(self.session.run(self.batch))]
        def close(self):
            self.closed = True
            self.outstanding = 0
        def reconcile(self, generation):
            return [Result(np.zeros((8,10), np.float32))] if self.timeout else []
        def observations(self):
            return [{"kind": "synthetic retirement"}]
    monkeypatch.setattr(diagnostic, "Session", Session)
    monkeypatch.setattr(diagnostic, "InferenceExecutor", Execution)
    return Session, Execution, instances


def test_late_result_retirement_is_retained_without_erasing_request_failure(synthetic_consumer, tmp_path):
    session, execution, instances = synthetic_consumer
    original_run = session.run
    execution.timeout = True
    report = diagnostic.run(SimpleNamespace(sha256="synthetic"), provider="cpu", instrumentation="worker",
        replicate=0, output=tmp_path / "late.json", max_seconds=8)
    assert report["status"] == "incomplete" and report["failure"]["type"] == "TimeoutError"
    close, = [row for row in report["rows"] if row["record"] == "close"]
    assert close["owner_retired"] and close["outstanding"] == 0 and len(close["retired_results"]) == 1
    assert close["observations"] and not any(row["record"] == "close_failure" for row in report["rows"])
    assert instances[0].closed and session.run is original_run


@pytest.mark.parametrize("bad_trace", ["", "missing.json"])
def test_trace_failure_does_not_relabel_successful_owner_retirement(synthetic_consumer, tmp_path, bad_trace):
    session, execution, instances = synthetic_consumer
    original_run = session.run
    execution.trace_path = str(tmp_path / bad_trace) if bad_trace else ""
    report = diagnostic.run(SimpleNamespace(sha256="synthetic"), provider="cpu", instrumentation="ort",
        replicate=0, output=tmp_path / "trace.json", max_seconds=8)
    assert report["status"] == "incomplete" and instances[0].closed
    assert any(row["record"] == "close" and row["owner_retired"] for row in report["rows"])
    assert any(row["record"] == "trace_failure" for row in report["rows"])
    assert not any(row["record"] == "close_failure" for row in report["rows"])
    assert session.run is original_run
