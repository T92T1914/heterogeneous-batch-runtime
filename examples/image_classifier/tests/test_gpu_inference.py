"""Explicitly enabled GPU execution of the existing ownership contracts."""
from functools import partial
import os
import pytest
import test_inference as contracts
from test_inference import model


pytestmark = pytest.mark.skipif(
    os.environ.get("HBR_INFERENCE_GPU_ACCEPTANCE") != "1",
    reason="requires explicit CUDA environment and a free permitted resource slot",
)


@pytest.mark.parametrize("case,count", [
    ("test_entire_logits_against_independent_onnx_engine", 0),
    ("test_entire_logits_against_independent_onnx_engine", 1),
    ("test_entire_logits_against_independent_onnx_engine", 2),
    ("test_entire_logits_against_independent_onnx_engine", 8),
    ("test_native_owner_and_closed_session", None),
    ("test_queued_cancel_running_cancel_stale_input_and_output_ownership", None),
    ("test_shutdown_with_work_and_failure_retirement", None),
    ("test_close_waits_for_running_physical_completion", None),
    ("test_duplicate_and_finite_session_history", None),
])
def test_required_cuda_existing_contract(model, monkeypatch, case, count):
    # Reuse the actual assertions and host gates, changing only the explicit
    # provider. A host gate does not prove interruption of an active GPU kernel.
    monkeypatch.setattr(contracts, "Session", partial(contracts.Session, provider="cuda-required"))
    monkeypatch.setattr(contracts, "InferenceExecutor", partial(contracts.InferenceExecutor, provider="cuda-required"))
    check = getattr(contracts, case)
    check(model, count) if count is not None else check(model)
