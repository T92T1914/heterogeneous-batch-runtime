"""Explicitly enabled GPU execution of the existing ownership contracts."""
from functools import partial
import os
import numpy as np
import pytest
import test_batched_inference as batched_contracts
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


@pytest.mark.parametrize("case,argument", [
    ("test_complete_batched_logits_and_one_actual_invocation", 0),
    ("test_complete_batched_logits_and_one_actual_invocation", 1),
    ("test_complete_batched_logits_and_one_actual_invocation", 2),
    ("test_complete_batched_logits_and_one_actual_invocation", 8),
    ("test_batch_order_duplicates_and_other_images_do_not_change_a_prediction", None),
    ("test_unaligned_batch_and_retained_completed_outputs", None),
    ("test_batched_native_owner_counter_and_close", None),
    ("test_batched_invalid_input_is_refused_before_model_invocation", np.zeros((1,1,28,28), np.float64)),
    ("test_batched_invalid_input_is_refused_before_model_invocation", np.zeros((9,1,28,28), np.float32)),
    ("test_batched_invalid_input_is_refused_before_model_invocation", np.zeros((1,28,28), np.float32)),
    ("test_batched_invalid_input_is_refused_before_model_invocation", np.zeros((1,1,28,27), np.float32)),
    ("test_batched_invalid_input_is_refused_before_model_invocation", np.full((1,1,28,28), np.nan, np.float32)),
    ("test_batched_invalid_input_is_refused_before_model_invocation", np.zeros((1,1,28,28), np.float32)[:,:,:,::-1]),
    ("test_batched_executor_retains_existing_ownership_contract", "test_queued_cancel_running_cancel_stale_input_and_output_ownership"),
    ("test_batched_executor_retains_existing_ownership_contract", "test_shutdown_with_work_and_failure_retirement"),
    ("test_batched_executor_retains_existing_ownership_contract", "test_close_waits_for_running_physical_completion"),
])
def test_required_cuda_batched_contract(model, monkeypatch, case, argument):
    monkeypatch.setattr(batched_contracts, "Session", partial(batched_contracts.Session, provider="cuda-required"))
    monkeypatch.setattr(batched_contracts, "InferenceExecutor", partial(batched_contracts.InferenceExecutor, provider="cuda-required"))
    check = getattr(batched_contracts, case)
    if case == "test_batched_executor_retains_existing_ownership_contract":
        check(model, monkeypatch, argument)
    elif argument is None:
        check(model)
    else:
        check(model, argument)
