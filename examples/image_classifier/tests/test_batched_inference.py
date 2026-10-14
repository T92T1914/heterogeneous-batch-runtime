"""Actual batched calls retain the original model's image semantics."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest

from hbr_image_classifier.batching import BATCH_RECIPE
from hbr_image_classifier.executor import InferenceExecutor
from hbr_image_classifier.model import MODEL_SHA256, Model, example_images, prepare_arrays
from hbr_image_classifier.session import Session
import test_inference as contracts
from test_inference import model


def original_logits(model, batch):
    graph = onnx.load_model_from_string(model.data)
    reference = ReferenceEvaluator(graph)
    return (np.concatenate([reference.run(None, {graph.graph.input[0].name: image[None]})[0]
                            for image in batch]) if len(batch) else np.empty((0,10), np.float32))


@pytest.mark.parametrize("count", [0,1,2,8])
def test_complete_batched_logits_and_one_actual_invocation(model, count):
    batch = prepare_arrays(example_images()[:count])
    session = Session(model, execution_mode="batched")
    try:
        assert session.model_invocations == 0
        actual = session.run(batch)
        assert session.model_invocations == (1 if count else 0)
        assert actual.shape == (count,10) and actual.dtype == np.float32
        assert actual.flags.owndata and not np.shares_memory(actual, batch)
        np.testing.assert_allclose(actual, original_logits(model, batch), rtol=1e-5, atol=2e-5)
    finally:
        session.close()


def test_batch_order_duplicates_and_other_images_do_not_change_a_prediction(model):
    images = example_images()
    ramp = (np.arange(784, dtype=np.uint16).reshape(28,28) % 256).astype(np.uint8)
    batch = prepare_arrays(np.stack((images[3], np.zeros((28,28), np.uint8), images[3],
                                    ramp, images[7], images[0])))
    session = Session(model, execution_mode="batched")
    try:
        actual = session.run(batch)
        np.testing.assert_allclose(actual, original_logits(model, batch), rtol=1e-5, atol=2e-5)
        np.testing.assert_array_equal(actual[0], actual[2])
        order = [4,0,5,2,1,3]
        reordered = session.run(np.ascontiguousarray(batch[order]))
        np.testing.assert_allclose(reordered, actual[order], rtol=1e-5, atol=2e-5)
        for index in (0,1,3):
            isolated = session.run(batch[index:index+1].copy())
            np.testing.assert_allclose(isolated, actual[index:index+1], rtol=1e-5, atol=2e-5)
        assert session.model_invocations == 5
    finally:
        session.close()


def test_unaligned_batch_and_retained_completed_outputs(model):
    batch = prepare_arrays(example_images()[:2])
    storage = np.empty(batch.nbytes + 1, np.uint8)
    unaligned = np.ndarray(batch.shape, dtype=np.float32, buffer=storage, offset=1)
    unaligned[:] = batch
    assert unaligned.flags.c_contiguous and not unaligned.flags.aligned
    session = Session(model, execution_mode="batched")
    try:
        actual = session.run(unaligned)
        expected = original_logits(model, batch)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)
        unaligned.fill(0)
        later = session.run(unaligned)
        assert actual.flags.owndata
        assert not np.shares_memory(actual, unaligned) and not np.shares_memory(actual, later)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)
    finally:
        session.close()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize("bad", [
    np.zeros((1,1,28,28), np.float64),
    np.zeros((1,1,28,28), dtype=">f4"),
    np.zeros((9,1,28,28), np.float32),
    np.zeros((1,28,28), np.float32),
    np.zeros((1,1,28,27), np.float32),
    np.full((1,1,28,28), np.nan, np.float32),
    np.full((1,1,28,28), 1.1, np.float32),
    np.zeros((1,1,28,28), np.float32)[:,:,:,::-1],
])
def test_batched_invalid_input_is_refused_before_model_invocation(model, bad):
    session = Session(model, execution_mode="batched")
    try:
        with pytest.raises((TypeError, ValueError)):
            session.run(bad)
        assert session.model_invocations == 0
    finally:
        session.close()


def test_batched_native_owner_counter_and_close(model):
    session = Session(model, execution_mode="batched")
    batch = prepare_arrays(example_images()[:2])
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for operation in (lambda: session.run(batch), lambda: session.model_invocations, session.close):
                with pytest.raises(RuntimeError, match="owning thread"):
                    pool.submit(operation).result()
        assert session.model_invocations == 0
        actual = session.run(batch)
        assert session.model_invocations == 1
        expected = actual.copy()
        session.close()
        session.close()
        with pytest.raises(RuntimeError, match="closed"):
            session.run(batch)
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.model_invocations
        np.testing.assert_array_equal(actual, expected)
    finally:
        session.close()


@pytest.mark.parametrize("case", [
    "test_queued_cancel_running_cancel_stale_input_and_output_ownership",
    "test_shutdown_with_work_and_failure_retirement",
    "test_close_waits_for_running_physical_completion",
])
def test_batched_executor_retains_existing_ownership_contract(model, monkeypatch, case):
    monkeypatch.setattr(contracts, "Session", partial(Session, execution_mode="batched"))
    monkeypatch.setattr(contracts, "InferenceExecutor", partial(InferenceExecutor, execution_mode="batched"))
    getattr(contracts, case)(model)


@pytest.mark.parametrize("mode", ["serial", "batched"])
def test_execution_provenance_describes_bytes_passed_to_native(model, monkeypatch, mode):
    from hbr_image_classifier import _session
    calls = []
    def construct(*arguments):
        calls.append(arguments)
        return SimpleNamespace(close=lambda: "")
    monkeypatch.setattr(_session, "Session", construct)
    options = {"execution_mode": mode} if mode == "batched" else {}
    session = Session(model, library="reviewed-runtime", **options)
    try:
        arguments, = calls
        assert arguments[0] == "reviewed-runtime" and arguments[2:] == (("cpu", "", True) if mode == "batched" else ("cpu", ""))
        assert session.model_sha256 == session.source_model_sha256 == MODEL_SHA256
        assert session.execution_model_sha256 == sha256(arguments[1]).hexdigest()
        assert session.execution_mode == mode
        assert session.execution_recipe == (BATCH_RECIPE if mode == "batched" else None)
        if mode == "serial":
            assert arguments[1] is model.data
        else:
            assert session.execution_model_sha256 != session.source_model_sha256
    finally:
        session.close()


def test_unknown_mode_and_unreviewed_source_cannot_select_a_native_session(model, monkeypatch):
    from hbr_image_classifier import _session
    monkeypatch.setattr(_session, "Session", lambda *args: pytest.fail("invalid request reached native construction"))
    with pytest.raises(ValueError, match="execution_mode"):
        Session(model, execution_mode="auto")
    with pytest.raises(ValueError, match="execution_mode"):
        InferenceExecutor(model, execution_mode="auto")
    corrupted = Model(model.data[:-1], model.sha256)
    with pytest.raises(ValueError, match="reviewed"):
        Session(corrupted, execution_mode="batched")


@pytest.mark.parametrize("mode", ["serial", "batched"])
def test_executor_exposes_qualified_provenance_without_changing_results(model, mode):
    execution = InferenceExecutor(model, execution_mode=mode)
    try:
        provenance = execution.execution_provenance
        assert provenance["model_sha256"] == provenance["source_model_sha256"] == MODEL_SHA256
        assert provenance["execution_mode"] == mode
        assert provenance["execution_recipe"] == (BATCH_RECIPE if mode == "batched" else None)
        assert (provenance["execution_model_sha256"] == MODEL_SHA256) == (mode == "serial")
    finally:
        execution.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = execution.execution_provenance
