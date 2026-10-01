import gc
import os
import threading
import weakref

import numpy as np
import pytest

from heterogeneous_batch_runtime import _native
from heterogeneous_batch_runtime.cuda import CudaHistogramContext


def expected(image, tr, tc):
    rows, cols = image.shape
    result = np.zeros(((rows + tr - 1) // tr, (cols + tc - 1) // tc, 256), dtype=np.uint64)
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            result[r, c] = np.bincount(image[r*tr:(r+1)*tr, c*tc:(c+1)*tc].ravel(), minlength=256)
    return result


def test_explicit_unavailable_never_falls_back():
    if not _native.cuda_built:
        with pytest.raises(RuntimeError, match="CUDA unavailable"):
            CudaHistogramContext(3, 4, 2, 2)
    else:
        with pytest.raises(RuntimeError, match="CUDA unavailable"):
            CudaHistogramContext(3, 4, 2, 2, device=2147483647)


def test_owner_teardown_and_ordinary_abandonment(monkeypatch):
    calls = []
    class Native:
        def __init__(self, *args):
            self.owner = threading.get_ident()
            calls.append(("create", self.owner))
            self.device_bytes = 42
        def run(self, image, *args):
            assert threading.get_ident() == self.owner
            raise RuntimeError("synthetic host failure")
        def close(self):
            assert threading.get_ident() == self.owner
            calls.append(("close", self.owner))
        def __del__(self):
            calls.append(("destroy", threading.get_ident()))
    monkeypatch.setattr(_native, "cuda_built", True)
    monkeypatch.setattr(_native, "_cuda_select_device", lambda device: "synthetic host owner", raising=False)
    monkeypatch.setattr(_native, "_CudaHistogramContext", Native, raising=False)
    with pytest.raises(ValueError, match="body"):
        with CudaHistogramContext(2, 2, 1, 1) as context:
            with pytest.raises(RuntimeError, match="synthetic host failure"):
                context.run(np.zeros((2, 2), dtype=np.uint8))
            raise ValueError("body")
    context.close()
    del context
    abandoned = CudaHistogramContext(2, 2, 1, 1)
    reference = weakref.ref(abandoned)
    # Its last reference may disappear on a different Python thread.
    holder = [abandoned]
    del abandoned
    thread = threading.Thread(target=holder.clear)
    thread.start()
    thread.join()
    gc.collect()
    assert reference() is None
    created = [owner for kind, owner in calls if kind == "create"]
    destroyed = [owner for kind, owner in calls if kind == "destroy"]
    assert created == destroyed


def test_pending_call_keeps_owned_snapshot_until_return(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    observed, failures, contexts = [], [], []
    image = np.full((2, 2), 3, dtype=np.uint8)
    class Native:
        device_bytes = 42
        def __init__(self, *args):
            self.owner = threading.get_ident()
        def run(self, owned, *args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("pending snapshot gate")
            observed.append(owned.copy())
            return {"value": owned.copy()}
        def close(self):
            assert self.owner == threading.get_ident()
    monkeypatch.setattr(_native, "cuda_built", True)
    monkeypatch.setattr(_native, "_cuda_select_device", lambda device: "synthetic host owner", raising=False)
    monkeypatch.setattr(_native, "_CudaHistogramContext", Native, raising=False)
    def caller():
        try:
            with CudaHistogramContext(2, 2, 1, 1) as context:
                contexts.append(weakref.ref(context))
                context.run(image)
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=caller)
    thread.start()
    try:
        assert entered.wait(1)
        # The gate is inside native execution, so the submission copy finished.
        image.fill(9)
        gc.collect()
        assert contexts[0]() is not None
    finally:
        release.set()
        thread.join()
    assert not failures
    np.testing.assert_array_equal(observed[0], np.full((2, 2), 3, dtype=np.uint8))


def test_reentrant_input_copy_cannot_queue_after_owner_shutdown(monkeypatch):
    calls = []
    class Native:
        device_bytes = 42
        def run(self, *args):
            calls.append("run")
            raise AssertionError("closed owner must not execute")
        def close(self):
            calls.append("close")
    monkeypatch.setattr(_native, "cuda_built", True)
    monkeypatch.setattr(_native, "_cuda_select_device", lambda device: "synthetic host owner", raising=False)
    monkeypatch.setattr(_native, "_CudaHistogramContext", lambda *args: Native(), raising=False)
    context = CudaHistogramContext(2, 2, 1, 1)
    original_put = context._queue.put
    def guarded_put(item, *args, **kwargs):
        # Turn the old unconsumed queue entry into a failure instead of hanging.
        if item[0] is not None and context._closed:
            raise AssertionError("work queued after native owner shutdown")
        return original_put(item, *args, **kwargs)
    monkeypatch.setattr(context._queue, "put", guarded_put)
    class ClosingImage(np.ndarray):
        def copy(self, *args, **kwargs):
            context.close()
            return np.zeros((2, 2), dtype=np.uint8)
    image = np.zeros((2, 2), dtype=np.uint8).view(ClosingImage)
    try:
        with pytest.raises(RuntimeError, match="closed"):
            context.run(image)
        assert "run" not in calls
        assert not context._thread.is_alive()
        assert context._queue.empty()
    finally:
        context.close()


@pytest.mark.parametrize("shape,tile,shared", [((17, 19), (4, 5), True), ((3, 4), (1, 2), False),
                                              ((0, 7), (2, 3), True)])
def test_real_cuda_correctness_and_contract(shape, tile, shared):
    if not _native.cuda_built:
        if os.environ.get("HBR_REQUIRE_CUDA") == "1":
            pytest.fail("required CUDA build is unavailable")
        pytest.skip("CPU build, no GPU coverage claimed")
    image = np.random.default_rng(73).integers(0, 256, shape, dtype=np.uint8)
    original = image.copy()
    with CudaHistogramContext(*shape, *tile, shared_bins=shared) as context:
        first = context.run(image)
        np.testing.assert_array_equal(first["value"], expected(original, *tile))
        image.fill(7)
        second = context.run(image)
        np.testing.assert_array_equal(second["value"], expected(image, *tile))
        np.testing.assert_array_equal(first["value"], expected(original, *tile))
        assert second["request_id"] == first["request_id"] + 1
        assert not np.shares_memory(first["value"], second["value"])
        for bad in (image.astype(np.int16), image.tolist()):
            with pytest.raises(TypeError):
                context.run(bad)
        with pytest.raises(ValueError):
            context.run(np.zeros((1, 1, 1), dtype=np.uint8))
        if shape[0] and shape[1] > 1:
            with pytest.raises(ValueError):
                context.run(image[:, ::-1])
        errors = []
        def foreign():
            try:
                context.run(image)
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        assert errors and "creating thread" in errors[0]
    context.close()
    with pytest.raises(RuntimeError, match="closed"):
        context.run(image)


def test_real_cuda_bounds():
    if not _native.cuda_built:
        pytest.skip("CPU build")
    for dimensions in ((2, 2, 0, 1), (-1, 2, 1, 1), (True, 2, 1, 1), (100000, 100000, 1, 1)):
        with pytest.raises(ValueError):
            CudaHistogramContext(*dimensions)
