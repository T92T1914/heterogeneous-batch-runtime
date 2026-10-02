"""The installed consumer must retire interrupted sessions on their worker."""
from concurrent.futures import Future
import gc
import threading
from unittest.mock import patch
import weakref

import pytest

import hbr_image_classifier.executor as consumer


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_interrupted_consumer_initialization(error_type, cleanup_fails):
    calls, references = [], []
    entered, release = threading.Event(), threading.Event()
    original_result = Future.result
    interrupted = False
    before = {thread.ident for thread in threading.enumerate()}
    error = error_type("caller stopped session initialization")

    class ThreadBoundSession:
        def __init__(self, *args, **kwargs):
            self.owner = threading.get_ident()
            calls.append(("create", self.owner))
            references.append(weakref.ref(self))
            entered.set()
            if not release.wait(2):
                raise TimeoutError("session factory gate")

        def close(self):
            current = threading.get_ident()
            calls.append(("close", current))
            assert current == self.owner
            if cleanup_fails:
                raise ValueError("inert profile retirement failed")
            return "inert-profile.json"

        def __del__(self):
            calls.append(("destroy", threading.get_ident()))

    def interrupted_result(future, timeout=None):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            assert entered.wait(1)
            release.set()
            raise error
        return original_result(future, timeout)

    try:
        with (patch.object(consumer, "Session", ThreadBoundSession),
              patch.object(Future, "result", interrupted_result),
              pytest.raises(error_type) as caught):
            consumer.InferenceExecutor(object())
        assert caught.value is error
        if cleanup_fails:
            assert any("ValueError: inert profile retirement failed" in note
                       for note in getattr(error, "__notes__", ()))
    finally:
        release.set()
        # A caller exception traceback is not part of the resource contract.
        # Release it before checking destruction and thread retirement.
        error.__traceback__ = None
    gc.collect()
    assert [name for name, _ in calls] == ["create", "close", "destroy"]
    assert len({ident for _, ident in calls}) == 1
    assert calls[0][1] != threading.get_ident()
    assert references[0]() is None
    assert not any(thread.ident not in before
                   and thread.name.startswith("timing-execution")
                   for thread in threading.enumerate())
