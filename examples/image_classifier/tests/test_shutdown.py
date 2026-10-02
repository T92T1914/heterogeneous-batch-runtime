"""Installed consumer cleanup diagnostics must not retain an owned session."""
import gc
import threading
from unittest.mock import patch
import weakref

import pytest

import hbr_image_classifier.executor as consumer


@pytest.mark.parametrize("chained", [False, True])
def test_failed_profile_close_retires_session_with_caller_error_retained(chained):
    calls, references = [], []

    class ThreadBoundSession:
        def __init__(self, *args, **kwargs):
            self.owner = threading.get_ident()
            calls.append(("create", self.owner))
            references.append(weakref.ref(self))

        def close(self):
            current = threading.get_ident()
            calls.append(("close", current))
            assert current == self.owner
            if chained:
                try:
                    raise LookupError("profile finalization detail", self)
                except LookupError as cause:
                    raise ValueError("inert profile retirement failed") from cause
            raise ValueError("inert profile retirement failed")

        def __del__(self):
            calls.append(("destroy", threading.get_ident()))

    error = None
    with patch.object(consumer, "Session", ThreadBoundSession):
        execution = consumer.InferenceExecutor(object(), profile_prefix="inert-profile")
        try:
            try:
                execution.close()
            except BaseException as caught:
                error = caught
            else:
                pytest.fail("profile close failure was hidden")

            # Keep the caller's diagnostic intact while checking retirement.
            assert references[0]() is None
            assert [name for name, _ in calls] == ["create", "close", "destroy"]
            assert len({ident for _, ident in calls}) == 1
            assert calls[0][1] != threading.get_ident()
            assert execution.profile_path == ""
            assert execution._resource is None
            assert execution._resource_closed
            assert all(not worker.is_alive() for worker in execution._pool._threads)
            assert type(error) is RuntimeError
            assert str(error) == (
                "Owner resource cleanup failed: ValueError: inert profile retirement failed"
            )
            assert error.__context__ is None and error.__cause__ is None
            frames = []
            traceback = error.__traceback__
            while traceback is not None:
                frames.append(traceback.tb_frame.f_code.co_name)
                traceback = traceback.tb_next
            assert "_close_resource" not in frames and "_retire_resource" not in frames
            execution.close()
            assert [name for name, _ in calls] == ["create", "close", "destroy"]
        finally:
            execution.close()
            # The old dependency keeps the session in its caller traceback.
            # Release it only after the lifetime assertions, including on failure.
            if error is not None:
                error.__traceback__ = None
    gc.collect()
