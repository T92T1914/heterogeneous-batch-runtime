"""Interrupted Python owner construction must signal retirement before joining."""
from concurrent.futures import Future
from queue import Queue
import threading
from types import SimpleNamespace
import weakref

import pytest

from heterogeneous_batch_runtime import _native
import heterogeneous_batch_runtime.cuda as cuda


@pytest.mark.parametrize("stage", ["pending", "ready"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_interrupted_initializer_retires_its_owner(
    monkeypatch, stage, error_type, cleanup_fails
):
    calls, references, threads, missing_stops = [], [], [], []
    entered, release = threading.Event(), threading.Event()
    original_result = Future.result
    original_thread = threading.Thread
    interruption = error_type("caller stopped owner initialization")
    injected = False

    class RecordingQueue(Queue):
        stops = 0

        def put(self, item, *args, **kwargs):
            if item[0] is None:
                self.stops += 1
            return super().put(item, *args, **kwargs)

    class ContainedThread(original_thread):
        def __init__(self, *args, **kwargs):
            self.owner_queue = kwargs["args"][0]
            super().__init__(*args, **kwargs)
            threads.append(self)

        def join(self, timeout=None):
            if self.is_alive() and not self.owner_queue.stops:
                missing_stops.append(True)
                # Contain the old implementation rather than hanging its test.
                self.owner_queue.put((None, Future()))
            super().join(2)
            assert not self.is_alive(), "inert owner did not retire"

    class Native:
        device_bytes = 42

        def __init__(self, *args):
            self.owner = threading.get_ident()
            calls.append(("create", self.owner))
            references.append(weakref.ref(self))
            entered.set()
            if stage == "pending" and not release.wait(2):
                raise TimeoutError("native factory gate")

        def close(self):
            current = threading.get_ident()
            calls.append(("close", current))
            assert current == self.owner
            if cleanup_fails:
                raise ValueError("inert initialization close failed")

        def __del__(self):
            calls.append(("destroy", threading.get_ident()))

    def interrupted_result(future, timeout=None):
        nonlocal injected
        if not injected:
            injected = True
            assert entered.wait(1)
            if stage == "ready":
                original_result(future, 1)
            release.set()
            raise interruption
        return original_result(future, timeout)

    monkeypatch.setattr(_native, "cuda_built", True)
    monkeypatch.setattr(_native, "_cuda_select_device", lambda device: "inert owner", raising=False)
    monkeypatch.setattr(_native, "_CudaHistogramContext", Native, raising=False)
    monkeypatch.setattr(cuda, "Queue", RecordingQueue)
    monkeypatch.setattr(threading, "Thread", ContainedThread)
    monkeypatch.setattr(Future, "result", interrupted_result)
    try:
        with pytest.raises(error_type) as caught:
            cuda.CudaHistogramContext(2, 2, 1, 1)
        assert caught.value is interruption
        assert not missing_stops
        assert not threads[0].is_alive()
        assert references[0]() is None
        assert calls[0][0] == "create" and calls[-1][0] == "destroy"
        assert len({ident for _, ident in calls}) == 1
        assert calls[0][1] != threading.get_ident()
        if cleanup_fails:
            assert any("ValueError: inert initialization close failed" in note
                       for note in getattr(interruption, "__notes__", ()))
    finally:
        release.set()
        interruption.__traceback__ = None


@pytest.mark.parametrize("failure_stage", ["device", "factory"])
def test_failed_initializer_does_not_wait_on_an_unconsumed_stop(monkeypatch, failure_stage):
    calls, threads, result_calls = [], [], []
    original_thread = threading.Thread
    original_result = Future.result
    failure = ValueError("inert initialization failed")

    class RecordingThread(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)

        def join(self, timeout=None):
            super().join(2)
            assert not self.is_alive(), "failed initializer owner did not retire"

    def select_device(device):
        calls.append("device")
        if failure_stage == "device":
            raise failure
        return "inert owner"

    def factory(*args):
        calls.append("factory")
        raise failure

    def checked_result(future, timeout=None):
        result_calls.append(future)
        # A failed factory cannot process a stop request. Never wait forever.
        return original_result(future, 1)

    monkeypatch.setattr(_native, "cuda_built", True)
    monkeypatch.setattr(_native, "_cuda_select_device", select_device, raising=False)
    monkeypatch.setattr(_native, "_CudaHistogramContext", factory, raising=False)
    monkeypatch.setattr(threading, "Thread", RecordingThread)
    monkeypatch.setattr(Future, "result", checked_result)
    try:
        with pytest.raises(ValueError) as caught:
            cuda.CudaHistogramContext(2, 2, 1, 1)
        assert caught.value is failure
        assert len(result_calls) == 1
        assert calls == (["device"] if failure_stage == "device" else ["device", "factory"])
        assert not threads[0].is_alive()
    finally:
        failure.__traceback__ = None


@pytest.mark.parametrize("phase", ["hostile-note", "detach-before-effect", "detach-after-effect"])
def test_secondary_acquisition_cleanup_failure_preserves_error_and_retires_owner(monkeypatch, phase):
    calls, references, threads, finalizers, queues, joins, notes = [], [], [], [], [], [], []
    callbacks = []
    original_thread = threading.Thread
    original_finalize = weakref.finalize
    secondary = RuntimeError("injected diagnostic or detach failure")

    class AcquisitionError(Exception):
        def add_note(self, note):
            notes.append(note)
            if phase == "hostile-note":
                raise secondary
            super().add_note(note)

    failure = AcquisitionError("original acquisition failure")

    class Native:
        device_bytes = 42

        def __init__(self, *args):
            calls.append(("create", threading.get_ident()))
            references.append(weakref.ref(self))

        def close(self):
            calls.append(("close", threading.get_ident()))
            if phase == "hostile-note":
                raise ValueError("inert cleanup failed")

        def __del__(self):
            calls.append(("destroy", threading.get_ident()))

    class RecordingQueue(Queue):
        stops = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            queues.append(self)

        def put(self, item, *args, **kwargs):
            if item[0] is None:
                self.stops += 1
            return super().put(item, *args, **kwargs)

    class RecordingThread(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            threads.append(self)

        def join(self, timeout=None):
            joins.append(True)
            for callback in callbacks:
                cells = dict(zip(callback.__code__.co_freevars, callback.__closure__))
                assert not cells["retired"].cell_contents.is_set()
            super().join(2)
            assert not self.is_alive(), "inert secondary-failure owner did not retire"

    futures = []

    class InterruptedReadiness(Future):
        def __init__(self):
            super().__init__()
            futures.append(self)

        def result(self, timeout=None):
            value = super().result(1)
            if self is futures[0] and phase == "hostile-note":
                raise failure
            return value

    class FailingDetach:
        def __init__(self, finalizer):
            self.finalizer = finalizer

        @property
        def atexit(self):
            return self.finalizer.atexit

        @atexit.setter
        def atexit(self, value):
            self.finalizer.atexit = value
            raise failure

        def detach(self):
            # Deliberate fault injection around a real registered finalizer.
            # This is not evidence that CPython detach has either failure mode.
            if phase == "detach-after-effect":
                self.finalizer.detach()
            raise secondary

    def finalize(*args, **kwargs):
        callback = args[1]
        callbacks.append(callback)
        assert callback.__code__.co_freevars == ("owner_queue", "owner_thread", "retired")
        assert tuple(cell.cell_contents for cell in callback.__closure__[:2]) == (queues[0], threads[0])
        finalizer = original_finalize(*args, **kwargs)
        finalizers.append(finalizer)
        return FailingDetach(finalizer)

    monkeypatch.setattr(_native, "cuda_built", True)
    monkeypatch.setattr(_native, "_cuda_select_device", lambda device: "inert owner", raising=False)
    monkeypatch.setattr(_native, "_CudaHistogramContext", Native, raising=False)
    monkeypatch.setattr(cuda, "Queue", RecordingQueue)
    monkeypatch.setattr(cuda, "Future", InterruptedReadiness)
    monkeypatch.setattr(threading, "Thread", RecordingThread)
    if phase != "hostile-note":
        monkeypatch.setattr(cuda, "weakref", SimpleNamespace(finalize=finalize))
    caught = None
    try:
        try:
            cuda.CudaHistogramContext(2, 2, 1, 1)
        except BaseException as error:
            caught = error
        assert caught is failure
        assert joins == [True]
        assert not threads[0].is_alive()
        assert references[0]() is None
        assert calls[0][0] == "create" and calls[-1][0] == "destroy"
        assert len({ident for _, ident in calls}) == 1
        assert calls[0][1] != threading.get_ident()
        assert queues[0].stops == 1
        assert len(notes) == 1
        if phase == "hostile-note":
            assert "ValueError: inert cleanup failed" in notes[0]
        else:
            assert "fallback detach failed: RuntimeError" in notes[0]
            actual = finalizers[0]
            assert actual.alive == (phase == "detach-before-effect")
            actual()
            # A still-registered callback cannot repeat completed retirement.
            assert not actual.alive
            assert queues[0].stops == 1 and joins == [True]
    finally:
        # Baseline failures must not strand the controlled owner or fallback.
        for finalizer in finalizers:
            finalizer.detach()
        for thread in threads:
            if thread.is_alive():
                queues[0].put((None, Future()))
                original_thread.join(thread, 2)
            assert not thread.is_alive(), "secondary-failure containment failed"
        if caught is not None:
            caught.__traceback__ = None
            caught.__context__ = None
        failure.__traceback__ = None
        secondary.__traceback__ = None


@pytest.mark.parametrize("phase", ["start-before", "start-after", "register", "configure"])
def test_owner_acquisition_failure_preserves_error_and_retires_launched_thread(monkeypatch, phase):
    calls, references, threads, finalizers, joins = [], [], [], [], []
    original_thread = threading.Thread
    original_finalize = weakref.finalize
    error_type = {"start-before": RuntimeError, "start-after": KeyboardInterrupt,
                  "register": MemoryError, "configure": SystemExit}[phase]
    failure = error_type("inert owner acquisition failed")

    class Native:
        device_bytes = 42

        def __init__(self, *args):
            calls.append(("create", threading.get_ident()))
            references.append(weakref.ref(self))

        def close(self):
            calls.append(("close", threading.get_ident()))

        def __del__(self):
            calls.append(("destroy", threading.get_ident()))

    class RecordingThread(original_thread):
        def __init__(self, *args, **kwargs):
            self.owner_queue = kwargs["args"][0]
            super().__init__(*args, **kwargs)
            threads.append(self)

        def start(self):
            if phase == "start-before":
                raise failure
            super().start()
            if phase == "start-after":
                # The public start completed, so an actual owner now exists.
                raise failure

        def join(self, timeout=None):
            joins.append(True)
            assert self.ident is not None, "must not join an unstarted owner"
            super().join(2)
            assert not self.is_alive(), "inert acquisition owner did not retire"

    class FailingConfiguration:
        def __init__(self, finalizer):
            self.finalizer = finalizer

        @property
        def atexit(self):
            return self.finalizer.atexit

        @atexit.setter
        def atexit(self, value):
            raise failure

        def detach(self):
            return self.finalizer.detach()

    def finalize(*args, **kwargs):
        if phase == "register":
            raise failure
        finalizer = original_finalize(*args, **kwargs)
        finalizers.append(finalizer)
        return FailingConfiguration(finalizer) if phase == "configure" else finalizer

    monkeypatch.setattr(_native, "cuda_built", True)
    monkeypatch.setattr(_native, "_cuda_select_device", lambda device: "inert owner", raising=False)
    monkeypatch.setattr(_native, "_CudaHistogramContext", Native, raising=False)
    monkeypatch.setattr(threading, "Thread", RecordingThread)
    monkeypatch.setattr(cuda, "weakref", SimpleNamespace(finalize=finalize))
    try:
        with pytest.raises(error_type) as caught:
            cuda.CudaHistogramContext(2, 2, 1, 1)
        assert caught.value is failure
        assert not threads[0].is_alive()
        if phase == "start-before":
            assert not calls and not joins
        else:
            assert joins == [True]
            assert references[0]() is None
            assert calls[0][0] == "create" and calls[-1][0] == "destroy"
            assert len({ident for _, ident in calls}) == 1
            assert calls[0][1] != threading.get_ident()
        assert all(not finalizer.alive for finalizer in finalizers)
    finally:
        # Contain an old implementation that never reached its join. Detach
        # fallback callbacks first so later collection cannot rejoin a dead owner.
        for finalizer in finalizers:
            finalizer.detach()
        if threads[0].is_alive():
            threads[0].owner_queue.put((None, Future()))
            original_thread.join(threads[0], 2)
            assert not threads[0].is_alive(), "baseline containment failed"
        failure.__traceback__ = None
