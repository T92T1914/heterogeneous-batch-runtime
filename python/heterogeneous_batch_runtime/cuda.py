"""Explicit optional CUDA interface with a dedicated native resource owner."""
from concurrent.futures import Future
from queue import Queue
import threading
import weakref

import numpy as np

from . import _native


def _owner(queue, ready, shape, tile, device, shared_bins):
    context = None
    try:
        name = _native._cuda_select_device(device)
        context = _native._CudaHistogramContext(*shape, *tile, shared_bins)
        ready.set_result((name, context.device_bytes))
        while True:
            image, result = queue.get()
            if image is None:
                try:
                    context.close()
                except BaseException as exc:
                    result.set_exception(exc.with_traceback(None))
                else:
                    result.set_result(None)
                break
            try:
                result.set_result(context.run(image, *shape, *tile))
            except BaseException as exc:
                result.set_exception(exc.with_traceback(None))
            finally:
                image = result = None
    except BaseException as exc:
        if not ready.done():
            ready.set_exception(exc.with_traceback(None))
    finally:
        # Native destruction always happens on this thread and its CUDA device.
        if context is not None:
            try:
                context.close()
            except BaseException:
                # A requested close already forwards its error. Destruction
                # must still release the resource on this native owner.
                pass
            finally:
                del context


def _shutdown(queue, thread):
    result = Future()
    queue.put((None, result))
    thread.join()
    result.result()


class CudaHistogramContext:
    """Fixed-shape synchronous histogram. Explicit close or a with block is required.

    Public calls belong to the creating Python thread. A dedicated native owner
    selects the requested device, creates the context, executes and destroys it.
    The input must have no concurrent writers until run has copied it. The GIL
    does not exclude external native writers. No CPU fallback is provided.
    """

    def __init__(self, rows, cols, tile_rows, tile_cols, *, device=0, shared_bins=True):
        if not _native.cuda_built:
            raise RuntimeError("CUDA unavailable: install an explicitly CUDA-enabled build")
        for value, name, minimum in ((rows, "rows", 0), (cols, "cols", 0),
                                     (tile_rows, "tile_rows", 1), (tile_cols, "tile_cols", 1),
                                     (device, "device", 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(shared_bins) is not bool:
            raise TypeError("shared_bins must be bool")
        tiles = ((rows + tile_rows - 1) // tile_rows) * ((cols + tile_cols - 1) // tile_cols)
        if rows * cols + tiles * 256 * 8 > 64 * 1024 * 1024:
            raise ValueError("histogram context exceeds 64 MiB device-storage limit")
        self._owner = threading.get_ident()
        self._shape, self._tile = (rows, cols), (tile_rows, tile_cols)
        self._closed = False
        self._queue = Queue(maxsize=1)
        ready = Future()
        self._thread = threading.Thread(target=_owner,
            args=(self._queue, ready, self._shape, self._tile, device, shared_bins),
            name="hbr-cuda-owner", daemon=True)
        self._thread.start()
        try:
            self.device_name, self.device_bytes = ready.result()
        except BaseException:
            self._thread.join()
            raise
        # Ordinary abandonment can signal the owner without destroying CUDA on
        # a finalizer thread. Interpreter termination is not a supported drain.
        self._finalizer = weakref.finalize(self, _shutdown, self._queue, self._thread)
        self._finalizer.atexit = False

    def _check_owner(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("CUDA Python context belongs to its creating thread")

    def run(self, image):
        self._check_owner()
        if self._closed:
            raise RuntimeError("histogram context is closed")
        if not isinstance(image, np.ndarray) or image.dtype != np.dtype("uint8"):
            raise TypeError("exact uint8 NumPy array required")
        if image.ndim != 2 or image.shape != self._shape:
            raise ValueError("fixed two-dimensional image shape required")
        if not image.flags.c_contiguous:
            raise ValueError("C contiguous input required")
        # Snapshot completes here before any native work or GIL release.
        owned = image.copy(order="C")
        # An ndarray subclass can close this context during its copy method.
        if self._closed:
            raise RuntimeError("histogram context is closed")
        result = Future()
        self._queue.put((owned, result))
        return result.result()

    def close(self):
        self._check_owner()
        if not self._closed:
            self._closed = True
            self._finalizer()

    def __enter__(self):
        self._check_owner()
        if self._closed:
            raise RuntimeError("histogram context is closed")
        return self

    def __exit__(self, *exc):
        self.close()
