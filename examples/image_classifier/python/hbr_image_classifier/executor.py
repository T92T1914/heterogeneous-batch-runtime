"""Reuse Adaptive Timing's existing admission and physical-completion path."""
import numpy as np
from adaptive_timing.execution import OwnerExecutor
from .model import MAX_BATCH
from .session import Session

class InferenceExecutor(OwnerExecutor):
    def __init__(self, model, *, provider="cpu", capacity=4, library=None, profile_prefix=""):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 4:
            raise ValueError("capacity must be an integer in [1,4]")
        self.profile_path = ""
        self._accepted = 0
        super().__init__(lambda: Session(model, provider=provider, library=library, profile_prefix=profile_prefix), capacity=capacity)

    def submit_batch(self, dispatch, generation, batch, *, gate=None):
        self._assert_owner()
        if self._accepted >= 256:
            raise RuntimeError("session request limit reached; close and start a new session")
        if not isinstance(batch, np.ndarray) or batch.dtype != np.dtype("float32"):
            raise TypeError("exact float32 input required")
        if batch.ndim != 4 or batch.shape[1:] != (1,28,28) or len(batch) > MAX_BATCH:
            raise ValueError("batch must have shape (N,1,28,28), N <= 8")
        if not np.isfinite(batch).all() or (batch < 0).any() or (batch > 1).any():
            raise ValueError("finite normalized pixels required")
        owned = np.array(batch, copy=True, order="C")
        owned.flags.writeable = False
        def execute(session, cancel):
            if gate is not None:
                gate(cancel)
            # Canceling running work is a request. Ordinary Run finishes before
            # storage is reusable. Its result remains inspectable if stale.
            return session.run(owned)
        self.submit_owned(dispatch, generation, execute)
        self._accepted += 1

    def _close_resource(self):
        try:
            self.profile_path = self._resource.close()
        finally:
            self._resource = None
