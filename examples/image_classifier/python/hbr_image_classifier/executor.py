"""Reuse Adaptive Timing's existing admission and physical-completion path."""
import numpy as np
from adaptive_timing.execution import OwnerExecutor
from .model import MAX_BATCH
from .session import Session

class InferenceExecutor(OwnerExecutor):
    def __init__(self, model, *, provider="cpu", capacity=4, library=None, profile_prefix="", execution_mode="serial"):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 4:
            raise ValueError("capacity must be an integer in [1,4]")
        if not isinstance(execution_mode, str) or execution_mode not in ("serial", "batched"):
            raise ValueError("execution_mode must be serial or batched")
        self.profile_path = ""
        self._accepted = 0
        self.execution_mode = execution_mode
        super().__init__(lambda: Session(model, provider=provider, library=library,
            profile_prefix=profile_prefix, execution_mode=execution_mode), capacity=capacity)

    @property
    def execution_provenance(self):
        self._assert_owner()
        if self._resource is None:
            raise RuntimeError("inference session is closed")
        return {name: getattr(self._resource, name) for name in (
            "model_sha256", "source_model_sha256", "execution_model_sha256",
            "execution_mode", "execution_recipe")}

    def submit_batch(self, dispatch, generation, batch, *, gate=None):
        self._assert_owner()
        if self._accepted >= 256:
            raise RuntimeError("session request limit reached; close and start a new session")
        if not isinstance(batch, np.ndarray):
            raise TypeError("exact float32 input required")
        actual = np.asarray(batch)
        if actual.dtype != np.dtype("float32"):
            raise TypeError("exact float32 input required")
        if actual.ndim != 4 or actual.shape[1:] != (1,28,28) or actual.shape[0] > MAX_BATCH:
            raise ValueError("batch must have shape (N,1,28,28), N <= 8")
        if not np.isfinite(actual).all() or (actual < 0).any() or (actual > 1).any():
            raise ValueError("finite normalized pixels required")
        owned = np.array(actual, copy=True, order="C")
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
