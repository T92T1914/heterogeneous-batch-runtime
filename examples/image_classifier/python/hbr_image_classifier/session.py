"""One synchronous C++ session, closed by its owning executor worker."""
from importlib.util import find_spec
from pathlib import Path
import os
import numpy as np
from .model import Model

def runtime_library():
    spec = find_spec("onnxruntime")
    if spec is None or spec.origin is None:
        raise RuntimeError("install the explicitly selected ONNX Runtime provider package")
    directory = Path(spec.origin).parent / "capi"
    names = ("onnxruntime.dll",) if os.name == "nt" else ("libonnxruntime.so.1.30.0", "libonnxruntime.so")
    for name in names:
        path = directory / name
        if path.is_file():
            return str(path.resolve())
    raise RuntimeError("provider package does not contain the expected standalone library")

class Session:
    def __init__(self, model, *, provider="cpu", library=None, profile_prefix=""):
        if not isinstance(model, Model):
            raise TypeError("a reviewed Model is required")
        from ._session import Session as NativeSession
        self._native = NativeSession(library or runtime_library(), model.data, provider, str(profile_prefix))
        self.provider = provider
        self.model_sha256 = model.sha256

    def run(self, batch):
        if not isinstance(batch, np.ndarray) or batch.dtype != np.dtype("float32"):
            raise TypeError("exact float32 NumPy input required")
        if not batch.flags.c_contiguous:
            raise ValueError("C contiguous input required")
        return self._native.run(batch)

    def close(self):
        return self._native.close()
