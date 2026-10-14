"""One synchronous C++ session, closed by its owning executor worker."""
from importlib.util import find_spec
from pathlib import Path
from hashlib import sha256
import os
import numpy as np
from .model import Model, MODEL_SHA256
from .batching import BATCH_RECIPE, execution_model_data

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
    def __init__(self, model, *, provider="cpu", library=None, profile_prefix="", execution_mode="serial"):
        if not isinstance(execution_mode, str) or execution_mode not in ("serial", "batched"):
            raise ValueError("execution_mode must be serial or batched")
        if not isinstance(model, Model):
            raise TypeError("a reviewed Model is required")
        if model.sha256 != MODEL_SHA256 or sha256(model.data).hexdigest() != MODEL_SHA256:
            raise ValueError("model does not match the reviewed artifact")
        execution_data = execution_model_data(model, execution_mode)
        if library is None and provider == "cuda-required":
            import onnxruntime as runtime
            if "CUDAExecutionProvider" not in runtime.get_available_providers():
                raise RuntimeError("installed runtime does not provide CUDAExecutionProvider")
            # Use ORT's documented vendor-wheel search, without changing PATH
            # or loading an unrelated Torch installation. Explicit library
            # callers retain responsibility for their chosen dependencies.
            runtime.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
        from ._session import Session as NativeSession
        arguments = (library or runtime_library(), execution_data, provider, str(profile_prefix))
        self._native = NativeSession(*arguments, True) if execution_mode == "batched" else NativeSession(*arguments)
        self.provider = provider
        self.model_sha256 = model.sha256
        self.source_model_sha256 = model.sha256
        self.execution_model_sha256 = sha256(execution_data).hexdigest()
        self.execution_mode = execution_mode
        self.execution_recipe = BATCH_RECIPE if execution_mode == "batched" else None

    @property
    def model_invocations(self):
        return self._native.model_invocations

    def run(self, batch):
        if not isinstance(batch, np.ndarray) or batch.dtype != np.dtype("float32"):
            raise TypeError("exact float32 NumPy input required")
        if not batch.flags.c_contiguous:
            raise ValueError("C contiguous input required")
        return self._native.run(batch)

    def close(self):
        return self._native.close()
