"""Separate bounded Windows required-CUDA memory observation."""
import argparse
import ctypes
import gc
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import time

from memory_check import CurrentProcessMemory, REQUESTS, CHECKPOINTS, RTOL, ATOL


class DeviceMemory:
    """Free device memory is shared OS accounting, not per-process allocation."""
    def __init__(self):
        import onnxruntime as ort
        ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
        distribution = metadata.distribution("nvidia-cuda-runtime")
        matches = [x for x in distribution.files or () if x.name == "cudart64_13.dll"]
        if len(matches) != 1:
            raise RuntimeError("one reviewed vendor-wheel CUDA runtime DLL required")
        path = Path(distribution.locate_file(matches[0])).resolve()
        self.library_sha256 = sha256(path.read_bytes()).hexdigest()
        self.library = ctypes.WinDLL(str(path), winmode=0x1100)
        self.read_info = self.library.cudaMemGetInfo
        self.read_info.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.read_info.restype = ctypes.c_int

    def read(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        status = self.read_info(ctypes.byref(free), ctypes.byref(total))
        if status != 0:
            raise RuntimeError(f"cudaMemGetInfo returned status {status}")
        if not 0 <= free.value <= total.value:
            raise ValueError("invalid shared-device memory accounting")
        return {"device_free_bytes": free.value, "device_total_bytes": total.value}


def probe(model_path, report):
    # This is a cooperative decision deadline. Initialization and retirement
    # cannot be interrupted safely merely because a wall-clock limit expires.
    deadline = time.perf_counter() + 30
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx.reference import ReferenceEvaluator
    from adaptive_timing.runtime import Dispatch
    from hbr_image_classifier.executor import InferenceExecutor
    from hbr_image_classifier.model import load_model, prepare_arrays, example_images
    if ort.__version__ != "1.30.0":
        raise ValueError("protocol requires ONNX Runtime 1.30.0")
    model = load_model(model_path)
    batch = prepare_arrays(example_images()[:8])
    reference = ReferenceEvaluator(onnx.load_model_from_string(model.data))
    expected = np.concatenate([reference.run(None, {"Input3": x[None]})[0] for x in batch])
    host, device = CurrentProcessMemory(), DeviceMemory()
    report.update({"provider_requested": "cuda-required", "model_sha256": model.sha256,
        "input_sha256": sha256(batch.tobytes()).hexdigest(), "input_shape": list(batch.shape),
        "ort_version": ort.__version__, "available_providers": ort.get_available_providers(),
        "cuda_runtime_library_sha256": device.library_sha256, "rtol": RTOL, "atol": ATOL,
        "completed_requests": 0, "full_output_checks": 0, "maximum_absolute_logit_difference": 0.0})
    def snapshot(phase, outstanding):
        report["snapshots"].append({"phase": phase, "completed_requests": report["completed_requests"],
            "outstanding_requests": outstanding, **device.read(), **host.read()})
    # The first device.read initializes CUDA runtime state before this baseline.
    snapshot("after_imports_reference_and_cuda_context_initialization", 0)
    execution = InferenceExecutor(model, provider="cuda-required", capacity=1)
    primary_error = None
    try:
        snapshot("after_session_creation", execution.outstanding)
        for number in range(1, REQUESTS + 1):
            if time.perf_counter() > deadline:
                raise TimeoutError("bounded memory observation incomplete")
            execution.submit_batch(Dispatch(str(number), "inference", 0, 1, "dispatched"), 1, batch)
            results = execution.wait(1, timeout=max(0, deadline-time.perf_counter()))
            if not results:
                raise TimeoutError("memory observation wait expired; physical retirement still required")
            result, = results
            if not result.usable or execution.outstanding != 0:
                raise RuntimeError("one physically completed reconciled result required")
            np.testing.assert_allclose(result.value, expected, rtol=RTOL, atol=ATOL)
            report["completed_requests"] += 1
            report["full_output_checks"] += 1
            report["maximum_absolute_logit_difference"] = max(report["maximum_absolute_logit_difference"], float(np.abs(result.value-expected).max()))
            execution.observations()
            del result
            if number in CHECKPOINTS:
                snapshot("after_completed_requests", execution.outstanding)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors = []
        def record_cleanup(phase, action):
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(error)
                report.setdefault("cleanup_errors", []).append({"phase": phase, "type": type(error).__name__})
        record_cleanup("physical_retirement", execution.close)
        record_cleanup("after_explicit_close", lambda:snapshot("after_explicit_close", execution.outstanding))
        del execution
        gc.collect()
        record_cleanup("after_close_executor_release_and_gc", lambda:snapshot("after_close_executor_release_and_gc", 0))
        if cleanup_errors and primary_error is None:
            raise RuntimeError("memory observation cleanup incomplete") from cleanup_errors[0]
    report["status"] = "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"schema_version": 1, "status": "incomplete", "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": "docs/inference-cuda-protocol.md", "python": platform.python_version(), "windows_version": platform.version(),
        "snapshots": [], "scope": "64 serial completed eight-image requests, one owner and no overlapping requests. Separate from timing collection.",
        "device_scope": "cudaMemGetInfo after context initialization reports shared-device OS free bytes, not this process's live allocations. Other applications can affect every sample.",
        "process_gpu_memory": "unavailable through this probe. No process allocation bound or GPU leak verdict is established.",
        "deadline": "30-second cooperative decision limit includes initialization. Synchronous initialization and safe retirement are not forcibly interrupted, so this is not a hard process wall-time cap.",
        "host_scope": "whole current-process working set, private commit and lifetime peaks. Imports, validator, loaded libraries and allocator caches are included.",
        "device_api": "https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html"}
    with Path(args.output).open("x", encoding="utf8") as stream:
        try:
            probe(args.model, report)
        except BaseException as error:
            report["failure"] = {"type": type(error).__name__}
            stream.write(json.dumps(report, indent=2)+"\n")
            raise
        stream.write(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "completed_requests": report["completed_requests"], "snapshots": len(report["snapshots"])}))


if __name__ == "__main__":
    main()
