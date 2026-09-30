"""Bounded Windows CPU process-memory observations, separate from timings."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import gc
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import sys


REQUESTS = 64
CHECKPOINTS = (8, 16, 32, 64)
RTOL = 1e-5
ATOL = 2e-5


class ProcessMemoryCountersEx(ctypes.Structure):
    """The documented PROCESS_MEMORY_COUNTERS_EX layout."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class CurrentProcessMemory:
    """Read only the current process through its documented pseudo handle."""

    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("this probe requires Windows")
        # LOAD_LIBRARY_SEARCH_SYSTEM32 avoids application-directory DLL lookup.
        kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True, winmode=0x800)
        self._psapi = ctypes.WinDLL("psapi.dll", use_last_error=True, winmode=0x800)
        current = kernel.GetCurrentProcess
        current.argtypes = []
        current.restype = wintypes.HANDLE
        self._handle = current()
        self._read = self._psapi.GetProcessMemoryInfo
        self._read.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        self._read.restype = wintypes.BOOL
        # The pseudo handle is never opened, attached to, duplicated or closed.

    def read(self):
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not self._read(self._handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "working_set_bytes": int(counters.WorkingSetSize),
            "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            "private_usage_bytes": int(counters.PrivateUsage),
            "peak_commit_bytes": int(counters.PeakPagefileUsage),
        }


def probe(model_path, report):
    """Mutate the report so a failure retains preceding observations."""
    if sys.platform != "win32":
        raise RuntimeError("this probe requires Windows")
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator
    import onnxruntime as ort
    from adaptive_timing.runtime import Dispatch
    from hbr_image_classifier.executor import InferenceExecutor
    from hbr_image_classifier.model import example_images, load_model, prepare_arrays

    if ort.__version__ != "1.30.0":
        raise ValueError("the declared probe requires ONNX Runtime 1.30.0")
    model = load_model(model_path)
    batch = prepare_arrays(example_images()[:8])
    reference = ReferenceEvaluator(onnx.load_model_from_string(model.data))
    assert_close = np.testing.assert_allclose  # Import the validator before baseline.
    prepared_reference = np.concatenate([
        reference.run(None, {"Input3": image[None]})[0] for image in batch
    ])
    del prepared_reference
    memory = CurrentProcessMemory()
    report.update({
        "model_sha256": model.sha256,
        "input_sha256": sha256(batch.tobytes()).hexdigest(),
        "input_shape": list(batch.shape),
        "provider_requested": "cpu",
        "ort_version": ort.__version__,
        "application_version": metadata.version("hbr-image-classifier"),
        "numpy_version": np.__version__,
        "onnx_version": onnx.__version__,
        "request_limit": REQUESTS,
        "completed_requests": 0,
        "full_output_checks": 0,
        "maximum_absolute_logit_difference": 0.0,
        "rtol": RTOL,
        "atol": ATOL,
        "observed_transitions": 0,
        "thread_budget": {
            "executor_workers": 1,
            "ort_intra_op_threads": 1,
            "ort_inter_op_threads": 1,
            "ort_execution": "sequential",
        },
    })

    def snapshot(phase, outstanding):
        report["snapshots"].append({
            "phase": phase,
            "completed_requests": report["completed_requests"],
            "outstanding_requests": outstanding,
            **memory.read(),
        })

    snapshot("after_imports_model_and_input_preparation", 0)
    execution = InferenceExecutor(model, provider="cpu", capacity=1)
    try:
        snapshot("after_session_creation", execution.outstanding)
        for number in range(1, REQUESTS + 1):
            execution.submit_batch(
                Dispatch(str(number), "inference", 0, 1, "dispatched"), 1, batch
            )
            results = execution.wait(1, timeout=30)
            if len(results) != 1 or not results[0].usable:
                raise RuntimeError("one usable completed inference result is required")
            result = results[0]
            report["completed_requests"] += 1
            if execution.outstanding != 0:
                raise RuntimeError("completed request was not fully reconciled")
            actual = result.value
            if actual.shape != (8, 10) or actual.dtype != np.dtype("float32"):
                raise ValueError("completed float32 (8,10) logits are required")
            # Recompute all independent reference logits for every request.
            expected = np.concatenate([
                reference.run(None, {"Input3": image[None]})[0] for image in batch
            ])
            assert_close(actual, expected, rtol=RTOL, atol=ATOL)
            report["full_output_checks"] += 1
            report["maximum_absolute_logit_difference"] = max(
                report["maximum_absolute_logit_difference"],
                float(np.abs(actual - expected).max()),
            )
            report["observed_transitions"] += len(execution.observations())
            # Do not retain previous result tensors or a growing event history.
            del result, results, actual, expected
            if number in CHECKPOINTS:
                snapshot("after_completed_requests", execution.outstanding)
    finally:
        # Close waits for physical completion. A timeout never frees active data.
        execution.close()
        snapshot("after_explicit_close", execution.outstanding)
        del execution
        gc.collect()
        snapshot("after_close_executor_release_and_gc", 0)
    report["status"] = "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "status": "incomplete",
        "probe": "windows_cpu_current_process_memory_64_requests",
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "windows_version": platform.version() if sys.platform == "win32" else None,
        "pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
        "snapshots": [],
        "definitions": {
            "working_set_bytes": "Current resident working set for the whole process, including shared pages.",
            "peak_working_set_bytes": "Cumulative process-lifetime working-set peak, including activity before this probe.",
            "private_usage_bytes": "Current private committed bytes for the whole process, not resident bytes or live tensor bytes.",
            "peak_commit_bytes": "Cumulative process-lifetime private-commit peak, including activity before this probe.",
        },
        "scope": "One CPU session and 64 serial completed/reconciled requests of eight prepared example images. No latency measurements or GPU execution.",
        "retained_at_final_snapshot": "Imports, model bytes, prepared input, independent ONNX evaluator and report rows remain alive. The executor and its session are explicitly retired.",
        "interpretation": "Totals include the independent validator, Python, libraries, executor bookkeeping and allocator caches. They do not isolate live session allocations, establish a strict memory cap, prove absence of leaks or promise lower memory after close.",
        "api_documentation": [
            "https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo",
            "https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex",
            "https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getcurrentprocess",
        ],
    }
    # Preserve a prior report even if this run fails. No automatic acquisition.
    with Path(args.output).open("x", encoding="utf-8") as output:
        try:
            probe(args.model, report)
        except BaseException as error:
            report["failure"] = {"type": type(error).__name__}
            output.write(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"status": report["status"], "failure": report["failure"]}))
            return 1
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "completed_requests": report["completed_requests"],
        "full_output_checks": report["full_output_checks"],
        "snapshots": len(report["snapshots"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
