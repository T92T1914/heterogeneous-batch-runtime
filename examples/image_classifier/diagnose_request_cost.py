"""Bounded host-clock diagnostics around the existing installed batch consumer."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

directory = Path(__file__).resolve().parent
if str(directory) not in sys.path:
    sys.path.insert(0, str(directory))

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from adaptive_timing.runtime import Dispatch
from hbr_image_classifier.executor import InferenceExecutor
from hbr_image_classifier.model import example_images, load_model, prepare_arrays
from hbr_image_classifier.session import Session
from compare_batching import Budget, serializable, source_identities
from memory_check import CurrentProcessMemory


FIRST = 1
WARMUPS = 5
SAMPLES = 20
RTOL = 1e-5
ATOL = 2e-5


def clock_record(name):
    info = time.get_clock_info(name)
    return {key: getattr(info, key) for key in ("implementation", "monotonic", "adjustable", "resolution")}


def fixtures():
    images = example_images()[:8]
    return (images, images[:, :, ::-1].copy(), images[[3, 0, 3, 7, 1, 5, 4, 2]],
            np.random.default_rng(10415).integers(0, 256, (8, 28, 28), dtype=np.uint8))


def oracle(model, frames):
    evaluator = ReferenceEvaluator(onnx.load_model_from_string(model.data))
    return [np.concatenate([evaluator.run(None, {"Input3": image[None]})[0]
                            for image in prepare_arrays(frame)]) for frame in frames]


def run(model, *, provider, instrumentation, replicate, output, max_seconds):
    budget = Budget(max_seconds)
    original_run = Session.run
    events = []
    report = {"schema_version": 1, "status": "incomplete", "provider": provider,
        "instrumentation": instrumentation, "replicate": replicate, "rows": [],
        "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_identities": source_identities(), "source_model_sha256": model.sha256,
        "clocks": {name: clock_record(name) for name in ("perf_counter", "monotonic")},
        "configuration": {"batch": 8, "mode": "batched", "first_requests": FIRST,
            "warmups": WARMUPS, "samples": SAMPLES, "capacity": 1, "max_seconds": max_seconds,
            "rtol": RTOL, "atol": ATOL},
        "scope": "Diagnostic host intervals and observer perturbation, not an optimization or device-cost measurement. Fixture construction is outside primary. Fresh uint8 preparation through usable owned host-result adoption is inside primary."}
    memory = CurrentProcessMemory()
    execution = None
    failure = None
    journal = output.with_suffix(output.suffix + ".jsonl")
    if output.exists() or journal.exists():
        raise ValueError("preserve previous evidence; select a new output identity")

    def observed_run(session, batch):
        started = time.perf_counter_ns()
        error_name = None
        try:
            return original_run(session, batch)
        except BaseException as error:
            error_name = type(error).__name__
            raise
        finally:
            finished = time.perf_counter_ns()
            events.append({"started_ns": started, "finished_ns": finished, "error_type": error_name})

    with output.open("x", encoding="utf8") as destination, journal.open("x", encoding="utf8") as stream:
        def retain(row):
            row = serializable(row)
            report["rows"].append(row)
            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
        try:
            frames = fixtures()
            expected = oracle(model, frames)
            retain({"record": "reference", "frames_sha256": [sha256(frame.tobytes()).hexdigest() for frame in frames],
                    "original_reference_logits": expected})
            retain({"record": "memory", "phase": "before_setup", **memory.read()})
            if instrumentation == "worker":
                Session.run = observed_run
            prefix = str(output.with_suffix("")) + "-ort" if instrumentation == "ort" else ""
            budget.remaining("setup")
            started = time.perf_counter_ns()
            execution = InferenceExecutor(model, provider=provider, execution_mode="batched", capacity=1, profile_prefix=prefix)
            retain({"record": "setup", "seconds": (time.perf_counter_ns()-started)/1e9,
                    "provenance": execution.execution_provenance})
            for request in range(FIRST + WARMUPS + SAMPLES):
                budget.remaining("request")
                frame_index = request % len(frames)
                caller_images = frames[frame_index].copy()
                start = time.perf_counter_ns()
                batch = prepare_arrays(caller_images)
                prepared = time.perf_counter_ns()
                ident = f"r{replicate}-{instrumentation}-{request}"
                dispatch = Dispatch(ident, "inference", 0, 1, "dispatched")
                admission_start = time.perf_counter_ns()
                execution.submit_batch(dispatch, 1, batch)
                admitted = time.perf_counter_ns()
                results = execution.wait(1, timeout=budget.remaining("wait"))
                end = time.perf_counter_ns()
                if len(results) != 1 or not results[0].usable or results[0].status != "completed" or execution.outstanding:
                    retain({"record": "request_failure", "request_id": ident, "results": results})
                    raise RuntimeError("completed usable independent result required")
                result = results[0]
                row = {"record": "request", "request_id": ident, "frame": frame_index,
                    "phase": "first" if request == 0 else ("warmup" if request <= WARMUPS else "sample"),
                    "sample": request-WARMUPS-1 if request>WARMUPS else None,
                    "started_ns": start, "prepared_ns": prepared, "admission_started_ns": admission_start,
                    "admitted_ns": admitted, "finished_ns": end,
                    "primary_seconds": (end-start)/1e9, "preparation_seconds": (prepared-start)/1e9,
                    "admission_seconds": (admitted-admission_start)/1e9,
                    "wait_reconciliation_seconds": (end-admitted)/1e9,
                    "caller_gap_seconds": (admission_start-prepared)/1e9, "result": result}
                if instrumentation == "worker":
                    if len(events) != request+1:
                        raise RuntimeError("one worker envelope per admitted request required")
                    event = events[-1]
                    if not start <= event["started_ns"] <= event["finished_ns"] <= end or event["error_type"]:
                        raise RuntimeError("worker/caller clock containment failed")
                    row["worker"] = {**event,
                        "session_run_host_seconds": (event["finished_ns"]-event["started_ns"])/1e9,
                        "prefix_seconds": (event["started_ns"]-start)/1e9,
                        "suffix_seconds": (end-event["finished_ns"])/1e9,
                        "starts_before_admission_return": event["started_ns"] < admitted}
                values = result.value
                validation_start = time.perf_counter_ns()
                try:
                    if values.shape != (8,10) or values.dtype != np.float32 or not values.flags.owndata or np.shares_memory(values, batch):
                        raise ValueError("owned complete float32 host result required")
                    np.testing.assert_allclose(values, expected[frame_index], rtol=RTOL, atol=ATOL)
                except BaseException as error:
                    row.update(validation_passed=False, error_type=type(error).__name__, error=str(error))
                    retain(row)
                    raise
                row.update(validation_passed=True, validation_seconds=(time.perf_counter_ns()-validation_start)/1e9,
                    max_absolute_difference=float(np.abs(values-expected[frame_index]).max()))
                retain(row)
            retain({"record": "memory", "phase": "after_requests", **memory.read()})
        except BaseException as error:
            failure = error
            retain({"record": "failure", "type": type(error).__name__, "message": str(error)})
        finally:
            if execution is not None:
                started = time.perf_counter_ns()
                closed_ok = False
                try:
                    execution.close()
                    retired = execution.reconcile(1)
                    if execution.outstanding:
                        raise RuntimeError("unresolved physical retirement")
                    retain({"record": "close", "seconds": (time.perf_counter_ns()-started)/1e9,
                            "owner_retired": True, "outstanding": execution.outstanding,
                            "retired_results": retired, "observations": execution.observations()})
                    closed_ok = True
                except BaseException as error:
                    retain({"record": "close_failure", "type": type(error).__name__, "message": str(error)})
                    failure = failure or error
                if closed_ok and instrumentation == "ort":
                    try:
                        if not execution.profile_path:
                            raise RuntimeError("ORT profiling did not produce a trace")
                        profile = Path(execution.profile_path)
                        trace = json.loads(profile.read_text())
                        categories = {}
                        for event in trace:
                            categories[event.get("cat", "unknown")] = categories.get(event.get("cat", "unknown"), 0)+1
                        model_runs = sum(e.get("name")=="model_run" for e in trace)
                        if failure is None and model_runs != FIRST + WARMUPS + SAMPLES:
                            raise RuntimeError("ORT trace model-run coverage incomplete")
                        providers = {e.get("args", {}).get("provider") for e in trace if e.get("cat")=="Node" and "provider" in e.get("args", {})}
                        wanted = "CPUExecutionProvider" if provider == "cpu" else "CUDAExecutionProvider"
                        if providers != {wanted}:
                            raise RuntimeError("ORT trace provider placement differs from request")
                        retain({"record": "ort_trace", "path": str(profile), "sha256": sha256(profile.read_bytes()).hexdigest(),
                            "categories": categories, "model_runs": model_runs, "node_providers": sorted(providers),
                            "scope": "Native ORT relative host timestamps. No cross-clock alignment or device duration is inferred."})
                    except BaseException as error:
                        retain({"record": "trace_failure", "type": type(error).__name__, "message": str(error)})
                        failure = failure or error
            Session.run = original_run
            report["source_identities_after"] = source_identities()
            if report["source_identities_after"] != report["source_identities"]:
                failure = failure or RuntimeError("installed source/library identities changed")
            report["status"] = "completed" if failure is None else "incomplete"
            report["elapsed_seconds"] = time.perf_counter()-budget.started
            if failure is not None:
                report["failure"] = {"type": type(failure).__name__, "message": str(failure)}
            elif report["elapsed_seconds"] > max_seconds:
                report["status"] = "incomplete"
                report["failure"] = {"type": "TimeoutError", "message": "collection budget exceeded"}
            destination.write(json.dumps(serializable(report), indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda-required"), required=True)
    parser.add_argument("--instrumentation", choices=("plain", "worker", "ort"), required=True)
    parser.add_argument("--replicate", type=int, choices=range(3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=8)
    args = parser.parse_args()
    report = run(load_model(args.model), provider=args.provider, instrumentation=args.instrumentation,
        replicate=args.replicate, output=args.output, max_seconds=args.max_seconds)
    print(json.dumps({"status": report["status"], "provider": args.provider,
        "instrumentation": args.instrumentation, "elapsed_seconds": report["elapsed_seconds"], "failure": report.get("failure")}))
    return 0 if report["status"]=="completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
