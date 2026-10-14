"""Compare complete serial and batched requests in a permitted performance slot."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from hashlib import sha256
from importlib import metadata, util
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
from adaptive_timing.runtime import Dispatch
from hbr_image_classifier.executor import InferenceExecutor
from hbr_image_classifier.model import example_images, load_model, prepare_arrays
from memory_check import CurrentProcessMemory


BATCHES = (1, 2, 8)
BLOCKS = 3
WARMUPS = 5
SAMPLES = 20
RTOL = 1e-5
ATOL = 2e-5
BOOTSTRAP_SEED = 10414
BOOTSTRAP_REPLICATES = 2000


def serializable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: serializable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unrecorded observation type: {type(value).__name__}")


class Budget:
    def __init__(self, seconds, started=None):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError("max_seconds must be finite and in (0,120]")
        self.started = time.perf_counter() if started is None else started
        self.deadline = self.started + seconds

    def remaining(self, phase):
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"declared collection budget exhausted during {phase}")
        return remaining


def source_identities():
    from hbr_image_classifier.session import runtime_library
    identities = {"collector": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).read_bytes()).hexdigest()}}
    for name in ("hbr_image_classifier.model", "hbr_image_classifier.batching", "hbr_image_classifier.session", "hbr_image_classifier.executor",
                 "hbr_image_classifier._session", "adaptive_timing.execution", "adaptive_timing.runtime", "memory_check"):
        spec = util.find_spec(name)
        if spec is None or spec.origin is None or not Path(spec.origin).is_file():
            raise RuntimeError(f"installed source identity unavailable: {name}")
        path = Path(spec.origin).resolve()
        identities[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()}
    library = Path(runtime_library()).resolve()
    for name, path in [("onnxruntime_library", library),
                       ("onnxruntime_shared_provider", library.parent / "onnxruntime_providers_shared.dll"),
                       ("onnxruntime_cuda_provider", library.parent / "onnxruntime_providers_cuda.dll")]:
        if path.is_file():
            digest = sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1048576), b""):
                    digest.update(chunk)
            identities[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return identities


def summarize(report):
    """Keep independent block variation separate from requests within a block."""
    summary = []
    paired = []
    requests = [row for row in report["rows"] if row["record"] == "request" and row["phase"] == "sample"]
    for count in BATCHES:
        block_differences = []
        for mode in ("serial", "batched"):
            values = [row["primary_seconds"] for row in requests if row["batch"] == count and row["mode"] == mode]
            if len(values) != BLOCKS * SAMPLES:
                raise ValueError("incomplete deciding sample group")
            block_medians = []
            for block in range(BLOCKS):
                block_values = [row["primary_seconds"] for row in requests if row["batch"] == count and row["mode"] == mode and row["block"] == block]
                if len(block_values) != SAMPLES:
                    raise ValueError("incomplete deciding block")
                block_medians.append({"block": block, "median_seconds": statistics.median(block_values),
                    "minimum_seconds": min(block_values), "maximum_seconds": max(block_values),
                    "p95_seconds": float(np.percentile(block_values, 95))})
            median = statistics.median(values)
            summary.append({"batch": count, "mode": mode, "samples": len(values), "blocks": BLOCKS,
                "median_seconds": median, "minimum_seconds": min(values), "maximum_seconds": max(values),
                "p95_seconds": float(np.percentile(values, 95)), "images_per_second_at_median": count / median,
                "images_per_second_over_samples": count * len(values) / sum(values), "block_summaries": block_medians})
        for block in range(BLOCKS):
            serial = {row["sample"]: row["primary_seconds"] for row in requests if row["batch"] == count and row["mode"] == "serial" and row["block"] == block}
            batched = {row["sample"]: row["primary_seconds"] for row in requests if row["batch"] == count and row["mode"] == "batched" and row["block"] == block}
            if set(serial) != set(range(SAMPLES)) or set(batched) != set(serial):
                raise ValueError("incomplete within-block sample pairing")
            difference = statistics.median(serial.values()) - statistics.median(batched.values())
            block_differences.append(difference)
            paired.append({"batch": count, "block": block, "serial_minus_batched_block_median_seconds": difference,
                "same_index_request_differences_seconds": [serial[index] - batched[index] for index in range(SAMPLES)]})
        rng = random.Random(BOOTSTRAP_SEED + count)
        bootstrap = [statistics.median(rng.choices(block_differences, k=BLOCKS)) for _ in range(BOOTSTRAP_REPLICATES)]
        lower, upper = [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])]
        paired.append({"batch": count, "blocks": BLOCKS,
            "median_serial_minus_batched_block_median_seconds": statistics.median(block_differences),
            "block_bootstrap_interval_seconds": [lower, upper], "bootstrap_seed": BOOTSTRAP_SEED + count,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "interpretation": "Positive favors batched. Three blocks give a coarse small-study interval. Requests within each block are not independent block replications.",
            "decision": "inconclusive_interval_spans_zero" if lower <= 0 <= upper else ("batched_faster_in_declared_blocks" if lower > 0 else "serial_faster_in_declared_blocks")})
    return {"conditions": summary, "paired": paired,
        "pairing": "Same batch/block/sample index across sequential reused mode sessions. Paired requests are nonadjacent within a block.",
        "p95_method": "NumPy percentile with linear interpolation",
        "scope": "Original geometric inputs and three ordered blocks on one shared workstation. No fleet, accuracy-dataset or universal speed claim."}


def collect(model, *, provider="cpu", max_seconds=120, on_record=None, started=None, report=None):
    import onnxruntime as ort
    if provider not in ("cpu", "cuda-required"):
        raise ValueError("provider must be cpu or cuda-required")
    budget = Budget(max_seconds, started)
    if ort.__version__ != "1.30.0":
        raise ValueError("declared comparison requires ONNX Runtime 1.30.0")
    report = {} if report is None else report
    report.update({"schema_version": 1, "status": "incomplete", "provider_requested": provider,
        "source_model_sha256": model.sha256, "ort_version": ort.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "application_version": metadata.version("hbr-image-classifier"),
        "numpy_version": np.__version__, "onnx_version": onnx.__version__, "source_identities": source_identities(),
        "configuration": {"batches": BATCHES, "blocks": BLOCKS, "warmups_per_mode_block": WARMUPS,
            "samples_per_mode_block": SAMPLES, "first_request_per_mode_block": 1, "max_seconds": max_seconds,
            "maximum_requests_per_session": 1 + WARMUPS + SAMPLES, "session_identity_limit": 256,
            "capacity": 1, "ort_intra_op_threads": 1, "ort_inter_op_threads": 1,
            "required_ort_execution": "sequential", "required_graph_optimization": "all", "rtol": RTOL, "atol": ATOL},
        "primary_boundary": "Original in-memory uint8 images before preparation through InferenceExecutor.submit_batch and completed usable wait/reconciliation host output. File decoding, reference validation, setup and close are separate.",
        "secondary_boundary": "Not collected. No prepared-tensor direct Session timing is inferred.",
        "memory_scope": "Current-process Windows working set/private commit and cumulative process-lifetime peaks. Not an allocation ledger, leak proof or total GPU bound.",
        "device_memory": "Not queried by this collector. Separate root-owned device observations are required.",
        "provider_placement": "Required-provider errors stop collection. Actual operation-placement and model-invocation evidence belong to separate correctness acceptance.",
        "budget_scope": "Clock checks include reference setup, session setup, all requests, validation, recording and close. Blocking construction/close rely on the separately bounded owned process for forced retirement.",
        "rows": []})

    def retain(row):
        row = serializable(row)
        report["rows"].append(row)
        if on_record is not None:
            on_record(row)

    memory = CurrentProcessMemory()

    def snapshot(phase, condition, outstanding):
        retain({"record": "process_memory", "phase": phase, **condition, "outstanding_requests": outstanding,
            "seconds_since_collection_start": time.perf_counter() - budget.started, **memory.read()})

    budget.remaining("reference setup")
    reference_started = time.perf_counter()
    images = example_images()
    reference = ReferenceEvaluator(onnx.load_model_from_string(model.data))
    expected = {}
    for count in BATCHES:
        prepared = prepare_arrays(images[:count])
        expected[count] = np.concatenate([reference.run(None, {"Input3": image[None]})[0] for image in prepared])
    retain({"record": "reference_setup", "seconds": time.perf_counter() - reference_started,
        "source_model_sha256": model.sha256, "reference": "original serial ONNX ReferenceEvaluator",
        "expected_logits": expected, "original_image_sha256": sha256(images.tobytes()).hexdigest()})
    budget.remaining("reference setup")
    for block in range(BLOCKS):
        order = BATCHES[block:] + BATCHES[:block]
        for position, count in enumerate(order):
            modes = ("serial", "batched") if (block + BATCHES.index(count)) % 2 == 0 else ("batched", "serial")
            retain({"record": "run_order", "block": block, "batch_position": position, "batch": count,
                "batch_order": order, "mode_order": modes})
            for mode in modes:
                condition = {"block": block, "batch": count, "mode": mode}
                execution = None
                failure = None
                snapshot("before_setup", condition, 0)
                budget.remaining("session setup")
                setup_started = time.perf_counter()
                try:
                    execution = InferenceExecutor(model, provider=provider, capacity=1, execution_mode=mode)
                    setup_seconds = time.perf_counter() - setup_started
                    provenance = execution.execution_provenance
                    if provenance["source_model_sha256"] != model.sha256 or provenance["execution_mode"] != mode:
                        raise ValueError("session provenance does not match the declared condition")
                    retain({"record": "setup", **condition, "seconds": setup_seconds, "execution_provenance": provenance})
                    snapshot("after_setup", condition, execution.outstanding)
                    budget.remaining("session setup")
                    for request in range(1 + WARMUPS + SAMPLES):
                        phase = "first_request" if request == 0 else ("warmup" if request <= WARMUPS else "sample")
                        sample = request - WARMUPS - 1 if phase == "sample" else None
                        ident = f"b{block}-n{count}-{mode}-{request}"
                        budget.remaining("request preparation")
                        primary_started = time.perf_counter()
                        batch = prepare_arrays(images[:count])
                        preparation_finished = time.perf_counter()
                        dispatch = Dispatch(ident, "inference", 0, 1, "dispatched")
                        admission_started = time.perf_counter()
                        execution.submit_batch(dispatch, 1, batch)
                        admission_finished = time.perf_counter()
                        results = execution.wait(1, timeout=budget.remaining("completed request wait"))
                        primary_finished = time.perf_counter()
                        if len(results) != 1 or not results[0].usable or results[0].status != "completed" or execution.outstanding != 0:
                            retain({"record": "request_failure", **condition, "phase": phase, "sample": sample,
                                "request_id": ident, "results": results, "observations": execution.observations(),
                                "primary_seconds": primary_finished - primary_started, "outstanding_requests": execution.outstanding})
                            raise RuntimeError("one completed usable reconciled request is required")
                        result = results[0]
                        values = result.value
                        row = {"record": "request", **condition, "phase": phase, "sample": sample, "request_id": ident,
                            "primary_seconds": primary_finished - primary_started,
                            "preparation_seconds": preparation_finished - primary_started,
                            "admission_seconds": admission_finished - admission_started,
                            "wait_reconciliation_seconds": primary_finished - admission_finished,
                            "result": result, "observations": execution.observations(), "input_sha256": sha256(batch.tobytes()).hexdigest(),
                            "execution_model_sha256": provenance["execution_model_sha256"]}
                        validation_started = time.perf_counter()
                        try:
                            if values.shape != (count, 10) or values.dtype != np.dtype("float32") or not values.flags.owndata or np.shares_memory(values, batch):
                                raise ValueError("complete independently owned float32 host logits required")
                            np.testing.assert_allclose(values, expected[count], rtol=RTOL, atol=ATOL)
                        except BaseException as error:
                            row.update(validation_seconds=time.perf_counter() - validation_started, validation_passed=False,
                                failure={"type": type(error).__name__, "message": str(error)})
                            retain(row)
                            raise
                        row.update(validation_seconds=time.perf_counter() - validation_started, validation_passed=True,
                            max_absolute_logit_difference=float(np.abs(values - expected[count]).max()))
                        retain(row)
                        budget.remaining("request validation and retention")
                    snapshot("after_requests", condition, execution.outstanding)
                except BaseException as error:
                    failure = error
                    retain({"record": "condition_failure", **condition, "type": type(error).__name__, "message": str(error),
                        "seconds_since_collection_start": time.perf_counter() - budget.started,
                        "setup_attempt_seconds": time.perf_counter() - setup_started if execution is None else None})
                finally:
                    if execution is not None:
                        close_started = time.perf_counter()
                        try:
                            execution.close()
                            close_finished = time.perf_counter()
                            retired_results = execution.reconcile(1)
                            close_observations = execution.observations()
                            if execution.outstanding != 0:
                                raise RuntimeError("owner retirement left outstanding requests")
                            retain({"record": "close", **condition, "seconds": close_finished - close_started,
                                "post_close_reconciliation_seconds": time.perf_counter() - close_finished,
                                "owner_retired": True, "outstanding_requests": execution.outstanding,
                                "retired_results": retired_results, "observations": close_observations})
                            snapshot("after_close", condition, execution.outstanding)
                        except BaseException as error:
                            retain({"record": "close_failure", **condition, "seconds": time.perf_counter() - close_started,
                                "type": type(error).__name__, "message": str(error), "owner_retired": False})
                            if failure is None:
                                failure = error
                if failure is not None:
                    raise failure
                budget.remaining("session close")
    report["analysis"] = summarize(report)
    report["source_identities_after"] = source_identities()
    if report["source_identities_after"] != report["source_identities"]:
        raise RuntimeError("executed installed source changed during collection")
    budget.remaining("analysis and completion")
    report["elapsed_seconds"] = time.perf_counter() - budget.started
    report["status"] = "completed"
    report["completed_requests"] = sum(row["record"] == "request" and row["validation_passed"] for row in report["rows"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda-required"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    destination = Path(args.output)
    journal = destination.with_suffix(destination.suffix + ".jsonl")
    if destination.exists() or journal.exists():
        raise ValueError("preserve the previous output and journal; choose a new result identity")
    started = time.perf_counter()
    report = {"schema_version": 1, "status": "incomplete", "rows": []}
    with destination.open("x", encoding="utf-8") as output, journal.open("x", encoding="utf-8") as stream:
        def record(row):
            stream.write(json.dumps(serializable(row), sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        try:
            Budget(args.max_seconds, started).remaining("model loading")
            model = load_model(args.model)
            record({"record": "metadata", "source_model_sha256": model.sha256,
                "collector_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "provider_requested": args.provider,
                "max_seconds": args.max_seconds, "retention": "Every record is appended outside primary request timing. Partial rows survive incomplete collection."})
            collect(model, provider=args.provider, max_seconds=args.max_seconds, on_record=record, started=started, report=report)
        except BaseException as error:
            report["status"] = "incomplete"
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            report["elapsed_seconds"] = time.perf_counter() - started
            record({"record": "incomplete", **report["failure"], "elapsed_seconds": report["elapsed_seconds"]})
        else:
            record({"record": "complete", "completed_requests": report["completed_requests"], "elapsed_seconds": report["elapsed_seconds"]})
        output.write(json.dumps(serializable(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "provider_requested": args.provider,
        "elapsed_seconds": report.get("elapsed_seconds"), "failure": report.get("failure")}, allow_nan=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
