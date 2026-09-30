"""Actual completed predictions and explicit execution provenance."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from adaptive_timing.runtime import Dispatch
from .executor import InferenceExecutor
from .model import example_images, load_model, prepare_arrays, probabilities, read_images

def summarize_profile(path):
    rows = json.loads(Path(path).read_text(encoding="utf8"))
    counts = {}
    nodes = []
    for row in rows:
        args = row.get("args", {})
        if row.get("cat") == "Node" and "provider" in args:
            provider = args["provider"]
            counts[provider] = counts.get(provider, 0) + 1
            nodes.append({"name": row.get("name"), "provider": provider, "op": args.get("op_name")})
    return {"provider_events": counts, "node_events": nodes,
            "raw_sha256": sha256(Path(path).read_bytes()).hexdigest(),
            "meaning": "ORT operation placement from this run; event counts are not hardware counters or elapsed inference time"}

def main():
    parser = argparse.ArgumentParser(description="Bounded digit classification using a reviewed ONNX model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda-required"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-prefix", required=True)
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()
    model = load_model(args.model)
    inputs = read_images(args.images) if args.images else prepare_arrays(example_images()[:8])
    input_digest = sha256(inputs.tobytes()).hexdigest()
    execution = InferenceExecutor(model, provider=args.provider, profile_prefix=args.profile_prefix)
    try:
        execution.submit_batch(Dispatch("image-batch-1", "inference", 0, 1, "dispatched"), 1, inputs)
        result, = execution.wait(1, timeout=30)
        if not result.usable:
            raise RuntimeError("inference did not produce a usable completed result")
        transitions = [asdict(row) for row in execution.observations()]
    finally:
        execution.close()
    logits = result.value
    report = {"schema_version": 1, "model_sha256": model.sha256, "input_sha256": input_digest,
              "provider_requested": args.provider, "batch_shape": list(inputs.shape),
              "input_kind": "user supplied bounded PNGs" if args.images else "original geometric example digits, not an accuracy dataset",
              "contract": "serial one-image calls, completed host outputs, no asynchronous GPU interface",
              "logits": logits.tolist(), "probabilities": probabilities(logits).tolist(),
              "predictions": np.argmax(logits, axis=1).tolist(), "status": result.status,
              "usable": result.usable, "observations": transitions,
              "placement": summarize_profile(execution.profile_path)}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"predictions": report["predictions"], "status": result.status, "provider_events": report["placement"]["provider_events"]}))
