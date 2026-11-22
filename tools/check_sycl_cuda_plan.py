"""Check that the SYCL and CUDA comparison's built plan equals its committed protocol."""
import json
import sys
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / "docs" / "sycl-cuda-protocol.json"
FIELDS = ("square_sides", "workloads", "backends", "tile_shape", "samples_per_condition",
          "additional_unrecorded_warmups", "run_rows")


def differences(plan, protocol):
    """Name every field where the binary's plan and the protocol disagree."""
    found = [field for field in FIELDS if plan.get(field) != protocol.get(field)]
    expected = len(protocol["square_sides"]) * len(protocol["workloads"]) * len(protocol["backends"]) \
        * protocol["samples_per_condition"]
    if protocol.get("run_rows") != expected:
        found.append("run_rows arithmetic")
    if protocol.get("warm_samples") != protocol["samples_per_condition"] - 1:
        found.append("warm_samples arithmetic")
    return found


def main(argv):
    if len(argv) != 2:
        print("usage: check_sycl_cuda_plan.py PLAN_JSON", file=sys.stderr)
        return 2
    plan = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    found = differences(plan, protocol)
    if found:
        print("Plan differs from docs/sycl-cuda-protocol.json: " + ", ".join(found), file=sys.stderr)
        return 1
    print("SYCL and CUDA comparison plan matches its protocol: " + str(protocol["run_rows"]) + " run rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
