"""Check that the SYCL and CUDA comparison's built plan and run order equal its committed protocol."""
import csv
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PROTOCOL = TOOLS.parent / "docs" / "sycl-cuda-protocol.json"
sys.path.insert(0, str(TOOLS))
import report_sycl_cuda  # noqa: E402

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
    if len(argv) not in (2, 3):
        print("usage: check_sycl_cuda_plan.py PLAN_JSON [SEQUENCE_CSV]", file=sys.stderr)
        return 2
    plan = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    found = differences(plan, protocol)
    if found:
        print("Plan differs from docs/sycl-cuda-protocol.json: " + ", ".join(found), file=sys.stderr)
        return 1
    print("SYCL and CUDA comparison plan matches its protocol: " + str(protocol["run_rows"]) + " run rows")
    if len(argv) == 3:
        with Path(argv[2]).open(newline="", encoding="utf-8") as stream:
            problem = report_sycl_cuda.sequence_differences(list(csv.DictReader(stream)))
        if problem:
            print("Run order differs from tools/report_sycl_cuda.py: " + problem, file=sys.stderr)
            return 1
        print("Run order matches the one the report expects")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
