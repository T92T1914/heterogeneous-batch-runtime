"""Check that the pinned benchmark's built plan equals its committed protocol."""
import json
import sys
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parents[1] / "docs" / "pinned-protocol.json"
FIELDS = ("square_sides", "workloads", "host_memory", "tile_shape", "samples_per_condition",
          "additional_unrecorded_warmups", "run_rows")


def differences(plan, protocol):
    """Name every field where the binary's plan and the protocol disagree."""
    found = [field for field in FIELDS if plan.get(field) != protocol.get(field)]
    expected = len(protocol["square_sides"]) * len(protocol["workloads"]) * len(protocol["host_memory"]) \
        * protocol["samples_per_condition"]
    if protocol.get("run_rows") != expected:
        found.append("run_rows arithmetic")
    return found


def main(argv):
    if len(argv) != 2:
        print("usage: check_pinned_plan.py PLAN_JSON", file=sys.stderr)
        return 2
    plan = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    found = differences(plan, protocol)
    if found:
        print("Plan differs from docs/pinned-protocol.json: " + ", ".join(found), file=sys.stderr)
        return 1
    print("Pinned benchmark plan matches its protocol: " + str(protocol["run_rows"]) + " run rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
