"""Validate pinned input samples against docs/pinned-protocol.json and summarize them.

This is written before any pinned run, so the checks and the summary are fixed
before the data exists. It reads the CSV that `hbr_cuda_pinned_bench --run`
prints and never runs anything on a device.
"""
import argparse
import csv
import functools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "docs" / "pinned-protocol.json").read_text(encoding="utf-8"))
SIDES = PROTOCOL["square_sides"]
WORKLOADS = PROTOCOL["workloads"]
MEMORIES = PROTOCOL["host_memory"]
SAMPLES = PROTOCOL["samples_per_condition"]
TILE = PROTOCOL["tile_shape"][0]
TIMINGS = ("host_ms", "upload_ms", "kernel_ms", "download_ms")
HEADER = ["phase", "side", "workload", "host_memory", "iteration", "order", *TIMINGS,
          "checksum", "upload_bytes", "download_bytes"]


def tiles(side):
    return (side // TILE + (side % TILE != 0)) ** 2


def transfer_bytes(workload, side):
    """Upload and download bytes of one call, as the benchmark computes them."""
    pixels = side * side
    if workload == "masked_reduce":
        return pixels * 9, 8
    if workload == "tile_histogram":
        return pixels, tiles(side) * 256 * 8
    return pixels * 8, pixels * 8


@functools.lru_cache(maxsize=None)
def histogram_checksum(side):
    """Sum of counts[i] * (i + 1) for the byte cycle i * 13, computed per pixel."""
    across = side // TILE + (side % TILE != 0)
    total = 0
    for r in range(side):
        for c in range(side):
            total += ((r // TILE) * across + c // TILE) * 256 + (r * side + c) * 13 % 256 + 1
    return float(total)


def validate(rows):
    """Check every row against the protocol and return the run rows by condition."""
    if not rows or list(rows[0]) != HEADER:
        raise ValueError("unexpected CSV header")
    expected = len(SIDES) * len(WORKLOADS) * len(MEMORIES) * SAMPLES
    if expected != PROTOCOL["run_rows"]:
        raise ValueError("protocol row count does not follow from its conditions")
    runs, lifecycle, positions = {}, {}, {}
    for position, row in enumerate(rows):
        if None in row or None in row.values():
            raise ValueError(f"row {position + 2} has the wrong number of fields")
        for field in TIMINGS:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field}: {row[field]}")
        if not math.isfinite(float(row["checksum"])):
            raise ValueError("nonfinite checksum")
        side = int(row["side"])
        if row["phase"] in ("setup", "retire"):
            if row["workload"] != "all" or row["host_memory"] != "pinned" or int(row["iteration"]) != -1:
                raise ValueError(f"malformed {row['phase']} row")
            if int(row["order"]) != -1 or int(row["download_bytes"]) != 0 or float(row["checksum"]) != 0:
                raise ValueError(f"malformed {row['phase']} row")
            if int(row["upload_bytes"]) != side * side * 10:
                raise ValueError("pinned byte count differs from three buffers")
            if any(float(row[f]) != 0 for f in TIMINGS[1:]):
                raise ValueError("lifecycle row carries device timing")
            key = (row["phase"], side)
            if key in lifecycle:
                raise ValueError(f"duplicate {row['phase']} row")
            lifecycle[key] = row
            positions[key] = position
            continue
        if row["phase"] != "run":
            raise ValueError(f"unknown phase {row['phase']}")
        runs.setdefault((side, row["workload"], row["host_memory"]), []).append(row)
        positions.setdefault(("first", side), position)
        positions[("last", side)] = position
    if set(lifecycle) != {(p, s) for p in ("setup", "retire") for s in SIDES}:
        raise ValueError("missing setup or retire row")
    if set(runs) != {(s, w, m) for s in SIDES for w in WORKLOADS for m in MEMORIES}:
        raise ValueError("unexpected or missing condition")
    # The benchmark prints each side as setup, its runs, retire, in the protocol's side order.
    sequence = [positions[(part, side)] for side in SIDES for part in ("setup", "first", "last", "retire")]
    if sequence != sorted(sequence):
        raise ValueError("rows are not in the order the benchmark prints them")
    for condition, side in enumerate(SIDES):
        for w_index, workload in enumerate(WORKLOADS):
            up, down = transfer_bytes(workload, side)
            reference = histogram_checksum(side) if workload == "tile_histogram" else None
            for m_index, memory in enumerate(MEMORIES):
                group = runs[(side, workload, memory)]
                if [int(r["iteration"]) for r in group] != list(range(SAMPLES)):
                    raise ValueError(f"iterations out of order or missing: {side} {workload} {memory}")
                for row in group:
                    order = (m_index * len(WORKLOADS) + w_index - int(row["iteration"]) - condition) % 6
                    if int(row["order"]) != order:
                        raise ValueError(f"order differs from the rotation: {side} {workload} {memory}")
                    if (int(row["upload_bytes"]), int(row["download_bytes"])) != (up, down):
                        raise ValueError(f"transfer bytes differ: {side} {workload}")
                    checksum = float(row["checksum"])
                    if reference is None:
                        reference = checksum
                    if checksum != reference:
                        raise ValueError(f"output differs across samples or memory kinds: {side} {workload}")
    return runs, lifecycle


def summarize(runs):
    """Warm sample statistics per condition. Iteration zero stays out, as the protocol says.

    Quartiles use the inclusive method, the same one the reuse report uses.
    """
    stats = {}
    for key, group in runs.items():
        warm = group[1:]
        entry = {"n": len(warm)}
        for field in TIMINGS:
            values = [float(r[field]) for r in warm]
            q = statistics.quantiles(values, n=4, method="inclusive")
            entry[field] = {"median": statistics.median(values), "p25": q[0], "p75": q[2],
                            "min": min(values), "max": max(values)}
        entry["first_host_ms"] = float(group[0]["host_ms"])
        stats[key] = entry
    return stats


def tables(stats, lifecycle):
    out = ["| Image | Workload | Upload bytes | Complete call, pageable | Complete call, pinned "
           "| Upload, pageable | Upload, pinned | Download, pageable | Download, pinned |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for side in SIDES:
        for workload in WORKLOADS:
            a, b = stats[(side, workload, "pageable")], stats[(side, workload, "pinned")]
            cells = [a["host_ms"], b["host_ms"], a["upload_ms"], b["upload_ms"], a["download_ms"], b["download_ms"]]
            out.append(f"| {side} × {side} | {workload} | {transfer_bytes(workload, side)[0]} | "
                       + " | ".join(f"{c['median']:.5f}" for c in cells) + " |")
    out += ["", "| Image | Workload | Memory | Warm samples | Median | Middle 50 percent | Full range "
            "| First call | Kernel median |", "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |"]
    for side in SIDES:
        for workload in WORKLOADS:
            for memory in MEMORIES:
                entry = stats[(side, workload, memory)]
                h = entry["host_ms"]
                out.append(f"| {side} × {side} | {workload} | {memory} | {entry['n']} | {h['median']:.5f} | "
                           f"{h['p25']:.5f} to {h['p75']:.5f} | {h['min']:.5f} to {h['max']:.5f} | "
                           f"{entry['first_host_ms']:.5f} | {entry['kernel_ms']['median']:.5f} |")
    out += ["", "| Image | Pinned bytes | Allocation | Release |", "| --- | ---: | ---: | ---: |"]
    for side in SIDES:
        setup, retire = lifecycle[("setup", side)], lifecycle[("retire", side)]
        out.append(f"| {side} × {side} | {setup['upload_bytes']} | {float(setup['host_ms']):.5f} | "
                   f"{float(retire['host_ms']):.5f} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="output of hbr_cuda_pinned_bench --run")
    args = parser.parse_args(argv)
    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    runs, lifecycle = validate(rows)
    sys.stdout.write(tables(summarize(runs), lifecycle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
