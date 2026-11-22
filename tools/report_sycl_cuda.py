"""Validate SYCL and CUDA comparison samples against docs/sycl-cuda-protocol.json and summarize them.

This is written before any comparison run, so the checks and the summary are fixed
before the data exists. It reads the CSV that `hbr_sycl_cuda_bench --run` prints and
never runs anything on a device.
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
PROTOCOL = json.loads((ROOT / "docs" / "sycl-cuda-protocol.json").read_text(encoding="utf-8"))
SIDES = PROTOCOL["square_sides"]
WORKLOADS = PROTOCOL["workloads"]
BACKENDS = PROTOCOL["backends"]
SAMPLES = PROTOCOL["samples_per_condition"]
TILE = PROTOCOL["tile_shape"][0]
COMBINATIONS = len(WORKLOADS) * len(BACKENDS)
DEVICE = ("upload_ms", "kernel_ms", "download_ms")
HEADER = ["side", "workload", "backend", "iteration", "order", "host_ms", *DEVICE,
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


def expected_sequence():
    """(side, workload, backend, iteration, order) for every run row, in the order the benchmark prints."""
    sequence = []
    for condition, side in enumerate(SIDES):
        for iteration in range(SAMPLES):
            for order in range(COMBINATIONS):
                index = (order + iteration + condition) % COMBINATIONS
                sequence.append((side, WORKLOADS[index % len(WORKLOADS)], BACKENDS[index // len(WORKLOADS)],
                                 iteration, order))
    return sequence


def nonnegative(row, field):
    value = float(row[field])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {field}: {row[field]}")
    return value


def validate(rows):
    """Check every row against the protocol and return the rows by condition.

    The reduction and stencil outputs are checked here only for staying the same within a
    backend. The benchmark checks each one against the scalar reference and exits nonzero
    on a mismatch, and the runner never passes a run that exited nonzero.
    """
    if not rows:
        raise ValueError("no run rows")
    if list(rows[0]) != HEADER:
        raise ValueError("unexpected CSV header")
    sequence = expected_sequence()
    if len(sequence) != PROTOCOL["run_rows"]:
        raise ValueError("protocol row count does not follow from its conditions")
    if len(rows) != len(sequence):
        raise ValueError(f"expected {len(sequence)} run rows, found {len(rows)}")
    runs, outputs = {}, {}
    for position, (row, expected) in enumerate(zip(rows, sequence)):
        if None in row or None in row.values():
            raise ValueError(f"row {position + 2} has the wrong number of fields")
        side, workload, backend, iteration, order = expected
        found = (int(row["side"]), row["workload"], row["backend"], int(row["iteration"]), int(row["order"]))
        if found != expected:
            raise ValueError(f"row {position + 2} is not in the rotation the protocol fixes: {found}")
        nonnegative(row, "host_ms")
        if backend == "cuda":
            for field in DEVICE:
                nonnegative(row, field)
        elif any(row[field] != "" for field in DEVICE):
            raise ValueError(f"a SYCL row carries device timing: row {position + 2}")
        if (int(row["upload_bytes"]), int(row["download_bytes"])) != transfer_bytes(workload, side):
            raise ValueError(f"transfer bytes differ: {side} {workload}")
        checksum = float(row["checksum"])
        if not math.isfinite(checksum):
            raise ValueError("nonfinite checksum")
        if workload == "tile_histogram" and checksum != histogram_checksum(side):
            raise ValueError(f"histogram differs from the expected counts: {side} {backend}")
        # The benchmark checks every output bit for bit against its backend's first one.
        first = outputs.setdefault((side, workload, backend), checksum)
        if checksum != first:
            raise ValueError(f"output differs across samples: {side} {workload} {backend}")
        runs.setdefault((side, workload, backend), []).append(row)
    return runs


def spread(values):
    """Median, quartiles by the inclusive method, as the reuse and pinned reports use, and range."""
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "p25": q[0], "p75": q[2], "min": min(values), "max": max(values)}


def summarize(runs):
    """Warm sample statistics per condition. Iteration zero stays out, as the protocol says."""
    stats = {}
    for key, group in runs.items():
        warm = group[1:]
        entry = {"n": len(warm), "host_ms": spread([float(r["host_ms"]) for r in warm]),
                 "first_host_ms": float(group[0]["host_ms"]), "checksum": float(group[0]["checksum"])}
        if key[2] == "cuda":
            for field in DEVICE:
                entry[field] = spread([float(r[field]) for r in warm])
        stats[key] = entry
    return stats


def tables(stats):
    out = ["| Image | Workload | CUDA median | SYCL median | SYCL ÷ CUDA | CUDA middle 50 percent "
           "| SYCL middle 50 percent | Same checksum |",
           "| --- | --- | ---: | ---: | ---: | --- | --- | --- |"]
    for side in SIDES:
        for workload in WORKLOADS:
            cuda, sycl = stats[(side, workload, "cuda")], stats[(side, workload, "sycl")]
            c, s = cuda["host_ms"], sycl["host_ms"]
            out.append(f"| {side} × {side} | {workload} | {c['median']:.5f} | {s['median']:.5f} | "
                       f"{s['median'] / c['median']:.2f} | {c['p25']:.5f} to {c['p75']:.5f} | "
                       f"{s['p25']:.5f} to {s['p75']:.5f} | {'yes' if cuda['checksum'] == sycl['checksum'] else 'no'} |")
    out += ["", "| Image | Workload | Backend | Warm samples | Median | Full range | Iteration zero "
            "| CUDA upload | CUDA kernel | CUDA download |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for side in SIDES:
        for workload in WORKLOADS:
            for backend in BACKENDS:
                entry = stats[(side, workload, backend)]
                h = entry["host_ms"]
                device = " | ".join(f"{entry[f]['median']:.5f}" if f in entry else "none" for f in DEVICE)
                out.append(f"| {side} × {side} | {workload} | {backend} | {entry['n']} | {h['median']:.5f} | "
                           f"{h['min']:.5f} to {h['max']:.5f} | {entry['first_host_ms']:.5f} | {device} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="output of hbr_sycl_cuda_bench --run")
    args = parser.parse_args(argv)
    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    sys.stdout.write(tables(summarize(validate(rows))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
