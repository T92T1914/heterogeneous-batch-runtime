"""Validate pinned input samples against docs/pinned-protocol.json and summarize them.

This is written before any pinned run, so the checks and the summary are fixed
before the data exists. It reads the CSV that `hbr_cuda_pinned_bench --run`
prints, and the receipt that `tools/run_pinned.py` writes when it is given, and
never runs anything on a device.
"""
import argparse
import csv
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "docs" / "pinned-protocol.json").read_text(encoding="utf-8"))
SIDES = PROTOCOL["square_sides"]
WORKLOADS = PROTOCOL["workloads"]
MEMORIES = PROTOCOL["host_memory"]
SAMPLES = PROTOCOL["samples_per_condition"]
TILE = PROTOCOL["tile_shape"][0]
COMBINATIONS = len(WORKLOADS) * len(MEMORIES)
TIMINGS = ("host_ms", "upload_ms", "kernel_ms", "download_ms")
LOAD = ("gpu_util_percent", "gpu_memory_mib")
HEADER = ["phase", "side", "workload", "host_memory", "iteration", "order", *TIMINGS,
          "checksum", "upload_bytes", "download_bytes", *LOAD]
SEQUENCE_HEADER = ["side", "workload", "host_memory", "iteration", "order"]
# What the benchmark prints to stderr for each idle reading, which the runner keeps.
IDLE = re.compile(r"GPU idle (before|after) (\d+): utilization (\d+) percent, memory (\d+) MiB")
# The idle readings a complete run prints, in order: before each side and after the last.
IDLE_POINTS = [f"before {side}" for side in SIDES] + [f"after {SIDES[-1]}"]
# The link readings the runner keeps under a protocol with a pcie_link entry, one per line of
# nvidia-smi in the order of the protocol's fields after the timestamp.
LINK_KEYS = ("gen_current", "gen_max", "gen_gpumax", "gen_hostmax", "width_current", "width_max")


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


def run_sequence():
    """(side, workload, memory, iteration, order) for every run row, in the order the benchmark prints.

    Each iteration takes the workloads in turn, rotated by iteration + condition, and runs
    each with pageable and pinned input back to back, pageable first in even iterations.
    """
    sequence = []
    for condition, side in enumerate(SIDES):
        for iteration in range(SAMPLES):
            for order in range(COMBINATIONS):
                workload = WORKLOADS[(iteration + condition + order // 2) % len(WORKLOADS)]
                memory = MEMORIES[(order + iteration) % len(MEMORIES)]
                sequence.append((side, workload, memory, iteration, order))
    return sequence


def expected_rows():
    """(phase, side, workload, memory, iteration, order) for every row the benchmark prints.

    Each side is a setup row, its run rows and a retire row, in the protocol's side order.
    """
    runs = run_sequence()
    per_side = len(runs) // len(SIDES)
    rows = []
    for index, side in enumerate(SIDES):
        rows.append(("setup", side, "all", "pinned", -1, -1))
        rows += [("run", *run) for run in runs[index * per_side:(index + 1) * per_side]]
        rows.append(("retire", side, "all", "pinned", -1, -1))
    return rows


def sequence_differences(rows):
    """Compare the rows of `hbr_cuda_pinned_bench --sequence` with run_sequence(). Returns a message or None."""
    if not rows or list(rows[0]) != SEQUENCE_HEADER:
        return "unexpected sequence header"
    found = [(int(r["side"]), r["workload"], r["host_memory"], int(r["iteration"]), int(r["order"])) for r in rows]
    expected = run_sequence()
    if len(found) != len(expected):
        return f"expected {len(expected)} sequence rows, found {len(found)}"
    for position, (a, b) in enumerate(zip(found, expected)):
        if a != b:
            return f"sequence row {position + 2} is {a}, the report expects {b}"
    return None


def validate(rows):
    """Check every row against the protocol and return the run rows by condition and the lifecycle rows."""
    if not rows:
        raise ValueError("no rows")
    if list(rows[0]) != HEADER:
        raise ValueError("unexpected CSV header")
    expected = expected_rows()
    if len(expected) - 2 * len(SIDES) != PROTOCOL["run_rows"]:
        raise ValueError("protocol row count does not follow from its conditions")
    if len(rows) != len(expected):
        raise ValueError(f"expected {len(expected)} rows, found {len(rows)}")
    runs, lifecycle, outputs = {}, {}, {}
    for position, (row, want) in enumerate(zip(rows, expected)):
        if None in row or None in row.values():
            raise ValueError(f"row {position + 2} has the wrong number of fields")
        found = (row["phase"], int(row["side"]), row["workload"], row["host_memory"],
                 int(row["iteration"]), int(row["order"]))
        if found != want:
            raise ValueError(f"row {position + 2} is not in the order the protocol fixes: {found}")
        for field in TIMINGS:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field}: {row[field]}")
        if not (row["gpu_util_percent"].isdigit() and int(row["gpu_util_percent"]) <= 100):
            raise ValueError(f"invalid gpu_util_percent: row {position + 2}")
        if not row["gpu_memory_mib"].isdigit():
            raise ValueError(f"invalid gpu_memory_mib: row {position + 2}")
        checksum = float(row["checksum"])
        if not math.isfinite(checksum):
            raise ValueError("nonfinite checksum")
        phase, side, workload, memory = want[:4]
        if phase == "run" and float(row["host_ms"]) == 0:
            raise ValueError(f"a complete call of zero time: row {position + 2}")
        if phase != "run":
            if int(row["download_bytes"]) != 0 or checksum != 0:
                raise ValueError(f"malformed {phase} row")
            if int(row["upload_bytes"]) != side * side * 10:
                raise ValueError("pinned byte count differs from three buffers")
            if any(float(row[f]) != 0 for f in TIMINGS[1:]):
                raise ValueError("lifecycle row carries device timing")
            lifecycle[(phase, side)] = row
            continue
        if (int(row["upload_bytes"]), int(row["download_bytes"])) != transfer_bytes(workload, side):
            raise ValueError(f"transfer bytes differ: {side} {workload}")
        if workload == "tile_histogram" and checksum != histogram_checksum(side):
            raise ValueError(f"output differs from the expected histogram: {side} {memory}")
        # The benchmark checks every output bit for bit against the first of its workload and side.
        if checksum != outputs.setdefault((side, workload), checksum):
            raise ValueError(f"output differs across samples or memory kinds: {side} {workload}")
        runs.setdefault((side, workload, memory), []).append(row)
    return runs, lifecycle


def spread(values):
    """Median, quartiles by the inclusive method, as the reuse report uses, and range."""
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "p25": q[0], "p75": q[2], "min": min(values), "max": max(values)}


def summarize(runs):
    """Warm sample statistics per condition. Iteration zero stays out, as the protocol says."""
    stats = {}
    for key, group in runs.items():
        warm = group[1:]
        entry = {"n": len(warm)}
        for field in TIMINGS:
            entry[field] = spread([float(r[field]) for r in warm])
        entry["first_host_ms"] = float(group[0]["host_ms"])
        stats[key] = entry
    return stats


def paired(runs):
    """Pinned divided by pageable within each warm pair, by side and workload.

    Rows of a condition come in iteration order, so the pageable and pinned rows at one
    index share side, workload and iteration. A pair is pageable first when its pageable
    row comes earlier in the run, which holds in even iterations.
    """
    result = {}
    for side in SIDES:
        for workload in WORKLOADS:
            pageable, pinned = runs[(side, workload, "pageable")], runs[(side, workload, "pinned")]
            ratios = {"all": [], "pageable_first": [], "pinned_first": []}
            for a, b in zip(pageable[1:], pinned[1:]):
                if a["iteration"] != b["iteration"]:
                    raise ValueError(f"unpaired rows: {side} {workload} iteration {a['iteration']}")
                ratio = float(b["host_ms"]) / float(a["host_ms"])
                ratios["all"].append(ratio)
                ratios["pageable_first" if int(a["order"]) < int(b["order"]) else "pinned_first"].append(ratio)
            result[(side, workload)] = {"n": len(ratios["all"]), "all": spread(ratios["all"]),
                                        "pageable_first": statistics.median(ratios["pageable_first"]),
                                        "pinned_first": statistics.median(ratios["pinned_first"])}
    return result


def load(runs):
    """Median and range of the reading taken before each recorded call, by side, over every run row."""
    result = {}
    for side in SIDES:
        rows = [r for (s, _, _), group in runs.items() if s == side for r in group]
        result[side] = {field: spread([int(r[field]) for r in rows]) for field in LOAD}
        result[side]["n"] = len(rows)
    return result


def idle_readings(receipt):
    """The idle readings the runner kept from the benchmark's stderr, in run order."""
    return [{"point": f"{m.group(1)} {m.group(2)}", "gpu_util_percent": int(m.group(3)),
             "gpu_memory_mib": int(m.group(4))}
            for line in receipt.get("stderr", []) for m in [IDLE.fullmatch(line)] if m]


def link_reading(line):
    """One line of nvidia-smi with the protocol's link fields, as a dict. A line that is not one raises ValueError."""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 1 + len(LINK_KEYS) or not all(part.isdigit() for part in parts[1:]):
        raise ValueError(f"not a link reading: {line!r}")
    return {"timestamp": parts[0], **{key: int(value) for key, value in zip(LINK_KEYS, parts[1:])}}


def link_problem(readings, generation, width):
    """Why the link readings do not qualify a run at this generation and width, or None when they do.

    An idle card drops its link to save power, so a current generation below the protocol's is
    allowed. Every reading must give the protocol's maximum, at least one must show the link
    running at the protocol's generation and width, and none may run above it.
    """
    if not readings:
        return "no link reading was taken while the benchmark ran"
    for reading in readings:
        if (reading["gen_max"], reading["width_max"]) != (generation, width):
            return (f"the link maximum read generation {reading['gen_max']} with {reading['width_max']} lanes "
                    f"at {reading['timestamp']}, the protocol names generation {generation} with {width}")
        if reading["gen_current"] > generation:
            return f"the link ran at generation {reading['gen_current']} at {reading['timestamp']}"
    if not any((r["gen_current"], r["width_current"]) == (generation, width) for r in readings):
        return f"no reading showed the link running at generation {generation} with {width} lanes"
    return None


def checked_link_readings(receipt):
    """The receipt's link readings and the protocol's link, after link_problem finds nothing wrong."""
    link = receipt["pcie_link"]
    problem = link_problem(receipt["pcie_link_readings"], link["generation"], link["width"])
    if problem:
        raise ValueError(problem)
    return receipt["pcie_link_readings"], link


def link_table(readings, link):
    counts = {}
    for r in readings:
        key = (r["gen_current"], r["width_current"])
        counts[key] = counts.get(key, 0) + 1
    out = [f"The protocol names PCIe generation {link['generation']} with {link['width']} lanes. "
           f"Every one of the {len(readings)} link readings gave that as the maximum.", "",
           "| Current link | Readings while the benchmark ran |", "| --- | ---: |"]
    for (generation, width), n in sorted(counts.items(), reverse=True):
        out.append(f"| generation {generation}, {width} lanes | {n} |")
    return "\n".join(out) + "\n"


def upload_rate(upload_bytes, upload_ms):
    """Bytes per millisecond divided by a million is gigabytes per second."""
    return f"{upload_bytes / upload_ms / 1e6:.2f} GB/s" if upload_ms > 0 else "not measured"


def compare(base_rows, rows, base_label, label):
    """Median complete call and upload of each condition in two runs under the same plan, beside each other."""
    base, other = summarize(validate(base_rows)[0]), summarize(validate(rows)[0])
    out = [f"| Image | Workload | Memory | Complete call, {base_label} | Complete call, {label} | Ratio "
           f"| Upload, {base_label} | Upload, {label} | Upload rate, {base_label} | Upload rate, {label} |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for side in SIDES:
        for workload in WORKLOADS:
            for memory in MEMORIES:
                a, b = base[(side, workload, memory)], other[(side, workload, memory)]
                ha, hb = a["host_ms"]["median"], b["host_ms"]["median"]
                ua, ub = a["upload_ms"]["median"], b["upload_ms"]["median"]
                up = transfer_bytes(workload, side)[0]
                out.append(f"| {side} × {side} | {workload} | {memory} | {ha:.5f} | {hb:.5f} | {hb / ha:.3f} | "
                           f"{ua:.5f} | {ub:.5f} | {upload_rate(up, ua)} | {upload_rate(up, ub)} |")
    return "\n".join(out) + "\n"


def tables(stats, lifecycle, pairs=None, loads=None, idle=None):
    out = ["| Image | Workload | Upload bytes | Complete call, pageable | Complete call, pinned "
           "| Upload, pageable | Upload, pinned | Download, pageable | Download, pinned |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for side in SIDES:
        for workload in WORKLOADS:
            a, b = stats[(side, workload, "pageable")], stats[(side, workload, "pinned")]
            cells = [a["host_ms"], b["host_ms"], a["upload_ms"], b["upload_ms"], a["download_ms"], b["download_ms"]]
            out.append(f"| {side} × {side} | {workload} | {transfer_bytes(workload, side)[0]} | "
                       + " | ".join(f"{c['median']:.5f}" for c in cells) + " |")
    if pairs is not None:
        out += ["", "| Image | Workload | Pairs | Median of pinned ÷ pageable in a pair | Middle 50 percent "
                "| Full range | Pageable first | Pinned first |",
                "| --- | --- | ---: | ---: | --- | --- | ---: | ---: |"]
        for side in SIDES:
            for workload in WORKLOADS:
                p = pairs[(side, workload)]
                a = p["all"]
                out.append(f"| {side} × {side} | {workload} | {p['n']} | {a['median']:.3f} | "
                           f"{a['p25']:.3f} to {a['p75']:.3f} | {a['min']:.3f} to {a['max']:.3f} | "
                           f"{p['pageable_first']:.3f} | {p['pinned_first']:.3f} |")
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
    if loads is not None:
        out += ["", "| Image | Readings | GPU utilization median | Utilization range | Memory in use median "
                "| Memory range |",
                "| --- | ---: | ---: | --- | ---: | --- |"]
        for side in SIDES:
            u, m = loads[side]["gpu_util_percent"], loads[side]["gpu_memory_mib"]
            out.append(f"| {side} × {side} | {loads[side]['n']} | {u['median']:g} percent | "
                       f"{u['min']:g} to {u['max']:g} percent | {m['median']:g} MiB | {m['min']:g} to {m['max']:g} MiB |")
    if idle is not None:
        out += ["", "| Idle reading | GPU utilization | Memory in use |", "| --- | ---: | ---: |"]
        out += [f"| {r['point']} | {r['gpu_util_percent']} percent | {r['gpu_memory_mib']} MiB |" for r in idle]
    return "\n".join(out) + "\n"


def checked_idle_readings(receipt):
    """The receipt's idle readings, only from a passed run that kept all four in order."""
    if receipt.get("status") != "passed":
        raise ValueError(f"the receipt's status is {receipt.get('status')!r}, not 'passed'")
    readings = idle_readings(receipt)
    points = [r["point"] for r in readings]
    if points != IDLE_POINTS:
        raise ValueError(f"idle readings {points}, expected {IDLE_POINTS}")
    return readings


def report(rows, receipt=None):
    runs, lifecycle = validate(rows)
    out = tables(summarize(runs), lifecycle, paired(runs), load(runs),
                 None if receipt is None else checked_idle_readings(receipt))
    # Only a run under a protocol that names the link has readings. The run of 2026-11-25 has none,
    # and its report stays as published.
    if receipt is not None and "pcie_link" in receipt:
        out += "\n" + link_table(*checked_link_readings(receipt))
    return out


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="output of hbr_cuda_pinned_bench --run")
    parser.add_argument("--receipt", type=Path, help="the run's receipt, for the idle readings")
    parser.add_argument("--compare", type=Path, metavar="BASE_CSV",
                        help="an earlier run under the same plan, set beside this one condition by condition")
    parser.add_argument("--labels", nargs=2, default=["generation 3", "generation 4"], metavar=("BASE", "THIS"),
                        help="the names of the earlier run and this one in the comparison table")
    args = parser.parse_args(argv)
    rows = read_rows(args.csv)
    receipt = None
    if args.receipt:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        # The receipt must describe these samples and not another attempt's.
        if receipt.get("csv_sha256") != hashlib.sha256(args.csv.read_bytes()).hexdigest():
            raise SystemExit(f"{args.receipt} does not name {args.csv} by its SHA-256.")
    sys.stdout.write(report(rows, receipt))
    if args.compare:
        sys.stdout.write("\n" + compare(read_rows(args.compare), rows, *args.labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
