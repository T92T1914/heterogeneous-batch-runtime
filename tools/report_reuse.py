"""Validate the retained reuse experiment and generate its tables from exact samples."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["cpu_optimized_1", "cpu_optimized_2", "cuda_simple_global", "cuda_simple_shared",
            "cuda_reuse_global", "cuda_reuse_shared"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    receipt = json.loads((ROOT / "docs/evidence/reuse-execution.json").read_text())
    assert sha(ROOT / "docs/evidence/reuse.csv") == receipt["csv_sha256"], "sample bytes changed"
    assert sha(ROOT / "docs/reuse-protocol.json") == receipt["protocol_sha256"], "protocol changed"
    assert sha(ROOT / "docs/reuse-contract.md") == receipt["contract_sha256"], "contract changed"
    assert receipt["status"] == "passed" and receipt["exit_code"] == 0 and receipt["source_unchanged"]
    rows = list(csv.DictReader((ROOT / "docs/evidence/reuse.csv").open(newline="")))
    assert len(rows) == 924, "expected 900 calls and 24 setup/retirement observations"
    groups = defaultdict(list)
    for row in rows:
        for field in ("host_ms", "upload_ms", "kernel_ms", "download_ms"):
            value = float(row[field])
            assert math.isfinite(value) and value >= 0
        groups[(row["phase"], int(row["side"]), row["distribution"], row["backend"])].append(row)
    expected_groups = set()
    condition = 0
    for side in (32, 256, 1024):
        for distribution in ("uniform", "single_bin"):
            nt = (side + 63) // 64
            expected_checksum = 0
            for r in range(side):
                for c in range(side):
                    byte = ((r * side + c) * 13) % 256 if distribution == "uniform" else 7
                    expected_checksum += ((r // 64) * nt + c // 64) * 256 + byte + 1
            for backend_index, backend in enumerate(BACKENDS):
                key = ("run", side, distribution, backend)
                expected_groups.add(key)
                group = groups[key]
                assert [int(r["iteration"]) for r in group] == list(range(25))
                for row in group:
                    iteration = int(row["iteration"])
                    assert int(row["order"]) == (backend_index - iteration - condition) % 6
                    assert int(row["checksum"]) == expected_checksum
                    reused = backend_index >= 4
                    assert int(row["request_id"]) == ((1 if iteration == 0 else iteration + 3) if reused else 0)
                    assert int(row["device_bytes"]) == (side * side + nt * nt * 256 * 8 if reused else 0)
                    if backend_index < 2:
                        assert all(float(row[f]) == 0 for f in ("upload_ms", "kernel_ms", "download_ms"))
                if backend_index >= 4:
                    for phase in ("setup", "retire"):
                        key = (phase, side, distribution, backend)
                        expected_groups.add(key)
                        assert len(groups[key]) == 1
                        row = groups[key][0]
                        assert int(row["iteration"]) == -1 and int(row["request_id"]) == 0
                        assert int(row["device_bytes"]) == side * side + nt * nt * 256 * 8
            condition += 1
    assert set(groups) == expected_groups, "unexpected condition group"
    return receipt, groups


def outputs(receipt, groups):
    stats = {}
    csv_output = io.StringIO(newline="")
    writer = csv.writer(csv_output, lineterminator="\n")
    writer.writerow(["side", "distribution", "backend", "n", "median_ms", "p25_ms", "p75_ms", "min_ms", "max_ms",
                     "upload_median_ms", "kernel_median_ms", "download_median_ms"])
    for side in (32, 256, 1024):
        for distribution in ("uniform", "single_bin"):
            for backend in BACKENDS:
                rows = groups[("run", side, distribution, backend)][1:]
                values = [float(r["host_ms"]) for r in rows]
                q = statistics.quantiles(values, n=4, method="inclusive")
                data = [statistics.median(values), q[0], q[2], min(values), max(values)]
                data += [statistics.median(float(r[f]) for r in rows) for f in ("upload_ms", "kernel_ms", "download_ms")]
                stats[(side, distribution, backend)] = data
                writer.writerow([side, distribution, backend, len(values), *[f"{v:.9f}" for v in data]])
    table = ["| Image | Input | CPU, one thread | CPU, two threads | Simple global | Reused global | Simple shared | Reused shared |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for side in (32, 256, 1024):
        for dist in ("uniform", "single_bin"):
            values = [stats[(side, dist, BACKENDS[i])][0] for i in (0, 1, 2, 4, 3, 5)]
            table.append(f"| {side} × {side} | {dist} | " + " | ".join(f"{v:.5f}" for v in values) + " |")
    spread = ["| 1024 × 1024 input | Reused method | Median | Middle 50 percent | Full range | Kernel median |",
              "| --- | --- | ---: | --- | --- | ---: |"]
    for dist in ("uniform", "single_bin"):
        for backend in BACKENDS[4:]:
            m, p25, p75, lo, hi, upload, kernel, download = stats[(1024, dist, backend)]
            spread.append(f"| {dist} | {backend.removeprefix('cuda_reuse_')} | {m:.5f} | {p25:.5f} to {p75:.5f} | {lo:.5f} to {hi:.5f} | {kernel:.5f} |")
    setup = ["| Image | Input | Method | Setup | Retirement | Explicit device bytes |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for side in (32, 256, 1024):
        for dist in ("uniform", "single_bin"):
            for backend in BACKENDS[4:]:
                a = groups[("setup", side, dist, backend)][0]
                b = groups[("retire", side, dist, backend)][0]
                setup.append(f"| {side} × {side} | {dist} | {backend.removeprefix('cuda_reuse_')} | {float(a['host_ms']):.5f} | {float(b['host_ms']):.5f} | {a['device_bytes']} |")
    report = f"""# Reusing CUDA histogram resources

I added one optional fixed-shape histogram context and measured it against the existing synchronous CUDA call and optimized CPU paths. Reusing allocations, the stream and events reduced repeated-call medians in this run. Small images still favored the CPU. At 1024 by 1024, the reused shared-bin uniform case and reused global-bin concentrated case had lower medians than the faster tested CPU path. This is a bounded repeated-work result, not a general GPU ranking or reason to change the default interface.

The measured source is `{receipt['source']}`. The [numerical and ownership contract](reuse-contract.md) and [protocol](reuse-protocol.json) were committed at `{receipt['protocol_commit']}` before collection. The [execution receipt](evidence/reuse-execution.json) identifies the source tree, executable, inputs/protocol and exact raw output. The original [504-row study](results-2026-09-29.md) remains unchanged.

## Completed calls

All durations below are milliseconds. Every cell is the median of 24 warm samples. Each condition also retains its first invocation and two additional unrecorded warmup calls. The first invocation is not a cold-process measurement. Backend order rotates across each iteration. The [raw CSV](evidence/reuse.csv) contains 900 complete calls and 24 setup/retirement observations. Every output, including warmups, was compared bin-for-bin with an independently indexed oracle.

{chr(10).join(table)}

The host clock starts before dispatch and ends with an owned usable host result. Every backend includes validation and host output allocation. Both CUDA paths include the current input upload, the same selected kernel, output download and synchronization. The simple API also releases its CUDA resources inside the call. The context keeps those resources until retirement. Full output comparison, checksum calculation and host result destruction occur outside every timed call. This boundary differs from the original report, which included host result destruction. These new samples do not replace or relabel the old comparison.

One and two CPU threads are both shown. Thread creation costs matter, so selecting only the slower two-thread result would exaggerate the GPU comparison for smaller images. The two bin methods remain separate because concentrated input changes their behavior.

## Variability and setup

{chr(10).join(spread)}

The [complete generated summary](evidence/reuse-summary.csv) includes counts, inclusive quartiles, minima, maxima and separate transfer/kernel medians for all 36 conditions. Device intervals are not added to host elapsed time. Wide maxima show why one fast observation is not a stable latency guarantee.

{chr(10).join(setup)}

The first context setup includes lazy CUDA startup in this process. It is not the incremental cost of persistent storage alone, and the other backends later benefit from that initialized process. Setup is therefore shown separately, without a universal amortization or crossover claim. Every repeated call still moves host data both ways. There is no prepared device input, pinned host memory, transfer overlap or zero-copy claim. The two largest simultaneous contexts hold 3 MiB of explicit device arrays together, excluding driver bookkeeping. The API rejects an individual context above 64 MiB.

## Correctness, ownership and limits

Windows verification executed the native suite, existing CUDA numerical suite, new reuse suite, and separate fault executable on the RTX 4090. The reuse suite covers changed input, exact bins, empty/partial shapes, both kernels, retained and independently modified old results, request IDs, invalid-input recovery, wrong-thread rejection, close twice and run after close. An installed C++ consumer executed the new context.

Seven acquisition and eight execution checkpoints injected host exceptions. Cleanup synchronized successfully and released acquired resources, including the path after an output download had been submitted. A failed context then rejected reuse. The test-only injection code is absent from the installed library and benchmark. These are host-exception cleanup tests. They do not establish real CUDA allocation-failure or device-loss recovery.

The context belongs to its creating thread and CUDA device. Inputs remain immutable through return, and each result owns a distinct host vector. Request IDs are scoped to that context. Running cancellation cannot interrupt the operation, and logical staleness does not permit early buffer release. The simple API and CPU-only Python interface remain the defaults. Adaptive Timing needs no new behavior for a synchronous callable, and MCTS's previous no-port result is unchanged.

Actual WSL2 Linux Release and ASan/UBSan verification and an installed CPU consumer are recorded separately in the receipt. That is Linux CPU evidence, not Linux GPU execution or a separate physical machine. Hosted checks must be verified at the delivered revision. GPU hardware counters remain restricted by `ERR_NVGPUCTRPERM`, and Windows Compute Sanitizer's WDDM initialization remains unresolved. No GPU sanitizer pass is claimed.

This run used the Ryzen 7 7800X3D, RTX 4090, driver 617.14, MSVC 19.44 and CUDA 13.4.59 on a shared Windows workstation. GPU utilization was 18 percent before and 32 percent after collection. Ordinary applications stayed active. Only one task benchmark ran, with at most two CPU threads and a 120 second cap. No clocks, power, drivers, display or security settings changed. A single short shared-host batch cannot establish statistical independence, general speedup or production latency.

## Reproduce

Build with `-DHBR_CUDA=ON` for the actual GPU architecture, run CTest, then run `hbr_cuda_reuse_bench --run` once under the committed protocol. Keep setup and retirement rows and retain slower cases. `python tools/report_reuse.py --check` checks artifact identities, sample groups, run order, independent checksums, request IDs and generated tables. `--write` regenerates the report and summary from the retained CSV without rerunning a workload.

The optional context is useful when a caller already has repeated fixed-shape work and can retain its resources. The original simple interface remains the easier default. The measurements support exposing this choice and its costs, not changing every workload or adding an asynchronous framework.
"""
    return report, csv_output.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    receipt, groups = load()
    report, summary = outputs(receipt, groups)
    for path, expected in [(ROOT / "docs/reuse-results.md", report),
                           (ROOT / "docs/evidence/reuse-summary.csv", summary)]:
        if args.write:
            path.write_text(expected, encoding="utf-8", newline="\n")
        else:
            assert path.read_text(encoding="utf-8") == expected, f"stale generated output: {path.name}"
    print("900 complete-call samples, 24 setup/retirement rows, identities and generated tables passed")


if __name__ == "__main__":
    main()
