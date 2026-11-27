"""Run the September CUDA benchmark once at PCIe generation 4 and write its samples and execution receipt.

docs/cuda-gen4-protocol.json fixes the run. The runner reads the card's link,
its utilization and its memory in use with nvidia-smi for as long as the
benchmark runs, keeps every reading, and passes the run only at the link the
protocol names. It refuses to run unless the forecast that
tools/cost_model.py --forecast wrote is committed in the source it runs from.

The receipt names the source, the protocol, the forecast commit, the binary and
the compilers. It records no process list, because the measuring computer is a
shared desktop. The person who starts the run confirms with
--confirm-desktop-load, after checking the processes privately, that no game is
running.

A run that fails, times out or breaks the protocol is still written, with its
status and its error text. The runner refuses to write over an earlier attempt.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import report_pinned  # noqa: E402
import run_pinned  # noqa: E402

PROTOCOL = ROOT / "docs" / "cuda-gen4-protocol.json"
FORECAST = "docs/evidence/cost-model-forecast.json"
HEADER = ["workload", "side", "distribution", "backend", "iteration", "upload_ms", "kernel_ms", "download_ms",
          "host_total_ms", "checksum"]
LOAD_KEYS = ("utilization_percent", "memory_used_mib")


def conditions():
    """The (workload, side, distribution, backend) groups hbr_cuda_bench --run prints, in its order."""
    found = []
    for side in (32, 256, 1024):
        for distribution in ("uniform", "single_bin"):
            for workload in ("masked_reduce", "tile_histogram", "stencil3x3"):
                if distribution == "single_bin" and workload != "tile_histogram":
                    continue
                for backend in ("cpu_optimized_2", "cuda_baseline", "cuda_shared"):
                    if backend == "cuda_shared" and workload != "tile_histogram":
                        continue
                    found.append((workload, side, distribution, backend))
    return found


def validate(rows, protocol):
    """Raise ValueError unless the rows are the protocol's complete, finite and consistent output."""
    if not rows or list(rows[0]) != HEADER:
        raise ValueError(f"the columns are not {','.join(HEADER)}")
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("a row has more or fewer fields than the header")
    if len(rows) != protocol["run_rows"]:
        raise ValueError(f"{len(rows)} rows, the protocol has {protocol['run_rows']}")
    expected = [(*c, i) for c in conditions() for i in range(protocol["iterations"])]
    actual = [(r["workload"], int(r["side"]), r["distribution"], r["backend"], int(r["iteration"])) for r in rows]
    if actual != expected:
        raise ValueError("the rows are not the protocol's conditions and iterations in the benchmark's order")
    references = {}
    for r in rows:
        for field in ("upload_ms", "kernel_ms", "download_ms", "host_total_ms"):
            value = float(r[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field} {r[field]!r} for {r['workload']} at {r['side']}")
        checksum = float(r["checksum"])
        if not math.isfinite(checksum):
            raise ValueError(f"nonfinite checksum for {r['workload']} at {r['side']}")
        key = (r["workload"], r["side"], r["distribution"])
        reference = references.setdefault(key, checksum)
        if abs(checksum - reference) > (1e-8 if r["workload"] == "masked_reduce" else 1e-12):
            raise ValueError(f"checksums differ across backends for {key}")


def reading(line, fields):
    """One nvidia-smi line as the link reading report_pinned.link_reading makes, plus the load fields.

    The link fields must all be read. A load field nvidia-smi cannot read, such as [N/A], is kept
    as None, so a line that shows the link is never dropped for want of a load reading.
    """
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(fields):
        raise ValueError(f"not a link and load reading: {line!r}")
    link_part = len(parts) - len(LOAD_KEYS)
    found = report_pinned.link_reading(",".join(parts[:link_part]))
    found.update({key: int(value) if value.isdigit() else None
                  for key, value in zip(LOAD_KEYS, parts[link_part:])})
    return found


def is_reading(line, fields):
    try:
        reading(line, fields)
        return True
    except ValueError:
        return False


class Watch:
    """nvidia-smi at the protocol's interval, read on a thread that notes when each line arrived.

    mark() gives the number of lines received so far. The runner marks right after the benchmark's
    process exists and right after it exits, and a line's position against those marks, not a clock,
    decides whether it arrived while the benchmark ran. Arrival times are kept for reading only.
    """

    def __init__(self, link, clock=time.perf_counter):
        self.command = run_pinned.LinkWatch(link).command
        self.fields = link["fields"]
        self.clock = clock
        self.received = []

    def start(self):
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for raw in self.process.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self.received.append((self.clock(), line))

    def mark(self):
        return len(self.received)

    def readings(self):
        """How many lines so far are link readings."""
        return sum(is_reading(line, self.fields) for _, line in list(self.received))

    def wait_for(self, count, timeout=10):
        """Wait until count lines are link readings. Other lines do not count."""
        deadline = time.monotonic() + timeout
        while self.readings() < count:
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"nvidia-smi gave fewer than {count} link readings")
            time.sleep(0.005)

    def stop(self):
        """Stop nvidia-smi and return every (arrival time, line) it printed, killing it if it does not stop."""
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=30)
        self.reader.join(timeout=30)
        return list(self.received)


def forecast_identity(source):
    """The commit that last wrote the forecast and its SHA-256, both as of source, or None when source lacks it."""
    if subprocess.run(["git", "cat-file", "-e", f"{source}:{FORECAST}"], cwd=ROOT, capture_output=True).returncode:
        return None
    data = subprocess.run(["git", "show", f"{source}:{FORECAST}"], cwd=ROOT, check=True, capture_output=True).stdout
    return {"forecast_commit": run_pinned.git("log", "-1", "--format=%H", source, "--", FORECAST),
            "forecast_sha256": run_pinned.sha256(data)}


def watch_problem(watch_errors, readings, unread, generation, width):
    """Why the watch does not qualify the run beyond report_pinned.link_problem, or None."""
    if watch_errors:
        return " Then ".join(watch_errors)
    if unread:
        return f"nvidia-smi printed {len(unread)} lines that are not link readings, so the link is unknown then"
    if not any(r["phase"] == "during" and r["gen_current"] == generation and r["width_current"] == width
               for r in readings):
        return f"no reading that arrived while the benchmark ran showed generation {generation} with {width} lanes"
    return None


def collect(command, binary, out_dir, build_dir, desktop_load, identity=run_pinned.source_identity,
            gpu=run_pinned.gpu_state, protocol_path=PROTOCOL, watch_type=Watch, forecast_check=forecast_identity,
            clock=time.perf_counter):
    """Run the benchmark with the link watched, then write the CSV and the receipt. Returns the receipt."""
    out_dir, protocol_path = Path(out_dir), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    link = protocol["pcie_link"]
    name = protocol["evidence_name"]
    samples, receipt_name = f"{name}.csv", f"{name}-execution.json"
    if (out_dir / samples).exists() or (out_dir / receipt_name).exists():
        raise SystemExit(f"{out_dir / samples} or its receipt exists. Rename and keep that attempt first.")
    source = identity(protocol=protocol_path)
    forecast = forecast_check(source["source"])
    if not forecast:
        raise SystemExit(f"{FORECAST} is not committed in the source. The forecast comes before any sample.")
    receipt = {"schema_version": 1, **source, **forecast,
               "protocol_sha256": run_pinned.sha256(protocol_path.read_bytes()),
               "binary_sha256": run_pinned.sha256(Path(binary).read_bytes()),
               "compilers": run_pinned.compilers(binary, build_dir), "cpu": platform.processor(),
               "os": platform.platform(), "desktop_load_confirmed": bool(desktop_load),
               "pcie_link": {"generation": link["generation"], "width": link["width"]}}
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt["gpu_before"] = gpu()
    watch = watch_type(link, clock=clock)
    watch_errors = []
    started = run_pinned.guarded(watch.start)
    if started is None:
        before = run_pinned.guarded(lambda: watch.wait_for(1))
        if before is not None:
            watch_errors.append(f"before the benchmark: {before}")
    else:
        watch_errors.append(f"nvidia-smi did not start: {started}")
    receipt["started_utc"] = run_pinned.now()
    try:
        benchmark = subprocess.Popen([*command, "--run"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        if started is None:
            run_pinned.guarded(watch.stop)
        raise
    # Marked once the benchmark's process exists and again once it has exited.
    first, begin = (watch.mark() if started is None else 0), clock()
    try:
        stdout, stderr = benchmark.communicate(timeout=protocol["limits"]["measurement_timeout_seconds"])
        exit_code = benchmark.returncode
    except subprocess.TimeoutExpired:
        benchmark.kill()
        stdout, stderr = benchmark.communicate()
        exit_code = "timeout"
    finally:
        last, end = (watch.mark() if started is None else 0), clock()
        entries = []
        if started is None:
            after = run_pinned.guarded(lambda: watch.wait_for(watch.readings() + 1))
            if after is not None:
                watch_errors.append(f"after the benchmark: {after}")
            stopped = run_pinned.guarded(watch.stop)
            if isinstance(stopped, list):
                entries = stopped
            else:
                watch_errors.append(f"stopping nvidia-smi: {stopped}")
    # The samples reach disk before any step that could still fail.
    (out_dir / samples).write_bytes(stdout)
    receipt["finished_utc"] = run_pinned.now()
    receipt["benchmark_seconds"] = round(end - begin, 4)
    receipt["exit_code"] = exit_code
    receipt["csv_sha256"] = run_pinned.sha256(stdout)
    receipt["stderr"] = stderr.decode("utf-8", "replace").strip().splitlines()
    receipt["gpu_after"] = run_pinned.guarded(gpu)
    readings, unread = [], []
    for index, (arrived, line) in enumerate(entries):
        phase = "before" if index < first else "during" if index < last else "after"
        try:
            readings.append({**reading(line, link["fields"]), "phase": phase,
                             "seconds_from_start": round(arrived - begin, 4)})
        except ValueError:
            unread.append(line)
    receipt["pcie_link_readings"] = readings
    if unread:
        receipt["pcie_link_unread"] = unread
    if watch_errors:
        receipt["pcie_link_watch_errors"] = watch_errors
    current = run_pinned.guarded(lambda: identity(check_clean=False, protocol=protocol_path)["source"])
    receipt["source_unchanged"] = current == receipt["source"]
    status = "passed" if exit_code == 0 else "failed"
    if status == "passed":
        try:
            validate(list(csv.DictReader(stdout.decode("utf-8-sig").splitlines())), protocol)
        except (ValueError, TypeError, UnicodeDecodeError) as error:
            status, receipt["validation_error"] = "invalid", str(error)
    if status == "passed" and not (receipt["source_unchanged"] and receipt["desktop_load_confirmed"]):
        status = "unqualified"
    if status == "passed":
        problem = (report_pinned.link_problem(readings, link["generation"], link["width"])
                   or watch_problem(watch_errors, readings, unread, link["generation"], link["width"]))
        if problem:
            status, receipt["link_problem"] = "unqualified", problem
    receipt["status"] = status
    (out_dir / receipt_name).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary", type=Path, help="the built hbr_cuda_bench")
    parser.add_argument("--build-dir", type=Path, required=True, help="the CMake build that produced it")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "evidence")
    parser.add_argument("--confirm-desktop-load", action="store_true",
                        help="I checked privately that no game is running and only the usual desktop applications use the GPU")
    args = parser.parse_args(argv)
    receipt = collect([str(args.binary)], args.binary, args.out_dir, args.build_dir, args.confirm_desktop_load)
    print(f"{receipt['status']}: exit {receipt['exit_code']}, receipt in {args.out_dir / 'cuda-gen4-execution.json'}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
