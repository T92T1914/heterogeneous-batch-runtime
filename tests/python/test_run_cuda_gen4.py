import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_cuda_gen4", ROOT / "tools" / "run_cuda_gen4.py")
PROTOCOL = json.loads(runner.PROTOCOL.read_text(encoding="utf-8"))
FIELDS = PROTOCOL["pcie_link"]["fields"]
SEPTEMBER = (ROOT / "docs" / "evidence" / "baseline-cuda.csv").read_bytes()
IDENTITY = {"source": "a" * 40, "source_tree": "b" * 40, "protocol_commit": "c" * 40}
FORECAST = {"forecast_commit": "d" * 40, "forecast_sha256": "e" * 64}
GPU = "Stand-in GPU, 1.0, 24564 MiB, 3 %, 100 MiB, 30"
IDLE = "2026/11/27 08:00:00.000, 1, 4, 4, 5, 16, 16, 9, 6100"
GEN4 = "2026/11/27 08:00:00.050, 4, 4, 4, 5, 16, 16, 31, 6900"
GEN3 = "2026/11/27 08:00:00.100, 3, 4, 4, 5, 16, 16, 30, 6900"
NO_LOAD_GEN3 = "2026/11/27 08:00:00.150, 3, 4, 4, 5, 16, 16, [N/A], 6900"
WARNING = "Unable to determine the device handle for one GPU"
# The fake clock: the benchmark starts at 10.0 and exits at 11.0.
BEGIN, END = 10.0, 11.0


def same_identity(check_clean=True, protocol=None):
    return dict(IDENTITY)


def september_rows():
    return list(csv.DictReader(SEPTEMBER.decode("utf-8").splitlines()))


def fake_clock():
    times = iter([BEGIN, END])
    return lambda: next(times, END)


class FakeWatch:
    """Stands in for nvidia-smi. Each line has the time it arrives on the fake clock. Lines before
    BEGIN are there once it starts, lines from BEGIN to END arrive by the second mark, the moment the
    benchmark has exited, and later ones arrive one at a time as the runner waits."""

    lines = [(9.9, IDLE), (10.3, GEN4), (11.05, IDLE)]

    def __init__(self, link, clock=None):
        self.fields = link["fields"]
        self.received = []
        self.marks = 0
        self.pending = [entry for entry in self.lines if entry[0] > END]

    def start(self):
        self.received.extend(entry for entry in self.lines if entry[0] < BEGIN)

    def mark(self):
        self.marks += 1
        if self.marks == 2:
            self.received.extend(entry for entry in self.lines if BEGIN <= entry[0] <= END)
        return len(self.received)

    def readings(self):
        return sum(runner.is_reading(line, self.fields) for _, line in self.received)

    def wait_for(self, count, timeout=10):
        while self.readings() < count:
            if not self.pending:
                raise RuntimeError(f"nvidia-smi gave fewer than {count} link readings")
            self.received.append(self.pending.pop(0))

    def stop(self):
        return list(self.received)


def watch_with(*lines):
    return type("Watch", (FakeWatch,), {"lines": list(lines)})


def stand_in(tmp_path, data=SEPTEMBER, exit_code=0):
    """A script inside a CMake build tree that prints the given CSV for --run, like hbr_cuda_bench."""
    build = tmp_path / "build"
    version = build / "CMakeFiles" / "4.4.4"
    version.mkdir(parents=True)
    (version / "CMakeCXXCompiler.cmake").write_text(
        'set(CMAKE_CXX_COMPILER_ID "MSVC")\nset(CMAKE_CXX_COMPILER_VERSION "19.44.35228.0")\n', encoding="utf-8")
    (version / "CMakeCUDACompiler.cmake").write_text(
        'set(CMAKE_CUDA_COMPILER_ID "NVIDIA")\nset(CMAKE_CUDA_COMPILER_VERSION "13.2.86")\n', encoding="utf-8")
    (tmp_path / "rows.csv").write_bytes(data)
    script = build / "stand_in.py"
    script.write_text(
        "import sys\n"
        "print('NVIDIA GeForce RTX 4090 SM 8.9', file=sys.stderr)\n"
        f"sys.stdout.buffer.write(open({str(tmp_path / 'rows.csv')!r}, 'rb').read())\n"
        f"sys.exit({exit_code})\n", encoding="utf-8")
    return [sys.executable, str(script)], script, build


def collect(tmp_path, data=SEPTEMBER, exit_code=0, watch=FakeWatch, desktop_load=True, forecast=FORECAST,
            identity=same_identity, out=None):
    command, script, build = stand_in(tmp_path, data, exit_code)
    out = out or tmp_path / "evidence"
    receipt = runner.collect(command, script, out, build, desktop_load, identity=identity, gpu=lambda: GPU,
                             watch_type=watch, forecast_check=lambda source: forecast, clock=fake_clock())
    return receipt, out


def rows_text(rows, extra=None):
    lines = [",".join(runner.HEADER)] + [",".join(r[k] for k in runner.HEADER) for r in rows]
    if extra is not None:
        lines[extra] += ",9"
    return ("\n".join(lines) + "\n").encode()


def test_conditions_follow_the_september_benchmark_output():
    runner.validate(september_rows(), PROTOCOL)
    assert len(runner.conditions()) * PROTOCOL["iterations"] == PROTOCOL["run_rows"] == 180


def test_protocol_keeps_the_september_plan():
    september = json.loads((ROOT / "docs" / "cuda-protocol.json").read_text(encoding="utf-8"))
    for key in ("command", "square_sides", "iterations", "first_iteration", "workloads", "histogram_tiles",
                "histogram_distributions", "host_timing"):
        assert PROTOCOL[key] == september[key], key
    assert PROTOCOL["pcie_link"]["generation"] == 4 and PROTOCOL["pcie_link"]["width"] == 16
    assert PROTOCOL["evidence_name"] == "cuda-gen4"
    assert PROTOCOL["forecast"]["file"] == runner.FORECAST
    assert (ROOT / runner.FORECAST).exists()


def test_a_run_at_generation_4_passes_and_keeps_every_reading(tmp_path):
    receipt, out = collect(tmp_path)
    assert receipt["status"] == "passed", receipt
    assert (out / "cuda-gen4.csv").read_bytes() == SEPTEMBER
    assert receipt["csv_sha256"] == runner.run_pinned.sha256(SEPTEMBER)
    assert [r["gen_current"] for r in receipt["pcie_link_readings"]] == [1, 4, 1]
    assert [r["seconds_from_start"] for r in receipt["pcie_link_readings"]] == [-0.1, 0.3, 1.05]
    assert [r["phase"] for r in receipt["pcie_link_readings"]] == ["before", "during", "after"]
    assert receipt["benchmark_seconds"] == 1.0
    assert receipt["pcie_link_readings"][1]["utilization_percent"] == 31
    assert receipt["pcie_link_readings"][1]["memory_used_mib"] == 6900
    assert receipt["pcie_link"] == {"generation": 4, "width": 16}
    assert receipt["forecast_commit"] == "d" * 40 and receipt["forecast_sha256"] == "e" * 64
    assert receipt["compilers"] == {"cxx": "MSVC 19.44.35228.0", "cuda": "NVIDIA 13.2.86"}
    assert receipt["stderr"] == ["NVIDIA GeForce RTX 4090 SM 8.9"]
    assert "pcie_link_unread" not in receipt and "pcie_link_watch_errors" not in receipt
    assert json.loads((out / "cuda-gen4-execution.json").read_text(encoding="utf-8")) == receipt


def test_a_link_that_trained_down_is_unqualified_and_kept(tmp_path):
    receipt, out = collect(tmp_path, watch=watch_with((9.9, IDLE), (10.2, GEN4), (10.6, GEN3), (11.05, IDLE)))
    assert receipt["status"] == "unqualified"
    assert "generation 3" in receipt["link_problem"]
    assert (out / "cuda-gen4.csv").exists()


def test_a_trained_down_reading_without_a_load_field_still_counts(tmp_path):
    # The review of the first version found that a missing load field dropped the line, so a
    # reading at generation 3 vanished and the run passed.
    receipt, _ = collect(tmp_path, watch=watch_with((9.9, IDLE), (10.2, GEN4), (10.6, NO_LOAD_GEN3),
                                                    (11.05, IDLE)))
    assert receipt["status"] == "unqualified" and "generation 3" in receipt["link_problem"]
    assert receipt["pcie_link_readings"][2]["utilization_percent"] is None


def test_generation_4_only_outside_the_benchmark_is_unqualified(tmp_path):
    receipt, _ = collect(tmp_path, watch=watch_with((9.9, GEN4), (10.5, IDLE), (11.05, GEN4)))
    assert receipt["status"] == "unqualified"
    assert "while the benchmark ran" in receipt["link_problem"]


def test_a_run_that_never_read_generation_4_is_unqualified(tmp_path):
    receipt, _ = collect(tmp_path, watch=watch_with((9.9, IDLE), (10.5, IDLE), (11.05, IDLE)))
    assert receipt["status"] == "unqualified"
    assert "no reading showed" in receipt["link_problem"]


def test_a_missing_reading_after_the_run_is_unqualified(tmp_path):
    receipt, _ = collect(tmp_path, watch=watch_with((9.9, IDLE), (10.3, GEN4)))
    assert receipt["status"] == "unqualified"
    assert receipt["pcie_link_watch_errors"] == [
        "after the benchmark: unavailable: RuntimeError: nvidia-smi gave fewer than 3 link readings"]


def test_lines_that_are_not_readings_do_not_count_and_unqualify(tmp_path):
    # A warning before the run is no reading, so the runner waits past it for a real one.
    receipt, _ = collect(tmp_path, watch=watch_with((9.8, WARNING), (9.9, IDLE), (10.3, GEN4), (11.05, IDLE)))
    assert receipt["status"] == "unqualified"
    assert receipt["pcie_link_unread"] == [WARNING]
    assert "not link readings" in receipt["link_problem"]
    receipt, _ = collect(tmp_path / "only", watch=watch_with((9.9, WARNING), (10.3, GEN4), (11.05, WARNING)))
    assert receipt["status"] == "unqualified"
    assert any("after the benchmark" in error for error in receipt["pcie_link_watch_errors"])


def test_missing_short_or_inconsistent_rows_are_invalid_and_still_written(tmp_path):
    rows = SEPTEMBER.decode("utf-8").splitlines(keepends=True)
    receipt, _ = collect(tmp_path / "count", data="".join(rows[:-1]).encode())
    assert receipt["status"] == "invalid" and "179 rows" in receipt["validation_error"]
    changed = september_rows()
    changed[7]["checksum"] = str(float(changed[7]["checksum"]) + 1)
    receipt, _ = collect(tmp_path / "checksum", data=rows_text(changed))
    assert receipt["status"] == "invalid" and "checksums differ" in receipt["validation_error"]
    short = rows[:5] + [rows[5].rsplit(",", 1)[0] + "\n"] + rows[6:]
    receipt, out = collect(tmp_path / "short", data="".join(short).encode())
    assert receipt["status"] == "invalid" and "fields" in receipt["validation_error"]
    assert (out / "cuda-gen4-execution.json").exists()
    receipt, _ = collect(tmp_path / "extra", data=rows_text(september_rows(), extra=3))
    assert receipt["status"] == "invalid" and "fields" in receipt["validation_error"]


def test_a_failed_run_is_written_with_its_exit_code(tmp_path):
    receipt, out = collect(tmp_path, exit_code=3)
    assert receipt["status"] == "failed" and receipt["exit_code"] == 3
    assert (out / "cuda-gen4-execution.json").exists()


def test_no_confirmation_or_a_changed_source_is_unqualified(tmp_path):
    receipt, _ = collect(tmp_path / "load", desktop_load=False)
    assert receipt["status"] == "unqualified"
    calls = []

    def moving(check_clean=True, protocol=None):
        calls.append(check_clean)
        return dict(IDENTITY, source=("f" if len(calls) > 1 else "a") * 40)

    receipt, _ = collect(tmp_path / "source", identity=moving)
    assert receipt["status"] == "unqualified" and receipt["source_unchanged"] is False


def test_nothing_runs_without_a_committed_forecast_or_over_an_earlier_attempt(tmp_path):
    with pytest.raises(SystemExit, match="forecast comes before any sample"):
        collect(tmp_path / "forecast", forecast=None)
    assert not (tmp_path / "forecast" / "evidence").exists()
    _, out = collect(tmp_path / "first")
    with pytest.raises(SystemExit, match="Rename and keep"):
        collect(tmp_path / "second", out=out)


def test_forecast_identity_reads_the_source_revision(monkeypatch):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()
    found = runner.forecast_identity(head)
    committed = subprocess.run(["git", "show", f"{head}:{runner.FORECAST}"], cwd=ROOT, check=True,
                               capture_output=True).stdout
    assert found["forecast_sha256"] == hashlib.sha256(committed).hexdigest()
    assert len(found["forecast_commit"]) == 40
    monkeypatch.setattr(runner, "FORECAST", "docs/evidence/no-such-forecast.json")
    assert runner.forecast_identity(head) is None


def test_reading_parses_link_and_load_and_refuses_other_lines():
    parsed = runner.reading(GEN4, FIELDS)
    assert parsed["gen_current"] == 4 and parsed["width_max"] == 16
    assert parsed["utilization_percent"] == 31 and parsed["memory_used_mib"] == 6900
    assert runner.reading(NO_LOAD_GEN3, FIELDS)["gen_current"] == 3
    for line in ("2026/11/27 08:00:00.000, 4, 4, 4, 5, 16, 16", WARNING, GEN4 + ", 1",
                 "2026/11/27 08:00:00.000, [N/A], 4, 4, 5, 16, 16, 9, 6100"):
        with pytest.raises(ValueError):
            runner.reading(line, FIELDS)


def test_watch_counts_only_readings_from_a_real_process(tmp_path):
    script = tmp_path / "ticker.py"
    script.write_text(f"import sys, time\nprint({WARNING!r}, flush=True)\nfor i in range(400):\n"
                      f"    print({GEN4!r}, flush=True)\n    time.sleep(0.02)\n", encoding="utf-8")
    watch = runner.Watch(PROTOCOL["pcie_link"])
    watch.command = [sys.executable, str(script)]
    watch.start()
    watch.wait_for(1)
    seen = watch.readings()
    watch.wait_for(seen + 2)
    entries = watch.stop()
    assert entries[0][1] == WARNING
    assert sum(runner.is_reading(line, FIELDS) for _, line in entries) >= seen + 2
    times = [arrived for arrived, _ in entries]
    assert times == sorted(times)


def real_watch(tmp_path, first, then):
    """runner.Watch on the real clock around a stand-in nvidia-smi that prints first once and then
    prints then every 50 milliseconds."""
    script = tmp_path / "smi.py"
    script.write_text(f"import time\nprint({first!r}, flush=True)\nwhile True:\n    time.sleep(0.05)\n"
                      f"    print({then!r}, flush=True)\n", encoding="utf-8")

    class StandIn(runner.Watch):
        def __init__(self, link, clock=runner.time.perf_counter):
            super().__init__(link, clock)
            self.command = [sys.executable, str(script)]

    return StandIn


def slow_stand_in(tmp_path):
    command, script, build = stand_in(tmp_path)
    text = script.read_text(encoding="utf-8").replace("import sys\n", "import sys, time\ntime.sleep(0.3)\n", 1)
    script.write_text(text, encoding="utf-8")
    return command, script, build


@pytest.mark.parametrize("trial", range(5))
def test_with_the_real_clock_a_reading_taken_before_the_run_never_counts_as_during(tmp_path, trial):
    # The second review found that on Windows time.monotonic ticks every 15.6 ms, so the reading the
    # runner waits for before the benchmark landed at 0.0 seconds and qualified a run that never read
    # generation 4 while it ran. The window is now set by line position, not by a clock.
    command, script, build = slow_stand_in(tmp_path)
    receipt = runner.collect(command, script, tmp_path / "evidence", build, True, identity=same_identity,
                             gpu=lambda: GPU, watch_type=real_watch(tmp_path, GEN4, IDLE),
                             forecast_check=lambda source: FORECAST)
    assert receipt["status"] == "unqualified", receipt.get("link_problem")
    assert "while the benchmark ran" in receipt["link_problem"]
    assert receipt["pcie_link_readings"][0]["phase"] == "before"
    assert all(r["gen_current"] == 1 for r in receipt["pcie_link_readings"] if r["phase"] == "during")


def test_with_the_real_clock_a_run_read_at_generation_4_passes(tmp_path):
    command, script, build = slow_stand_in(tmp_path)
    receipt = runner.collect(command, script, tmp_path / "evidence", build, True, identity=same_identity,
                             gpu=lambda: GPU, watch_type=real_watch(tmp_path, IDLE, GEN4),
                             forecast_check=lambda source: FORECAST)
    assert receipt["status"] == "passed", receipt.get("link_problem")
    phases = [r["phase"] for r in receipt["pcie_link_readings"]]
    assert phases[0] == "before" and "during" in phases and phases[-1] == "after"
    assert phases == sorted(phases, key=["before", "during", "after"].index)
    assert receipt["benchmark_seconds"] >= 0.3


def test_a_launch_that_fails_stops_nvidia_smi(tmp_path):
    stopped = []

    class Tracked(FakeWatch):
        def stop(self):
            stopped.append(True)
            return super().stop()

    _, script, build = stand_in(tmp_path)
    with pytest.raises(OSError):
        runner.collect([str(tmp_path / "no-such-benchmark.exe")], script, tmp_path / "evidence", build, True,
                       identity=same_identity, gpu=lambda: GPU, watch_type=Tracked,
                       forecast_check=lambda source: FORECAST)
    assert stopped == [True]
