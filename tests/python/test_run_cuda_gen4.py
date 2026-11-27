import csv
import importlib.util
import json
from pathlib import Path
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
SEPTEMBER = (ROOT / "docs" / "evidence" / "baseline-cuda.csv").read_bytes()
IDENTITY = {"source": "a" * 40, "source_tree": "b" * 40, "protocol_commit": "c" * 40}
GPU = "Stand-in GPU, 1.0, 24564 MiB, 3 %, 100 MiB, 30"
IDLE = "2026/11/27 08:00:00.000, 1, 4, 4, 5, 16, 16, 9, 6100"
GEN4 = "2026/11/27 08:00:00.050, 4, 4, 4, 5, 16, 16, 31, 6900"
GEN3 = "2026/11/27 08:00:00.100, 3, 4, 4, 5, 16, 16, 30, 6900"


def same_identity(check_clean=True, protocol=None):
    return dict(IDENTITY)


def september_rows():
    return list(csv.DictReader(SEPTEMBER.decode("utf-8").splitlines()))


class FakeWatch:
    """Stands in for nvidia-smi: the lines in at_start are there once it starts, as if read while the
    benchmark ran, and the lines in later arrive one at a time as the runner waits for them."""

    at_start = [IDLE, GEN4]
    later = [IDLE]

    def __init__(self, link):
        self.link = link
        self.received = []
        self.pending = list(self.later)

    def start(self):
        self.received.extend(self.at_start)

    def lines(self):
        return len(self.received)

    def wait_for(self, count, timeout=10):
        while len(self.received) < count:
            if not self.pending:
                raise RuntimeError(f"nvidia-smi gave fewer than {count} readings")
            self.received.append(self.pending.pop(0))

    def stop(self):
        return list(self.received)


def watch_with(at_start, later):
    return type("Watch", (FakeWatch,), {"at_start": list(at_start), "later": list(later)})


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


def collect(tmp_path, data=SEPTEMBER, exit_code=0, watch=FakeWatch, desktop_load=True, forecast=True,
            identity=same_identity):
    command, script, build = stand_in(tmp_path, data, exit_code)
    out = tmp_path / "evidence"
    receipt = runner.collect(command, script, out, build, desktop_load, identity=identity, gpu=lambda: GPU,
                             watch_type=watch, forecast_check=lambda source: forecast)
    return receipt, out


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
    assert receipt["pcie_link_readings"][1]["utilization_percent"] == 31
    assert receipt["pcie_link_readings"][1]["memory_used_mib"] == 6900
    assert receipt["pcie_link"] == {"generation": 4, "width": 16}
    assert receipt["compilers"] == {"cxx": "MSVC 19.44.35228.0", "cuda": "NVIDIA 13.2.86"}
    assert receipt["stderr"] == ["NVIDIA GeForce RTX 4090 SM 8.9"]
    assert json.loads((out / "cuda-gen4-execution.json").read_text(encoding="utf-8")) == receipt


def test_a_link_that_trained_down_is_unqualified_and_kept(tmp_path):
    receipt, out = collect(tmp_path, watch=watch_with([IDLE, GEN4, GEN3], [IDLE]))
    assert receipt["status"] == "unqualified"
    assert "generation 3" in receipt["link_problem"]
    assert (out / "cuda-gen4.csv").exists()


def test_a_run_that_never_read_generation_4_is_unqualified(tmp_path):
    receipt, _ = collect(tmp_path, watch=watch_with([IDLE], [IDLE]))
    assert receipt["status"] == "unqualified"
    assert "no reading showed" in receipt["link_problem"]


def test_a_missing_reading_after_the_run_is_recorded(tmp_path):
    receipt, _ = collect(tmp_path, watch=watch_with([IDLE, GEN4], []))
    assert receipt["status"] == "passed"
    assert receipt["pcie_link_unread"] == ["unavailable: RuntimeError: nvidia-smi gave fewer than 3 readings"]


def test_missing_or_inconsistent_rows_are_invalid(tmp_path):
    rows = SEPTEMBER.decode("utf-8").splitlines(keepends=True)
    receipt, _ = collect(tmp_path / "short", data="".join(rows[:-1]).encode())
    assert receipt["status"] == "invalid" and "179 rows" in receipt["validation_error"]
    changed = september_rows()
    changed[7]["checksum"] = str(float(changed[7]["checksum"]) + 1)
    text = ",".join(runner.HEADER) + "\n" + "".join(",".join(r[k] for k in runner.HEADER) + "\n" for r in changed)
    receipt, _ = collect(tmp_path / "checksum", data=text.encode())
    assert receipt["status"] == "invalid" and "checksums differ" in receipt["validation_error"]


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
        return dict(IDENTITY, source=("d" if len(calls) > 1 else "a") * 40)

    receipt, _ = collect(tmp_path / "source", identity=moving)
    assert receipt["status"] == "unqualified" and receipt["source_unchanged"] is False


def test_nothing_runs_without_a_committed_forecast_or_over_an_earlier_attempt(tmp_path):
    with pytest.raises(SystemExit, match="forecast comes before any sample"):
        collect(tmp_path / "forecast", forecast=False)
    assert not (tmp_path / "forecast" / "evidence").exists()
    collect(tmp_path / "again")
    with pytest.raises(SystemExit, match="Rename and keep"):
        command, script, build = stand_in(tmp_path / "again2")
        runner.collect(command, script, tmp_path / "again" / "evidence", build, True, identity=same_identity,
                       gpu=lambda: GPU, watch_type=FakeWatch, forecast_check=lambda source: True)


def test_reading_parses_link_and_load_and_refuses_other_lines():
    fields = PROTOCOL["pcie_link"]["fields"]
    parsed = runner.reading(GEN4, fields)
    assert parsed["gen_current"] == 4 and parsed["width_max"] == 16
    assert parsed["utilization_percent"] == 31 and parsed["memory_used_mib"] == 6900
    for line in ("2026/11/27 08:00:00.000, 4, 4, 4, 5, 16, 16", "No devices were found", GEN4 + ", 1"):
        with pytest.raises(ValueError):
            runner.reading(line, fields)


def test_watch_collects_lines_from_a_process_and_waits_for_more(tmp_path):
    script = tmp_path / "ticker.py"
    script.write_text("import sys, time\nfor i in range(400):\n    print(f'line {i}', flush=True)\n"
                      "    time.sleep(0.02)\n", encoding="utf-8")
    watch = runner.Watch({"fields": ["timestamp"], "interval_milliseconds": 50})
    watch.command = [sys.executable, str(script)]
    watch.start()
    watch.wait_for(1)
    seen = watch.lines()
    watch.wait_for(seen + 2)
    lines = watch.stop()
    assert len(lines) >= seen + 2 and lines[0] == "line 0"
