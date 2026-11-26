import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_pinned", ROOT / "tools" / "run_pinned.py")
synthetic = load("test_report_pinned_rows", ROOT / "tests" / "python" / "test_report_pinned.py")

PLAN = {"square_sides": [32, 256, 1024], "workloads": ["masked_reduce", "tile_histogram", "stencil3x3"],
        "host_memory": ["pageable", "pinned"], "tile_shape": [64, 64], "samples_per_condition": 121,
        "additional_unrecorded_warmups": 2, "run_rows": 2178}
STDERR = synthetic.IDLE_STDERR
IDENTITY = {"source": "a" * 40, "source_tree": "b" * 40, "protocol_commit": "c" * 40}
GPU = "Stand-in GPU, 1.0, 24564 MiB, 3 %, 100 MiB, 30"


def same_identity(check_clean=True, protocol=None):
    return dict(IDENTITY)


def sequence_text(swap=False):
    """What --sequence prints. With swap, the first two rows trade their memory kinds."""
    rows = [list(map(str, entry)) for entry in synthetic.report.run_sequence()]
    if swap:
        rows[0][2], rows[1][2] = rows[1][2], rows[0][2]
    return "\n".join([",".join(synthetic.report.SEQUENCE_HEADER)] + [",".join(r) for r in rows]) + "\n"


def stand_in(tmp_path, rows, exit_code=0, plan=PLAN, error=None, sleep=0, stderr=STDERR, swap=False):
    """A script inside a CMake build tree that answers --plan, --sequence and --run like the benchmark, without a device."""
    build = tmp_path / "build"
    version = build / "CMakeFiles" / "4.4.4"
    version.mkdir(parents=True)
    (version / "CMakeCXXCompiler.cmake").write_text(
        'set(CMAKE_CXX_COMPILER_ID "MSVC")\nset(CMAKE_CXX_COMPILER_VERSION "19.44.35228.0")\n', encoding="utf-8")
    (version / "CMakeCUDACompiler.cmake").write_text(
        'set(CMAKE_CUDA_COMPILER_ID "NVIDIA")\nset(CMAKE_CUDA_COMPILER_VERSION "13.2.86")\n', encoding="utf-8")
    data = tmp_path / "rows.csv"
    with data.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=synthetic.report.HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sequence = tmp_path / "sequence.csv"
    sequence.write_text(sequence_text(swap), encoding="utf-8")
    script = build / "stand_in.py"
    script.write_text(
        "import sys, time\n"
        f"if sys.argv[1] == '--plan':\n    print({json.dumps(json.dumps(plan))})\n    sys.exit(0)\n"
        f"if sys.argv[1] == '--sequence':\n    sys.stdout.write(open({str(sequence)!r}).read())\n    sys.exit(0)\n"
        f"print({chr(10).join(stderr)!r}, file=sys.stderr, flush=True)\n"
        f"sys.stdout.write(open({str(data)!r}, newline='').read())\n"
        "sys.stdout.flush()\n"
        f"time.sleep({sleep})\n"
        + (f"print({error!r}, file=sys.stderr)\n" if error else "")
        + f"sys.exit({exit_code})\n", encoding="utf-8")
    return [sys.executable, str(script)], script, build


def collect(tmp_path, rows, desktop_load=True, identity=same_identity, gpu=lambda: GPU, protocol_path=runner.PROTOCOL,
            link_watch=None, **kwargs):
    command, script, build = stand_in(tmp_path, rows, **kwargs)
    out = tmp_path / "evidence"
    extra = {} if link_watch is None else {"link_watch": link_watch}
    return runner.collect(command, script, out, build, desktop_load, identity=identity, gpu=gpu,
                          protocol_path=protocol_path, **extra), out


@pytest.fixture(scope="module")
def rows():
    return synthetic.synthetic_rows()


def test_a_valid_run_writes_its_samples_and_a_passed_receipt(tmp_path, rows):
    receipt, out = collect(tmp_path, rows)
    assert receipt["status"] == "passed" and receipt["exit_code"] == 0
    assert receipt["csv_sha256"] == hashlib.sha256((out / "pinned.csv").read_bytes()).hexdigest()
    assert receipt["protocol_sha256"] == hashlib.sha256(runner.PROTOCOL.read_bytes()).hexdigest()
    assert receipt["compilers"] == {"cxx": "MSVC 19.44.35228.0", "cuda": "NVIDIA 13.2.86"}
    assert receipt["stderr"] == STDERR and receipt["schema_version"] == 2
    assert [r["point"] for r in receipt["gpu_idle_readings"]] == synthetic.report.IDLE_POINTS
    assert receipt["gpu_idle_readings"][-1] == {"point": "after 1024", "gpu_util_percent": 30, "gpu_memory_mib": 6090}
    assert receipt["gpu_before"] == receipt["gpu_after"] == GPU
    assert receipt["source_unchanged"] is True and receipt["desktop_load_confirmed"] is True
    assert json.loads((out / "pinned-execution.json").read_text(encoding="utf-8")) == receipt


def test_rows_that_break_the_protocol_are_kept_and_marked_invalid(tmp_path, rows):
    broken = [dict(r) for r in rows]
    broken[3]["upload_bytes"] = "1"
    receipt, out = collect(tmp_path, broken)
    assert receipt["status"] == "invalid"
    assert "transfer bytes differ" in receipt["validation_error"]
    assert (out / "pinned.csv").read_bytes().count(b"\n") == len(rows) + 1


def test_a_failed_run_keeps_its_partial_samples_and_error_text(tmp_path, rows):
    receipt, out = collect(tmp_path, rows[:10], exit_code=1, error="histogram mismatch", stderr=STDERR[:3])
    assert receipt["status"] == "failed" and receipt["exit_code"] == 1
    assert receipt["stderr"] == STDERR[:3] + ["histogram mismatch"]
    assert (out / "pinned.csv").read_bytes().count(b"\n") == 11


def test_a_timed_out_run_is_kept(tmp_path, rows, monkeypatch):
    monkeypatch.setattr(runner, "TIMEOUT", 2)
    receipt, out = collect(tmp_path, rows[:10], sleep=30)
    assert receipt["status"] == "failed" and receipt["exit_code"] == "timeout"
    assert (out / "pinned-execution.json").exists() and (out / "pinned.csv").exists()


def test_a_failing_step_after_the_run_does_not_lose_the_samples(tmp_path, rows):
    calls = []

    def gpu():
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError("nvidia-smi hung")
        return GPU

    receipt, out = collect(tmp_path, rows, gpu=gpu)
    assert receipt["gpu_after"].startswith("unavailable: TimeoutError")
    assert receipt["status"] == "passed" and (out / "pinned.csv").exists()


def test_a_run_without_all_four_idle_readings_is_invalid(tmp_path, rows):
    receipt, _ = collect(tmp_path, rows, stderr=STDERR[:-1])
    assert receipt["status"] == "invalid"
    assert "idle readings" in receipt["validation_error"]


def test_passed_needs_an_unchanged_source_and_a_confirmed_desktop_load(tmp_path, rows):
    sources = iter(["a" * 40, "d" * 40])

    def moved(check_clean=True, protocol=None):
        return dict(IDENTITY, source=next(sources))

    receipt, _ = collect(tmp_path / "moved", rows, identity=moved)
    assert receipt["source_unchanged"] is False and receipt["status"] == "unqualified"
    receipt, _ = collect(tmp_path / "busy", rows, desktop_load=False)
    assert receipt["desktop_load_confirmed"] is False and receipt["status"] == "unqualified"


def test_an_earlier_attempt_is_never_overwritten(tmp_path, rows):
    collect(tmp_path, rows[:10], exit_code=1)
    before = (tmp_path / "evidence" / "pinned-execution.json").read_bytes()
    build = tmp_path / "build"
    with pytest.raises(SystemExit, match="Rename and keep"):
        runner.collect([sys.executable, str(build / "stand_in.py")], build / "stand_in.py",
                       tmp_path / "evidence", build, True, identity=same_identity, gpu=lambda: GPU)
    assert (tmp_path / "evidence" / "pinned-execution.json").read_bytes() == before


def test_a_binary_outside_the_build_or_a_plan_mismatch_runs_nothing(tmp_path, rows):
    command, script, build = stand_in(tmp_path, rows, plan=dict(PLAN, samples_per_condition=120))
    with pytest.raises(SystemExit, match="Nothing ran"):
        runner.collect(command, script, tmp_path / "evidence", build, True, identity=same_identity, gpu=lambda: GPU)
    with pytest.raises(SystemExit, match="not inside the build"):
        runner.collect(command, script, tmp_path / "evidence", tmp_path / "elsewhere", True,
                       identity=same_identity, gpu=lambda: GPU)
    assert not (tmp_path / "evidence").exists()


def test_source_identity_returns_full_revisions():
    identity = runner.source_identity(check_clean=False)
    assert all(len(identity[key]) == 40 for key in ("source", "source_tree", "protocol_commit"))


def test_a_run_order_mismatch_runs_nothing(tmp_path, rows):
    command, script, build = stand_in(tmp_path, rows, swap=True)
    with pytest.raises(SystemExit, match="run order differs"):
        runner.collect(command, script, tmp_path / "evidence", build, True, identity=same_identity, gpu=lambda: GPU)
    assert not (tmp_path / "evidence").exists()


GEN4 = ROOT / "docs" / "pinned-gen4-protocol.json"


def link_line(current, width=16, maximum=4):
    return f"2026/11/26 02:00:00.000, {current}, {maximum}, 4, 5, {width}, 16"


class FakeWatch:
    """Stands in for LinkWatch and returns the given nvidia-smi lines, or raises on start or stop."""

    def __init__(self, lines, fail_start=False):
        self.lines, self.fail_start, self.calls = lines, fail_start, []

    def __call__(self, link):
        self.calls.append(("made", link["interval_milliseconds"]))
        return self

    def start(self):
        self.calls.append("start")
        if self.fail_start:
            raise FileNotFoundError("nvidia-smi")

    def stop(self):
        self.calls.append("stop")
        return list(self.lines)


def collect_gen4(tmp_path, rows, watch, **kwargs):
    return collect(tmp_path, rows, protocol_path=GEN4, link_watch=watch, **kwargs)


def test_the_gen4_protocol_keeps_the_plan_of_the_pinned_protocol():
    protocol = json.loads(GEN4.read_text(encoding="utf-8"))
    assert runner.plan_differences(protocol) == []
    assert runner.evidence_names(protocol) == ("pinned-gen4.csv", "pinned-gen4-execution.json")
    assert runner.evidence_names(json.loads(runner.PROTOCOL.read_text(encoding="utf-8"))) == \
        ("pinned.csv", "pinned-execution.json")
    assert protocol["pcie_link"]["generation"] == 4 and protocol["pcie_link"]["width"] == 16
    assert protocol["pcie_link"]["fields"][1:] == [f"pcie.link.{key.replace('_', '.')}"
                                                   for key in synthetic.report.LINK_KEYS]


def test_a_repeat_that_changes_the_plan_runs_nothing(tmp_path, rows):
    changed = json.loads(GEN4.read_text(encoding="utf-8"))
    changed["samples_per_condition"], changed["host_timing"] = 120, "something else"
    path = tmp_path / "changed-protocol.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    watch = FakeWatch([link_line(4)])
    with pytest.raises(SystemExit, match="host_timing, samples_per_condition. Nothing ran"):
        collect(tmp_path, rows, protocol_path=path, link_watch=watch)
    assert watch.calls == [] and not (tmp_path / "evidence").exists()


def test_a_gen4_run_keeps_every_link_reading_and_passes_at_generation_4(tmp_path, rows):
    watch = FakeWatch([link_line(1), link_line(4), link_line(4), link_line(1)])
    receipt, out = collect_gen4(tmp_path, rows, watch)
    assert watch.calls == [("made", 250), "start", "stop"]
    assert receipt["status"] == "passed" and receipt["schema_version"] == 3
    assert receipt["pcie_link"] == {"generation": 4, "width": 16}
    assert [r["gen_current"] for r in receipt["pcie_link_readings"]] == [1, 4, 4, 1]
    assert "pcie_link_unread" not in receipt and "link_problem" not in receipt
    assert receipt["protocol_sha256"] == hashlib.sha256(GEN4.read_bytes()).hexdigest()
    assert receipt["csv_sha256"] == hashlib.sha256((out / "pinned-gen4.csv").read_bytes()).hexdigest()
    assert json.loads((out / "pinned-gen4-execution.json").read_text(encoding="utf-8")) == receipt
    assert not (out / "pinned.csv").exists()
    assert "| generation 4, 16 lanes | 2 |" in synthetic.report.report(rows, receipt)


@pytest.mark.parametrize("lines, message", [
    ([link_line(3, maximum=3)], "maximum read generation 3"),
    ([link_line(1), link_line(1)], "no reading showed the link running"),
    (["nvidia-smi has failed", link_line(4, width=8)], "ran at generation 4 with 8 lanes"),
    ([link_line(4), link_line(3), link_line(4)], "ran at generation 3 with 16 lanes"),
])
def test_a_run_at_another_link_is_kept_and_unqualified(tmp_path, rows, lines, message):
    receipt, out = collect_gen4(tmp_path, rows, FakeWatch(lines))
    assert receipt["status"] == "unqualified" and message in receipt["link_problem"]
    assert (out / "pinned-gen4.csv").exists()
    if lines[0] == "nvidia-smi has failed":
        assert receipt["pcie_link_unread"] == ["nvidia-smi has failed"]


def test_a_link_watch_that_cannot_start_leaves_the_run_unqualified(tmp_path, rows):
    watch = FakeWatch([link_line(4)], fail_start=True)
    receipt, _ = collect_gen4(tmp_path, rows, watch)
    assert watch.calls == [("made", 250), "start"]
    assert receipt["pcie_link_readings"] == []
    assert receipt["pcie_link_unread"][0].startswith("unavailable: FileNotFoundError")
    assert receipt["status"] == "unqualified" and "no link reading" in receipt["link_problem"]


def test_a_timed_out_gen4_run_still_stops_the_link_watch(tmp_path, rows, monkeypatch):
    monkeypatch.setattr(runner, "TIMEOUT", 2)
    watch = FakeWatch([link_line(4)])
    receipt, out = collect_gen4(tmp_path, rows[:10], watch, sleep=30)
    assert watch.calls == [("made", 250), "start", "stop"]
    assert receipt["exit_code"] == "timeout" and receipt["status"] == "failed"
    assert len(receipt["pcie_link_readings"]) == 1 and (out / "pinned-gen4.csv").exists()


def test_the_pinned_protocol_runs_without_a_link_watch(tmp_path, rows):
    watch = FakeWatch([link_line(4)])
    receipt, _ = collect(tmp_path, rows, link_watch=watch)
    assert watch.calls == [] and "pcie_link" not in receipt and receipt["schema_version"] == 2


def test_link_watch_keeps_what_its_command_prints_until_it_is_stopped(tmp_path):
    # The stand in writes a marker file once its lines are out, so the test never moves the file
    # position that it shares with the child while the child still writes.
    script, marker = tmp_path / "smi.py", tmp_path / "printed"
    script.write_text("import sys, time\nfor i in range(3):\n    print(f'line {i}', flush=True)\n"
                      f"open({str(marker)!r}, 'w').close()\ntime.sleep(60)\n", encoding="utf-8")
    watch = runner.LinkWatch({"fields": ["timestamp"], "interval_milliseconds": 250})
    assert watch.command == ["nvidia-smi", "--query-gpu=timestamp", "--format=csv,noheader,nounits", "-lms", "250"]
    watch.command = [sys.executable, str(script)]
    watch.start()
    for _ in range(200):
        if marker.exists():
            break
        time.sleep(0.05)
    assert watch.stop() == ["line 0", "line 1", "line 2"]
    assert watch.process.poll() is not None


def test_the_old_confirmation_flag_is_refused(monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("collect ran with the old flag")

    monkeypatch.setattr(runner, "collect", never)
    with pytest.raises(SystemExit) as refused:
        runner.main(["bench.exe", "--build-dir", "build", "--confirm-gpu-free"])
    assert refused.value.code == 2
