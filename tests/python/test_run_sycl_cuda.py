import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PureWindowsPath
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_sycl_cuda", ROOT / "tools" / "run_sycl_cuda.py")
synthetic = load("test_report_sycl_cuda_rows", ROOT / "tests" / "python" / "test_report_sycl_cuda.py")

PLAN = {"square_sides": [32, 256, 1024], "workloads": ["masked_reduce", "tile_histogram", "stencil3x3"],
        "backends": ["cuda", "sycl"], "tile_shape": [64, 64], "samples_per_condition": 25,
        "additional_unrecorded_warmups": 2, "run_rows": 450}
IDENTITY = {"source": "a" * 40, "source_tree": "b" * 40, "protocol_commit": "c" * 40}
GPU = "Stand-in GPU, 1.0, 24564 MiB, 3 %, 100 MiB, 30"
BUILD = {"build_source_dir": "/mnt/c/Work/hbr", "build_dry_run": "ninja: no work to do.",
         "cuda_host_compiler": "/usr/bin/g++: g++ 13.3.0", "wsl_distro": "Stand-in", "wsl_kernel": "6.0.0-stand-in"}


def same_identity(check_clean=True):
    return dict(IDENTITY)


def stand_in(tmp_path, rows, exit_code=0, plan=PLAN, error=None, sleep=0):
    """A script inside a CMake build tree that answers --plan and --run like the benchmark, without a device."""
    build = tmp_path / "build"
    version = build / "CMakeFiles" / "3.28.3"
    version.mkdir(parents=True)
    (version / "CMakeCXXCompiler.cmake").write_text(
        'set(CMAKE_CXX_COMPILER_ID "Clang")\nset(CMAKE_CXX_COMPILER_VERSION "23.1.1")\n', encoding="utf-8")
    (version / "CMakeCUDACompiler.cmake").write_text(
        'set(CMAKE_CUDA_COMPILER_ID "NVIDIA")\nset(CMAKE_CUDA_COMPILER_VERSION "13.2.86")\n', encoding="utf-8")
    data = tmp_path / "rows.csv"
    with data.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=synthetic.report.HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    script = build / "stand_in.py"
    script.write_text(
        "import sys, time\n"
        f"if sys.argv[1] == '--plan':\n    print({json.dumps(json.dumps(plan))})\n    sys.exit(0)\n"
        "print('CUDA: Stand-in GPU SM 8.9', file=sys.stderr, flush=True)\n"
        f"sys.stdout.write(open({str(data)!r}, newline='').read())\n"
        "sys.stdout.flush()\n"
        f"time.sleep({sleep})\n"
        + (f"print({error!r}, file=sys.stderr)\n" if error else "")
        + f"sys.exit({exit_code})\n", encoding="utf-8")
    return [sys.executable, str(script)], script, build


def collect(tmp_path, rows, gpu_free=True, identity=same_identity, gpu=lambda: GPU, **kwargs):
    command, script, build = stand_in(tmp_path, rows, **kwargs)
    out = tmp_path / "evidence"
    return runner.collect(command, script, out, build, gpu_free, identity=identity, gpu=gpu), out


@pytest.fixture(scope="module")
def rows():
    return synthetic.synthetic_rows()


def test_a_valid_run_writes_its_samples_and_a_passed_receipt(tmp_path, rows):
    receipt, out = collect(tmp_path, rows)
    assert receipt["status"] == "passed" and receipt["exit_code"] == 0
    assert receipt["csv_sha256"] == hashlib.sha256((out / "sycl-cuda.csv").read_bytes()).hexdigest()
    assert receipt["protocol_sha256"] == hashlib.sha256(runner.PROTOCOL.read_bytes()).hexdigest()
    assert receipt["compilers"] == {"cxx": "Clang 23.1.1", "cuda": "NVIDIA 13.2.86"}
    assert receipt["command"][0] == sys.executable and receipt["command"][1].endswith("stand_in.py")
    assert receipt["stderr"] == ["CUDA: Stand-in GPU SM 8.9"]
    assert receipt["gpu_before"] == receipt["gpu_after"] == GPU
    assert receipt["source_unchanged"] is True and receipt["gpu_free_confirmed"] is True
    assert json.loads((out / "sycl-cuda-execution.json").read_text(encoding="utf-8")) == receipt


def test_rows_that_break_the_protocol_are_kept_and_marked_invalid(tmp_path, rows):
    broken = [dict(r) for r in rows]
    broken[3]["upload_bytes"] = "1"
    receipt, out = collect(tmp_path, broken)
    assert receipt["status"] == "invalid"
    assert "transfer bytes differ" in receipt["validation_error"]
    assert (out / "sycl-cuda.csv").read_bytes().count(b"\n") == len(rows) + 1


def test_a_failed_run_keeps_its_partial_samples_and_error_text(tmp_path, rows):
    receipt, out = collect(tmp_path, rows[:10], exit_code=1, error="sycl histogram mismatch")
    assert receipt["status"] == "failed" and receipt["exit_code"] == 1
    assert receipt["stderr"] == ["CUDA: Stand-in GPU SM 8.9", "sycl histogram mismatch"]
    assert (out / "sycl-cuda.csv").read_bytes().count(b"\n") == 11


def test_a_timed_out_run_is_kept_with_the_rows_it_printed(tmp_path, rows, monkeypatch):
    monkeypatch.setattr(runner, "TIMEOUT", 2)
    monkeypatch.setattr(runner, "GRACE", 0)
    receipt, out = collect(tmp_path, rows[:10], sleep=30)
    assert receipt["status"] == "failed" and receipt["exit_code"] == "timeout"
    assert (out / "sycl-cuda-execution.json").exists()
    assert (out / "sycl-cuda.csv").read_bytes().count(b"\n") == 11


def test_a_failing_step_after_the_run_does_not_lose_the_samples(tmp_path, rows):
    calls = []

    def gpu():
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError("nvidia-smi hung")
        return GPU

    receipt, out = collect(tmp_path, rows, gpu=gpu)
    assert receipt["gpu_after"].startswith("unavailable: TimeoutError")
    assert receipt["status"] == "passed" and (out / "sycl-cuda.csv").exists()


def test_passed_needs_an_unchanged_source_and_a_confirmed_free_gpu(tmp_path, rows):
    sources = iter(["a" * 40, "d" * 40])

    def moved(check_clean=True):
        return dict(IDENTITY, source=next(sources))

    receipt, _ = collect(tmp_path / "moved", rows, identity=moved)
    assert receipt["source_unchanged"] is False and receipt["status"] == "unqualified"
    receipt, _ = collect(tmp_path / "busy", rows, gpu_free=False)
    assert receipt["gpu_free_confirmed"] is False and receipt["status"] == "unqualified"


def test_an_earlier_attempt_is_never_overwritten(tmp_path, rows):
    collect(tmp_path, rows[:10], exit_code=1)
    before = (tmp_path / "evidence" / "sycl-cuda-execution.json").read_bytes()
    build = tmp_path / "build"
    with pytest.raises(SystemExit, match="Rename and keep"):
        runner.collect([sys.executable, str(build / "stand_in.py")], build / "stand_in.py",
                       tmp_path / "evidence", build, True, identity=same_identity, gpu=lambda: GPU)
    assert (tmp_path / "evidence" / "sycl-cuda-execution.json").read_bytes() == before


def test_a_binary_outside_the_build_or_a_plan_mismatch_runs_nothing(tmp_path, rows):
    command, script, build = stand_in(tmp_path, rows, plan=dict(PLAN, backends=["cuda"]))
    with pytest.raises(SystemExit, match="Nothing ran"):
        runner.collect(command, script, tmp_path / "evidence", build, True, identity=same_identity, gpu=lambda: GPU)
    with pytest.raises(SystemExit, match="not inside the build"):
        runner.collect(command, script, tmp_path / "evidence", tmp_path / "elsewhere", True,
                       identity=same_identity, gpu=lambda: GPU)
    assert not (tmp_path / "evidence").exists()


def test_paths_inside_the_distro_map_to_its_share():
    path = runner.share("HBR-SYCL-20261115", "/opt/hbr-sycl-cuda/build/hbr_sycl_cuda_bench")
    assert PureWindowsPath(path) == PureWindowsPath(
        r"\\wsl.localhost\HBR-SYCL-20261115\opt\hbr-sycl-cuda\build\hbr_sycl_cuda_bench")
    assert runner.in_distro("D", "uname", "-r") == ["wsl", "-d", "D", "--exec", "uname", "-r"]


def test_mounted_paths_map_to_windows_and_distro_paths_do_not():
    assert runner.windows_path("/mnt/c/Work/hbr") == PureWindowsPath(r"C:\Work\hbr")
    assert runner.windows_path("/opt/hbr-sycl-cuda") is None
    assert runner.windows_path("/mnt/wsl/x") is None


def test_cache_entries_reads_only_the_named_values(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text(
        "# comment\nCMAKE_HOME_DIRECTORY:INTERNAL=/mnt/c/Work/hbr\n"
        "CMAKE_CUDA_HOST_COMPILER:UNINITIALIZED=/usr/bin/g++\nOTHER:STRING=x\n", encoding="utf-8")
    assert runner.cache_entries(tmp_path, ("CMAKE_HOME_DIRECTORY", "CMAKE_CUDA_HOST_COMPILER")) == {
        "CMAKE_HOME_DIRECTORY": "/mnt/c/Work/hbr", "CMAKE_CUDA_HOST_COMPILER": "/usr/bin/g++"}


def test_a_build_from_another_tree_or_with_work_left_is_refused():
    root = PureWindowsPath(r"c:\work\HBR")
    runner.check_build(BUILD, root=root)
    with pytest.raises(SystemExit, match="not this checkout"):
        runner.check_build(dict(BUILD, build_source_dir="/opt/hbr"), root=root)
    with pytest.raises(SystemExit, match="not this checkout"):
        runner.check_build(dict(BUILD, build_source_dir="/mnt/c/Work/other"), root=root)
    with pytest.raises(SystemExit, match="not up to date"):
        runner.check_build(dict(BUILD, build_dry_run="[1/2] Building CXX object x.o"), root=root)


def test_a_refused_build_runs_nothing_and_an_accepted_one_is_recorded(tmp_path, rows, monkeypatch):
    command, script, build = stand_in(tmp_path, rows)
    with pytest.raises(SystemExit, match="Nothing ran"):
        runner.collect(command, script, tmp_path / "evidence", build, True, identity=same_identity,
                       gpu=lambda: GPU, build_state=lambda: dict(BUILD, build_source_dir="/opt/elsewhere"))
    assert not (tmp_path / "evidence").exists()
    monkeypatch.setattr(runner, "check_build", lambda state: None)
    receipt = runner.collect(command, script, tmp_path / "evidence", build, True, identity=same_identity,
                             gpu=lambda: GPU, build_state=lambda: dict(BUILD))
    assert receipt["status"] == "passed" and receipt["cuda_host_compiler"] == "/usr/bin/g++: g++ 13.3.0"
    assert receipt["build_dry_run"] == "ninja: no work to do." and receipt["wsl_kernel"] == "6.0.0-stand-in"


def test_source_identity_names_the_protocol_commit():
    identity = runner.source_identity(check_clean=False)
    assert all(len(identity[key]) == 40 for key in ("source", "source_tree", "protocol_commit"))
