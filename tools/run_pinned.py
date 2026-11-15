"""Run the pinned input benchmark once and write its samples and execution receipt.

The receipt names the source, the protocol, the binary and the compilers, and
records the GPU state before and after the run. It records no process list,
because the measuring computer is a shared desktop. The person who starts the
run confirms the GPU precondition with --confirm-gpu-free after checking the
processes privately.

A run that fails, times out or breaks the protocol is still written, with its
status and its error text. The runner refuses to write over an earlier attempt,
so every attempt has to be renamed and kept before the next one starts.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_pinned_plan  # noqa: E402
import report_pinned  # noqa: E402

PROTOCOL = ROOT / "docs" / "pinned-protocol.json"
GPU_FIELDS = "name,driver_version,memory.total,utilization.gpu,memory.used,temperature.gpu"
TIMEOUT = json.loads(PROTOCOL.read_text(encoding="utf-8"))["limits"]["measurement_timeout_seconds"]
SAMPLES, RECEIPT = "pinned.csv", "pinned-execution.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def source_identity(check_clean=True):
    """Revision, tree and the commit that last changed the protocol. A dirty tree is refused."""
    if check_clean and git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("The working tree has uncommitted changes. Commit or set them aside first.")
    return {"source": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
            "protocol_commit": git("log", "-1", "--format=%H", "--", "docs/pinned-protocol.json")}


def gpu_state():
    result = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else f"nvidia-smi exit {result.returncode}"


def compilers(binary, build_dir):
    """Host and CUDA compiler identities that CMake recorded for the build holding this binary."""
    build_dir = Path(build_dir).resolve()
    if build_dir not in Path(binary).resolve().parents:
        raise SystemExit("The binary is not inside the build directory given with --build-dir.")
    found = {}
    for language in ("CXX", "CUDA"):
        paths = list(build_dir.joinpath("CMakeFiles").glob(f"*/CMake{language}Compiler.cmake"))
        if len(paths) != 1:
            raise SystemExit(f"Expected one CMake{language}Compiler.cmake in the build, found {len(paths)}.")
        text = paths[0].read_text(encoding="utf-8")
        ident = re.search(rf'set\(CMAKE_{language}_COMPILER_ID "([^"]*)"\)', text)
        version = re.search(rf'set\(CMAKE_{language}_COMPILER_VERSION "([^"]*)"\)', text)
        if not (ident and version):
            raise SystemExit(f"CMake recorded no {language} compiler identity.")
        found[language.lower()] = f"{ident.group(1)} {version.group(1)}"
    return found


def guarded(step):
    """Run a step after the benchmark without letting its failure lose the samples."""
    try:
        return step()
    except Exception as error:  # noqa: BLE001 - recorded, never silent
        return f"unavailable: {type(error).__name__}: {error}"


def collect(command, binary, out_dir, build_dir, gpu_free, identity=source_identity, gpu=gpu_state):
    """Run the plan check and the benchmark, then write the CSV and the receipt. Returns the receipt."""
    out_dir = Path(out_dir)
    if (out_dir / SAMPLES).exists() or (out_dir / RECEIPT).exists():
        raise SystemExit(f"{out_dir / SAMPLES} or its receipt exists. Rename and keep that attempt first.")
    receipt = {"schema_version": 1, **identity(),
               "protocol_sha256": sha256(PROTOCOL.read_bytes()),
               "binary_sha256": sha256(Path(binary).read_bytes()),
               "compilers": compilers(binary, build_dir), "cpu": platform.processor(), "os": platform.platform(),
               "gpu_free_confirmed": bool(gpu_free)}
    plan = subprocess.run([*command, "--plan"], capture_output=True, text=True, timeout=30)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if plan.returncode != 0 or check_pinned_plan.differences(json.loads(plan.stdout), protocol):
        raise SystemExit("The binary's plan differs from docs/pinned-protocol.json. Nothing ran.")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt["gpu_before"] = gpu()
    receipt["started_utc"] = now()
    try:
        run = subprocess.run([*command, "--run"], capture_output=True, timeout=TIMEOUT)
        exit_code, stdout, stderr = run.returncode, run.stdout, run.stderr
    except subprocess.TimeoutExpired as expired:
        exit_code, stdout, stderr = "timeout", expired.stdout or b"", expired.stderr or b""
    # The samples reach disk before any step that could still fail.
    (out_dir / SAMPLES).write_bytes(stdout)
    receipt["finished_utc"] = now()
    receipt["exit_code"] = exit_code
    receipt["csv_sha256"] = sha256(stdout)
    receipt["stderr"] = stderr.decode("utf-8", "replace").strip().splitlines()
    receipt["gpu_after"] = guarded(gpu)
    current = guarded(lambda: identity(check_clean=False)["source"])
    receipt["source_unchanged"] = current == receipt["source"]
    status = "passed" if exit_code == 0 else "failed"
    if status == "passed":
        try:
            report_pinned.validate(list(csv.DictReader(stdout.decode("utf-8-sig").splitlines())))
        except (ValueError, UnicodeDecodeError) as error:
            status, receipt["validation_error"] = "invalid", str(error)
    if status == "passed" and not (receipt["source_unchanged"] and receipt["gpu_free_confirmed"]):
        status = "unqualified"
    receipt["status"] = status
    (out_dir / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary", type=Path, help="the built hbr_cuda_pinned_bench")
    parser.add_argument("--build-dir", type=Path, required=True, help="the CMake build that produced it")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "evidence")
    parser.add_argument("--confirm-gpu-free", action="store_true",
                        help="I checked privately that no game or other GPU workload is running")
    args = parser.parse_args(argv)
    receipt = collect([str(args.binary)], args.binary, args.out_dir, args.build_dir, args.confirm_gpu_free)
    print(f"{receipt['status']}: exit {receipt['exit_code']}, receipt in {args.out_dir / RECEIPT}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
