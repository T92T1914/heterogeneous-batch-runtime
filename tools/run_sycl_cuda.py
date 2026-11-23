"""Run the SYCL and CUDA comparison once in its WSL distro and write its samples and receipt.

The runner itself runs on Windows, where git can read the worktree. It starts the
Linux binary with `wsl -d <distro> --exec` and reads the binary and the CMake records
of its build through the distro's \\\\wsl.localhost share. Before anything runs, it
requires that the build was configured from this checkout and that ninja has no work
left in it, so the binary matches the committed source. The receipt records that the
build came from this checkout but not the checkout's path, which stays private. It names the source,
the protocol, the binary, the command, the compilers, the WSL kernel and the host
compiler for nvcc, and records the GPU state before and after the run and the idle
readings the benchmark prints. It records no process list, because the measuring
computer is a shared desktop. The run goes ahead at the usual desktop load, and the
person who starts it confirms with --confirm-desktop-load, after checking the processes
privately, that no game is running.

A run that fails, times out or breaks the protocol is still written, with its status
and its error text. The time limit is enforced inside the distro, so a hung run stops
there and does not keep the GPU. The runner refuses to write over an earlier attempt,
so every attempt has to be renamed and kept before the next one starts.
"""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_sycl_cuda_plan  # noqa: E402
import report_sycl_cuda  # noqa: E402
from run_pinned import compilers, git, gpu_state, guarded, now, sha256  # noqa: E402

PROTOCOL = ROOT / "docs" / "sycl-cuda-protocol.json"
TIMEOUT = json.loads(PROTOCOL.read_text(encoding="utf-8"))["limits"]["measurement_timeout_seconds"]
# Python waits this much longer than the limit inside the distro, as a last resort.
GRACE = 30
SAMPLES, RECEIPT = "sycl-cuda.csv", "sycl-cuda-execution.json"
DISTRO = "HBR-SYCL-20261115"
LIBRARY_PATH = "/opt/dpcpp-7.2.0/lib:/usr/lib/wsl/lib"


def source_identity(check_clean=True):
    """Revision, tree and the commit that last changed the protocol. A dirty tree is refused."""
    if check_clean and git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("The working tree has uncommitted changes. Commit or set them aside first.")
    return {"source": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
            "protocol_commit": git("log", "-1", "--format=%H", "--", "docs/sycl-cuda-protocol.json")}


def share(distro, path):
    """The Windows path of a file inside the distro, through the \\\\wsl.localhost share."""
    return Path(f"\\\\wsl.localhost\\{distro}") / PurePosixPath(path).relative_to("/")


def windows_path(path):
    """The Windows path of a /mnt/<drive>/ path, or None for a path that stays inside the distro."""
    parts = PurePosixPath(path).parts
    if len(parts) < 3 or parts[:2] != ("/", "mnt") or len(parts[2]) != 1:
        return None
    return PureWindowsPath(f"{parts[2].upper()}:\\", *parts[3:])


def in_distro(distro, *command):
    # --exec runs the command with no Linux shell in between, so no argument is reinterpreted.
    return ["wsl", "-d", distro, "--exec", *command]


def cache_entries(build_dir, names):
    """Selected entries of the build's CMakeCache.txt, read through a path this process can open."""
    found = {}
    for line in (Path(build_dir) / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        name, _, rest = line.partition(":")
        if name in names and "=" in rest:
            found[name] = rest.split("=", 1)[1]
    return found


def distro_build(distro, build_dir):
    """Where the distro build was configured from, whether ninja has work left and its nvcc host compiler."""
    def first_line(*command):
        result = subprocess.run(in_distro(distro, *command), capture_output=True, text=True, timeout=120)
        lines = result.stdout.strip().splitlines()
        return lines[0] if result.returncode == 0 and lines else f"exit {result.returncode}"
    entries = cache_entries(share(distro, build_dir), ("CMAKE_HOME_DIRECTORY", "CMAKE_CUDA_HOST_COMPILER"))
    host = entries.get("CMAKE_CUDA_HOST_COMPILER", "")
    return {"build_source_dir": entries.get("CMAKE_HOME_DIRECTORY", ""),
            "build_dry_run": first_line("cmake", "--build", build_dir, "--", "-n"),
            "cuda_host_compiler": f"{host}: {first_line(host, '--version')}" if host else "not recorded",
            "wsl_distro": distro, "wsl_kernel": first_line("uname", "-r")}


def check_build(state, root=ROOT):
    """Refuse a build configured from another source tree or one that ninja would still change.

    Windows paths compare without regard to case, so C:\\Users and /mnt/c/users match.
    """
    source = windows_path(state.get("build_source_dir", ""))
    if source is None or source != PureWindowsPath(root):
        raise SystemExit(f"The build was configured from {state.get('build_source_dir')!r}, not this checkout. Nothing ran.")
    if state.get("build_dry_run") != "ninja: no work to do.":
        raise SystemExit("The build is not up to date with this checkout. Build it again first. Nothing ran.")


def collect(command, binary, out_dir, build_dir, desktop_load, identity=source_identity, gpu=gpu_state,
            build_state=None):
    """Run the plan check and the benchmark, then write the CSV and the receipt. Returns the receipt.

    binary and build_dir are paths this process can read. command starts the binary.
    build_state returns what distro_build returns. Without it, no build check runs, as in tests.
    """
    out_dir = Path(out_dir)
    if (out_dir / SAMPLES).exists() or (out_dir / RECEIPT).exists():
        raise SystemExit(f"{out_dir / SAMPLES} or its receipt exists. Rename and keep that attempt first.")
    receipt = {"schema_version": 2, **identity(),
               "protocol_sha256": sha256(PROTOCOL.read_bytes()),
               "binary_sha256": sha256(Path(binary).read_bytes()),
               "command": list(command),
               "compilers": compilers(binary, build_dir),
               "cpu": platform.processor(), "os": platform.platform(),
               "desktop_load_confirmed": bool(desktop_load)}
    if build_state is not None:
        state = build_state()
        check_build(state)
        # The check above compares the full path. The receipt keeps only its outcome, because
        # the path of the checkout on the measuring computer is private and the receipt is not.
        state = {key: value for key, value in state.items() if key != "build_source_dir"}
        receipt.update(state, build_source="this checkout")
    plan = subprocess.run([*command, "--plan"], capture_output=True, text=True, timeout=120)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if plan.returncode != 0 or check_sycl_cuda_plan.differences(json.loads(plan.stdout), protocol):
        raise SystemExit("The binary's plan differs from docs/sycl-cuda-protocol.json. Nothing ran.")
    sequence = subprocess.run([*command, "--sequence"], capture_output=True, text=True, timeout=120)
    if sequence.returncode != 0 or report_sycl_cuda.sequence_differences(
            list(csv.DictReader(sequence.stdout.splitlines()))):
        raise SystemExit("The binary's run order differs from the one the report expects. Nothing ran.")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt["gpu_before"] = gpu()
    receipt["started_utc"] = now()
    try:
        run = subprocess.run([*command, "--run"], capture_output=True, timeout=TIMEOUT + GRACE)
        exit_code, stdout, stderr = run.returncode, run.stdout, run.stderr
    except subprocess.TimeoutExpired as expired:
        exit_code, stdout, stderr = "timeout", expired.stdout or b"", expired.stderr or b""
    # The samples reach disk before any step that could still fail.
    (out_dir / SAMPLES).write_bytes(stdout)
    receipt["finished_utc"] = now()
    receipt["exit_code"] = exit_code
    receipt["csv_sha256"] = sha256(stdout)
    receipt["stderr"] = stderr.decode("utf-8", "replace").strip().splitlines()
    receipt["gpu_idle_readings"] = report_sycl_cuda.idle_readings(receipt)
    receipt["gpu_after"] = guarded(gpu)
    current = guarded(lambda: identity(check_clean=False)["source"])
    receipt["source_unchanged"] = current == receipt["source"]
    status = "passed" if exit_code == 0 else "failed"
    if status == "passed":
        try:
            report_sycl_cuda.validate(list(csv.DictReader(stdout.decode("utf-8-sig").splitlines())))
            points = [reading["point"] for reading in receipt["gpu_idle_readings"]]
            if points != report_sycl_cuda.IDLE_POINTS:
                raise ValueError(f"idle readings {points}, expected {report_sycl_cuda.IDLE_POINTS}")
        except (ValueError, UnicodeDecodeError) as error:
            status, receipt["validation_error"] = "invalid", str(error)
    if status == "passed" and not (receipt["source_unchanged"] and receipt["desktop_load_confirmed"]):
        status = "unqualified"
    receipt["status"] = status
    (out_dir / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary", help="hbr_sycl_cuda_bench, as a path inside the distro")
    parser.add_argument("--build-dir", required=True, help="the CMake build that produced it, inside the distro")
    parser.add_argument("--distro", default=DISTRO)
    parser.add_argument("--library-path", default=LIBRARY_PATH, help="LD_LIBRARY_PATH for the binary")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "evidence")
    parser.add_argument("--confirm-desktop-load", action="store_true",
                        help="I checked privately that no game is running and only the usual desktop applications use the GPU")
    args = parser.parse_args(argv)
    # timeout stops the benchmark inside the distro at the protocol's limit and kills it 5 seconds later.
    command = in_distro(args.distro, "timeout", "-k", "5", str(TIMEOUT), "env",
                        f"LD_LIBRARY_PATH={args.library_path}", args.binary)
    receipt = collect(command, share(args.distro, args.binary), args.out_dir, share(args.distro, args.build_dir),
                      args.confirm_desktop_load, build_state=lambda: distro_build(args.distro, args.build_dir))
    print(f"{receipt['status']}: exit {receipt['exit_code']}, receipt in {args.out_dir / RECEIPT}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
