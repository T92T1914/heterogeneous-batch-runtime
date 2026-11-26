"""Run the pinned input benchmark once and write its samples and execution receipt.

The receipt names the source, the protocol, the binary and the compilers, and
records the GPU state before and after the run and the idle readings the
benchmark prints. It records no process list, because the measuring computer is
a shared desktop. The run goes ahead at the usual desktop load, and the person
who starts it confirms with --confirm-desktop-load, after checking the
processes privately, that no game is running.

Given a protocol with a pcie_link entry, such as docs/pinned-gen4-protocol.json,
it also reads the card's PCIe link with nvidia-smi for as long as the benchmark
runs and keeps every reading, and the run passes only at the link that protocol
names. Such a protocol must keep the plan of docs/pinned-protocol.json, because
the benchmark and the report are built for that plan.

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
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_pinned_plan  # noqa: E402
import report_pinned  # noqa: E402

PROTOCOL = ROOT / "docs" / "pinned-protocol.json"
GPU_FIELDS = "name,driver_version,memory.total,utilization.gpu,memory.used,temperature.gpu"
TIMEOUT = json.loads(PROTOCOL.read_text(encoding="utf-8"))["limits"]["measurement_timeout_seconds"]
# What a repeat of the pinned protocol may change. Everything else must equal docs/pinned-protocol.json.
DESCRIPTIVE = {"purpose", "amendment", "repeats", "evidence_name", "pcie_link", "preconditions", "summary",
               "environment", "decision", "historical_evidence"}


def evidence_names(protocol):
    """The sample and receipt file names a protocol's run writes."""
    name = protocol.get("evidence_name", "pinned")
    return f"{name}.csv", f"{name}-execution.json"


def plan_differences(protocol, base=None):
    """Fields outside DESCRIPTIVE where a protocol differs from docs/pinned-protocol.json."""
    base = json.loads(PROTOCOL.read_text(encoding="utf-8")) if base is None else base
    return sorted(key for key in set(protocol) | set(base)
                  if key not in DESCRIPTIVE and protocol.get(key) != base.get(key))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def source_identity(check_clean=True, protocol=PROTOCOL):
    """Revision, tree and the commit that last changed the protocol. A dirty tree is refused."""
    if check_clean and git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("The working tree has uncommitted changes. Commit or set them aside first.")
    relative = Path(protocol).resolve().relative_to(ROOT).as_posix()
    commit = git("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise SystemExit(f"{relative} is not committed. A protocol is committed before any sample.")
    return {"source": git("rev-parse", "HEAD"), "source_tree": git("rev-parse", "HEAD^{tree}"),
            "protocol_commit": commit}


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


class LinkWatch:
    """nvidia-smi reading the protocol's link fields at its interval, from start() until stop()."""

    def __init__(self, link):
        self.command = ["nvidia-smi", "--query-gpu=" + ",".join(link["fields"]),
                        "--format=csv,noheader,nounits", "-lms", str(link["interval_milliseconds"])]

    def start(self):
        self.output = tempfile.TemporaryFile()
        self.process = subprocess.Popen(self.command, stdout=self.output, stderr=subprocess.STDOUT)

    def stop(self):
        """Stop nvidia-smi and return the lines it printed."""
        self.process.terminate()
        self.process.wait(timeout=30)
        self.output.seek(0)
        lines = self.output.read().decode("utf-8", "replace").splitlines()
        self.output.close()
        return [line.strip() for line in lines if line.strip()]


def guarded(step):
    """Run a step after the benchmark without letting its failure lose the samples."""
    try:
        return step()
    except Exception as error:  # noqa: BLE001 - recorded, never silent
        return f"unavailable: {type(error).__name__}: {error}"


def collect(command, binary, out_dir, build_dir, desktop_load, identity=source_identity, gpu=gpu_state,
            protocol_path=PROTOCOL, link_watch=LinkWatch):
    """Run the plan check and the benchmark, then write the CSV and the receipt. Returns the receipt."""
    out_dir, protocol_path = Path(out_dir), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    changed = plan_differences(protocol)
    if changed:
        raise SystemExit(f"{protocol_path.name} differs from docs/pinned-protocol.json in {', '.join(changed)}. "
                         "Nothing ran.")
    samples, receipt_name = evidence_names(protocol)
    if (out_dir / samples).exists() or (out_dir / receipt_name).exists():
        raise SystemExit(f"{out_dir / samples} or its receipt exists. Rename and keep that attempt first.")
    receipt = {"schema_version": 2, **identity(protocol=protocol_path),
               "protocol_sha256": sha256(protocol_path.read_bytes()),
               "binary_sha256": sha256(Path(binary).read_bytes()),
               "compilers": compilers(binary, build_dir), "cpu": platform.processor(), "os": platform.platform(),
               "desktop_load_confirmed": bool(desktop_load)}
    link = protocol.get("pcie_link")
    if link:
        receipt["schema_version"] = 3
        receipt["pcie_link"] = {"generation": link["generation"], "width": link["width"]}
    plan = subprocess.run([*command, "--plan"], capture_output=True, text=True, timeout=30)
    if plan.returncode != 0 or check_pinned_plan.differences(json.loads(plan.stdout), protocol):
        raise SystemExit(f"The binary's plan differs from {protocol_path.name}. Nothing ran.")
    sequence = subprocess.run([*command, "--sequence"], capture_output=True, text=True, timeout=30)
    if sequence.returncode != 0 or report_pinned.sequence_differences(
            list(csv.DictReader(sequence.stdout.splitlines()))):
        raise SystemExit("The binary's run order differs from the one the report expects. Nothing ran.")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt["gpu_before"] = gpu()
    watch = link_watch(link) if link else None
    # None once nvidia-smi started, or the text of the error that kept it from starting.
    start_error = guarded(watch.start) if watch else None
    receipt["started_utc"] = now()
    try:
        run = subprocess.run([*command, "--run"], capture_output=True, timeout=TIMEOUT)
        exit_code, stdout, stderr = run.returncode, run.stdout, run.stderr
    except subprocess.TimeoutExpired as expired:
        exit_code, stdout, stderr = "timeout", expired.stdout or b"", expired.stderr or b""
    finally:
        link_lines = None
        if watch:
            link_lines = [start_error] if start_error is not None else guarded(watch.stop)
    # The samples reach disk before any step that could still fail.
    (out_dir / samples).write_bytes(stdout)
    receipt["finished_utc"] = now()
    receipt["exit_code"] = exit_code
    receipt["csv_sha256"] = sha256(stdout)
    receipt["stderr"] = stderr.decode("utf-8", "replace").strip().splitlines()
    receipt["gpu_idle_readings"] = report_pinned.idle_readings(receipt)
    receipt["gpu_after"] = guarded(gpu)
    if watch:
        readings, unread = [], []
        # A string here is the text of an error from stop(), kept like any line that is no reading.
        for line in link_lines if isinstance(link_lines, list) else [link_lines]:
            try:
                readings.append(report_pinned.link_reading(line))
            except ValueError:
                unread.append(line)
        receipt["pcie_link_readings"] = readings
        if unread:
            receipt["pcie_link_unread"] = unread
    current = guarded(lambda: identity(check_clean=False, protocol=protocol_path)["source"])
    receipt["source_unchanged"] = current == receipt["source"]
    status = "passed" if exit_code == 0 else "failed"
    if status == "passed":
        try:
            report_pinned.validate(list(csv.DictReader(stdout.decode("utf-8-sig").splitlines())))
            points = [reading["point"] for reading in receipt["gpu_idle_readings"]]
            if points != report_pinned.IDLE_POINTS:
                raise ValueError(f"idle readings {points}, expected {report_pinned.IDLE_POINTS}")
        except (ValueError, UnicodeDecodeError) as error:
            status, receipt["validation_error"] = "invalid", str(error)
    if status == "passed" and not (receipt["source_unchanged"] and receipt["desktop_load_confirmed"]):
        status = "unqualified"
    if status == "passed" and watch:
        problem = report_pinned.link_problem(receipt["pcie_link_readings"], link["generation"], link["width"])
        if problem:
            status, receipt["link_problem"] = "unqualified", problem
    receipt["status"] = status
    (out_dir / receipt_name).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("binary", type=Path, help="the built hbr_cuda_pinned_bench")
    parser.add_argument("--build-dir", type=Path, required=True, help="the CMake build that produced it")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "evidence")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL,
                        help="docs/pinned-protocol.json, or a repeat of it such as docs/pinned-gen4-protocol.json")
    parser.add_argument("--confirm-desktop-load", action="store_true",
                        help="I checked privately that no game is running and only the usual desktop applications use the GPU")
    args = parser.parse_args(argv)
    receipt = collect([str(args.binary)], args.binary, args.out_dir, args.build_dir, args.confirm_desktop_load,
                      protocol_path=args.protocol)
    receipt_name = evidence_names(json.loads(args.protocol.read_text(encoding="utf-8")))[1]
    print(f"{receipt['status']}: exit {receipt['exit_code']}, receipt in {args.out_dir / receipt_name}")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
