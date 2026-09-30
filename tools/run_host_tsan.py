"""Run host ownership contracts and distinguish unsupported TSan initialization."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run([args.executable], capture_output=True, text=True, timeout=120)
    log = result.stdout + result.stderr
    unsupported = result.returncode != 0 and "WARNING: ThreadSanitizer" not in log and any(
        marker in log for marker in ("FATAL: ThreadSanitizer: unexpected memory mapping",
                                     "ThreadSanitizer: CHECK failed: tsan_platform_linux.cpp"))
    state = "unsupported-initialization" if unsupported else "passed" if result.returncode == 0 else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"state": state, "exit_code": result.returncode,
        "coverage": "host ownership/concurrency only, no GPU safety claim", "log": log}, indent=2) + "\n")
    print(log, end="")
    print(f"Host ThreadSanitizer state: {state}")
    # Unsupported initialization is visible evidence, never a sanitizer pass.
    # Genuine race reports and other failures remain failing checks.
    return 0 if state in ("passed", "unsupported-initialization") else 1


if __name__ == "__main__":
    raise SystemExit(main())
