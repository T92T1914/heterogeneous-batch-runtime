"""Exercise installed CMake discovery and a completed CPU consumer.

The optional CUDA capability is checked at configuration only. No GPU executable
is launched by this check, including when --cuda-built is supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile


def verify(prefix: Path, output: Path, cmake: str, generator: str | None,
           architecture: str | None, cuda_built: bool) -> None:
    configure = [cmake]
    if generator:
        configure += ["-G", generator]
    if architecture:
        configure += ["-A", architecture]
    package = prefix / "lib" / "cmake" / "hbr"
    if not (package / "hbrConfig.cmake").is_file():
        raise ValueError("prefix must contain an installed hbrConfig.cmake")

    def run(name: str, argv: list[str], rejection: str | None = None) -> None:
        result = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120, check=False)
        (output / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        if rejection is not None:
            if result.returncode == 0 or rejection not in result.stdout:
                raise AssertionError(f"{name}: expected configure rejection containing {rejection!r}\n{result.stdout}")
        elif result.returncode != 0:
            raise AssertionError(f"{name}: command exited {result.returncode}\n{result.stdout}")
        print(f"{name}: passed")

    def probe(name: str, body: str, rejection: str | None = None) -> None:
        source = output / name
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.24)\n'
            f'project({name} LANGUAGES CXX)\n' + body, encoding="utf-8")
        run(name, configure + ["-S", str(source), "-B", str(output / f"{name}-build"),
                               f"-Dhbr_DIR={package}"], rejection)

    if not cuda_built:
        probe("required_cuda", "find_package(hbr CONFIG REQUIRED COMPONENTS cuda)\n",
              "The hbr cuda component was not built")
    else:
        probe("required_cuda", "find_package(hbr CONFIG REQUIRED COMPONENTS cuda)\n"
              "if(NOT hbr_cuda_FOUND OR NOT TARGET hbr::hbr_cuda)\n"
              '  message(FATAL_ERROR "Compiled CUDA discovery lost")\nendif()\n')
    for component in ("hip", "sycl", "CUDA"):
        name = "required_uppercase_cuda" if component == "CUDA" else f"required_{component}"
        probe(name, f"set(hbr_{component}_FOUND TRUE)\n"
              f"find_package(hbr CONFIG REQUIRED COMPONENTS {component})\n",
              f"Unsupported hbr component '{component}'")
    probe("required_cpu", "find_package(hbr CONFIG REQUIRED COMPONENTS cpu)\n"
          "if(NOT hbr_cpu_FOUND OR NOT TARGET hbr::hbr)\n"
          '  message(FATAL_ERROR "CPU discovery lost")\nendif()\n')
    wrong_cuda = "NOT hbr_cuda_FOUND" if cuda_built else "hbr_cuda_FOUND"
    probe("optional_backends", "find_package(hbr CONFIG REQUIRED COMPONENTS cpu OPTIONAL_COMPONENTS cuda hip sycl)\n"
          "if(NOT hbr_FOUND OR NOT hbr_cpu_FOUND OR hbr_hip_FOUND OR hbr_sycl_FOUND)\n"
          '  message(FATAL_ERROR "Optional backend discovery invalid")\nendif()\n'
          f"if({wrong_cuda})\n"
          '  message(FATAL_ERROR "Compiled CUDA capability invalid")\nendif()\n')
    probe("legacy", "find_package(hbr CONFIG REQUIRED)\n"
          "if(NOT TARGET hbr::hbr)\n"
          '  message(FATAL_ERROR "Legacy discovery lost")\nendif()\n')
    probe("rediscovery", "find_package(hbr CONFIG QUIET COMPONENTS unsupported)\n"
          "if(hbr_FOUND)\n"
          '  message(FATAL_ERROR "Unsupported backend accepted")\nendif()\n'
          "find_package(hbr CONFIG REQUIRED COMPONENTS cpu)\n"
          "if(NOT hbr_FOUND OR hbr_NOT_FOUND_MESSAGE)\n"
          '  message(FATAL_ERROR "Earlier failure poisoned CPU discovery")\nendif()\n')
    consumer = Path(__file__).resolve().parent / "consumer"
    build = output / "consumer-build"
    run("consumer_configure", configure + ["-S", str(consumer), "-B", str(build),
                                          f"-Dhbr_DIR={package}", "-DHBR_CONSUMER_CUDA=OFF"])
    run("consumer_build", [cmake, "--build", str(build), "--config", "Release", "--parallel", "2"])
    executable = build / ("Release/consumer.exe" if (build / "Release/consumer.exe").exists()
                          else "consumer.exe" if (build / "consumer.exe").exists() else "consumer")
    run("consumer_run", [str(executable)])
    if not cuda_built:
        run("consumer_requires_cuda", configure + ["-S", str(consumer), "-B", str(output / "cuda-consumer-build"),
                                                   f"-Dhbr_DIR={package}", "-DHBR_CONSUMER_CUDA=ON"],
            "The hbr cuda component was not built")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--generator")
    parser.add_argument("--architecture")
    parser.add_argument("--cuda-built", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        verify(args.prefix.resolve(), args.output.resolve(), args.cmake,
               args.generator, args.architecture, args.cuda_built)
    else:
        with tempfile.TemporaryDirectory(prefix="hbr-components-") as temporary:
            verify(args.prefix.resolve(), Path(temporary), args.cmake,
                   args.generator, args.architecture, args.cuda_built)


if __name__ == "__main__":
    main()
