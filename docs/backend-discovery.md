# Installed backend discovery

An application that requires CUDA must fail at package discovery when it receives an installation built without CUDA. Previously, `find_package(hbr CONFIG REQUIRED COMPONENTS cuda)` succeeded without a CUDA target because the package ignored components. That deferred a missing backend to a later target error, or let an application continue without its required capability.

The installed CMake package now provides three components:

| Component | Compiled interface | Discovery meaning |
| --- | --- | --- |
| `cpu` | `hbr::hbr` with scalar and optimized operations | Always part of an installation |
| `cuda` | `hbr::hbr_cuda` with synchronous operations and the reusable histogram | Present only when the producer built `HBR_CUDA=ON` |
| `sycl` | `hbr::hbr_sycl`, a shared library that selects a GPU and reports it, with no workload yet | Present only when the producer built `HBR_SYCL=ON` |

```cmake
find_package(hbr 0.1 CONFIG REQUIRED COMPONENTS cpu cuda)
target_link_libraries(application PRIVATE hbr::hbr hbr::hbr_cuda)
```

For an optional backend, use `find_package(hbr CONFIG REQUIRED COMPONENTS cpu OPTIONAL_COMPONENTS cuda sycl)` and inspect `hbr_cuda_FOUND` or `hbr_sycl_FOUND` before selecting its target. An unavailable optional component leaves CPU discovery usable. An unsupported required component fails with its name in the diagnostic. Component names are case sensitive. A request without components retains the existing behavior of importing the compiled targets and finding their dependencies. An installation built with CUDA still requires its CUDA toolkit dependency even when the consumer requests only CPU. An installation built with SYCL needs no SYCL compiler in the application, because `hbr/sycl.hpp` includes no SYCL header and the shared library carries its own device code. The DPC++ runtime libraries must be on the loader path when the application links and runs.

Component discovery describes the installed build. It does not establish device availability, architecture compatibility, numerical correctness or physical completion. CUDA initialization and execution retain their existing explicit error and ownership contracts. Discovery never falls back from a required GPU component to CPU execution. The separate `tests/consumer` application runs CPU work by default and requires `HBR_CONSUMER_CUDA=ON` or `HBR_CONSUMER_SYCL=ON` to include the checks of that backend.

## The SYCL build

The SYCL backend is built and tested only on Linux with the DPC++ compiler. Configure the whole project with that compiler, because CMake uses one C++ compiler for every target. `-fsycl` is applied to the SYCL library alone, and the other targets compile as ordinary C++.

```sh
cmake -S . -B build/sycl -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=<dpcpp>/bin/clang++ \
  -DHBR_SYCL=ON -DHBR_SYCL_FLAGS=--cuda-path=<cuda-toolkit>
cmake --build build/sycl --parallel 2
ctest --test-dir build/sycl --output-on-failure
```

`HBR_SYCL_TARGETS` sets `-fsycl-targets` and defaults to `nvidia_gpu_sm_89`, which compiles the device code ahead of time for an RTX 4090. `HBR_SYCL_FLAGS` adds options such as the CUDA toolkit path that the NVIDIA target needs for `ptxas` and libdevice. In this revision `hbr::sycl_device()` selects a GPU with the SYCL GPU selector, which honors `ONEAPI_DEVICE_SELECTOR`, and reports its backend, platform, name, driver, memory and its `fp64` and `atomic64` aspects. It never selects a CPU device and throws `std::runtime_error` when no GPU is offered. It runs none of the three workloads.

Tests whose names start with `sycl_` need a SYCL GPU. `host_sycl_selection` hides every GPU through `ONEAPI_DEVICE_SELECTOR` and expects the visible rejection, so it runs on any host. The hosted [`SYCL compile`](../.github/workflows/sycl-compile.yml) workflow fetches the DPC++ 7.2.0 release and the CUDA 13.2.2 device tools by SHA-256, builds the backend for `sm_89` and checks that the library carries that device image. It then runs every test whose name does not start with `sycl_`, followed by the installed discovery check. It executes nothing on a device.

## Portable backend boundary

There is no HIP/ROCm implementation in this revision, and a required `hip` request rejects visibly. The SYCL build targets NVIDIA hardware. No AMD/Intel execution or speedup across vendors is implied by this contract.

Each SYCL workload, and any later backend, must preserve the [numerical contract](contracts.md): finite binary64 input, the declared reduction range check, exact uint64 histogram counts, partial tiles, copied stencil borders and independently owned host results. Its completed host call must retain borrowed immutable input through synchronization and failure cleanup. Reusable resources need a bounded allocation policy, explicit owner/device rules, nonwrapping request identity and retirement after execution failure. Logical cancellation and staleness cannot release physically active storage.

Each workload of a backend therefore requires a supported compiler/runtime, suitable device capabilities and its own executed correctness and lifecycle evidence. Translating a kernel or creating an imported target does not establish those results. The existing [negative CUDA comparisons](results-2026-09-29.md), [reuse results](reuse-results.md) and separate ONNX consumer evidence remain unchanged. The SYCL build adds no kernel for a workload and no performance claim.

For the SYCL workloads, the [device contract](https://github.khronos.org/SYCL_Reference/iface/device.html) exposes `fp64` and `atomic64` capabilities that matter to these binary64 workloads and an implementation using 64 bit histogram atomics, and `hbr::sycl_device()` reports both. Its [queue contract](https://github.khronos.org/SYCL_Reference/iface/queue.html#wait-and-throw) requires explicit completion and asynchronous error handling. A portable implementation could choose different histogram arithmetic, but must preserve exact counts and report failure before adopting a result. These are design prerequisites, not executed workload evidence.

The configuration uses CMake's documented [handling of required components](https://cmake.org/cmake/help/latest/module/CMakePackageConfigHelpers.html) and [semantics of optional components](https://cmake.org/cmake/help/latest/command/find_package.html). Run `python tests/package_components.py --prefix <installed-prefix> --cmake <cmake-executable>` to exercise actual installed discovery, rejection and the CPU consumer. Add `--cuda-built` when the prefix holds an installation built with CUDA. The check then expects `COMPONENTS cuda` to succeed with an `hbr::hbr_cuda` target, where the default form expects that request to be rejected. Add `--sycl-built` for an installation built with SYCL. The check then expects `COMPONENTS sycl` to succeed with an `hbr::hbr_sycl` target, and it configures and links the consumer with `HBR_CONSUMER_SYCL=ON` using the default compiler, without running it. The check builds at most two jobs, executes no GPU work and retains the output of each case when `--output` is supplied.
