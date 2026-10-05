# Installed backend discovery

An application that requires CUDA must fail at package discovery when it receives a CPU-only installation. Previously, `find_package(hbr CONFIG REQUIRED COMPONENTS cuda)` succeeded without a CUDA target because the package ignored components. That deferred a missing backend to a later target error, or let an application continue without its required capability.

The installed CMake package now provides two components:

| Component | Compiled interface | Discovery meaning |
| --- | --- | --- |
| `cpu` | `hbr::hbr` with scalar and optimized operations | Always part of an installation |
| `cuda` | `hbr::hbr_cuda` with synchronous operations and the reusable histogram | Present only when the producer built `HBR_CUDA=ON` |

```cmake
find_package(hbr 0.1 CONFIG REQUIRED COMPONENTS cpu cuda)
target_link_libraries(application PRIVATE hbr::hbr hbr::hbr_cuda)
```

For an optional backend, use `find_package(hbr CONFIG REQUIRED COMPONENTS cpu OPTIONAL_COMPONENTS cuda)` and inspect `hbr_cuda_FOUND` before selecting its target. An unavailable optional component leaves CPU discovery usable. An unsupported required component fails with its name in the diagnostic. Component names are case sensitive. A request without components retains the existing behavior of importing the compiled targets and finding their dependencies. A CUDA-enabled installation still requires its CUDA toolkit dependency even when the consumer requests only CPU.

Component discovery describes the installed build. It does not establish device availability, architecture compatibility, numerical correctness or physical completion. CUDA initialization and execution retain their existing explicit error and ownership contracts. Discovery never falls back from a required GPU component to CPU execution. The separate `tests/consumer` application runs CPU work by default and requires `HBR_CONSUMER_CUDA=ON` to include its CUDA checks.

## Portable backend boundary

There is no HIP/ROCm or SYCL/oneAPI implementation in this revision. Required `hip` and `sycl` requests reject visibly. No target, supported device, AMD/Intel execution or cross-vendor speedup is implied by this contract.

A future backend must preserve the [numerical contract](contracts.md): finite binary64 input, the declared reduction range check, exact uint64 histogram counts, partial tiles, copied stencil borders and independently owned host results. Its completed host call must retain borrowed immutable input through synchronization and failure cleanup. Reusable resources need a bounded allocation policy, explicit owner/device rules, nonwrapping request identity and retirement after execution failure. Logical cancellation and staleness cannot release physically active storage.

Implementing another backend therefore requires a supported compiler/runtime, suitable device capabilities and its own executed correctness and lifecycle evidence. Translating a kernel or creating an imported target does not establish those results. The existing [negative CUDA comparisons](results-2026-09-29.md), [reuse results](reuse-results.md) and separate ONNX consumer evidence remain unchanged. This milestone makes unavailable capabilities explicit for consumers. It introduces no new kernel, performance claim or vendor adapter.

For a future SYCL implementation, the [device contract](https://github.khronos.org/SYCL_Reference/iface/device.html) exposes `fp64` and `atomic64` capabilities that matter to these binary64 workloads and an implementation using 64-bit histogram atomics. Its [queue contract](https://github.khronos.org/SYCL_Reference/iface/queue.html#wait-and-throw) requires explicit completion and asynchronous error handling. A portable implementation could choose different histogram arithmetic, but must preserve exact counts and report failure before adopting a result. These are design prerequisites, not executed SYCL evidence.

The configuration uses CMake's documented [required-component handling](https://cmake.org/cmake/help/latest/module/CMakePackageConfigHelpers.html) and [optional-component semantics](https://cmake.org/cmake/help/latest/command/find_package.html). Run `python tests/package_components.py --prefix <installed-prefix> --cmake <cmake-executable>` to exercise actual installed discovery, rejection and the CPU consumer. The check builds at most two jobs, executes no GPU work and retains its per-case output when `--output` is supplied.
