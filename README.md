# Heterogeneous Batch Runtime

I am building a small native runtime to compare three numerical workloads across CPU and GPU execution. The first implementation provides C++20 scalar references, optimized CPU paths and a narrow Python interface. The workloads expose different costs: masked reduction, tiled histogram counts and a 3 by 3 image stencil.

The runtime keeps submitted work alive until it actually finishes. Cancelling a running operation records a request. It does not pretend that the operation stopped or that its storage can already be reused.

## Build and inspect

Use CMake 3.24 or later and a C++20 compiler. Build with a conservative parallel limit on a shared machine.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 2
ctest --test-dir build -C Release --output-on-failure
```

On GCC or Clang, configure a separate Debug build with `-DHBR_SANITIZERS=ON` for AddressSanitizer and UndefinedBehaviorSanitizer. Do not treat a configured job as executed evidence.

The optional Python package uses pybind11 and NumPy:

```sh
python -m pip install .
python -m pytest tests/python
```

```python
import numpy as np
from heterogeneous_batch_runtime import masked_reduce

values = np.array([2.0, -3.0, 5.0], dtype=np.float64)
mask = np.array([1, 0, 1], dtype=np.uint8)
print(masked_reduce(values, mask))  # 7.0
```

`tile_histogram` accepts a two-dimensional uint8 array and tile height and width. `stencil3x3` accepts a two-dimensional float64 array and copies its outer border. Every operation accepts `backend="scalar"` or `backend="optimized"` and a thread count. Python inputs are snapshotted before native work releases the GIL. No implicit dtype conversion occurs.

## What the implementation compares

The CPU path uses runtime-dispatched AVX2 for reduction and stencil work, with a portable fallback. Histogram workers own separate tiles and use private bin banks. Parallelism is bounded by the requested thread count. The [contracts](docs/contracts.md) define layout, edge behavior, cancellation and numerical tolerances.

The [CPU protocol](docs/cpu-protocol.json) fixes the initial measurement conditions. It retains the first invocation and every warm sample, including slower optimized cases. Native operation time includes validation and allocation. It excludes Python conversion, queueing and GPU transfers. Those require separate measurements.

The optional CUDA backend uses a completed synchronous C++ interface. Configure with `-DHBR_CUDA=ON` and the CUDA architecture for the actual device. Its tests execute on a compatible GPU. The [GPU protocol](docs/cuda-protocol.json) compares global and shared histogram atomics and retains transfer, kernel and completed host timings separately. The Python wheel remains a CPU interface.

Native installation exports `hbr::hbr` and, when enabled, `hbr::hbr_cuda`. Configure `tests/consumer` against the installed prefix to check a separate application. CUDA consumers also need a compatible CUDA toolkit. No toolkit is bundled.

Executed results are recorded separately from source capabilities. Cross-vendor comparisons are not established by one CUDA implementation. No hardware-independent speedup is claimed.

## Source and dependencies

This runtime is independently authored for these numerical examples. It contains no source from private applications. C++ standard library facilities provide the runtime. pybind11, scikit-build-core and NumPy are optional packaging/interface dependencies with their own licenses. No toolkit, driver, font or third-party binary is vendored.
