# Optional Python CUDA histograms

The default package builds and imports without a CUDA toolkit or GPU. Importing
`heterogeneous_batch_runtime.cuda` is also safe in that build. Creating a CUDA
context then raises an explicit unavailable error. It never falls back to CPU.

Build the optional wheel with a compatible installed CUDA toolkit and C++20
compiler. Choose the architecture for the actual device:

```sh
python -m pip install . -Ccmake.define.HBR_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=89
```

If the toolkit is outside the normal compiler search path, supply
`-Ccmake.define.CUDAToolkit_ROOT=/path/to/toolkit` and
`-Ccmake.define.CMAKE_CUDA_COMPILER=/path/to/toolkit/bin/nvcc`.
No toolkit or driver is bundled. A CPU build is the default even on a GPU host.

```python
import numpy as np
from heterogeneous_batch_runtime.cuda import CudaHistogramContext

image = np.full((17, 19), 7, dtype=np.uint8)
with CudaHistogramContext(17, 19, 4, 5, device=0) as context:
    first = context.run(image)
    image[0, 0] = 19
    second = context.run(image)
    assert first["request_id"] + 1 == second["request_id"]
    assert first["value"].sum() == image.size
```

`run` accepts an exact NumPy uint8 array with two dimensions, the configured
shape and C contiguous storage. It rejects lists, conversion requests, wrong
rank and noncontiguous layouts. Zero dimensions are supported. Tile dimensions
must be positive. The existing 64 MiB device-storage bound still applies.
Each output has shape `(tile_row_count, tile_col_count, 256)` and uint64 counts.
Returned arrays own their storage and do not alias inputs, previous results or
the context. IDs are scoped to one native context.

The public Python object belongs to its creating Python thread. Calls from
other threads are rejected. A dedicated native owner selects the requested
current CUDA device once and creates, runs, closes and destroys the native
context on that same thread. The caller's current device does not redirect it.
The private binding retains the native creating-thread/current-device checks.
Do not use the private binding directly or change its owner's current device.

`run` converts an accepted array subclass to a base ndarray and copies that
array before admission to its native owner. It does not call the subclass's
overridable copy method. The owned snapshot cannot share caller storage.
The binding copies
that owned array to a C++ vector while holding the GIL, then releases the GIL
around synchronous native execution. The input has to remain free of writes
through the copy. The GIL cannot exclude external native array writers. Caller
mutation during the snapshot is unsupported even if a Python reference stays
alive. The Adaptive adapter adds a submission snapshot so caller mutation after
successful submission returns is supported.

If an array subclass closes the context during input validation, `run` rejects
the copied input before queueing it. No result is left waiting for an owner
that has already shut down.

Native return follows upload, kernel execution, download and stream
synchronization. The result array is constructed after the GIL is reacquired.
Host asynchronous submission does not establish concurrent GPU work or transfer
overlap. Device-stage event times and the native host total are separate clocks.
The host total excludes Python copies, admission and result-array construction.
Do not add overlapping intervals or describe these observations as application
latency or a speedup experiment.

Use an explicit `close()` or a `with` block. Close waits for owner teardown,
propagates its synchronization error and is idempotent on the public owner.
The context is closed even if close raises. Execution failure preserves native
permanent retirement and drains before submitted storage is released. Input
validation failures do not execute work. Existing native fault tests exercise
the execution retirement path with synthetic host exceptions, not device loss.

Ordinary object abandonment signals and joins the dedicated owner, including
when the last reference is released by a different Python thread. This is a
fallback for ordinary garbage collection, not the primary shutdown protocol.
An in-progress method holds its object and snapshot until its synchronous
result or exception returns. Arbitrary interpreter termination, process killing
and physical device loss are outside the supported recovery contract.

## Host diagnostics

Clang Release coverage is separate from GCC/MSVC coverage. ASan/UBSan remains
available through `HBR_SANITIZERS`. A separate host-only GCC/Clang build can use
`HBR_THREAD_SANITIZER=ON`. It rejects combination with ASan/UBSan, Python, CUDA
or MSVC. The native tests include gated running cancellation, queued cancellation,
capacity, exception propagation and capture release. Host diagnostics do not
establish GPU safety.

```sh
cmake -S . -B build/clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build/clang --parallel 1
ctest --test-dir build/clang --output-on-failure
cmake -S . -B build/tsan -DCMAKE_CXX_COMPILER=clang++ -DHBR_THREAD_SANITIZER=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/tsan --parallel 1
python tools/run_host_tsan.py build/tsan/hbr_tests --output reports/host-tsan.json
```

The diagnostic runner records an initialization restriction explicitly as
`unsupported-initialization`, with the actual exit code and log. That state is
not a TSan pass. Race warnings and other failures return failure. No race is
suppressed and no security setting is changed. GPU counter permissions and
Windows WDDM sanitizer initialization remain separate gates.
