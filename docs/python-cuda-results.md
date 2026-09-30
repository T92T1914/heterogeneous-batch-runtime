# Python CUDA application verification

The optional interface source is `fa7afb4b798a1481ffa34a85248ba63d2ea029a2`.
The connected Adaptive application source is
`3ce1b6571899faed99fc7a2eabdf5862cf5639ad`.
The native CUDA numerical implementation and existing reuse failure-retirement
contract are unchanged from the delivered PR2 baseline.

## Executed coverage

Windows CPU and CUDA wheels were built separately with MSVC, Python 3.11 and
the existing CUDA toolkit. New independent virtual environments installed those
wheels. The CPU wheel imported with CUDA disabled and rejected explicit CUDA
creation. It requires neither toolkit discovery nor GPU initialization.
The CUDA wheel imported with CUDA enabled and executed on an RTX 4090.

The final Python contract suite has 22 cases. Actual CUDA execution passes all
22. The CPU build executes the host cases and skips four CUDA-only cases.
Tests cover exact dtype, shape, contiguity, independent repeated outputs,
unavailable CUDA, bounds, creating-thread rejection, explicit close, ordinary
abandonment from a different Python thread, exception teardown and pending
snapshot ownership. Synthetic resource tests are host tests and do not imply
GPU execution. An initial exception-lifetime test exposed retained worker
tracebacks. Clearing those tracebacks before forwarding exceptions corrected
the ownership issue, and the regression now passes.

All four existing Windows native groups passed: CPU numerical/runtime contracts,
simple CUDA contracts, reusable CUDA contracts and reuse fault contracts.
The fault group executed seven acquisition and eight execution host-exception
checkpoints. These are synthetic host exceptions around actual CUDA work,
not device-loss, OOM or GPU sanitizer coverage.

The [exact-source CI run](https://github.com/T92T1914/heterogeneous-batch-runtime/actions/runs/36672953632)
passed Windows and Ubuntu CPU builds/packages, ASan/UBSan, Clang 18.1.3 Release
tests and separate Clang host ThreadSanitizer execution. Its host TSan result
was `passed`. The tests exercise native concurrency, cancellation and owned
capture release. A CPU CI runner does not provide CUDA execution evidence.

The dedicated WSL GCC host build passed Release and ASan/UBSan at the same
source revision. Separate GCC TSan compiled but exited 66 during initialization
with `FATAL: ThreadSanitizer: unexpected memory mapping`. Its state is
`unsupported-initialization`, not passed. No race suppression or OS/security
change was applied. This differs from the successful hosted Clang TSan run.

## Connected application

The [retained Adaptive trace](https://github.com/T92T1914/adaptive-timing-engine/blob/main/docs/cuda-application-trace.json)
contains four actual completed CUDA calls and seven reconciled outcomes.
Full output arrays match independent NumPy bincount and separate scalar-loop
oracles. Queued cancellation avoids CUDA execution. Running cancellation
requests retain live storage through completed return. Obsolete completion
remains inspectable and is not current state. Explicit shutdown drains its
running call, cancels queued work and closes the resource before final
reconciliation. A synthetic host exception is retained separately.

The caller image changes after verified submission return while a deterministic
host gate prevents native execution. The original snapshot still produces the
original full histogram. This verifies input ownership, not a GPU concurrency
or transfer-overlap claim. The [runnable example and ownership rules](https://github.com/T92T1914/adaptive-timing-engine/blob/main/docs/cuda-application.md)
explain how to reproduce and inspect the trace.

The original 504 timing rows, reuse measurements, protocols, historical reports
and failure evidence remain unchanged. Both historical validators pass.
No old timing study was rerun, and no new benchmark or speedup claim is made.
Python copies, queueing and setup were not measured as an application latency
experiment. Host monotonic observations and device event times remain separate.

GPU counters remain unpassed after `ERR_NVGPUCTRPERM`. Windows GPU sanitizer
initialization remains separately unpassed under WDDM. These gates were not
retried or changed. No second-vendor execution is established. Explicit normal
shutdown is supported. Arbitrary interpreter termination and physical device
loss are outside the recovery contract.
