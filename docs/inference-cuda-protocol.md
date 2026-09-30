# Required CUDA consumer follow-up

This follows the existing completed-host-output protocol. It does not replace the retained CPU comparison or change the model, preparation, tolerance or default installation. The first question is whether a fresh optional GPU environment can run the same application through the required provider without a silent CPU fallback.

## Environment and acquisition

Use an isolated Windows x64 Python 3.11 environment with ONNX Runtime GPU 1.30.0. The official provider documentation identifies CUDA 13.0 and cuDNN 9 for this release. Record the exact resolved runtime library wheels, their release URLs and hashes, and the observed GPU and driver before execution. Keep dependency licenses separate from the model artifact and application source. Do not redistribute model weights or provider binaries.

Use only the documented ONNX Runtime preload operation if the independently installed NVIDIA libraries require it. Do not change system paths, drivers, settings or counter permissions. Retain an original-path initialization failure before fixing a demonstrated loader gap. CPU installation and operation must retain their separate path.

## Execution and comparison

One resource owner executes this unit after other video, native acceptance and performance operations finish. The smoke application is capped at 30 seconds and eight serial images. It retains complete logits and an actual operator profile. Full outputs must pass the existing independent ONNX reference at `rtol=1e-5, atol=2e-5`. Labels alone cannot establish numerical agreement. Failures and missing placement remain explicit.

For the existing comparison runner, use the same prepared tensors and host outputs at batch sizes 8, 1 and 2. Keep its separate first invocation, five warmups and twenty samples per path. The two C++ paths use required CUDA and the direct Python baseline uses optimized CPU, exactly as each row records. The reused setup and final close remain separate. Fresh calls include construction and close. Reference checks and journal writes remain outside every timed boundary. The comparison is capped at two minutes. Preserve any incomplete rows and slower outcomes. The prior CPU measurements remain a different collection.

Operator placement comes from ONNX Runtime's actual node profile. Available providers are build capabilities, not session execution proof. Profile event durations are not CUDA kernel intervals or hardware counters. No device interval, overlap, kernel cancellation or GPU sanitizer result is inferred from this run.

## Bounded memory observation

Run a separate fresh process with one required CUDA session, one outstanding request and eight images per request. Observe 64 physically completed requests and validate every complete output. Sample current process working set, cumulative peak working set and private committed bytes before session creation, after creation, after 8, 16, 32 and 64 requests, after close and after garbage collection. These are process totals, not a ledger of live cache allocations.

Read `cudaMemGetInfo` on device zero at the same boundaries when its documented runtime entry point is available. Its initial call creates a CUDA context before the baseline. Report that boundary explicitly. Free and total device bytes describe the shared device, including other applications and allocator retention. They do not identify this process's live allocations. Do not subtract them and call the result a private memory bound or a leak. Record unavailable process-specific GPU memory as unavailable. The provider arena limit remains 128 MiB, which does not cap driver, library or total device memory. No deliberate device failure is performed.

## Acceptance

Required CUDA must initialize or return a visible failure. Actual full outputs, repeated ownership and operator placement must be retained. Recheck the existing installed CPU contracts after a loader correction. Host cancellation and stale-result tests retain their host scope. This evidence cannot close the separate hardware-counter, WDDM sanitizer, second-vendor or native application gates.
