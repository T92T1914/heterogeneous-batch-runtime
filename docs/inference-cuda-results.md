# Actual required CUDA image inference

The optional classifier now completed the same bounded image application on an RTX 4090 through required CUDA. All 80 logits agreed with the independent ONNX reference within `rtol=1e-5, atol=2e-5`. The maximum absolute difference was `5.7220458984375e-06`. Its actual ONNX Runtime profile retained 64 node events, all on `CUDAExecutionProvider`. Provider enumeration alone was not used as execution proof.

The outputs were `2, 3, 2, 3, 4, 5, 5, 2` for the original geometric drawings labelled zero through seven. The incorrect classifications remain visible. These drawings demonstrate application behavior, not a handwriting accuracy study. Application batches remain serial one-image calls through one session. There is no claim of fused GPU batching or an asynchronous GPU interface.

The [machine-readable evidence](inference-cuda-evidence.json) retains the complete outputs, independent reference, operator placement, every timing row, memory observations, dependency identities and distinct executed source revisions. The [protocol](inference-cuda-protocol.md) was committed at `635e5d2caf7d2786e6d98a2f4696277a36d2d6f3` before collection. The original [CPU collection](inference-results.md) is unchanged and remains separately identified.

## Dependency loading and executed contracts

This run used Windows 11 build 26200, Python 3.11.8, ONNX Runtime GPU 1.30.0, an RTX 4090 and driver 617.14. The isolated provider environment installed reviewed, hash-pinned NVIDIA wheels, including CUDA runtime 13.4.92 and cuDNN 9.27.0.42. No driver, system path, counter permission or setting was changed. These dependencies are not bundled or redistributed. The [official provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) specifies CUDA 13.0 and cuDNN 9 for this ONNX Runtime release, and documents loading installed NVIDIA wheels through `preload_dlls(directory="")`.

The original automatic session path failed with Windows loader error 126 for `cublasLt64_13.dll`, despite the vendor wheels being installed. That initialization failure produced no usable result. The corrected automatic required-CUDA path now performs the documented preload before constructing its C++ owner. CPU operation does not preload GPU dependencies. A caller supplying an explicit runtime library keeps its own dependency-loading contract. Two of four focused regressions failed before the correction. All four passed afterward.

Nine explicitly opted-in CUDA contract cases then passed. They checked every logit at zero, one, two and eight images, owner-thread and closed-session behavior, admitted input snapshots, independently owned outputs, duplicate identities, the finite request limit, host-gated queued/running cancellation, staleness, host failure retirement and waiting close. The gates exercise host ownership around actual required-CUDA sessions. They do not interrupt a running kernel or simulate device loss. The default test run skips these cases rather than pretending a CPU runner executed them.

A fresh CPU-only installation separately passed 35 current contract cases, with the nine GPU cases skipped. The GPU installation also passed 31 existing CPU and loader cases, with its CPU-package rejection case skipped because CUDA was available. Three synthetic memory-probe failure fixtures passed. Those fixtures use host faults to test deadline and cleanup reporting. They are not GPU sanitizer evidence.

Six additional fixtures reject relabelled evidence without starting inference. They cover mismatched models and inputs, a CPU label on the GPU memory record, missing session boundaries, relaxed tolerance and an invalid numerical summary. Independent review reproduced the original checker gap, which is corrected before publication. These reporting checks are separate from actual GPU execution.

## Completed-call comparison

The comparison retains 240 rows, including 180 samples. Each of nine conditions has twenty samples after five warmups, with the first invocation separate. All 234 outputs produced during first invocations, warmups and samples passed full-logit reference validation. Tables are generated from the retained rows by [the evidence checker](../tools/check_inference_cuda_evidence.py).

<!-- timing-begin -->
| Images | Path | Samples | Median ms | Minimum ms | Maximum ms |
| --- | --- | --- | --- | --- | --- |
| 1 | cpp-fresh | 20 | 7.00700 | 6.42280 | 9.66150 |
| 1 | cpp-reused | 20 | 0.39140 | 0.34770 | 0.80870 |
| 1 | python-cpu-reused | 20 | 0.08870 | 0.05500 | 0.14750 |
| 2 | cpp-fresh | 20 | 7.90435 | 7.00230 | 10.13210 |
| 2 | cpp-reused | 20 | 0.84625 | 0.48270 | 1.11620 |
| 2 | python-cpu-reused | 20 | 0.12975 | 0.08880 | 0.18810 |
| 8 | cpp-fresh | 20 | 8.76590 | 7.85450 | 10.63780 |
| 8 | cpp-reused | 20 | 2.22550 | 1.57760 | 3.65930 |
| 8 | python-cpu-reused | 20 | 0.33425 | 0.28320 | 0.54740 |
<!-- timing-end -->

The C++ fresh path includes session creation, input admission, completed inference, reconciliation and close. The reused path retains the completed host-result contract while setup and final retirement are separate rows. Both require CUDA. The direct Python baseline uses an optimized CPU session in the same provider installation, copies the same prepared tensors and returns completed logits. It lacks the wrapper's request identity and generation accounting, so this is also a wrapper comparison. PNG preparation, reference validation and journal writes are outside all timed boundaries. These are prepared-tensor calls, not file-to-result latency.

Reused setup at batches eight, one and two took 238.7053, 17.5940 and 18.2185 ms. Final retirement took 11.7311, 10.4230 and 11.6510 ms in that order. The first setup includes initial native provider loading. Neither it nor the first-invocation rows is a cold-process measurement. A short-lived session must account for those costs.

Reuse reduced repeated complete-call cost in this collection. The direct Python CPU baseline remained faster in all three tested batches. The slower GPU outcomes remain in the record, and the simpler CPU default stays available. Twenty samples per condition on a workstation shared with ordinary applications do not establish extreme tails, a universal hardware ranking or a general CPU/GPU crossover. This collection did not overlap another task's performance workload. Operator profile events establish placement, not kernel intervals, hardware counters or overlap.

## Separate memory observation

The corrected probe used a fresh process, one required-CUDA owner and one outstanding request at a time. It completed 64 eight-image requests and validated all 80 logits for every request. All 64 checks passed, with eight snapshots and no cleanup errors. Its initial `cudaMemGetInfo` call initialized a CUDA context before the first baseline below. That is a different boundary from a process with no CUDA context.

<!-- memory-begin -->
| Phase | Completed requests | Working set MiB | Private commit MiB | Shared device free MiB |
| --- | --- | --- | --- | --- |
| after_imports_reference_and_cuda_context_initialization | 0 | 417.492 | 1506.949 | 23036.000 |
| after_session_creation | 0 | 454.215 | 1547.363 | 23024.000 |
| after_completed_requests | 8 | 702.594 | 1854.504 | 23014.000 |
| after_completed_requests | 16 | 702.594 | 1854.504 | 23014.000 |
| after_completed_requests | 32 | 702.598 | 1854.504 | 23014.000 |
| after_completed_requests | 64 | 702.602 | 1854.504 | 23014.000 |
| after_explicit_close | 64 | 693.352 | 1846.602 | 23032.000 |
| after_close_executor_release_and_gc | 64 | 693.352 | 1846.602 | 23032.000 |
<!-- memory-end -->

Host columns include Python, model/input preparation, the independent evaluator, loaded libraries and allocator retention. They are process totals. NVIDIA documents `cudaMemGetInfo` as a [shared-device free/total memory observation](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html). Other applications affect it, and a free-byte sample is not an allocation guarantee. Process-specific live GPU allocation was unavailable. These samples do not establish an allocation cap or absence of leaks. The provider's 128 MiB arena limit also does not bound driver, library or total device memory.

The probe's 30-second cooperative decision limit includes initialization and gives each wait the remaining time. Synchronous initialization and safe retirement are not forcibly interrupted, so it is not a hard process wall-time cap. Cleanup records secondary failures while preserving the primary exception and attempting the final observation. A cleanup-only failure cannot publish a completed result.

## Source identities and remaining gates

The application and comparison ran at `7a40f5fd43a213dab94ea5345fb87c697d3f4612`. The corrected memory probe and nine GPU cases ran at `6d0254bf99709b66c7435f9178c49f2c1889c579`. The initial memory record keeps its earlier source identity privately rather than being relabelled as execution of the corrected probe. Current source and historical records are checked separately.

Linux GPU execution, second-vendor validation, hardware-counter profiling and GPU sanitizer acceptance remain unverified. The earlier `ERR_NVGPUCTRPERM` and WDDM debugger initialization failures remain separate restrictions. Host sanitizer and host fault fixtures do not close them. No deliberate GPU reset or device-loss test was performed.
