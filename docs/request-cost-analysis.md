# Completed-request cost after model batching

The October 15, 2026 investigation explains more of the existing classifier's host request cost but does not justify a Runtime optimization. Preparation is a small observed component. A same-clock worker probe places about 65 to 70 percent of the diagnostic request inside the synchronous Python `Session.run` envelope, with the remaining 30 to 35 percent around that envelope. Neither interval identifies a specific avoidable operation. Ordinary ONNX Runtime profiling adds observable disturbance and produces host operation events without device activity events in these installed builds.

The outcome is a completed cost-attribution investigation with no consumer change. I/O binding, pinned buffers and removal of ownership or admission checks are not supported by this evidence. No request speedup is claimed. The accepted [model-batching result](batching-results.md), its original data and the default serial interface retain their own source identities and conclusions.

## Source, protocol and evidence

The reference baseline is PR19, merged at `9fcc2ecc7bed1f0bc84a3aca33212f619bef40e0`. Diagnostic source and the [prospective protocol](request-cost-protocol.md) were committed before collection at `6ec762f4a81d172c797f03557aaeffa05f69f6ad`. The driver SHA-256 is `dfddceef0c452772e9e97e25352673a5093958442faf3ddc9a16c9c956ba099f`. The source inventory fingerprint is `ac47e7812c8f937008a641c8b6dca39d51095afd16ff3cae6071b750b0fbbeea`.

The [complete evidence](request-cost-evidence.json) contains all 18 diagnostic reports, all 468 request outputs, all six diagnostic ORT traces, qualifications for every supported batch length, resource receipts and recomputable summaries. Absolute local path fields are removed from this public derivative. The private original report, journal and trace hashes remain recorded. Those original-text hashes identify retained raw files, rather than the reserialized public JSON. `tools/check_inference_evidence.py` replays the source relationships, complete numerical outputs and arithmetic without executing a model. These checks establish consistency with the recorded identities, not signed origin authentication or a new hardware execution.

The actual separately installed CPU and required-CUDA consumers are the accepted PR19 packages. Their installed source and library identities were recorded before and after each diagnostic. The native session, graph recipe, package implementation and pinned Adaptive Timing dependency were unchanged. Both provider environments use ONNX Runtime 1.30.0. No new package build is needed for an unchanged implementation. The new diagnostic calls the installed consumer through the existing executor and reuses the accepted collector's budget, serialization, identity and memory helpers.

The original model SHA-256 remains `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`. Explicit batched execution uses derived graph `cfba8152caa873db9563fef0daa1bebe97d2f3dfc3ec23cf72cc5db31a4ee539`. The [graph and ownership contract](model-batching.md) is unchanged. Original and derived model bytes are not bundled. The unresolved Apache/MIT model-card disagreement still leaves redistribution unqualified.

## Existing caller fields answer the first question

Before collecting new data, the original two provider reports were reprocessed as files. They contain 936 total requests, including 720 deciding requests. The table below uses the 60 batched deciding samples per provider and batch length. These are same-request arithmetic means in microseconds, rounded to three decimals.

| Provider | Images | Preparation | Admission | Wait and reconciliation | Caller gap | Completed request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 1 | 11.970 | 39.363 | 132.507 | 1.280 | 185.120 |
| CPU | 2 | 12.220 | 39.462 | 161.663 | 1.230 | 214.575 |
| CPU | 8 | 15.867 | 44.148 | 315.720 | 1.378 | 377.113 |
| Required CUDA | 1 | 12.028 | 41.350 | 669.320 | 1.297 | 723.995 |
| Required CUDA | 2 | 11.257 | 39.633 | 691.877 | 1.245 | 744.012 |
| Required CUDA | 8 | 15.720 | 42.680 | 661.402 | 1.335 | 721.137 |

The caller intervals are disjoint. Their means add to the mean completed request because they use the same rows. Their marginal medians need not add. The caller gap is exactly the interval from preparation return to admission start, including construction of the dispatch record and timing work. It is not a transfer remainder. Its median across all 936 retained requests is 1.2 microseconds, with a range of 0.9 to 6.7 microseconds.

At eight images, preparation accounts for 4.2 percent of summed CPU request time and 2.2 percent of CUDA request time. Even an impossible complete elimination of that preparation would fall below the prospective five-percent admission threshold. Admission accounts for about 11.7 percent CPU and 5.9 percent CUDA. The dominant caller interval is wait and reconciliation, at 83.7 percent CPU and 91.7 percent CUDA. That interval includes waiting for owner work, inference and result adoption. It is not synonymous with native execution or GPU time. Worker execution can start during admission, so adding a separate worker duration to caller admission and wait would double count work.

## The observable clocks have different limits

All 18 new reports record `time.perf_counter` as `QueryPerformanceCounter()` with reported resolution 100 nanoseconds. Both caller timestamps and the optional worker probe use `perf_counter_ns`, allowing same-request containment and partition checks. Reported resolution is not an experimentally established timing-accuracy guarantee.

The existing owner observations use `time.monotonic`, reported here as `GetTickCount64()` with resolution 15.625 milliseconds. In 352 of the 360 new deciding samples, owner start and completion have the same timestamp. Those fields preserve lifecycle observations but cannot resolve these submillisecond costs. A zero owner timestamp difference does not mean zero execution. Dispatch times are planner metadata and do not repair this clock limitation.

ORT's host profiler records session-relative microsecond timestamps using its own host clock. Its parent and child events are nested and cannot simply be summed. No subtraction across Python caller, owner and ORT clock domains is used. The versioned implementation supplies the clock and event-recording semantics. [ORT 1.30.0 host profiler](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/common/profiler.cc).

## One bounded diagnostic, without repeating the batching campaign

Only explicit batched N8 was timed. Each provider has three fresh process/session replications of three conditions: plain, worker probe and ordinary ORT profiling. Condition order rotates by replication as declared. Providers and processes are serialized. Each session records one first request, five warmups and twenty deciding samples. This gives 18 processes, 468 checked requests and 360 deciding samples. Setup, first request, validation and close remain separate from deciding latency.

Four input frames rotate through each session: the original eight geometric images, horizontal reflections, duplicate/permuted rows and seeded uint8 images. Each request starts with a fresh in-memory uint8 array. The primary boundary includes preparation into an owned normalized tensor, admission, any queueing and native/provider work, output materialization and reconciliation through a usable independent host result. Frame construction, the fresh source-array copy, reference evaluation, file reading and PNG decoding are outside that boundary. No upload or necessary output copy is moved outside the request. No device-resident result substitutes for a completed host result.

Every request's full 80 scores is compared with the original one-image graph's independent ONNX `ReferenceEvaluator` outputs at `rtol=1e-5, atol=2e-5`. The reference uses the preserved preprocessing contract, rather than the derived execution graph. Existing preparation tests and the installed qualification separately challenge dtype, layout, normalized values, mutation and row semantics.

Plain runs share caller timers, full-output validation and journaling with the instrumented runs. They are unprofiled diagnostic controls, not observer-free production latency. The worker condition adds two timestamps around the original Python `Session.run`. The ORT condition uses the existing native session profiler without that worker wrapper. No instrumented condition supplies an optimization decision.

All declared operations completed on the first collection. The owned runner reports 25.153 seconds aggregate elapsed, including startup, output and cleanup, within the explicitly granted 300-second slot. Each diagnostic had a ten-second process cap and eight-second internal collection budget. Two correctness operations had twenty-second process caps. CPU committed host memory was bounded at 512 MiB and CUDA at 2 GiB, with one owner and one inference/BLAS thread. All twenty owned Jobs retired empty. These are aggregate committed host-memory bounds, not device-memory ceilings. The process memory snapshots include libraries, references, allocator state and retained report data. They do not isolate tensor allocations or process-specific device memory.

## All process results, including observer disturbance

The following table retains every process median and deciding-sample range, in microseconds. Each row contains twenty samples. Complete samples, first requests, setup, memory and retirement records remain in the evidence artifact.

| Provider | Replication | Plain median [min, max] | Worker median [min, max] | ORT median [min, max] |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0 | 402.70 [320.40, 970.20] | 408.50 [343.00, 651.00] | 421.10 [359.40, 519.10] |
| CPU | 1 | 336.90 [285.50, 395.70] | 345.30 [297.70, 430.70] | 395.75 [354.40, 578.60] |
| CPU | 2 | 345.35 [323.10, 515.60] | 335.20 [301.30, 596.40] | 375.05 [334.70, 547.40] |
| Required CUDA | 0 | 361.25 [294.10, 1057.20] | 375.65 [321.40, 938.00] | 474.45 [376.50, 1125.00] |
| Required CUDA | 1 | 338.10 [309.70, 1047.00] | 353.00 [319.60, 536.50] | 404.40 [366.60, 1743.60] |
| Required CUDA | 2 | 322.15 [289.20, 511.60] | 356.50 [320.50, 928.10] | 401.75 [371.80, 1121.50] |

Paired differences below subtract the corresponding plain process median from the instrumented process median. Pairing is by declared replication, not simultaneous or adjacent requests. Three rotated process replications retain ambient variation and do not establish a calibrated population interval or a causal bare-timer cost.

| Provider | Additional observer | Differences for replications 0, 1, 2 (microseconds) | Median process difference |
| --- | --- | --- | ---: |
| CPU | Worker timestamps | +5.80, +8.40, -10.15 | +5.80 |
| CPU | ORT profiling | +18.40, +58.85, +29.70 | +29.70 |
| Required CUDA | Worker timestamps | +14.40, +14.90, +34.35 | +14.90 |
| Required CUDA | ORT profiling | +113.20, +66.30, +79.60 | +79.60 |

The negative CPU worker difference is retained. ORT process medians are 4.6 to 17.5 percent above the matched plain CPU medians and 19.6 to 31.3 percent above the matched plain CUDA medians. These are observed condition differences under this small protocol. They are not a universal profiler-overhead correction to subtract from another run.

The new CUDA plain medians are also lower than the historical N8 CUDA result. There is no implementation change behind that difference. The workload now rotates additional frames, the processes and sessions are fresh, and collection occurred in a different shared-machine period. This comparison cannot be relabeled as an optimization or a reversal of the historical matched CPU/CUDA study. The historical finding that CPU had the lower median in each tested condition remains attached to that original experiment.

## What the worker split actually measures

The worker probe brackets Python `Session.run`, including its validation, the native input snapshot, synchronous ORT/provider work, output construction and final checks. The prefix begins at fresh input preparation and ends at worker entry. The suffix begins at worker return and ends at usable caller reconciliation. Their same-request intervals are disjoint. Means below use each replication's twenty deciding samples, in microseconds.

| Provider | Replication | Prefix | Session.run host envelope | Suffix | Completed request | Envelope share of summed request time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 0 | 98.570 | 276.945 | 52.750 | 428.265 | 64.7% |
| CPU | 1 | 83.725 | 225.630 | 35.660 | 345.015 | 65.4% |
| CPU | 2 | 79.000 | 242.040 | 38.915 | 359.955 | 67.2% |
| Required CUDA | 0 | 94.095 | 305.385 | 43.215 | 442.695 | 69.0% |
| Required CUDA | 1 | 80.805 | 248.220 | 41.060 | 370.085 | 67.1% |
| Required CUDA | 2 | 86.430 | 304.635 | 42.670 | 433.735 | 70.2% |

All 468 caller phase partitions and all 156 worker partitions, including first and warmup requests, close to their recorded primary intervals. Every worker interval is inside its caller interval and has no worker error. No worker entry in this collection precedes admission return. The source contract still permits that overlap, so the result does not establish that admission and execution always occur sequentially.

This split locates material host cost around the model call, but does not say which prefix, suffix or native operation is avoidable. It does not isolate scheduling from reconciliation, native snapshotting from ORT, or CUDA transfers from device execution. A synchronous host envelope cannot become a kernel duration by changing its label.

## Instrumentation support and unobserved device costs

Every new ORT trace has 208 `Node` events and 54 `Session` events, including 26 `model_run` events. CPU Node placement is exclusively `CPUExecutionProvider`. Required-CUDA Node placement is exclusively `CUDAExecutionProvider`. There are zero `Kernel` category events and no stream fields in these six traces. Names ending in `_kernel_time` remain host Node names. Eight operation events per model invocation are not eight measured CUDA kernels, nor proof that all host work disappeared.

The actual CPU and CUDA library hashes match the retained installed-consumer identities. GPU wheel metadata reports CUDA 13.0.88 and cuDNN 9, but does not establish `ENABLE_CUDA_PROFILING`. Read-only PE import inspection found no CUPTI dependency in the GPU core/provider libraries. That is consistent with the observed host-only output but does not conclusively establish every build flag or loading route. CUDA profiling support for these installed bytes remains unconfirmed.

The documented ORT device profiling route needs CUPTI and a build made with CUDA profiling enabled. The retained toolkit's CUPTI headers/library/DLL do not retrofit that build option. No usable native timeline profiler was found in the bounded approved inventory. No profiler was installed, no ORT wheel was rebuilt, and no driver, counter, privilege or system setting was changed. [ORT profiling tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html), [versioned CUDA profiler](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/providers/cuda/cuda_profiler.h), [CUDA provider build configuration](https://github.com/microsoft/onnxruntime/blob/v1.30.0/cmake/onnxruntime_providers_cuda.cmake).

Even a future device trace needs activity classification. ORT 1.30.0's CUPTI path can put both compute and memcpy activities in the `Kernel` category. Host/device clock correlation and overlapping streams also require explicit handling. Instrumentation itself can change short workloads. [ORT CUPTI conversion](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/providers/cuda/cupti_manager.cc), [ORT device clock correlation](https://github.com/microsoft/onnxruntime/blob/v1.30.0/include/onnxruntime/core/common/gpu_profiler_common.h), [NVIDIA CUPTI overhead](https://docs.nvidia.com/cupti/main/main.html#cupti-overhead).

Actual device execution, transfers, launches, synchronization contributions, stream overlap and process-specific peak device allocation remain unobserved. No unexplained host remainder is assigned to one of them. A supported, already installed activity-tracing route with an applicable device grant would be a prerequisite for that distinct attribution question. It is not required to accept the present no-change result.

## Correctness, ownership and retirement

The bounded installed-consumer qualification covers both execution modes, every N from zero through eight, and original, duplicate/permuted and seeded frames. Each provider records 54 complete-output cases against the independent original reference. The actual invocation deltas remain N for serial, one for nonempty batched and zero for empty input. All scores, rather than only the winning label, are retained.

Both installed providers additionally passed unaligned contiguous input, input snapshot/mutation and independent retained-output checks across later runs and close. Invalid dtype, shape, nonfinite, negative and noncontiguous inputs are refused before invocation. Wrong-thread run, counter access and close are refused. Closed and repeated-close behavior is retained. The targeted existing CPU lifecycle cases passed four tests. The required-CUDA suite passed 25 tests on the actual compatible device, including its refusal and batched controls.

These results reuse the existing cancellation, staleness, failed-invocation and physical-retirement contracts. They do not claim injection of an active CUDA kernel failure or proof that a running device kernel can be interrupted. Three synthetic diagnostic cases separately verify that a late reconciled result retains successful cleanup without erasing the request failure, and that trace failure is recorded distinctly from successful owner retirement. Synthetic reporting controls are not native CUDA acceptance.

Independent file-only reviews recomputed all outputs, host partitions and observer differences, checked raw trace hashes and challenged the event interpretation. They did not rerun hardware. The exposed Code Review action supplies CI diagnostics, rather than a dedicated source-audit action. Source review and CI results are distinct evidence.

## Decision and stopping point

The prospective rule required a measured component plus a compatible concrete change supporting at least five percent and ten microseconds improvement to the complete N8 host request. Preparation fails the percentage threshold even under a complete-removal upper bound. The worker envelope is material but combines several necessary and unknown costs. The surrounding host work is also material, yet these data do not show which safe narrow change would remove enough of it. Removing snapshots or ownership would change the supported semantics. Moving upload outside `Run` or leaving results on the device would change the measured task.

No Runtime optimization was admitted or implemented. The diagnostic, complete evidence and replay checks are delivered so the finding is inspectable. This is an inconclusive optimization opportunity with a qualified no-change disposition, rather than a failed profiler initialization presented as a performance result. The absent device timeline is explicit. The adverse profiler observations and historical negative comparisons remain available.

Independent-request microbatching has not been started. It would need its own useful arrival workload, logical request/adoption rules, bounded storage and authority. A bounded assembly wait would not establish a hard real-time guarantee. No Adaptive Timing change, serving stack, second-vendor backend, model redistribution or public personal-profile update follows from this investigation.
