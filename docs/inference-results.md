# Actual CPU image inference

The optional application built and executed on Windows with a C++20 MSVC extension, Python 3.11 and ONNX Runtime 1.30.0. It is separately installed from the numerical runtime. A pinned public MNIST model produced completed predictions for eight original geometric digit drawings. The full logits agreed with ONNX's independent reference evaluator within the committed tolerance.

This is the retained CPU collection. The later [required CUDA collection](inference-cuda-results.md) records its own executed outputs, placement, comparison and memory observations. Pending GPU statements below describe the CPU collection's original boundary rather than the current status of that later collection.

The [machine readable record](inference-cpu-evidence.json) retains identities for the model, inputs and source content, along with dependencies, every output, normalized probabilities, lifecycle observations and actual provider placement. It records the maximum difference over the full logits instead of comparing only the winning class. An actual ONNX Runtime profile contained 64 CPU provider node events. Those events establish this run's operator placement. They are not hardware counters, a latency result or a GPU result.

The predicted labels were `2, 3, 2, 3, 4, 5, 5, 2` for the geometric examples labelled zero through seven. Several drawings were misclassified. These examples are not a handwriting dataset held out from training, and the demonstration claims no accuracy score. The outputs are retained without selecting only the correct examples.

## Executed checks

Twenty six tests passed with the installed consumer in a clean environment. They exercised every logit at batch sizes zero, one, two and eight, independent returned arrays, PNG normalization and alpha handling, file and dimension limits, exact dtype/layout, nonfinite input and input out of range, corrupted model rejection and operations on the owner thread. Lifecycle cases checked immutable admitted inputs, capacity, duplicate identities, the finite session request limit, queued cancellation, running cancellation, stale result retention, error retirement and waiting shutdown. Acquisition tests checked interrupted flush, retry, conflicting content and identity validation. Synthetic reporting fixtures checked tables of complete samples and rejection of incomplete groups without collecting performance data.

The failure fixture raises a host error before inference. It does not simulate a GPU fault. The CPU package also rejected required CUDA initialization rather than silently satisfying that request on CPU. Profiling verified actual CPU node placement. None of those checks closes native GPU correctness, provider failure recovery or device memory acceptance.

The fresh application wheel was built and installed in a clean environment with its declared dependencies. Package dependency validation passed. The command line application ran from a directory outside its source tree. Adaptive Timing was installed from public revision `f6e0564474b053212f722e107b55b619aa8e50d9`. Its existing completion adapter was reused without changing that repository. The default numerical runtime's dependencies and source interfaces are unchanged.

## Pending evidence

The [measurement protocol](inference-protocol.md) was committed before any performance collection. The performance slot was occupied initially. After a later check that changed nothing found no game or named profiling process, a bounded CPU comparison on a single thread executed. Its separately identified results appear below. No GPU workload was started.

The installed CPU provider is not a CUDA package. Compatible CUDA provider dependencies, actual GPU output and placement checks, GPU comparisons and device memory measurements remain pending. The installed CUDA toolkit from the earlier workload phase does not establish compatibility with a newer inference provider. No driver, counter permission, Windows feature or GPU setting was changed. Prior Nsight counter and GPU sanitizer restrictions remain separate.

Hosted Windows and Ubuntu consumer jobs also built and installed the optional package, executed its contract tests and checked the retained evidence at source revision `631a546f781625a27f7b9e9cf94f641293dd55d5`. Both jobs in [run 36701644311](https://github.com/T92T1914/heterogeneous-batch-runtime/actions/runs/36701644311) completed successfully. The separate core Windows, Ubuntu and host sanitizer jobs in [run 36701644479](https://github.com/T92T1914/heterogeneous-batch-runtime/actions/runs/36701644479) also passed. These are hosted CPU executions. They do not establish Linux GPU execution or GPU sanitizer coverage. The record of full outputs and provider placement above remains the separately identified local Windows run.

## Completed CPU session reuse comparison

The [retained comparison](inference-cpu-reuse.json) contains 240 rows, including 180 timing samples. Each of the nine conditions has twenty samples after five warmups, with first invocation kept separately. The table is generated from those rows. All 234 produced outputs, including first invocations and warmups, passed the reference check over the full logits. The maximum absolute difference was `1.33514404296875e-05`. The protocol was already committed at the executed source revision before collection.

| Images | Path | Samples | Median ms | Minimum ms | Maximum ms |
| --- | --- | --- | --- | --- | --- |
| 1 | cpp-fresh | 20 | 1.9271 | 1.7290 | 2.3880 |
| 1 | cpp-reused | 20 | 0.1649 | 0.1307 | 0.2706 |
| 1 | python-cpu-reused | 20 | 0.0670 | 0.0540 | 0.1177 |
| 2 | cpp-fresh | 20 | 1.8951 | 1.7643 | 2.5662 |
| 2 | cpp-reused | 20 | 0.1962 | 0.1789 | 0.2264 |
| 2 | python-cpu-reused | 20 | 0.1042 | 0.0869 | 0.1296 |
| 8 | cpp-fresh | 20 | 2.1773 | 2.0660 | 3.2534 |
| 8 | cpp-reused | 20 | 0.4388 | 0.3700 | 0.8355 |
| 8 | python-cpu-reused | 20 | 0.3261 | 0.2882 | 0.5168 |

Fresh calls include owner/session creation, input admission, completed inference, reconciliation and close. Reused calls retain that contract for completed results while session creation and final close are separate. Direct Python ONNX Runtime uses the same optimized graph and configuration with one thread, copies its input and returns completed logits. It does not supply the C++ wrapper's request identity and generation accounting. It is a wrapper comparison, not an identical scheduler implementation. PNG decoding, normalization and reference validation are outside all three timing boundaries. These are calls on prepared tensors, not the latency from file to result.

The first setup of a reused session was 33.1041 ms at batch eight. The later setups at batch one and batch two were 2.0437 and 2.1689 ms. Final retirement rows were 0.2746, 0.1819 and 0.1597 ms in that order. The first setup includes the process's first native initialization of the provider library. Neither it nor the separate rows for first invocations is a study of a cold process. Setup costs must be included when assessing a short session.

Session reuse reduced repeated setup cost under these conditions. The direct Python reused session was faster than the C++ owner wrapper in all three tested batch sizes. That unfavorable wrapper comparison remains visible. The existing simple interfaces stay available. These small samples from a shared workstation do not establish latency at the extreme tail, a general language ranking or GPU benefit.

## Separate CPU process memory observation

The [retained memory record](inference-cpu-memory.json) comes from a fresh Windows process using the separately committed probe and protocol. One session completed and reconciled 64 requests, each with eight images and an independent check of all 80 logits. All 64 checks passed. Snapshots below come from Windows process counters, after prior request tensors and event rows were released. There was one request outstanding at a time.

| Phase | Completed requests | Working set MiB | Private commit MiB | Lifetime peak working set MiB |
| --- | --- | --- | --- | --- |
| after_imports_model_and_input_preparation | 0 | 81.953 | 60.777 | 81.953 |
| after_session_creation | 0 | 99.074 | 71.559 | 99.074 |
| after_completed_requests | 8 | 100.391 | 73.422 | 100.391 |
| after_completed_requests | 16 | 100.398 | 73.422 | 100.398 |
| after_completed_requests | 32 | 100.465 | 73.422 | 100.465 |
| after_completed_requests | 64 | 100.719 | 73.609 | 100.719 |
| after_explicit_close | 64 | 100.680 | 72.586 | 100.738 |
| after_close_executor_release_and_gc | 64 | 100.676 | 72.523 | 100.738 |

This process retains Python imports, model bytes, prepared input, the independent evaluator and its report after session retirement. Current process totals and lifetime peaks include those objects and allocator caches. The observations are not a native allocation ledger. They do not establish a strict cap on process memory or prove absence of leaks. CPU memory results do not establish device memory safety.

The timing runner later gained failing regressions for Python baseline initialization and failure of the journal setup row after creation of the C++ owner. Its cleanup now encloses both operations, so either failure still retires the owner and attempts the retirement row. The recorded successful timings above keep their original source identity. The evidence checker verifies older changed source files against the recorded Git snapshot. Current contracts execute separately and do not turn historical measurements into a rerun of later code.

The isolated installed provider supports only the CPU. The current [CUDA provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) lists CUDA 13.0 and cuDNN 9.x for the default ONNX Runtime 1.30 GPU package, with a separate CUDA 12.8 build. The older numerical runtime's toolkit does not establish those inference dependencies. GPU telemetry, read passively, showed ongoing GPU activity on the shared workstation during this prerequisite check, so no GPU inference or timing workload was started. Configuration with the provider required remains separately unexecuted. AMD validation remains conditional on supported GPU hardware and the current [MIGraphX provider route](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html). The host's AMD CPU is not AMD GPU evidence.
