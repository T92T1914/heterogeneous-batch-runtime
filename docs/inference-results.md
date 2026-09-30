# Actual CPU image inference

The optional application built and executed on Windows with a C++20 MSVC extension, Python 3.11 and ONNX Runtime 1.30.0. It is separately installed from the numerical runtime. A pinned public MNIST model produced completed predictions for eight original geometric digit drawings. The full logits agreed with ONNX's independent reference evaluator within the committed tolerance.

The [machine-readable record](inference-cpu-evidence.json) retains model, input and source-content identities, dependencies, every output, normalized probabilities, lifecycle observations and actual provider placement. It records the maximum full-logit difference instead of comparing only the winning class. An actual ONNX Runtime profile contained 64 CPU provider node events. Those events establish this run's operator placement. They are not hardware counters, a latency result or a GPU result.

The predicted labels were `2, 3, 2, 3, 4, 5, 5, 2` for the geometric examples labelled zero through seven. Several drawings were misclassified. These examples are not a held-out handwriting dataset, and the demonstration claims no accuracy score. The outputs are retained without selecting only the correct examples.

## Executed checks

Twenty-six tests passed with the installed consumer in a clean environment. They exercised every logit at batch sizes zero, one, two and eight, independent returned arrays, PNG normalization and alpha handling, file and dimension limits, exact dtype/layout, nonfinite and out-of-range input, corrupted model rejection and owner-thread operations. Lifecycle cases checked immutable admitted inputs, capacity, duplicate identities, the finite session request limit, queued cancellation, running cancellation, stale result retention, error retirement and waiting shutdown. Acquisition tests checked interrupted flush, retry, conflicting content and identity validation. Synthetic reporting fixtures checked complete-sample tables and rejection of incomplete groups without collecting performance data.

The failure fixture raises a host error before inference. It does not simulate a GPU fault. The CPU package also rejected required CUDA initialization rather than silently satisfying that request on CPU. Profiling verified actual CPU node placement. None of those checks closes native GPU correctness, provider failure recovery or device memory acceptance.

The fresh application wheel was built and installed in a clean environment with its declared dependencies. Package dependency validation passed. The command line application ran from a directory outside its source tree. Adaptive Timing was installed from public revision `f6e0564474b053212f722e107b55b619aa8e50d9`; its existing completion adapter was reused without changing that repository. The default numerical runtime's dependencies and source interfaces are unchanged.

## Pending evidence

The [measurement protocol](inference-protocol.md) was committed before any performance collection. Fresh-session versus reused-session timings have not been collected. The shared workstation's performance slot was occupied, so no GPU work or performance measurements were started. This source provides session reuse and tests its lifecycle. It does not establish a speedup.

The installed CPU provider is not a CUDA package. Compatible CUDA provider dependencies, actual GPU output and placement checks, end-to-end comparisons and process/device memory measurements remain pending. The installed CUDA toolkit from the earlier workload phase does not establish compatibility with a newer inference provider. No driver, counter permission, Windows feature or GPU setting was changed. Prior Nsight counter and GPU sanitizer restrictions remain separate.

Hosted Windows and Linux consumer checks are configured separately from the existing core workflow. Their actual conclusions must be inspected at the delivered revision. Until those jobs execute, the Windows local result above is the available consumer evidence. Prior WSL and hosted numerical-runtime checks are not relabeled as Linux inference executions.
