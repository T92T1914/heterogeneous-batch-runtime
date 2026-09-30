# Bounded image consumer protocol

This follow-up asks whether a reusable inference session avoids setup work for repeated small image requests. It keeps the existing numerical workloads and their retained results unchanged. The optional consumer uses the public ONNX Model Zoo MNIST model at the artifact identity recorded in `examples/image_classifier/artifacts.json`.

The model accepts one float32 grayscale image of shape `(1,1,28,28)` and returns ten logits. An application batch contains zero to eight images. It performs serial one-image calls and returns an owned `(N,10)` array. This is not a fused model batch. Inputs are black-background, white-foreground digit images normalized to `[0,1]`. PNG decoding, alpha composition over black, grayscale conversion and bilinear resizing are part of the file-input contract.

The original geometric examples are application fixtures. They are not MNIST test images, an accuracy estimate or evidence of handwriting generalization. Full outputs must agree with the independent ONNX reference evaluator at `rtol=1e-5, atol=2e-5`. Shape, empty input, partial batch, nonfinite values and independent output ownership are checked separately. The model's published accuracy does not describe these executions.

## Ownership and completion

The admission call copies the caller's prepared tensor. The caller must exclude external native writers during that copy. After successful admission, mutation of the original tensor cannot affect the request. One worker creates, runs and closes the C++ ONNX Runtime session through Adaptive Timing's existing `OwnerExecutor`. At most four requests and 256 total request identities belong to one session. Completed unreconciled requests still occupy capacity.

Ordinary synchronous `Run` produces completed host outputs. The consumer does not use custom streams, I/O binding or disabled provider synchronization. Queued cancellation can prevent execution. Running cancellation is a request, and the session retains its storage until `Run` returns. A stale generation controls adoption, not buffer lifetime. Shutdown waits for completion and resource retirement. Failures become inspectable outcomes. No worker is killed.

The model is 26,454 bytes. Model loading rejects a different hash, external tensor files and custom operator domains. An input file is limited to one MiB, one PNG frame and dimensions from 1 to 256. Each request has at most 25,088 prepared input bytes and 320 output bytes. Capacity bounds admitted tensors and results. ONNX Runtime allocator overhead is additional. The requested CUDA arena limit is 128 MiB, which is not a cap on every driver or provider allocation. Process and device memory must be measured separately before claiming a complete memory bound.

## Comparison decided before collection

Use the same model, input values, normalization, numerical check and completed host outputs for fresh-session and reused-session paths. Compare batch sizes 1, 2 and 8 with five warmups and twenty retained samples per condition. Alternate the two modes per condition and rotate batch order. Run only one performance workload on the shared machine. Cap the experiment at two minutes and retain every neutral or slower condition.

Record session setup, tensor preparation and complete request duration separately. The fresh-session complete call includes construction, execution, output production and close. The reusable-session complete call includes submission copy, execution, output production and reconciliation. Its setup and final retirement remain separate rows. Report setup amortization explicitly. Do not call submission time completion, or subtract copying and synchronization from one mode while retaining them in another.

Run CPU with one intra-op and one inter-op thread. A serious baseline uses the same ONNX Runtime graph optimizations and outputs. A direct Python ONNX Runtime session provides a wrapper comparison, while the independent ONNX evaluator provides correctness evidence, not the performance baseline. The prepared-input comparison excludes PNG decoding and normalization equally from all three paths. Mandatory full-output reference validation follows every result outside all three intervals, with its own duration. Preserve runtime, compiler, provider, input, protocol and source identities. Include sample counts, median, range, run order and shared-workstation limitations. First invocation is separate and is not a cold-process measurement.

CUDA is an optional required-provider mode. Failed initialization must fail visibly. Successful configuration alone does not establish placement. Retain an ONNX Runtime operator profile from actual execution, including any CPU nodes. The required mode disables CPU fallback. Host timestamps and operator profile durations are distinct from hardware counters. GPU measurement waits for a compatible provider toolchain and a free permitted machine. Prior counter and sanitizer restrictions remain open.

## Acceptance and publication

Execute preprocessing and entire-logit reference checks, repeated calls, mutable input after admission, independent results, duplicate identity, capacity, queued and running cancellation, stale adoption, failure retirement and waiting shutdown. A CPU-only provider must reject required CUDA. Actual GPU correctness, placement, memory and timings are separate acceptance items.

Build the default CPU runtime without these dependencies. Build and install the optional consumer separately, then exercise it from outside the source directory. Retain actual CPU evidence even when GPU or measurement prerequisites are unavailable. Generate comparison tables from machine-readable rows. No result is filled in before execution, and an unexecuted comparison remains pending.
