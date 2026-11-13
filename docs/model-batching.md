# Model batching in the optional classifier

The classifier has two explicit execution modes. `serial` remains the default and invokes the original graph, which takes one image, once per image. `batched` prepares one tensor holding several images and invokes a narrowly derived graph once for a nonempty request. Both modes return a completed, independently owned float32 `(N,10)` array. Empty input returns `(0,10)` without invoking the model. The limit remains eight images.

Use the existing installed application with `hbr-classify --model <original-reviewed-model> --execution-mode batched --provider cpu --output predictions.json --profile-prefix batch-profile`. The same mode is available through `Session(model, execution_mode="batched")` and `InferenceExecutor(model, execution_mode="batched")`. Provider selection is separate. `cuda-required` preserves required placement and refuses CPU fallback. There is no automatic execution mode or silent serial fallback.

## What changes in the graph

Admission still requires the original MNIST artifact of 26,454 bytes with SHA256 `2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf`. Batch mode derives its execution graph in memory. It changes `Pooling160_Output_0_reshape0_shape` from `[1,256]` to `[0,256]` and changes the leading dimension of twelve data tensors (the input, intermediate and output values) from one to symbolic `N`. Under the Reshape semantics of the graph's opset 8, zero preserves the incoming batch dimension.

The final data path is `[N,16,4,4]` to `[N,256]`, then multiplication by the same `[256,10]` weight matrix and addition of the same `[1,10]` bias. Every trained weight, node, operator attribute, IR/opset and preprocessing rule is preserved. Changing only the declared input shape would leave the fixed flattening operation wrong, so the transformation qualifies that actual join.

The reviewed derived hash is `cfba8152caa873db9563fef0daa1bebe97d2f3dfc3ec23cf72cc5db31a4ee539`. An unexpected source or derived hash is refused. `model_sha256` remains the original source identity. Reports additionally retain `source_model_sha256`, `execution_model_sha256`, `execution_mode` and `execution_recipe` rather than labeling the derived graph as the original artifact.

The original model card's Apache-2.0 metadata and MIT SPDX/body discrepancy remains unresolved. The model is acquired separately. Original and derived model bytes are neither committed nor included in the wheel. Local derivation retains the original evidence and notices. This change does not resolve artifact licensing or authorize redistribution. Hash identity establishes the selected bytes, not signed origin authentication or general safety for hostile models.

## Output and lifetime checks

Complete logits are compared at the existing `rtol=1e-5, atol=2e-5` with independent ONNX reference outputs of the original graph, which takes one image. Graph tests reverse the transformation and check unchanged operator/weight bytes. Batch tests also challenge row order, duplicates, mixed images and the same image alone versus inside a larger batch.

The native session checks the selected graph schema and the returned tensor's dtype, shape and exact element count before copying. Its `model_invocations` property, which only the owning thread may read, counts attempted actual ONNX Runtime `Run` calls after input admission. Serial adds `N` calls per valid request. Batched adds one for nonempty input and none for empty input. A failed attempted invocation still counts.

The existing consumer snapshots, owner thread, cancellation requests, adoption of stale results and retirement on physical completion remain in use. Batched execution is synchronous through completed host outputs. No result left on the device, asynchronous GPU API, new owner framework or backend spanning several vendors is introduced.

## Measuring the request

`examples/image_classifier/compare_batching.py` compares explicit modes through the installed consumer. It times preparation of the original images in memory, submission, queueing, inference, output materialization and reconciliation through a usable host result. It reports setup, first request, validation and retirement separately. The boundary excludes file reading and PNG decoding and must not be described as latency from file to result.

Use one free permitted performance slot and one active owner at a time. The protocol runs each mode in blocks on one reused owner session, alternates block order and rotates batch order. Its paired samples are nonadjacent within corresponding blocks. Process memory observations are process counters, and the CUDA arena configuration is not a total GPU memory ceiling. Slower and inconclusive cases remain results. Original timing reports retain their original revisions and boundaries.

The [declared protocol](batching-performance-protocol.md) and [completed comparison](batching-results.md) retain the source revision, complete rows and outputs, setup/retirement, memory scope and the limits of a small study. The earlier [serial inference protocol](inference-protocol.md) remains a pinned input to the historical evidence page. Its current bytes are not the ones the serial CPU session reuse collection in `inference-cpu-reuse.json` recorded. That collection recorded the protocol as it stood at its source revision `8868e9f`. A Windows CPU memory probe section and two Adaptive Timing cleanup sections were added afterward without changing the earlier text. The evidence page links each collection's protocol at that collection's executed source.
