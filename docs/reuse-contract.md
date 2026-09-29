# Repeated histogram execution

The fixed-shape `CudaHistogramContext` is an optional C++ interface for repeated tiled histograms. The simple `cuda_tile_histogram` interface and both histogram kernels remain available. Python remains CPU-only. There is no device-resident input mode or asynchronous result in this experiment.

## Ownership and completion

Construction fixes the image and tile dimensions, bin implementation and current CUDA device. The context owns one nonblocking stream, four timing events, a device image and device counts. Explicit device storage is bounded to 64 MiB per context. This limit excludes driver bookkeeping. Construction checks shape multiplication, the tile grid and storage size before acquiring CUDA resources. Empty images own stream and events but no device arrays.

Use, close and destroy a context on its creating thread with its original CUDA device current. `run` and `close` reject another thread or current device. Concurrent destruction or access is unsupported. This simple ownership rule avoids a second execution queue or implicit device switching. Create the context inside a dedicated worker if a worker owns its calls.

Each `run` repeats shape checks and allocates a separate host result. Input is a borrowed contiguous byte span. The caller keeps it alive and immutable from entry until return, including failure cleanup. Each call uploads the current pixels, computes all bins, downloads them and synchronizes the stream. It never assumes that a previous input is still valid. The returned vector owns its memory and remains unchanged after later calls or context destruction. Numerical behavior is identical to the existing histogram contract: exact uint64 counts, row-major tiles and partial edges, with no padded pixels.

Accepted calls have increasing request IDs starting at one. Identity is the pair of context instance and request ID, not a process-wide identifier. Invalid shape or host allocation failure before admission does not consume an ID. An execution failure may consume an ID without returning a result. There is no wraparound. Empty valid calls complete with empty output and advance identity.

Only successful stream synchronization permits a completed result and another invocation. A host or CUDA exception after submission drains the stream while the input and host result still exist, releases context resources and permanently retires that context. Partial construction releases resources already acquired. Cleanup is best effort if CUDA itself cannot synchronize or release a resource. That case is an error, not successful completion, recovery or permission to reuse storage. Actual device loss is not deliberately induced on a shared desktop.

`close` is idempotent and attempts synchronization and cleanup. Its synchronization error is reported. The destructor also attempts synchronization and cleanup without throwing. The caller must not destroy a context while one of its methods is active. There is no outstanding asynchronous work after a successful call, and shutdown does not require polling a result queue.

## Cancellation and staleness

The context has no cancellation API. If it is called by the existing `Runtime`, queued cancellation can prevent the callable from starting. A cancellation request during execution cannot interrupt CUDA work. The callable must retain the context and input until `run` finishes. Its ticket reaches physical completion only after the synchronous call returns and its captures retire. Consumer staleness does not change these ownership rules. No Adaptive Timing extension is needed for this synchronous contract.

## Experiment boundaries

Context setup and retirement are recorded separately from repeated execution. Each measured call includes validation, fresh host result allocation, current-input upload, the same histogram kernel, download and completed synchronization. It ends with an owned usable host result. Host result comparison and destruction occur afterward for every backend. This differs from the earlier report's outer boundary, which included host result destruction. Compare the new paths within this experiment and preserve the older report as its own evidence.

The context retains device allocation, streams and events between calls. No pinned memory, multiple streams, transfer overlap or zero-copy behavior is claimed. The separate fault-test executable injects host exceptions around acquisition and submission, including after a download is queued. It is excluded from the installed library and benchmark. Those tests do not establish real device-loss recovery or replace the unavailable GPU sanitizer.
