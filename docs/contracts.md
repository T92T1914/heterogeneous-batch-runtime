# Numerical and ownership contracts

These are the initial implementation decisions for the three named workloads. They make the API testable before backend or timing comparisons.

## Inputs and results

`masked_reduce` sums finite binary64 values selected by a byte mask. Zero means excluded. Any nonzero byte means included. Empty input returns positive zero. Lengths must match. NaN and infinity are rejected even when masked out. The scalar reference accumulates in input order. The optimized path combines fixed partitions and four SIMD lanes, so reassociation can change rounding. Bitwise equality across backends is not promised. Verification compares against `math.fsum` using `8 * epsilon * max(1, n) * sum(abs(selected_values))`, with an absolute floor of `1e-12`. Before execution, an upward-rounded bound checks the selected absolute sum. It rejects over-range inputs and can conservatively reject some inputs extremely close to that limit. It does not rely on extended precision being available.

`tile_histogram` accepts a row-major byte image and positive tile dimensions. It returns row-major tiles, each with 256 uint64 bins. Partial edge tiles include only real pixels. Empty dimensions yield an empty result with the corresponding tile shape. Counts and total conservation are exact. Optimized CPU workers own disjoint tiles and use four private banks to reduce repeated-bin dependencies.

`stencil3x3` computes the mean of each interior 3 by 3 neighborhood, dividing each input by nine before summation. The outer border is copied unchanged. Images smaller than 3 in either dimension are copied. NaN and infinity are rejected. The scalar and AVX2 implementations preserve the addition order within each output. A mean of finite values is finite. At the representable limits, the final rounded addition can overflow, so the output is bounded to the largest finite magnitude. Portable checks allow relative and absolute error of `1e-13` against an independently accumulated oracle for the declared bounded fixtures. This tolerance is not a universal accuracy guarantee for arbitrary ill-conditioned input.

Shape multiplication is checked before allocation. Thread counts must be between 1 and 64. The shared-machine examples use at most two kernel threads. Allocation failures propagate. Inputs are never modified. Results own their storage.

The native synchronous API borrows immutable spans for the duration of the call. The caller must prevent concurrent mutation and keep those spans alive until return. Python accepts only exact native float64 or uint8 arrays with C contiguous layout and the documented rank. It does not silently cast or reshape. Each input is copied while the GIL is held, and native work releases the GIL only after that snapshot. The caller must still prevent mutation from external native threads during the copy. Output arrays own copies of completed results. Conversion and snapshot costs belong in Python end-to-end measurements.

## Runtime lifetime

`Runtime` owns a bounded queue and joins its worker threads on destruction. Submitted callables must capture shared or owned storage, rather than references whose lifetime ends before completion. Queue overflow is a visible rejection. Shutdown drains accepted jobs and waits for running work. Calling runtime destruction from its own worker is unsupported.

A queued ticket can confirm cancellation before its callable starts. Its captures remain retained until the queue removes that cancelled job. A running ticket only records a cancellation request. It does not interrupt numerical work, release resources early or change physical completion to cancellation. After successful work, captures are destroyed before `completed` is published. Failures retain their exception for `wait` and report `failed`. Staleness belongs to the consumer's reconciliation policy, not to physical execution state.

AVX2 instructions live in a separate compilation unit. Runtime detection checks CPU and OS support. Unsupported machines take the portable implementation. The whole library is not compiled with AVX2 enabled.

## Evidence boundaries

Correctness tests, instrumented profiles, warm native kernel timings, queue latency and Python end-to-end timings are separate evidence. An optimized backend name is not a measured speedup. Validation and result allocation are included in the native benchmark. A GPU callable must synchronize completed work before publishing physical completion. Merely submitting a launch does not satisfy this contract.
