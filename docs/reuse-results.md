# Reusing CUDA histogram resources

I added one optional fixed-shape histogram context and measured it against the existing synchronous CUDA call and optimized CPU paths. Reusing allocations, the stream and events reduced repeated-call medians in this run. Small images still favored the CPU. At 1024 by 1024, the reused shared-bin uniform case and reused global-bin concentrated case had lower medians than the faster tested CPU path. This is a bounded repeated-work result, not a general GPU ranking or reason to change the default interface.

The measured source is `caa7391fdf20efe0a712d500043333e05541f555`. The [numerical and ownership contract](reuse-contract.md) and [protocol](reuse-protocol.json) were committed at `7f742572aebeff14be4321b532d00ad91301e150` before collection. The [execution receipt](evidence/reuse-execution.json) identifies the source tree, executable, inputs/protocol and exact raw output. The original [504-row study](results-2026-09-29.md) remains unchanged.

## Completed calls

All durations below are milliseconds. Every cell is the median of 24 warm samples. Each condition also retains its first invocation and two additional unrecorded warmup calls. The first invocation is not a cold-process measurement. Backend order rotates across each iteration. The [raw CSV](evidence/reuse.csv) contains 900 complete calls and 24 setup/retirement observations. Every output, including warmups, was compared bin-for-bin with an independently indexed oracle.

| Image | Input | CPU, one thread | CPU, two threads | Simple global | Reused global | Simple shared | Reused shared |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 × 32 | uniform | 0.00060 | 0.00050 | 0.04695 | 0.04205 | 0.05035 | 0.05005 |
| 32 × 32 | single_bin | 0.00050 | 0.00050 | 0.04510 | 0.03990 | 0.07050 | 0.06580 |
| 256 × 256 | uniform | 0.01845 | 0.12275 | 0.10265 | 0.05600 | 0.08530 | 0.05970 |
| 256 × 256 | single_bin | 0.01860 | 0.12170 | 0.09710 | 0.05370 | 0.15230 | 0.13995 |
| 1024 × 1024 | uniform | 0.37270 | 0.31800 | 0.70770 | 0.30890 | 0.65315 | 0.27150 |
| 1024 × 1024 | single_bin | 0.38155 | 0.35620 | 0.67180 | 0.29795 | 0.73015 | 0.36600 |

The host clock starts before dispatch and ends with an owned usable host result. Every backend includes validation and host output allocation. Both CUDA paths include the current input upload, the same selected kernel, output download and synchronization. The simple API also releases its CUDA resources inside the call. The context keeps those resources until retirement. Full output comparison, checksum calculation and host result destruction occur outside every timed call. This boundary differs from the original report, which included host result destruction. These new samples do not replace or relabel the old comparison.

One and two CPU threads are both shown. Thread creation costs matter, so selecting only the slower two-thread result would exaggerate the GPU comparison for smaller images. The two bin methods remain separate because concentrated input changes their behavior.

## Variability and setup

| 1024 × 1024 input | Reused method | Median | Middle 50 percent | Full range | Kernel median |
| --- | --- | ---: | --- | --- | ---: |
| uniform | global | 0.30890 | 0.30083 to 0.33988 | 0.29310 to 0.50070 | 0.05755 |
| uniform | shared | 0.27150 | 0.26950 to 0.28123 | 0.26490 to 0.60200 | 0.03682 |
| single_bin | global | 0.29795 | 0.29050 to 0.35915 | 0.28500 to 0.54340 | 0.05653 |
| single_bin | shared | 0.36600 | 0.36455 to 0.37420 | 0.36130 to 0.69290 | 0.13766 |

The [complete generated summary](evidence/reuse-summary.csv) includes counts, inclusive quartiles, minima, maxima and separate transfer/kernel medians for all 36 conditions. Device intervals are not added to host elapsed time. Wide maxima show why one fast observation is not a stable latency guarantee.

| Image | Input | Method | Setup | Retirement | Explicit device bytes |
| --- | --- | --- | ---: | ---: | ---: |
| 32 × 32 | uniform | global | 62.93670 | 0.00440 | 3072 |
| 32 × 32 | uniform | shared | 0.02310 | 0.14650 | 3072 |
| 32 × 32 | single_bin | global | 0.05400 | 0.00390 | 3072 |
| 32 × 32 | single_bin | shared | 0.01660 | 0.10920 | 3072 |
| 256 × 256 | uniform | global | 0.05360 | 0.01430 | 98304 |
| 256 × 256 | uniform | shared | 0.01660 | 0.14520 | 98304 |
| 256 × 256 | single_bin | global | 0.07250 | 0.00410 | 98304 |
| 256 × 256 | single_bin | shared | 0.01830 | 0.13910 | 98304 |
| 1024 × 1024 | uniform | global | 0.13060 | 0.00820 | 1572864 |
| 1024 × 1024 | uniform | shared | 0.06520 | 0.14650 | 1572864 |
| 1024 × 1024 | single_bin | global | 0.05990 | 0.00700 | 1572864 |
| 1024 × 1024 | single_bin | shared | 0.04880 | 0.19240 | 1572864 |

The first context setup includes lazy CUDA startup in this process. It is not the incremental cost of persistent storage alone, and the other backends later benefit from that initialized process. Setup is therefore shown separately, without a universal amortization or crossover claim. Every repeated call still moves host data both ways. There is no prepared device input, pinned host memory, transfer overlap or zero-copy claim. The two largest simultaneous contexts hold 3 MiB of explicit device arrays together, excluding driver bookkeeping. The API rejects an individual context above 64 MiB.

## Correctness, ownership and limits

Windows verification executed the native suite, existing CUDA numerical suite, new reuse suite, and separate fault executable on the RTX 4090. The reuse suite covers changed input, exact bins, empty/partial shapes, both kernels, retained and independently modified old results, request IDs, invalid-input recovery, wrong-thread rejection, close twice and run after close. An installed C++ consumer executed the new context.

Seven acquisition and eight execution checkpoints injected host exceptions. Cleanup synchronized successfully and released acquired resources, including the path after an output download had been submitted. A failed context then rejected reuse. The test-only injection code is absent from the installed library and benchmark. These are host-exception cleanup tests. They do not establish real CUDA allocation-failure or device-loss recovery.

The context belongs to its creating thread and CUDA device. Inputs remain immutable through return, and each result owns a distinct host vector. Request IDs are scoped to that context. Running cancellation cannot interrupt the operation, and logical staleness does not permit early buffer release. The simple API and CPU-only Python interface remain the defaults. Adaptive Timing needs no new behavior for a synchronous callable, and MCTS's previous no-port result is unchanged.

Actual WSL2 Linux Release and ASan/UBSan verification and an installed CPU consumer are recorded separately in the receipt. That is Linux CPU evidence, not Linux GPU execution or a separate physical machine. Hosted checks must be verified at the delivered revision. GPU hardware counters remain restricted by `ERR_NVGPUCTRPERM`, and Windows Compute Sanitizer's WDDM initialization remains unresolved. No GPU sanitizer pass is claimed.

This run used the Ryzen 7 7800X3D, RTX 4090, driver 617.14, MSVC 19.44 and CUDA 13.4.59 on a shared Windows workstation. GPU utilization was 18 percent before and 32 percent after collection. Ordinary applications stayed active. Only one task benchmark ran, with at most two CPU threads and a 120 second cap. No clocks, power, drivers, display or security settings changed. A single short shared-host batch cannot establish statistical independence, general speedup or production latency.

## Reproduce

Build with `-DHBR_CUDA=ON` for the actual GPU architecture, run CTest, then run `hbr_cuda_reuse_bench --run` once under the committed protocol. Keep setup and retirement rows and retain slower cases. `python tools/report_reuse.py --check` checks artifact identities, sample groups, run order, independent checksums, request IDs and generated tables. `--write` regenerates the report and summary from the retained CSV without rerunning a workload.

The optional context is useful when a caller already has repeated fixed-shape work and can retain its resources. The original simple interface remains the easier default. The measurements support exposing this choice and its costs, not changing every workload or adding an asynchronous framework.
