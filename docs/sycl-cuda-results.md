# SYCL and CUDA on one RTX 4090, measured on a working desktop

In two runs of the same build, the complete SYCL call had a lower median than the complete CUDA call for all three workloads at all three image sizes. The gap was between 0.7 and 11.3 percent of the CUDA median. It was smallest at 1024 by 1024 for every workload in both runs, where transfers and kernels take more of each call, while 32 by 32 and 256 by 256 showed no consistent order. The calls vary a lot around those medians. In each condition, between 34 and 60 of the 120 back to back pairs had SYCL slower than CUDA. In the first run's 1024 by 1024 stencil exactly half did, and its pairs with CUDA first had a median ratio above 1. Both backends gave the same checksums in every condition.

I measured this on a working desktop. The usual applications kept using the card during both runs. The idle readings, taken after a pause of two seconds with no call from the benchmark, showed 14 to 36 percent utilization, which is the desktop's alone. The readings taken before each call showed 14 to 54 percent. Memory is different, because NVML counts every process on the card, this benchmark included. The readings taken inside the process showed 7809 to 7981 MiB in use, while nvidia-smi on Windows read 7237 and 7030 MiB just before and after the first run and 7141 and 7146 MiB around the second, with 33, 31, 26 and 24 percent utilization. Within a run, the readings taken inside the process sat 572 to 908 MiB above the outside ones in the first and 806 to 840 MiB above them in the second. That difference is most likely what the benchmark's two runtimes held, though the two kinds of reading come through different interfaces. Each CUDA call ran back to back with the SYCL call of the same workload, so both calls of a pair met nearly the same load, but that load sits inside every timing. These are not the numbers an idle card would give. This compares the complete synchronous calls of the two libraries on one card under WSL. It makes no general claim about SYCL, CUDA, native Linux or an idle card, and it is not evidence from a second GPU vendor.

The [protocol](sycl-cuda-protocol.json) was committed at `92179878456986940d6e60a6c87b726a80edb775` and amended at `2db1fab2cab940c457131b4f790fc5e8d4c8301b`, both before any sample. The amendment let the run go ahead at the desktop load, paired the calls, raised the samples to 121 for each condition and added a reading of the card before every recorded call. Both runs used the binary with SHA-256 `2932070fdfbfdd3a17c172d34104316c1aa8b86d2f2e801829e6239bc2dc87fc`, built from the amended source, and the protocol file with SHA-256 `1017a32df752f6812240db0b250bbb2e532eda9c5383ea5d67feb010397aed51`.

## Two runs

The first run started at 06:56:38 UTC on November 23, 2026, from source `1c31bb781251e6561bb2e0a63c84b1b9ae357ff2`, and passed. Its receipt recorded the absolute path of the checkout on the measuring computer, which names a local account, so that receipt stays out of this public repository. Its [samples](evidence/sycl-cuda-attempt-1.csv) are published unchanged, with SHA-256 `4d5ee831b4a92c18c83c8883f1e22eac4206a7a35ba176267dde3c533c475d79`, the value its receipt records. Its idle readings were 15, 17, 18 and 36 percent, with 7810, 7912, 7887 and 7844 MiB in use.

I changed the runner to record only that the build came from the same checkout, and ran once more. Before that second run I decided to publish both runs and not to choose between them by their numbers. The second run started at 06:59:12 UTC from source `968c0e825b9ae4536b1a51d0ff9705fa327bbecb` and passed. Its [execution receipt](evidence/sycl-cuda-execution.json) names the source, the protocol, the binary, the command, the compilers, the WSL kernel, the GPU state before and after the run and the idle readings. Its [samples](evidence/sycl-cuda.csv) hold all 2178 rows, with SHA-256 `2e6f08ea7287bf7b35d4ca3c2e4c368b793af72b299d80677f9bc218b6f63d4f`. `python tools/validate_evidence.py` checks both CSV files against their recorded hashes and every row against the protocol. It checks that each run's tables below are the ones `tools/report_sycl_cuda.py` makes from that run's samples, that the hashes and revisions on this page are the recorded ones, and that the range of the gap and the count of pairs with SYCL slower in the first paragraph follow from the samples.

## Second run

All durations are milliseconds on the host clock. Each median is over 120 warm samples. Iteration zero of each condition is kept in the CSV and shown in the third table, and two unrecorded warmup rounds follow it. The first call of each backend in the process falls in iteration zero of the 32 by 32 condition, so it may include loading device code. Iteration zero of every size also follows a pause of two seconds for the idle reading. It is neither a warm measurement nor a cold process measurement.

The first table puts the medians side by side. The second divides the SYCL call by the CUDA call within each pair and takes the median of those 120 ratios, then the medians of the 60 pairs where CUDA went first and the 60 where SYCL went first. The fourth table summarizes the reading taken before each recorded call. That reading covers the last NVML sample period, between 1/6 and 1 second, so it includes kernels of this benchmark's own earlier calls as well as the desktop's. The idle utilization shows the desktop alone. Every memory figure also counts what this process holds. The protocol calls the idle reading the desktop load alone, which holds for utilization and not for memory.

| Image | Workload | CUDA median | SYCL median | SYCL ÷ CUDA | CUDA middle 50 percent | SYCL middle 50 percent | Same checksum |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 32 × 32 | masked_reduce | 0.74815 | 0.67154 | 0.90 | 0.66301 to 0.87908 | 0.61884 to 0.74766 | yes |
| 32 × 32 | tile_histogram | 0.74320 | 0.66213 | 0.89 | 0.63987 to 0.86344 | 0.60704 to 0.74212 | yes |
| 32 × 32 | stencil3x3 | 0.74068 | 0.68802 | 0.93 | 0.65842 to 0.90667 | 0.62458 to 0.76862 | yes |
| 256 × 256 | masked_reduce | 0.90963 | 0.87494 | 0.96 | 0.83891 to 1.02938 | 0.81293 to 0.93081 | yes |
| 256 × 256 | tile_histogram | 0.66164 | 0.62055 | 0.94 | 0.61228 to 0.81824 | 0.56114 to 0.70303 | yes |
| 256 × 256 | stencil3x3 | 0.89294 | 0.82190 | 0.92 | 0.80399 to 1.04872 | 0.77350 to 0.89914 | yes |
| 1024 × 1024 | masked_reduce | 4.35679 | 4.24679 | 0.97 | 4.15613 to 4.70289 | 4.12147 to 4.55764 | yes |
| 1024 × 1024 | tile_histogram | 1.00534 | 0.94644 | 0.94 | 0.90456 to 1.15614 | 0.87139 to 1.09964 | yes |
| 1024 × 1024 | stencil3x3 | 3.76095 | 3.63574 | 0.97 | 3.58525 to 3.97136 | 3.45275 to 3.83871 | yes |

| Image | Workload | Pairs | Median of SYCL ÷ CUDA in a pair | Middle 50 percent | Full range | CUDA first | SYCL first |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 32 × 32 | masked_reduce | 120 | 0.901 | 0.793 to 1.060 | 0.252 to 2.636 | 0.892 | 0.901 |
| 32 × 32 | tile_histogram | 120 | 0.892 | 0.745 to 1.075 | 0.353 to 1.830 | 0.865 | 0.937 |
| 32 × 32 | stencil3x3 | 120 | 0.919 | 0.765 to 1.072 | 0.161 to 2.197 | 0.925 | 0.915 |
| 256 × 256 | masked_reduce | 120 | 0.937 | 0.818 to 1.037 | 0.327 to 2.041 | 0.934 | 0.937 |
| 256 × 256 | tile_histogram | 120 | 0.923 | 0.784 to 1.041 | 0.356 to 1.568 | 0.952 | 0.896 |
| 256 × 256 | stencil3x3 | 120 | 0.923 | 0.829 to 1.036 | 0.467 to 1.862 | 0.936 | 0.910 |
| 1024 × 1024 | masked_reduce | 120 | 0.978 | 0.912 to 1.061 | 0.617 to 1.535 | 0.974 | 0.991 |
| 1024 × 1024 | tile_histogram | 120 | 0.962 | 0.846 to 1.073 | 0.361 to 2.149 | 0.950 | 0.977 |
| 1024 × 1024 | stencil3x3 | 120 | 0.973 | 0.905 to 1.021 | 0.621 to 1.896 | 0.978 | 0.961 |

| Image | Workload | Backend | Warm samples | Median | Full range | Iteration zero | CUDA upload | CUDA kernel | CUDA download |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 32 × 32 | masked_reduce | cuda | 120 | 0.74815 | 0.51102 to 2.79493 | 3.84005 | 0.00614 | 0.00819 | 0.04045 |
| 32 × 32 | masked_reduce | sycl | 120 | 0.67154 | 0.46806 to 3.52694 | 1.94139 | none | none | none |
| 32 × 32 | tile_histogram | cuda | 120 | 0.74320 | 0.51354 to 1.99024 | 0.93922 | 0.00410 | 0.00512 | 0.03955 |
| 32 × 32 | tile_histogram | sycl | 120 | 0.66213 | 0.45581 to 1.74288 | 1.02918 | none | none | none |
| 32 × 32 | stencil3x3 | cuda | 120 | 0.74068 | 0.50962 to 5.35838 | 0.78128 | 0.00410 | 0.00922 | 0.03949 |
| 32 × 32 | stencil3x3 | sycl | 120 | 0.68802 | 0.47963 to 2.16517 | 1.34296 | none | none | none |
| 256 × 256 | masked_reduce | cuda | 120 | 0.90963 | 0.69484 to 2.21016 | 0.82325 | 0.10291 | 0.00922 | 0.03440 |
| 256 × 256 | masked_reduce | sycl | 120 | 0.87494 | 0.65971 to 2.40975 | 0.83738 | none | none | none |
| 256 × 256 | tile_histogram | cuda | 120 | 0.66164 | 0.48571 to 2.17728 | 0.88378 | 0.00819 | 0.00922 | 0.03891 |
| 256 × 256 | tile_histogram | sycl | 120 | 0.62055 | 0.46713 to 1.34956 | 0.76109 | none | none | none |
| 256 × 256 | stencil3x3 | cuda | 120 | 0.89294 | 0.67231 to 1.71297 | 1.22182 | 0.06946 | 0.03730 | 0.11325 |
| 256 × 256 | stencil3x3 | sycl | 120 | 0.82190 | 0.63221 to 1.83882 | 0.84594 | none | none | none |
| 1024 × 1024 | masked_reduce | cuda | 120 | 4.35679 | 3.74517 to 7.56823 | 4.55692 | 0.75690 | 0.14462 | 0.03712 |
| 1024 × 1024 | masked_reduce | sycl | 120 | 4.24679 | 3.76685 to 7.25979 | 4.75798 | none | none | none |
| 1024 × 1024 | tile_histogram | cuda | 120 | 1.00534 | 0.75242 to 3.20460 | 1.15056 | 0.10986 | 0.04238 | 0.11786 |
| 1024 × 1024 | tile_histogram | sycl | 120 | 0.94644 | 0.66603 to 2.68287 | 0.91389 | none | none | none |
| 1024 × 1024 | stencil3x3 | cuda | 120 | 3.76095 | 3.31170 to 6.91840 | 8.02572 | 0.68216 | 0.31816 | 0.77584 |
| 1024 × 1024 | stencil3x3 | sycl | 120 | 3.63574 | 3.25364 to 7.12196 | 7.42759 | none | none | none |

| Image | Readings | GPU utilization median | Utilization range | Memory in use median | Memory range |
| --- | ---: | ---: | --- | ---: | --- |
| 32 × 32 | 726 | 27 percent | 21 to 31 percent | 7962 MiB | 7952 to 7963 MiB |
| 256 × 256 | 726 | 29 percent | 14 to 31 percent | 7962 MiB | 7958 to 7965 MiB |
| 1024 × 1024 | 726 | 41 percent | 23 to 44 percent | 7965 MiB | 7963 to 7981 MiB |

| Idle reading | GPU utilization | Memory in use |
| --- | ---: | ---: |
| before 32 | 21 percent | 7952 MiB |
| before 256 | 14 percent | 7962 MiB |
| before 1024 | 23 percent | 7966 MiB |
| after 1024 | 25 percent | 7960 MiB |

## First run

The same tables from the first run's samples. Its idle readings are given above, since its receipt is not published.

| Image | Workload | CUDA median | SYCL median | SYCL ÷ CUDA | CUDA middle 50 percent | SYCL middle 50 percent | Same checksum |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 32 × 32 | masked_reduce | 0.73741 | 0.67852 | 0.92 | 0.65399 to 0.89740 | 0.62575 to 0.76876 | yes |
| 32 × 32 | tile_histogram | 0.71994 | 0.63833 | 0.89 | 0.64192 to 0.85955 | 0.59409 to 0.71088 | yes |
| 32 × 32 | stencil3x3 | 0.72100 | 0.66846 | 0.93 | 0.65242 to 0.82003 | 0.61245 to 0.73785 | yes |
| 256 × 256 | masked_reduce | 1.02540 | 0.92964 | 0.91 | 0.92891 to 1.16930 | 0.87022 to 1.00375 | yes |
| 256 × 256 | tile_histogram | 0.76492 | 0.69720 | 0.91 | 0.67461 to 0.91055 | 0.65034 to 0.78744 | yes |
| 256 × 256 | stencil3x3 | 0.99520 | 0.90186 | 0.91 | 0.89060 to 1.09982 | 0.83068 to 0.98343 | yes |
| 1024 × 1024 | masked_reduce | 4.54416 | 4.51441 | 0.99 | 4.25525 to 5.37172 | 4.15386 to 5.40045 | yes |
| 1024 × 1024 | tile_histogram | 1.05451 | 0.99879 | 0.95 | 0.94984 to 1.21168 | 0.89289 to 1.16034 | yes |
| 1024 × 1024 | stencil3x3 | 3.80084 | 3.70303 | 0.97 | 3.57100 to 4.19352 | 3.53794 to 4.11947 | yes |

| Image | Workload | Pairs | Median of SYCL ÷ CUDA in a pair | Middle 50 percent | Full range | CUDA first | SYCL first |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 32 × 32 | masked_reduce | 120 | 0.929 | 0.766 to 1.076 | 0.447 to 8.231 | 0.882 | 0.962 |
| 32 × 32 | tile_histogram | 120 | 0.891 | 0.772 to 1.055 | 0.260 to 2.327 | 0.898 | 0.877 |
| 32 × 32 | stencil3x3 | 120 | 0.936 | 0.784 to 1.061 | 0.275 to 2.032 | 0.935 | 0.936 |
| 256 × 256 | masked_reduce | 120 | 0.899 | 0.751 to 1.029 | 0.220 to 2.304 | 0.902 | 0.896 |
| 256 × 256 | tile_histogram | 120 | 0.919 | 0.810 to 1.063 | 0.302 to 1.593 | 0.869 | 0.951 |
| 256 × 256 | stencil3x3 | 120 | 0.918 | 0.796 to 1.050 | 0.312 to 1.927 | 0.918 | 0.916 |
| 1024 × 1024 | masked_reduce | 120 | 0.979 | 0.899 to 1.092 | 0.604 to 1.793 | 0.985 | 0.975 |
| 1024 × 1024 | tile_histogram | 120 | 0.962 | 0.811 to 1.137 | 0.336 to 2.114 | 0.916 | 0.978 |
| 1024 × 1024 | stencil3x3 | 120 | 0.999 | 0.907 to 1.055 | 0.604 to 1.937 | 1.022 | 0.967 |

| Image | Workload | Backend | Warm samples | Median | Full range | Iteration zero | CUDA upload | CUDA kernel | CUDA download |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 32 × 32 | masked_reduce | cuda | 120 | 0.73741 | 0.52590 to 1.52219 | 3.91793 | 0.00614 | 0.00819 | 0.04411 |
| 32 × 32 | masked_reduce | sycl | 120 | 0.67852 | 0.47693 to 5.77656 | 2.27144 | none | none | none |
| 32 × 32 | tile_histogram | cuda | 120 | 0.71994 | 0.46748 to 2.82194 | 1.27303 | 0.00410 | 0.00512 | 0.04045 |
| 32 × 32 | tile_histogram | sycl | 120 | 0.63833 | 0.46924 to 2.22077 | 1.24258 | none | none | none |
| 32 × 32 | stencil3x3 | cuda | 120 | 0.72100 | 0.49215 to 2.13838 | 0.82069 | 0.00410 | 0.00922 | 0.05584 |
| 32 × 32 | stencil3x3 | sycl | 120 | 0.66846 | 0.46857 to 1.47699 | 0.74738 | none | none | none |
| 256 × 256 | masked_reduce | cuda | 120 | 1.02540 | 0.73364 to 4.32285 | 1.16775 | 0.11110 | 0.00922 | 0.03686 |
| 256 × 256 | masked_reduce | sycl | 120 | 0.92964 | 0.73879 to 2.48022 | 1.00500 | none | none | none |
| 256 × 256 | tile_histogram | cuda | 120 | 0.76492 | 0.55077 to 2.37682 | 1.64345 | 0.00819 | 0.00922 | 0.04317 |
| 256 × 256 | tile_histogram | sycl | 120 | 0.69720 | 0.51103 to 2.29265 | 0.76273 | none | none | none |
| 256 × 256 | stencil3x3 | cuda | 120 | 0.99520 | 0.73032 to 3.15234 | 1.23266 | 0.07299 | 0.04402 | 0.11659 |
| 256 × 256 | stencil3x3 | sycl | 120 | 0.90186 | 0.68007 to 1.91053 | 0.95941 | none | none | none |
| 1024 × 1024 | masked_reduce | cuda | 120 | 4.54416 | 3.84614 to 12.66106 | 4.24174 | 0.77453 | 0.12318 | 0.03891 |
| 1024 × 1024 | masked_reduce | sycl | 120 | 4.51441 | 3.76167 to 11.16202 | 4.13862 | none | none | none |
| 1024 × 1024 | tile_histogram | cuda | 120 | 1.05451 | 0.71760 to 3.59695 | 1.18729 | 0.11198 | 0.04147 | 0.12365 |
| 1024 × 1024 | tile_histogram | sycl | 120 | 0.99879 | 0.65346 to 2.53588 | 0.85128 | none | none | none |
| 1024 × 1024 | stencil3x3 | cuda | 120 | 3.80084 | 3.21360 to 8.23059 | 7.59120 | 0.68670 | 0.23939 | 0.76858 |
| 1024 × 1024 | stencil3x3 | sycl | 120 | 3.70303 | 3.17418 to 9.36253 | 7.41507 | none | none | none |

| Image | Readings | GPU utilization median | Utilization range | Memory in use median | Memory range |
| --- | ---: | ---: | --- | ---: | --- |
| 32 × 32 | 726 | 28 percent | 15 to 34 percent | 7810 MiB | 7809 to 7817 MiB |
| 256 × 256 | 726 | 28 percent | 17 to 39 percent | 7874 MiB | 7866 to 7912 MiB |
| 1024 × 1024 | 726 | 41 percent | 18 to 54 percent | 7892 MiB | 7842 to 7938 MiB |

## What each timing includes

The host clock starts when the call is entered and stops when it returns an owned, completed result. Both backends validate the input, allocate host results, make one device allocation call for each buffer, upload, run the kernels, download, wait and release device memory inside that interval. The CUDA call also creates a stream and four events, records the events and reads three elapsed times. The SYCL call also selects a GPU among those that can run every kernel of the library, checks its `fp64` or `atomic64` aspect and creates and drains an in order queue, on every call. Input generation, device discovery, the NVML readings and idle pauses, comparison with the references, the checksum, result destruction and the CSV output stay outside it. The protocol lists each step.

The CUDA call does more of its own bookkeeping, a stream, four events and three elapsed time queries, and that may account for part of the gap at the small sizes. Nothing here separates those costs, so that stays a possible explanation and not a finding.

CUDA rows also carry the library's own upload, kernel and download event times. They are never added to the host time. The SYCL library records no device events, so only the complete call is compared across backends, and nothing here compares kernels alone.

## Correctness

At 00:52 CST on November 23, 2026, with this binary on this card, the eight `cuda_` and `sycl_` tests passed and `hbr_sycl_cuda_bench --verify` passed all nine conditions against the scalar reference, with the two backends equal bit for bit in all nine. During both runs every recorded and warmup output was compared with the scalar reference, exactly for histogram counts, within `1e-9` for the reduction and within `1e-12` per stencil value, and every reduction and stencil output stayed equal bit for bit to the first one of its backend, workload and size.

## Conditions and limits

The computer has an AMD processor that Windows reports as AMD64 Family 25 Model 97 and an NVIDIA GeForce RTX 4090 with Windows driver 621.24, which the SYCL runtime reports as CUDA 13.5. Both backends were built in one CMake build in the WSL distro, with DPC++ 7.2.0 (Clang 23.1.1) as the C++ compiler and nvcc 13.2.86 with g++ 13.3.0 for host code, and ran in one process on one thread under WSL kernel 6.18.33.2. No clock, power, driver or display setting changed. Each run took about 15 seconds.

These are three calls at three sizes in two short runs on one card. The fixed costs of each call, such as allocation, queue or stream setup and synchronization, are a large part of what this measures, and the larger images show more of the transfer and kernel time. Nothing here pins host memory, keeps device memory between calls or overlaps transfers. The CUDA rows were built with nvcc and g++ under WSL, so they are not comparable with the earlier CUDA runs, which were built with MSVC on Windows. The [open gates](limits.md) stay as they are.

## Reproduce

Build both backends in one WSL build as [`sycl.md`](sycl.md) describes, with `-DHBR_CUDA=ON` and `-DHBR_SYCL=ON`. Check that no game is running, run the `cuda_` and `sycl_` tests and `hbr_sycl_cuda_bench --verify` with that build, then run it once from a clean checkout on Windows.

```text
python tools/run_sycl_cuda.py /opt/hbr-sycl-cuda/build/hbr_sycl_cuda_bench --build-dir /opt/hbr-sycl-cuda/build --confirm-desktop-load
python tools/report_sycl_cuda.py docs/evidence/sycl-cuda.csv --receipt docs/evidence/sycl-cuda-execution.json
```

From Git Bash, set `MSYS_NO_PATHCONV=1` first, or the shell rewrites the `/opt` paths before Python sees them.
