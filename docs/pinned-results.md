# Pinned and pageable input on one RTX 4090, measured on a working desktop

Pinned host input made no consistent difference to the complete CUDA call of any of the three workloads at any of the three image sizes. Within each condition I divided the pinned call by the pageable call in each back to back pair, and the median of the 120 paired ratios was between 0.963 and 1.045. In all nine conditions the middle half of those ratios includes 1, and between 40 and 74 of the 120 back to back pairs had the pinned call slower. The pinned median was lower in six conditions and higher in three. The largest gaps were 6.6 percent higher for the 32 by 32 masked reduction and 5.0 percent higher for the 1024 by 1024 histogram, and the largest gap the other way was 3.2 percent lower, for the 256 by 256 stencil. Pinned and pageable input gave the same checksums in every condition.

The upload alone shows no consistent gain either. At 1024 by 1024 the masked reduction's median upload moved 9437184 bytes in 0.751 ms from pageable memory and in 0.735 ms from pinned memory, about 12.6 and 12.8 GB/s. The stencil moved 8388608 bytes at about 12.5 and 12.8 GB/s. At the two smaller sizes the pinned upload median was higher in four of six conditions, by 37 to 83 percent, and lower for the 256 by 256 masked reduction and stencil, by 23 and 0.4 percent. Those uploads take 0.004 to 0.095 ms, and every event time at those two sizes is a multiple of 0.000032 ms. The library copies straight from the caller's buffer with `cudaMemcpyAsync`, so the pinned buffer does reach the copy. After the run nvidia-smi read the card's link as PCIe generation 3 with 16 lanes, both currently and as the maximum this system allows. That reading is not in the receipt, and this run does not test whether the link is what holds both kinds of memory to the same rate.

I corrected the sentence on the event times on November 27, 2026. It first said they move in steps of about 0.001 ms, which the samples do not show. At 32 by 32 and 256 by 256 every upload, kernel and download time is a multiple of 0.000032 ms, and two neighboring distinct times are most often exactly one such step apart. No sample, table or other number on this page changed.

Pinning has its own cost. Allocating the three pinned buffers of the 1024 by 1024 size, 10485760 bytes, took 0.900 ms and releasing them took 0.588 ms. Together that is about 32 times the 0.046 ms by which the pinned median of the 1024 by 1024 masked reduction was lower, and these samples do not show that gap to be real.

I measured this on a working desktop with no game running. The usual applications kept using the card. The four idle readings, each taken after a pause of two seconds with no call from the benchmark, showed 4 to 9 percent utilization, which is the desktop's alone. The readings taken before each recorded call had medians of 5, 11 and 30 percent for the three sizes and ranged from 4 to 35 percent, and they include the benchmark's own earlier kernels. Memory is different, because NVML counts every process on the card, this benchmark included. The readings taken inside the process showed 6597 to 6603 MiB in use, while nvidia-smi read 5836 MiB just before and just after the run, at 15 and 9 percent utilization. That difference of 761 to 767 MiB is most likely what the benchmark's CUDA context held, though the two kinds of reading are not taken the same way.

The [protocol](pinned-protocol.json) was amended at `982ce814103e56e9749eb145e9ce35f5c3e03a0c` on November 24, 2026, before any sample, and no sample was taken under its earlier version. The amendment let the run go ahead at the desktop load, paired the pageable and pinned calls back to back with the first kind alternating, raised the samples to 121 for each condition and added a reading of the card before every recorded call. The run used the binary with SHA-256 `e644ed7a9c33d98417d35538d27684accdef784e828f2d82e5d297598f136685`, built from the amended source, and the protocol file with SHA-256 `78ce235842c971b5559e53c579e1db56955c548065dbc86af9ac830080af4eaf`.

## The run

The run started at 10:24:48 UTC on November 25, 2026, from source `2efa1d5bf6d52098b3182f667bd150f50ef9ba6a`, and passed in about 13 seconds. Its [execution receipt](evidence/pinned-execution.json) names the source, the protocol, the binary, the compilers, the GPU state before and after the run and the idle readings. Its [samples](evidence/pinned.csv) hold all 2178 run rows and one setup and one retire row for each size, with SHA-256 `5c90cfd4b2f044a23d815b7d5199814341e984b401029ec044b26d3a143a8e25`. `python tools/validate_evidence.py` checks the CSV against that hash and every row against the protocol. It checks that the tables below are the ones `tools/report_pinned.py` makes from the samples and the receipt, that this page cites the recorded hashes and revisions, and that the range of the paired ratio medians, the count of pairs with the pinned call slower and the middle halves that include 1 follow from the samples.

## Results

All durations are milliseconds. Each median is over 120 warm samples. Iteration zero of each condition is kept in the CSV and shown as the first call in the third table, and two unrecorded warmup rounds follow it. Iteration zero of every size also follows a pause of two seconds for the idle reading, so it is neither a warm measurement nor a cold process measurement.

The first table puts the medians side by side. The complete call is on the host clock, while the upload and download columns are the library's own stream event times. Every call downloads into memory the runtime owns, so the download is pageable for both kinds and serves as a control. The second table divides the pinned call by the pageable call within each pair and takes the median of those 120 ratios, then the medians of the 60 pairs where pageable went first and the 60 where pinned went first. The fourth table times the pinned allocation and release for each size, outside every run row. The last two summarize the reading taken before each recorded call and the four idle readings.

| Image | Workload | Upload bytes | Complete call, pageable | Complete call, pinned | Upload, pageable | Upload, pinned | Download, pageable | Download, pinned |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 × 32 | masked_reduce | 9216 | 0.35995 | 0.38360 | 0.00614 | 0.01126 | 0.01229 | 0.01786 |
| 32 × 32 | tile_histogram | 1024 | 0.35175 | 0.34300 | 0.00410 | 0.00704 | 0.01126 | 0.01126 |
| 32 × 32 | stencil3x3 | 8192 | 0.35075 | 0.36140 | 0.00499 | 0.00717 | 0.01229 | 0.01229 |
| 256 × 256 | masked_reduce | 589824 | 0.80540 | 0.78415 | 0.09523 | 0.07317 | 0.01126 | 0.01126 |
| 256 × 256 | tile_histogram | 65536 | 0.38515 | 0.37485 | 0.00819 | 0.01126 | 0.01843 | 0.01536 |
| 256 × 256 | stencil3x3 | 524288 | 0.66705 | 0.64590 | 0.06630 | 0.06603 | 0.08952 | 0.08963 |
| 1024 × 1024 | masked_reduce | 9437184 | 6.71690 | 6.67065 | 0.75082 | 0.73491 | 0.01126 | 0.01286 |
| 1024 × 1024 | tile_histogram | 1048576 | 0.64700 | 0.67910 | 0.10642 | 0.10685 | 0.09653 | 0.09597 |
| 1024 × 1024 | stencil3x3 | 8388608 | 5.39570 | 5.35515 | 0.66941 | 0.65528 | 0.72763 | 0.72232 |

| Image | Workload | Pairs | Median of pinned ÷ pageable in a pair | Middle 50 percent | Full range | Pageable first | Pinned first |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 32 × 32 | masked_reduce | 120 | 1.045 | 0.954 to 1.163 | 0.538 to 1.953 | 1.029 | 1.052 |
| 32 × 32 | tile_histogram | 120 | 0.989 | 0.885 to 1.056 | 0.501 to 1.742 | 0.949 | 1.017 |
| 32 × 32 | stencil3x3 | 120 | 1.023 | 0.946 to 1.106 | 0.493 to 2.060 | 1.025 | 1.008 |
| 256 × 256 | masked_reduce | 120 | 0.989 | 0.931 to 1.065 | 0.570 to 1.799 | 1.017 | 0.970 |
| 256 × 256 | tile_histogram | 120 | 0.996 | 0.842 to 1.101 | 0.196 to 1.603 | 0.977 | 1.013 |
| 256 × 256 | stencil3x3 | 120 | 0.963 | 0.909 to 1.040 | 0.650 to 1.682 | 0.961 | 0.967 |
| 1024 × 1024 | masked_reduce | 120 | 0.997 | 0.949 to 1.053 | 0.758 to 1.991 | 0.990 | 0.999 |
| 1024 × 1024 | tile_histogram | 120 | 1.025 | 0.903 to 1.208 | 0.223 to 2.978 | 0.966 | 1.099 |
| 1024 × 1024 | stencil3x3 | 120 | 0.994 | 0.946 to 1.026 | 0.817 to 1.276 | 1.009 | 0.988 |

| Image | Workload | Memory | Warm samples | Median | Middle 50 percent | Full range | First call | Kernel median |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 32 × 32 | masked_reduce | pageable | 120 | 0.35995 | 0.33145 to 0.40687 | 0.30520 to 0.64310 | 0.99790 | 0.00819 |
| 32 × 32 | masked_reduce | pinned | 120 | 0.38360 | 0.34860 to 0.42863 | 0.31790 to 0.80240 | 0.50400 | 0.00922 |
| 32 × 32 | tile_histogram | pageable | 120 | 0.35175 | 0.31830 to 0.39785 | 0.29920 to 0.62870 | 0.42940 | 0.00512 |
| 32 × 32 | tile_histogram | pinned | 120 | 0.34300 | 0.31982 to 0.37627 | 0.30250 to 0.61200 | 0.59830 | 0.00512 |
| 32 × 32 | stencil3x3 | pageable | 120 | 0.35075 | 0.32720 to 0.39488 | 0.30460 to 0.76840 | 0.43700 | 0.00922 |
| 32 × 32 | stencil3x3 | pinned | 120 | 0.36140 | 0.33083 to 0.40595 | 0.30650 to 0.69820 | 0.48640 | 0.00819 |
| 256 × 256 | masked_reduce | pageable | 120 | 0.80540 | 0.75073 to 0.85548 | 0.70170 to 2.44320 | 0.90440 | 0.00922 |
| 256 × 256 | masked_reduce | pinned | 120 | 0.78415 | 0.73723 to 0.86288 | 0.69270 to 1.45980 | 0.88340 | 0.02475 |
| 256 × 256 | tile_histogram | pageable | 120 | 0.38515 | 0.35757 to 0.44575 | 0.32470 to 2.29700 | 0.74420 | 0.00922 |
| 256 × 256 | tile_histogram | pinned | 120 | 0.37485 | 0.35170 to 0.43830 | 0.31750 to 0.59850 | 0.52030 | 0.00922 |
| 256 × 256 | stencil3x3 | pageable | 120 | 0.66705 | 0.63077 to 0.72600 | 0.57850 to 0.95550 | 0.76340 | 0.03530 |
| 256 × 256 | stencil3x3 | pinned | 120 | 0.64590 | 0.60732 to 0.70638 | 0.57240 to 0.99100 | 0.84910 | 0.03486 |
| 1024 × 1024 | masked_reduce | pageable | 120 | 6.71690 | 6.52323 to 6.94630 | 6.18670 to 8.57060 | 7.12380 | 0.09282 |
| 1024 × 1024 | masked_reduce | pinned | 120 | 6.67065 | 6.49522 to 6.98420 | 6.33510 to 13.10430 | 6.83510 | 0.09382 |
| 1024 × 1024 | tile_histogram | pageable | 120 | 0.64700 | 0.60267 to 0.77195 | 0.54980 to 3.36200 | 0.66020 | 0.04144 |
| 1024 × 1024 | tile_histogram | pinned | 120 | 0.67910 | 0.61262 to 0.79115 | 0.55350 to 2.38250 | 0.56690 | 0.04288 |
| 1024 × 1024 | stencil3x3 | pageable | 120 | 5.39570 | 5.27055 to 5.59392 | 5.04280 to 6.51330 | 5.92930 | 0.46971 |
| 1024 × 1024 | stencil3x3 | pinned | 120 | 5.35515 | 5.20625 to 5.53185 | 4.95060 to 6.74670 | 5.30380 | 0.48406 |

| Image | Pinned bytes | Allocation | Release |
| --- | ---: | ---: | ---: |
| 32 × 32 | 10240 | 0.42770 | 0.19290 |
| 256 × 256 | 655360 | 0.32250 | 0.15030 |
| 1024 × 1024 | 10485760 | 0.89980 | 0.58770 |

| Image | Readings | GPU utilization median | Utilization range | Memory in use median | Memory range |
| --- | ---: | ---: | --- | ---: | --- |
| 32 × 32 | 726 | 5 percent | 5 to 27 percent | 6597 MiB | 6597 to 6597 MiB |
| 256 × 256 | 726 | 11 percent | 4 to 28 percent | 6603 MiB | 6597 to 6603 MiB |
| 1024 × 1024 | 726 | 30 percent | 7 to 35 percent | 6599 MiB | 6597 to 6599 MiB |

| Idle reading | GPU utilization | Memory in use |
| --- | ---: | ---: |
| before 32 | 5 percent | 6597 MiB |
| before 256 | 4 percent | 6597 MiB |
| before 1024 | 7 percent | 6599 MiB |
| after 1024 | 9 percent | 6597 MiB |

## What each timing includes

The host clock starts when the call is entered and stops when it returns an owned, completed result. Inside that interval the call validates the input, allocates the host result, uploads, runs the kernels, downloads, synchronizes and cleans up its own resources. Input generation, the NVML readings and idle pauses, the pinned allocation, filling and release, comparison with the references, the checksum, result destruction and the CSV output stay outside it. The protocol lists each step. The upload, kernel and download event times are never added to the host time.

## Correctness

Right before the run, at 04:24 CST on November 25, 2026, `cuda_contract`, `cuda_reuse_contract` and `cuda_reuse_faults` passed with this build on this card. That result is in my own records and not in the receipt. Before any run row the benchmark compared each pinned buffer with its pageable source byte for byte. During the run every recorded and warmup output was compared with the scalar reference, exactly for histogram counts, within `1e-9` for the reduction and within `1e-12` per stencil value, and every reduction and stencil output stayed equal bit for bit to the first one of its workload and size across both kinds of memory.

## Conditions and limits

The computer has an AMD processor that Windows reports as AMD64 Family 25 Model 97 and an NVIDIA GeForce RTX 4090 with Windows driver 621.24. The benchmark was built on Windows with MSVC 19.44 for host code and nvcc 13.2.86, and ran in one process. I changed no clock, power, driver or display setting.

This is pinned input only, through the runtime's unchanged interface. Outputs stay in memory the runtime owns, and nothing here pins output, keeps device memory between calls, overlaps transfers, uses more than one stream or reads host memory from the device. The largest input is 9437184 bytes, so this says nothing about larger transfers, and it makes no general bandwidth claim. These are three calls at three sizes in one short run on one card. The [open gates](limits.md) stay as they are.

## Reproduce

Configure on Windows with `-DHBR_CUDA=ON`, build `hbr_cuda_pinned_bench` and run the `cuda_` tests with that build. Check that no game is running, then run it once from a clean checkout. `tools/run_pinned.py` refuses to overwrite an earlier attempt, so give it another `--out-dir` to keep the retained files.

```text
python tools/run_pinned.py build/cuda/hbr_cuda_pinned_bench.exe --build-dir build/cuda --confirm-desktop-load
python tools/report_pinned.py docs/evidence/pinned.csv --receipt docs/evidence/pinned-execution.json
```
