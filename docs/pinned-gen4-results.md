# Pinned and pageable input at PCIe generation 4, measured on a working desktop

I ran the pinned input protocol a second time on the same RTX 4090, with the card's link at PCIe generation 4 and 16 lanes. The [first run](pinned-results.md) on November 25, 2026 had the link at generation 3, because the motherboard's setting for the card's slot held it there until that evening. That run, its samples and its page stay as they are.

Pinned host input still made no consistent difference to the complete CUDA call. Within each condition I divided the pinned call by the pageable call in each back to back pair, and the median of the 120 paired ratios was between 0.969 and 1.020. In all nine conditions the middle half of those ratios includes 1, and between 39 and 67 of the 120 back to back pairs had the pinned call slower. The pinned median was lower in six conditions and higher in three. The largest gap was 3.4 percent lower, for the 256 by 256 stencil, and the largest the other way was 1.9 percent higher, for the 32 by 32 masked reduction. Pinned and pageable input gave the same checksum in every condition, and each one equals the checksum of the first run.

The upload is where the faster link shows. At 1024 by 1024 the masked reduction moved 9437184 bytes at about 22.8 GB/s from pageable memory and 24.6 GB/s from pinned memory, where the first run moved them at about 12.6 and 12.8 GB/s. The stencil moved 8388608 bytes at about 22.8 and 24.4 GB/s, against 12.5 and 12.8 GB/s before. At this size the pinned upload median was 7.0 percent lower than the pageable one for the masked reduction and 6.6 percent lower for the stencil, and 0.8 percent higher for the histogram, whose upload is 1048576 bytes. That gap in the upload did not carry into a consistent gap in the complete call, whose paired medians at 1024 by 1024 were 0.991 to 0.995 with every middle half including 1. At the two smaller sizes the uploads take 0.004 to 0.074 ms, and the event times move in steps of about 0.001 ms.

Beside the first run, the complete call medians at 32 by 32 were 0.986 to 1.031 times the first run's, at 256 by 256 they were 0.909 to 0.980 times and at 1024 by 1024 they were 0.849 to 0.933 times. The upload medians at 1024 by 1024 were 0.52 to 0.64 times the first run's. The stencil's download at that size, which goes to memory the runtime owns for both kinds of input, took 0.465 and 0.469 ms against 0.728 and 0.722 ms before. The kernel medians at 1024 by 1024 went the other way, at 1.17 to 1.58 times the first run's. The two runs differ in more than the link. They ran on different days at different desktop loads, one run each, and only this one had nvidia-smi reading the link beside every timed row. So this page sets them beside each other and names no cause for any difference between them.

Pinning has its own cost. Allocating the three pinned buffers of the 1024 by 1024 size, 10485760 bytes, took 0.817 ms and releasing them took 0.497 ms. Together that is about 16 times the 0.082 ms by which the pinned median of the 1024 by 1024 masked reduction was lower, and these samples do not show that gap to be real.

I measured this on a working desktop with no game running. The usual applications kept using the card. The four idle readings, each taken after a pause of two seconds with no call from the benchmark, showed 1 to 9 percent utilization, which is the desktop's alone. The readings taken before each recorded call had medians of 8, 17 and 25 percent for the three sizes and ranged from 1 to 30 percent, and they include the benchmark's own earlier kernels. In the first run those medians were 5, 11 and 30 percent. Memory is different, because NVML counts every process on the card, this benchmark included. The readings taken inside the process showed 5265 MiB in use throughout, while nvidia-smi read 4504 MiB just before and just after the run, at 1 and 13 percent utilization. That difference of 761 MiB is most likely what the benchmark's CUDA context held, though the two kinds of reading are not taken the same way.

The [protocol](pinned-gen4-protocol.json) repeats the plan of the [first protocol](pinned-protocol.json) and adds the link. It was committed at `219e03b9e732de2859beb65d9d4fc348c012d8fd` and tightened at `2f8396fe6583c1d28693c5ea2ddb8198e96262fc` on November 26, 2026, both before any sample. The run used the binary with SHA-256 `e644ed7a9c33d98417d35538d27684accdef784e828f2d82e5d297598f136685`, the same bytes as the first run, and the protocol file with SHA-256 `e4aa315fb358245d81f37fc6e234c0a4eea84fb44ee80400aea674a0a50a1455`.

## The run

The run started at 12:10:46 UTC on November 26, 2026, from source `175fcd3abb3f1775b36c61fc603e0768d8341e51`, and passed in about 12 seconds. Its [execution receipt](evidence/pinned-gen4-execution.json) names the source, the protocol, the binary, the compilers, the GPU state before and after the run, the idle readings and every link reading. Its [samples](evidence/pinned-gen4.csv) hold all 2178 run rows and one setup and one retire row for each size, with SHA-256 `c22b79b33ae8c122afab2f2b081a23db265fdf7f3190c1c0007333d2f1e17e3c`. nvidia-smi read the link every 250 ms while the benchmark ran. All 48 readings gave generation 4 with 16 lanes, both as the current link and as its maximum, so the link never dropped to the idle generation 1 during the run. The host's own maximum read as generation 5.

`python tools/validate_evidence.py` checks the CSV against that hash and every row against the plan. It checks that the tables under Results are the ones `tools/report_pinned.py` makes from the samples and the receipt, link readings included, that the table under the comparison heading is the one it makes from both runs, that this page cites the recorded hashes and revisions, and that the range of the paired ratio medians, the count of pairs with the pinned call slower and the middle halves that include 1 follow from the samples.

## Results

All durations are milliseconds. Each median is over 120 warm samples. Iteration zero of each condition is kept in the CSV and shown as the first call in the third table, and two unrecorded warmup rounds follow it. Iteration zero of every size also follows a pause of two seconds for the idle reading, so it is neither a warm measurement nor a cold process measurement.

The first table puts the medians side by side. The complete call is on the host clock, while the upload and download columns are the library's own stream event times. Every call downloads into memory the runtime owns, so the download is pageable for both kinds and serves as a control. The second table divides the pinned call by the pageable call within each pair and takes the median of those 120 ratios, then the medians of the 60 pairs where pageable went first and the 60 where pinned went first. The fourth table times the pinned allocation and release for each size, outside every run row. The next two summarize the reading taken before each recorded call and the four idle readings, and the last counts the link readings.

| Image | Workload | Upload bytes | Complete call, pageable | Complete call, pinned | Upload, pageable | Upload, pinned | Download, pageable | Download, pinned |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 × 32 | masked_reduce | 9216 | 0.37120 | 0.37825 | 0.00610 | 0.01024 | 0.01645 | 0.02150 |
| 32 × 32 | tile_histogram | 1024 | 0.35125 | 0.34970 | 0.00410 | 0.00614 | 0.01126 | 0.01126 |
| 32 × 32 | stencil3x3 | 8192 | 0.36170 | 0.36750 | 0.00410 | 0.00701 | 0.01126 | 0.01219 |
| 256 × 256 | masked_reduce | 589824 | 0.74990 | 0.72515 | 0.07373 | 0.05254 | 0.01054 | 0.01024 |
| 256 × 256 | tile_histogram | 65536 | 0.36715 | 0.36735 | 0.00706 | 0.00915 | 0.01331 | 0.01741 |
| 256 × 256 | stencil3x3 | 524288 | 0.60800 | 0.58720 | 0.04781 | 0.04814 | 0.07000 | 0.07046 |
| 1024 × 1024 | masked_reduce | 9437184 | 6.26790 | 6.18600 | 0.41328 | 0.38426 | 0.01131 | 0.01536 |
| 1024 × 1024 | tile_histogram | 1048576 | 0.57845 | 0.57640 | 0.06797 | 0.06854 | 0.07398 | 0.07419 |
| 1024 × 1024 | stencil3x3 | 8388608 | 5.00885 | 4.91540 | 0.36760 | 0.34317 | 0.46517 | 0.46885 |

| Image | Workload | Pairs | Median of pinned ÷ pageable in a pair | Middle 50 percent | Full range | Pageable first | Pinned first |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 32 × 32 | masked_reduce | 120 | 1.015 | 0.935 to 1.117 | 0.371 to 2.348 | 1.042 | 1.012 |
| 32 × 32 | tile_histogram | 120 | 0.993 | 0.933 to 1.064 | 0.781 to 1.623 | 0.976 | 0.999 |
| 32 × 32 | stencil3x3 | 120 | 1.020 | 0.937 to 1.106 | 0.390 to 1.836 | 1.006 | 1.036 |
| 256 × 256 | masked_reduce | 120 | 0.974 | 0.926 to 1.018 | 0.313 to 2.956 | 0.977 | 0.968 |
| 256 × 256 | tile_histogram | 120 | 1.001 | 0.935 to 1.077 | 0.753 to 2.597 | 0.972 | 1.039 |
| 256 × 256 | stencil3x3 | 120 | 0.969 | 0.910 to 1.030 | 0.430 to 1.783 | 0.960 | 0.978 |
| 1024 × 1024 | masked_reduce | 120 | 0.995 | 0.949 to 1.042 | 0.738 to 1.357 | 0.995 | 0.993 |
| 1024 × 1024 | tile_histogram | 120 | 0.991 | 0.921 to 1.075 | 0.341 to 2.842 | 0.948 | 1.048 |
| 1024 × 1024 | stencil3x3 | 120 | 0.993 | 0.939 to 1.035 | 0.757 to 1.764 | 0.993 | 0.994 |

| Image | Workload | Memory | Warm samples | Median | Middle 50 percent | Full range | First call | Kernel median |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| 32 × 32 | masked_reduce | pageable | 120 | 0.37120 | 0.34728 to 0.41010 | 0.31450 to 1.01950 | 1.03970 | 0.00819 |
| 32 × 32 | masked_reduce | pinned | 120 | 0.37825 | 0.36153 to 0.42017 | 0.31770 to 0.88440 | 0.51400 | 0.00819 |
| 32 × 32 | tile_histogram | pageable | 120 | 0.35125 | 0.33508 to 0.38218 | 0.29640 to 0.68150 | 0.46600 | 0.00512 |
| 32 × 32 | tile_histogram | pinned | 120 | 0.34970 | 0.33117 to 0.37657 | 0.29520 to 1.04690 | 0.35240 | 0.00512 |
| 32 × 32 | stencil3x3 | pageable | 120 | 0.36170 | 0.33688 to 0.39658 | 0.29840 to 1.55880 | 0.40600 | 0.00824 |
| 32 × 32 | stencil3x3 | pinned | 120 | 0.36750 | 0.34098 to 0.40295 | 0.30430 to 0.84880 | 0.33720 | 0.00819 |
| 256 × 256 | masked_reduce | pageable | 120 | 0.74990 | 0.72200 to 0.78048 | 0.67380 to 2.59850 | 0.73010 | 0.00819 |
| 256 × 256 | masked_reduce | pinned | 120 | 0.72515 | 0.70515 to 0.76710 | 0.65890 to 2.12480 | 0.69520 | 0.02418 |
| 256 × 256 | tile_histogram | pageable | 120 | 0.36715 | 0.34690 to 0.39363 | 0.30420 to 1.49650 | 0.76890 | 0.00922 |
| 256 × 256 | tile_histogram | pinned | 120 | 0.36735 | 0.35157 to 0.39515 | 0.31370 to 1.36790 | 0.53300 | 0.00822 |
| 256 × 256 | stencil3x3 | pageable | 120 | 0.60800 | 0.58065 to 0.64420 | 0.55060 to 1.42540 | 0.76020 | 0.03238 |
| 256 × 256 | stencil3x3 | pinned | 120 | 0.58720 | 0.56640 to 0.63135 | 0.53720 to 1.09560 | 0.64870 | 0.03235 |
| 1024 × 1024 | masked_reduce | pageable | 120 | 6.26790 | 6.05480 to 6.50292 | 5.79280 to 8.19150 | 5.93510 | 0.12971 |
| 1024 × 1024 | masked_reduce | pinned | 120 | 6.18600 | 5.96997 to 6.56935 | 5.76560 to 8.16480 | 6.09110 | 0.13658 |
| 1024 × 1024 | tile_histogram | pageable | 120 | 0.57845 | 0.55237 to 0.61820 | 0.51710 to 1.88120 | 0.57580 | 0.04970 |
| 1024 × 1024 | tile_histogram | pinned | 120 | 0.57640 | 0.54740 to 0.61947 | 0.50080 to 1.75860 | 0.53580 | 0.05024 |
| 1024 × 1024 | stencil3x3 | pageable | 120 | 5.00885 | 4.86165 to 5.29560 | 4.51300 to 6.38440 | 5.06230 | 0.74083 |
| 1024 × 1024 | stencil3x3 | pinned | 120 | 4.91540 | 4.77233 to 5.13622 | 4.49540 to 8.60210 | 4.72150 | 0.75781 |

| Image | Pinned bytes | Allocation | Release |
| --- | ---: | ---: | ---: |
| 32 × 32 | 10240 | 0.25260 | 0.15280 |
| 256 × 256 | 655360 | 0.25630 | 0.15090 |
| 1024 × 1024 | 10485760 | 0.81690 | 0.49740 |

| Image | Readings | GPU utilization median | Utilization range | Memory in use median | Memory range |
| --- | ---: | ---: | --- | ---: | --- |
| 32 × 32 | 726 | 8 percent | 1 to 23 percent | 5265 MiB | 5265 to 5265 MiB |
| 256 × 256 | 726 | 17 percent | 2 to 17 percent | 5265 MiB | 5265 to 5265 MiB |
| 1024 × 1024 | 726 | 25 percent | 2 to 30 percent | 5265 MiB | 5265 to 5265 MiB |

| Idle reading | GPU utilization | Memory in use |
| --- | ---: | ---: |
| before 32 | 1 percent | 5265 MiB |
| before 256 | 2 percent | 5265 MiB |
| before 1024 | 2 percent | 5265 MiB |
| after 1024 | 9 percent | 5265 MiB |

The protocol names PCIe generation 4 with 16 lanes. Every one of the 48 link readings gave that as the maximum.

| Current link | Readings while the benchmark ran |
| --- | ---: |
| generation 4, 16 lanes | 48 |

## Beside the generation 3 run

Each row sets the median complete call and the median upload of one condition in the first run beside the same condition in this one. The ratio divides this run's complete call median by the first run's. An upload rate is the upload bytes divided by the median upload time. Neither run is a reference for the other, and the conditions that differ between them are listed above.

| Image | Workload | Memory | Complete call, generation 3 | Complete call, generation 4 | Ratio | Upload, generation 3 | Upload, generation 4 | Upload rate, generation 3 | Upload rate, generation 4 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 × 32 | masked_reduce | pageable | 0.35995 | 0.37120 | 1.031 | 0.00614 | 0.00610 | 1.50 GB/s | 1.51 GB/s |
| 32 × 32 | masked_reduce | pinned | 0.38360 | 0.37825 | 0.986 | 0.01126 | 0.01024 | 0.82 GB/s | 0.90 GB/s |
| 32 × 32 | tile_histogram | pageable | 0.35175 | 0.35125 | 0.999 | 0.00410 | 0.00410 | 0.25 GB/s | 0.25 GB/s |
| 32 × 32 | tile_histogram | pinned | 0.34300 | 0.34970 | 1.020 | 0.00704 | 0.00614 | 0.15 GB/s | 0.17 GB/s |
| 32 × 32 | stencil3x3 | pageable | 0.35075 | 0.36170 | 1.031 | 0.00499 | 0.00410 | 1.64 GB/s | 2.00 GB/s |
| 32 × 32 | stencil3x3 | pinned | 0.36140 | 0.36750 | 1.017 | 0.00717 | 0.00701 | 1.14 GB/s | 1.17 GB/s |
| 256 × 256 | masked_reduce | pageable | 0.80540 | 0.74990 | 0.931 | 0.09523 | 0.07373 | 6.19 GB/s | 8.00 GB/s |
| 256 × 256 | masked_reduce | pinned | 0.78415 | 0.72515 | 0.925 | 0.07317 | 0.05254 | 8.06 GB/s | 11.23 GB/s |
| 256 × 256 | tile_histogram | pageable | 0.38515 | 0.36715 | 0.953 | 0.00819 | 0.00706 | 8.00 GB/s | 9.29 GB/s |
| 256 × 256 | tile_histogram | pinned | 0.37485 | 0.36735 | 0.980 | 0.01126 | 0.00915 | 5.82 GB/s | 7.16 GB/s |
| 256 × 256 | stencil3x3 | pageable | 0.66705 | 0.60800 | 0.911 | 0.06630 | 0.04781 | 7.91 GB/s | 10.97 GB/s |
| 256 × 256 | stencil3x3 | pinned | 0.64590 | 0.58720 | 0.909 | 0.06603 | 0.04814 | 7.94 GB/s | 10.89 GB/s |
| 1024 × 1024 | masked_reduce | pageable | 6.71690 | 6.26790 | 0.933 | 0.75082 | 0.41328 | 12.57 GB/s | 22.83 GB/s |
| 1024 × 1024 | masked_reduce | pinned | 6.67065 | 6.18600 | 0.927 | 0.73491 | 0.38426 | 12.84 GB/s | 24.56 GB/s |
| 1024 × 1024 | tile_histogram | pageable | 0.64700 | 0.57845 | 0.894 | 0.10642 | 0.06797 | 9.85 GB/s | 15.43 GB/s |
| 1024 × 1024 | tile_histogram | pinned | 0.67910 | 0.57640 | 0.849 | 0.10685 | 0.06854 | 9.81 GB/s | 15.30 GB/s |
| 1024 × 1024 | stencil3x3 | pageable | 5.39570 | 5.00885 | 0.928 | 0.66941 | 0.36760 | 12.53 GB/s | 22.82 GB/s |
| 1024 × 1024 | stencil3x3 | pinned | 5.35515 | 4.91540 | 0.918 | 0.65528 | 0.34317 | 12.80 GB/s | 24.44 GB/s |

## What each timing includes

The host clock starts when the call is entered and stops when it returns an owned, completed result. Inside that interval the call validates the input, allocates the host result, uploads, runs the kernels, downloads, synchronizes and cleans up its own resources. Input generation, the NVML readings and idle pauses, the pinned allocation, filling and release, comparison with the references, the checksum, result destruction and the CSV output stay outside it. The nvidia-smi process that read the link ran beside the benchmark for the whole run, outside the benchmark's process. The protocol lists each step. The upload, kernel and download event times are never added to the host time.

## Correctness

Right before the run, at 06:10 CST on November 26, 2026, `cuda_contract`, `cuda_reuse_contract` and `cuda_reuse_faults` passed with this build on this card. That result is in my own records and not in the receipt. Before any run row the benchmark compared each pinned buffer with its pageable source byte for byte. During the run every recorded and warmup output was compared with the scalar reference, exactly for histogram counts, within `1e-9` for the reduction and within `1e-12` per stencil value, and every reduction and stencil output stayed equal bit for bit to the first one of its workload and size across both kinds of memory.

## Conditions and limits

The computer has an AMD processor that Windows reports as AMD64 Family 25 Model 97 and an NVIDIA GeForce RTX 4090 with Windows driver 621.24. The benchmark was built on Windows with MSVC 19.44 for host code and nvcc 13.2.86, and ran in one process. Between the two runs the motherboard setting for the card's slot moved from its generation 3 step to automatic, and the driver stayed at 621.24. I changed no clock, power, driver or display setting for either run.

This is pinned input only, through the runtime's unchanged interface. Outputs stay in memory the runtime owns, and nothing here pins output, keeps device memory between calls, overlaps transfers, uses more than one stream or reads host memory from the device. The largest input is 9437184 bytes, so this says nothing about larger transfers, and it makes no general bandwidth claim. These are three calls at three sizes in one short run on one card, set beside one earlier run. The [open gates](limits.md) stay as they are.

## Reproduce

Configure on Windows with `-DHBR_CUDA=ON`, build `hbr_cuda_pinned_bench` and run the `cuda_` tests with that build. Check that no game is running and that nvidia-smi gives the link maximum as generation 4 with 16 lanes, then run it once from a clean checkout. `tools/run_pinned.py` refuses to overwrite an earlier attempt, so give it another `--out-dir` to keep the retained files.

```text
python tools/run_pinned.py build/cuda/hbr_cuda_pinned_bench.exe --build-dir build/cuda --protocol docs/pinned-gen4-protocol.json --confirm-desktop-load
python tools/report_pinned.py docs/evidence/pinned-gen4.csv --receipt docs/evidence/pinned-gen4-execution.json --compare docs/evidence/pinned.csv
```
