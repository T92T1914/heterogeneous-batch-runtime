# A transfer and fixed cost model of the CUDA calls

This page asks how much of each retained CUDA complete call is a fixed cost and how much grows with the bytes it moves, and whether a two-term model fitted on two image sizes predicts the third. It runs nothing on the GPU. Every number here is recomputed by `python tools/cost_model.py` from the [original CUDA samples](evidence/baseline-cuda.csv), whose bytes are pinned by the [execution receipt](evidence/execution.json) of the run at `675bd82`. The [results page](results-2026-09-29.md) describes that run, its machine and its limits.

The short answer is mixed, and the misses are the useful part. A latency plus per-byte line predicts the held-out 1024 uploads within 14%, but it misses five of the six 32 by 32 uploads and five of the six 32 by 32 downloads by 40% or more, and the download line fitted on the two smaller sizes overshoots the large stencil download by 84%. The complete call model looks better than it is at the small sizes, because a fixed cost near 0.4 ms is most of every small call. At 1024 it underpredicts masked reduce by 44%, overpredicts the stencil by 26% and overpredicts two of the four histogram conditions by more than 25%. Most of every complete call happens outside the device event intervals, so bytes moved are not enough to explain it.

## What the model is

The inputs are the medians of the five warm samples for each of six CUDA conditions at three sizes, 18 medians in all. The conditions are masked reduce, the stencil and four histogram conditions (uniform or single-bin input, global or shared bins), and the sizes are 32, 256 and 1024 pixels on a side. The first invocation of each condition is left out, as it is in the results page. Byte counts come from the copy calls in `src/cuda.cu`, which are the same at the measured revision and today.

| Workload | Upload | Download |
| --- | --- | --- |
| masked reduce | two copies, 9 bytes per element (values and mask) | one copy, 8 bytes |
| stencil | one copy, 8 bytes per element | one copy, 8 bytes per element |
| histogram | one copy, 1 byte per element | one copy, 2,048 bytes per 64 by 64 tile |

Three models share one form, `a * x0 + b * x1`:

1. Upload interval = copies × per-copy latency + upload bytes × time per byte.
2. Download interval = the same, with download copies and bytes.
3. Complete call (host clock) = a fixed cost + all bytes moved × time per byte.

Each model is fitted by least squares on relative error, so a 0.01 ms interval counts as much as a 1 ms one. Leaving one size out gives three fits per model. Each fit uses the 12 medians at the other two sizes and predicts the 6 medians at the held-out size. Holding out 256 tests interpolation. Holding out 32 or 1024 tests extrapolation. A prediction more than 25% from the measured median is marked as a miss. The method, the error measure and the miss threshold were written into the script before it first ran. The condition medians had already been read, so this is a held-out check, not a blind one.

## Fitted terms

| Model | Held-out side | Fixed term, ms | Per MiB, ms | Implied rate, GiB/s |
| --- | ---: | ---: | ---: | ---: |
| upload | 32 | 0.0022 | 0.1013 | 9.64 |
| upload | 256 | 0.0036 | 0.0951 | 10.26 |
| upload | 1024 | 0.0035 | 0.0954 | 10.24 |
| download | 32 | 0.0187 | 0.1223 | 7.99 |
| download | 256 | 0.0142 | 0.1256 | 7.77 |
| download | 1024 | 0.0140 | 0.1669 | 5.85 |
| complete | 32 | 0.3893 | 0.3021 | 3.23 |
| complete | 256 | 0.3994 | 0.2868 | 3.41 |
| complete | 1024 | 0.4064 | 0.3864 | 2.53 |

The implied rate is only a reading of the per-byte term, not a bandwidth measurement. For the complete call it folds host work that grows with the image into the same term as the copies.

## Held-out predictions

Upload: median absolute error 17%, worst 43%, 6 of 18 held-out predictions off by more than 25% (median by held-out side: 43% at 32, 17% at 256, 10% at 1024).

| Condition | Held-out side | Measured median, ms | Predicted, ms | Error |
| --- | ---: | ---: | ---: | ---: |
| masked reduce | 32 | 0.0060 | 0.0054 | -11% |
| stencil | 32 | 0.0051 | 0.0030 | -40% miss |
| histogram, uniform, global bins | 32 | 0.0041 | 0.0023 | -43% miss |
| histogram, uniform, shared bins | 32 | 0.0041 | 0.0023 | -43% miss |
| histogram, one bin, global bins | 32 | 0.0041 | 0.0023 | -43% miss |
| histogram, one bin, shared bins | 32 | 0.0041 | 0.0023 | -43% miss |
| masked reduce | 256 | 0.0993 | 0.0608 | -39% miss |
| stencil | 256 | 0.0677 | 0.0512 | -24% |
| histogram, uniform, global bins | 256 | 0.0090 | 0.0096 | +7% |
| histogram, uniform, shared bins | 256 | 0.0082 | 0.0096 | +17% |
| histogram, one bin, global bins | 256 | 0.0082 | 0.0096 | +17% |
| histogram, one bin, shared bins | 256 | 0.0082 | 0.0096 | +17% |
| masked reduce | 1024 | 0.7692 | 0.8656 | +13% |
| stencil | 1024 | 0.6731 | 0.7667 | +14% |
| histogram, uniform, global bins | 1024 | 0.1138 | 0.0989 | -13% |
| histogram, uniform, shared bins | 1024 | 0.1048 | 0.0989 | -6% |
| histogram, one bin, global bins | 1024 | 0.1077 | 0.0989 | -8% |
| histogram, one bin, shared bins | 1024 | 0.1080 | 0.0989 | -8% |

Download: median absolute error 26%, worst 84%, 9 of 18 held-out predictions off by more than 25% (median by held-out side: 42% at 32, 22% at 256, 5% at 1024).

| Condition | Held-out side | Measured median, ms | Predicted, ms | Error |
| --- | ---: | ---: | ---: | ---: |
| masked reduce | 32 | 0.0207 | 0.0187 | -10% |
| stencil | 32 | 0.0133 | 0.0197 | +48% miss |
| histogram, uniform, global bins | 32 | 0.0133 | 0.0189 | +42% miss |
| histogram, uniform, shared bins | 32 | 0.0126 | 0.0189 | +51% miss |
| histogram, one bin, global bins | 32 | 0.0133 | 0.0189 | +42% miss |
| histogram, one bin, shared bins | 32 | 0.0133 | 0.0189 | +42% miss |
| masked reduce | 256 | 0.0205 | 0.0142 | -31% miss |
| stencil | 256 | 0.0955 | 0.0770 | -19% |
| histogram, uniform, global bins | 256 | 0.0226 | 0.0181 | -20% |
| histogram, uniform, shared bins | 256 | 0.0256 | 0.0181 | -29% miss |
| histogram, one bin, global bins | 256 | 0.0236 | 0.0181 | -23% |
| histogram, one bin, shared bins | 256 | 0.0154 | 0.0181 | +18% |
| masked reduce | 1024 | 0.0257 | 0.0140 | -46% miss |
| stencil | 1024 | 0.7335 | 1.3494 | +84% miss |
| histogram, uniform, global bins | 1024 | 0.1051 | 0.0974 | -7% |
| histogram, uniform, shared bins | 1024 | 0.0977 | 0.0974 | 0% |
| histogram, one bin, global bins | 1024 | 0.0968 | 0.0974 | +1% |
| histogram, one bin, shared bins | 1024 | 0.0941 | 0.0974 | +4% |

Complete call: median absolute error 18%, worst 50%, 6 of 18 held-out predictions off by more than 25% (median by held-out side: 8% at 32, 12% at 256, 31% at 1024).

| Condition | Held-out side | Measured median, ms | Predicted, ms | Error |
| --- | ---: | ---: | ---: | ---: |
| masked reduce | 32 | 0.5120 | 0.3920 | -23% |
| stencil | 32 | 0.3682 | 0.3941 | +7% |
| histogram, uniform, global bins | 32 | 0.5298 | 0.3902 | -26% miss |
| histogram, uniform, shared bins | 32 | 0.3958 | 0.3902 | -1% |
| histogram, one bin, global bins | 32 | 0.3660 | 0.3902 | +7% |
| histogram, one bin, shared bins | 32 | 0.4304 | 0.3902 | -9% |
| masked reduce | 256 | 0.7621 | 0.5607 | -26% miss |
| stencil | 256 | 0.7396 | 0.6861 | -7% |
| histogram, uniform, global bins | 256 | 0.4543 | 0.4263 | -6% |
| histogram, uniform, shared bins | 256 | 0.3473 | 0.4263 | +23% |
| histogram, one bin, global bins | 256 | 0.5088 | 0.4263 | -16% |
| histogram, one bin, shared bins | 256 | 0.4646 | 0.4263 | -8% |
| masked reduce | 1024 | 6.9292 | 3.8845 | -44% miss |
| stencil | 1024 | 5.2319 | 6.5896 | +26% miss |
| histogram, uniform, global bins | 1024 | 0.8524 | 0.9861 | +16% |
| histogram, uniform, shared bins | 1024 | 0.6562 | 0.9861 | +50% miss |
| histogram, one bin, global bins | 1024 | 0.7198 | 0.9861 | +37% miss |
| histogram, one bin, shared bins | 1024 | 0.8253 | 0.9861 | +19% |

## Time outside the device intervals

The device events bracket only the upload, the kernels and the download. This table shows the share of each complete call median that falls outside those three intervals. That share covers what the protocol puts inside the host clock around the device work: validation, host and device allocation, stream and event creation, synchronization, checksum selection, result destruction and CUDA cleanup.

| Condition | 32 | 256 | 1024 |
| --- | ---: | ---: | ---: |
| masked reduce | 93% | 83% | 86% |
| stencil | 93% | 73% | 68% |
| histogram, uniform, global bins | 96% | 91% | 62% |
| histogram, uniform, shared bins | 94% | 88% | 64% |
| histogram, one bin, global bins | 94% | 92% | 64% |
| histogram, one bin, shared bins | 88% | 72% | 59% |

## What this shows and what it does not

The copy model is a small version of the transfer models in the GPU performance literature. Boyer, Meng and Kumaran's [data transfer modeling](https://doi.org/10.1109/IPDPSW.2013.236) (IPDPSW 2013) found that adding transfer time cut their GPU speedup prediction error from about 255% to 9%. This repository points the same way from the other side. The kernels were short, and even the copies were a minority of each complete call.

The copy lines are most accurate at 1024, which is also where copies are their largest share of a complete call. Large copies sit near one per-byte rate. The smallest copies do not fit one latency term in either direction. Every 32 by 32 upload took longer than the fit predicted, the 32 by 32 stencil and histogram downloads took less, and the large stencil download moved its 8 MiB faster per byte than the 256 by 256 copies suggested. Extrapolating down to 32 is where the copy lines are weakest, and extrapolating up to 1024 is where the complete call model is weakest. Interpolating at 256 is not safe either. It missed four times, three of them on masked reduce.

The complete call model is where the form breaks. One fixed cost and one per-byte rate cannot be right for masked reduce and the stencil at the same time. Masked reduce moves fewer bytes than the stencil at every size, yet its complete call took longer at every size. The code suggests why, though this run did not time it separately. Both workloads check every input value for finiteness. Masked reduce also makes a serial pass that bounds the selected absolute sum (`detail::reduction_range` in `include/hbr/detail/validation.hpp`), and it makes four device allocations to the stencil's two. The histogram shows the opposite limit. Its four conditions move the same bytes at each size, yet their 1024 complete calls measured from 0.656 to 0.852 ms, so a model that sees only bytes cannot follow that spread. Its worst miss, histogram, uniform, shared bins at 1024, is +50%. A model with a host term per element, or one fit per workload, would follow the data more closely. With three sizes per workload it would mostly be drawing lines through points, so the two-term form stays and the misses are reported.

These fits describe one shared Windows workstation, one RTX 4090 with pageable host memory, one driver and one short run with five warm samples per condition. They are not a bandwidth measurement, a prediction for other sizes, other cards or pinned memory, or a reason to expect a crossover with the CPU at some larger image. Hardware counters were not available to the measuring account (`ERR_NVGPUCTRPERM`), so the model works from timing alone. The [limits](results-2026-09-29.md) of the original run apply unchanged.

## Reproduce

`python tools/cost_model.py --check` checks the CSV hash against the receipt, recomputes every fit and prediction and fails if this page or [the full prediction table](evidence/cost-model.csv) differs. `--write` regenerates both from the retained CSV. Neither runs a workload.
