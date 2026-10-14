# Completed serial and batched requests

The new batched mode delivers one model invocation for each admitted nonempty multi-image request, with complete outputs equivalent to the original one-image reference. In this declared study, batching reduced request cost at two and eight images within both providers. CPU remained faster than required CUDA for every tested batch, even after CUDA batching. The one-image CPU comparison is inconclusive and its pooled batched median is slightly slower.

Measured source: `188a7ec0480d5ec43d7f5e69b541ad8ebbc1b130`. The [predeclared protocol](batching-performance-protocol.md) was committed before collection. [Evidence index](batching-evidence.json), [complete CPU rows](batching-cpu-evidence.json) and [complete required-CUDA rows](batching-cuda-evidence.json) retain all first requests, warmups, deciding requests, outputs, observations, source/runtime binary identities and retirement records. The public derivatives remove only private source path fields; original-file and journal hashes are retained in the index.

Each condition has three sequential reused-session blocks, five warmups and twenty deciding requests per block. Batch order rotates and every batch sees both mode orders. Each provider completed 468 independently validated requests, including 360 deciding requests. Paired requests use the same block/sample index and are nonadjacent. The primary timer starts before preparing the original in-memory images and ends at usable completed host-result reconciliation. File reading and PNG decoding are excluded.

| Provider | Images | Mode | Median ms | Range ms | p95 ms | Images/s over deciding requests |
|---|---:|---|---:|---|---:|---:|
| cpu | 1 | serial | 0.1764 | 0.1413 to 0.3186 | 0.2457 | 5462.3 |
| cpu | 1 | batched | 0.1802 | 0.1547 to 0.2414 | 0.2286 | 5401.9 |
| cpu | 2 | serial | 0.2294 | 0.1902 to 0.3899 | 0.3046 | 8354.5 |
| cpu | 2 | batched | 0.2052 | 0.1773 to 0.2933 | 0.2719 | 9320.8 |
| cpu | 8 | serial | 0.4209 | 0.3803 to 0.6945 | 0.6282 | 17487.8 |
| cpu | 8 | batched | 0.3635 | 0.3254 to 0.5364 | 0.4914 | 21213.8 |
| cuda | 1 | serial | 0.5592 | 0.2822 to 2.9127 | 1.9455 | 1322.5 |
| cuda | 1 | batched | 0.4268 | 0.2846 to 2.0328 | 1.7430 | 1381.2 |
| cuda | 2 | serial | 1.0423 | 0.4501 to 8.2642 | 2.1559 | 1607.0 |
| cuda | 2 | batched | 0.5015 | 0.2948 to 2.3216 | 2.0515 | 2688.1 |
| cuda | 8 | serial | 3.1774 | 1.7664 to 6.3351 | 6.1052 | 2279.8 |
| cuda | 8 | batched | 0.6371 | 0.3132 to 1.6682 | 1.5930 | 11093.6 |

Throughput is total images divided by the sum of deciding request durations. It excludes validation, journal writing, warmups, setup and retirement. It is an in-memory completed-request measure, not sustained service capacity.

The decision statistic is the median of three serial block medians minus their corresponding batched block medians. Positive favors batching. The fixed-seed 2,000-replicate bootstrap resamples those three differences. This coarse interval describes the declared blocks and does not establish calibrated population confidence or statistical significance.

| Provider | Images | Median block difference ms | Coarse interval ms | Declared decision |
|---|---:|---:|---|---|
| cpu | 1 | -0.0032 | -0.0068 to 0.0001 | Inconclusive |
| cpu | 2 | 0.0186 | 0.0114 to 0.0317 | Batched faster in declared blocks |
| cpu | 8 | 0.0624 | 0.0446 to 0.0692 | Batched faster in declared blocks |
| cuda | 1 | 0.0303 | 0.0030 to 0.1884 | Batched faster in declared blocks |
| cuda | 2 | 0.5336 | 0.4075 to 0.6165 | Batched faster in declared blocks |
| cuda | 8 | 2.6261 | 2.4416 to 2.6331 | Batched faster in declared blocks |

Setup and retirement remain separate. These phase summaries combine nine sessions per mode/provider across batch sizes. The initial provider load belongs to the first serial session, so these distributions cannot attribute all setup differences to execution mode. The raw rows preserve each session and its first request.

| Provider | Mode | Setup median ms (range) | Close median ms (range) |
|---|---|---|---|
| cpu | serial | 2.0596 (1.7016 to 40.5909) | 0.2301 (0.1558 to 0.4556) |
| cpu | batched | 2.4246 (2.2254 to 2.9769) | 0.2102 (0.1665 to 0.2816) |
| cuda | serial | 18.7014 (15.8503 to 235.7608) | 11.1971 (9.8586 to 12.1623) |
| cuda | batched | 18.2987 (16.1506 to 20.6153) | 10.6465 (9.4778 to 14.1831) |

The owned CPU and CUDA collection operations returned in 1.219 and 2.984 seconds respectively, within separate 60-second collection budgets and 75-second process safeguards. Every owner closed with zero outstanding requests, and the enclosing assigned Windows Jobs retired empty. The study is deliberately short and cannot qualify a sustained workload or tail guarantee.

Process memory is sampled at before-setup, after-setup, after-request and after-close boundaries. It includes the collector, reference evaluator, retained report rows, runtime libraries and allocator caches. Peaks are cumulative for the process. The observations establish neither a leak nor a per-mode allocation ledger.

| Provider | Maximum sampled working set MiB | Maximum sampled private commit MiB | Cumulative process commit peak MiB | Last after-close private commit MiB |
|---|---:|---:|---:|---:|
| cpu | 99.90 | 76.61 | 76.62 | 74.58 |
| cuda | 712.64 | 1407.04 | 1407.10 | 1399.05 |

The owned Job bounds were 512 MiB committed host memory for CPU and 2 GiB for required CUDA. Earlier 512 MiB and 1 GiB CUDA setup attempts failed before model execution and remain retained diagnostics. The 2 GiB setup qualification succeeded before deciding collection. These bounds and the existing 128 MiB CUDA arena option are distinct from total GPU memory.

The shared workstation used a Ryzen 7 7800X3D and RTX 4090 with driver 617.42. Separate device-global snapshots showed 3,684/3,690 MiB used and 39/42 percent utilization before/after the study. They include ordinary other activity, do not isolate this process and do not establish an idle or exclusive GPU. CUDA variation is substantial. CPU and GPU packages both use ONNX Runtime 1.30.0, while their actual library/provider binary hashes are separately recorded.

Correctness acceptance is separate from timing. The installed CPU contracts passed, the required-CUDA suite passed all 25 cases, and installed CLI profiles independently confirmed one model run for an eight-image batched request. CUDA's profile recorded eight CUDA operation events and no CPU operation events. Every collected logit was checked against the original serial ONNX ReferenceEvaluator at unchanged rtol=1e-5 and atol=2e-5. Model calls remain synchronous through independently owned host results.

The admitted original model and locally derived graph retain separate identities. Trained weights and operator bodies are unchanged. No model bytes are committed or included in the wheel. The original model card's Apache/MIT discrepancy remains unresolved. This evidence qualifies a bounded local execution capability and comparison, not model redistribution, an accuracy dataset, another platform backend or an automatic optimization queue.
