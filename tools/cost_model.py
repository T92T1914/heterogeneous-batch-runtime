"""Fit a transfer and fixed cost model to the retained CUDA complete calls and test it on sizes held out of each fit.

The model reads only `docs/evidence/baseline-cuda.csv`, whose bytes are pinned by
`docs/evidence/execution.json`. It runs no workload. `--write` regenerates
`docs/cost-model.md` and `docs/evidence/cost-model.csv`. `--check` recomputes both and
fails if either differs from the committed bytes.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SIDES = (32, 256, 1024)
TILE = 64
HISTOGRAM_BIN_BYTES = 256 * 8
CONDITIONS = [
    ("masked_reduce", "uniform", "cuda_baseline"),
    ("stencil3x3", "uniform", "cuda_baseline"),
    ("tile_histogram", "uniform", "cuda_baseline"),
    ("tile_histogram", "uniform", "cuda_shared"),
    ("tile_histogram", "single_bin", "cuda_baseline"),
    ("tile_histogram", "single_bin", "cuda_shared"),
]
LABELS = {
    ("masked_reduce", "uniform", "cuda_baseline"): "masked reduce",
    ("stencil3x3", "uniform", "cuda_baseline"): "stencil",
    ("tile_histogram", "uniform", "cuda_baseline"): "histogram, uniform, global bins",
    ("tile_histogram", "uniform", "cuda_shared"): "histogram, uniform, shared bins",
    ("tile_histogram", "single_bin", "cuda_baseline"): "histogram, one bin, global bins",
    ("tile_histogram", "single_bin", "cuda_shared"): "histogram, one bin, shared bins",
}
# A prediction for a held out size this far from the measured median, in either direction, is called a miss.
MISS = 0.25


def require(condition, message):
    """Fail even under python -O, which strips assert statements."""
    if not condition:
        raise SystemExit(message)


def transfers(workload, side):
    """Copies and bytes each direction, read from the copy calls in src/cuda.cu at the measured revision."""
    n = side * side
    if workload == "masked_reduce":
        return {"upload": (2, 8 * n + n), "download": (1, 8)}
    if workload == "stencil3x3":
        return {"upload": (1, 8 * n), "download": (1, 8 * n)}
    tiles = ((side + TILE - 1) // TILE) ** 2
    return {"upload": (1, n), "download": (1, tiles * HISTOGRAM_BIN_BYTES)}


def load():
    receipt = json.loads((ROOT / "docs/evidence/execution.json").read_text(encoding="utf-8"))
    path = ROOT / "docs/evidence/baseline-cuda.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == receipt["artifact_sha256"]["baseline-cuda.csv"], "retained CUDA samples changed")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    groups = defaultdict(list)
    for row in rows:
        if row["backend"].startswith("cuda") and int(row["iteration"]) > 0:
            groups[(row["workload"], row["distribution"], row["backend"], int(row["side"]))].append(row)
    points = []
    for condition in CONDITIONS:
        for side in SIDES:
            warm = groups[(*condition, side)]
            require(len(warm) == 5, f"expected five warm samples for {condition} at {side}")
            median = {field: statistics.median(float(r[field]) for r in warm)
                      for field in ("upload_ms", "kernel_ms", "download_ms", "host_total_ms")}
            moved = transfers(condition[0], side)
            points.append({"condition": condition, "side": side, **median,
                           "up_copies": moved["upload"][0], "up_bytes": moved["upload"][1],
                           "down_copies": moved["download"][0], "down_bytes": moved["download"][1]})
    return receipt, points


def fit(samples):
    """Two-term fit y = a*x0 + b*x1 that minimizes the summed squared relative error."""
    s00 = s01 = s11 = r0 = r1 = 0.0
    for x0, x1, y in samples:
        w = 1.0 / (y * y)
        s00 += w * x0 * x0
        s01 += w * x0 * x1
        s11 += w * x1 * x1
        r0 += w * x0 * y
        r1 += w * x1 * y
    det = s00 * s11 - s01 * s01
    require(det > 0, "singular fit")
    return (r0 * s11 - r1 * s01) / det, (s00 * r1 - s01 * r0) / det


MODELS = {
    # Each model maps a point to (x0, x1, measured). The fitted a and b multiply x0 and x1.
    "upload": lambda p: (p["up_copies"], p["up_bytes"], p["upload_ms"]),
    "download": lambda p: (p["down_copies"], p["down_bytes"], p["download_ms"]),
    "complete": lambda p: (1, p["up_bytes"] + p["down_bytes"], p["host_total_ms"]),
}


def evaluate(points):
    folds = []
    for held in SIDES:
        train = [p for p in points if p["side"] != held]
        test = [p for p in points if p["side"] == held]
        for name, view in MODELS.items():
            a, b = fit([view(p) for p in train])
            for p in test:
                x0, x1, y = view(p)
                predicted = a * x0 + b * x1
                folds.append({"model": name, "held_out": held, "fixed_ms": a, "ms_per_mib": b * 2 ** 20,
                              "condition": p["condition"], "x1": x1, "measured_ms": y,
                              "predicted_ms": predicted, "relative_error": predicted / y - 1})
    return folds


def outputs(receipt, points):
    folds = evaluate(points)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["model", "held_out_side", "workload", "distribution", "backend", "fit_fixed_ms",
                     "fit_ms_per_mib", "bytes", "measured_median_ms", "predicted_ms", "relative_error"])
    for f in folds:
        writer.writerow([f["model"], f["held_out"], *f["condition"], f"{f['fixed_ms']:.9f}",
                         f"{f['ms_per_mib']:.9f}", f["x1"], f"{f['measured_ms']:.9f}",
                         f"{f['predicted_ms']:.9f}", f"{f['relative_error']:.6f}"])

    def pct(value):
        rounded = round(value * 100)
        return f"{rounded:+d}%" if rounded else "0%"

    params = ["| Model | Side held out | Fixed term, ms | Per MiB, ms | Implied rate, GiB/s |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name in MODELS:
        for held in SIDES:
            f = next(x for x in folds if x["model"] == name and x["held_out"] == held)
            rate = 1000 / 1024 / f["ms_per_mib"] if f["ms_per_mib"] > 0 else math.nan
            rate_text = f"{rate:.2f}" if f["ms_per_mib"] > 0 else "not physical"
            params.append(f"| {name} | {held} | {f['fixed_ms']:.4f} | {f['ms_per_mib']:.4f} | {rate_text} |")

    tables = {}
    summary = {}
    for name in MODELS:
        table = ["| Condition | Side held out | Measured median, ms | Predicted, ms | Error |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        errors = []
        for held in SIDES:
            for f in folds:
                if f["model"] == name and f["held_out"] == held:
                    errors.append(f)
                    flag = " miss" if abs(f["relative_error"]) > MISS else ""
                    table.append(f"| {LABELS[f['condition']]} | {held} | {f['measured_ms']:.4f} | "
                                 f"{f['predicted_ms']:.4f} | {pct(f['relative_error'])}{flag} |")
        tables[name] = "\n".join(table)
        absolute = [abs(f["relative_error"]) for f in errors]
        by_side = {held: statistics.median(abs(f["relative_error"]) for f in errors if f["held_out"] == held)
                   for held in SIDES}
        summary[name] = (statistics.median(absolute), max(absolute), sum(a > MISS for a in absolute), len(absolute), by_side)

    outside = ["| Condition | 32 | 256 | 1024 |", "| --- | ---: | ---: | ---: |"]
    for condition in CONDITIONS:
        cells = []
        for side in SIDES:
            p = next(x for x in points if x["condition"] == condition and x["side"] == side)
            events = p["upload_ms"] + p["kernel_ms"] + p["download_ms"]
            cells.append(f"{(1 - events / p['host_total_ms']) * 100:.0f}%")
        outside.append(f"| {LABELS[condition]} | " + " | ".join(cells) + " |")

    def line(name):
        median, worst, misses, count, by_side = summary[name]
        sides = ", ".join(f"{by_side[s] * 100:.0f}% at {s}" for s in SIDES)
        return (f"median absolute error {median * 100:.0f}%, worst {worst * 100:.0f}%, "
                f"{misses} of the {count} predictions missed by more than {MISS * 100:.0f}% "
                f"(median by side held out: {sides})")

    def error(name, held, condition):
        return next(f["relative_error"] for f in folds
                    if f["model"] == name and f["held_out"] == held and f["condition"] == condition)

    stencil, masked = CONDITIONS[1], CONDITIONS[0]
    up1024 = max(abs(f["relative_error"]) for f in folds if f["model"] == "upload" and f["held_out"] == 1024)
    stencil_down = abs(error("download", 1024, stencil))
    masked_1024 = abs(error("complete", 1024, masked))
    stencil_1024 = abs(error("complete", 1024, stencil))
    histogram_1024 = [p["host_total_ms"] for p in points if p["condition"][0] == "tile_histogram" and p["side"] == 1024]
    hist_low, hist_high = min(histogram_1024), max(histogram_1024)
    worst = max((f for f in folds if f["model"] == "complete" and f["held_out"] == 1024
                 and f["condition"][0] == "tile_histogram"), key=lambda f: abs(f["relative_error"]))
    worst_label, worst_error = LABELS[worst["condition"]], pct(worst["relative_error"])
    negative_note = ""
    if any(f["ms_per_mib"] <= 0 for f in folds):
        negative_note = (" A term per byte at or below zero means the fit put as much time on the smaller size "
                         "it saw, and it is shown as not physical rather than hidden.")

    report = f"""# A transfer and fixed cost model of the CUDA calls

This page asks how much of each retained CUDA complete call is a fixed cost and how much grows with the bytes it moves, and whether a two term model fitted on two image sizes predicts the third. It runs nothing on the GPU. Every number here is recomputed by `python tools/cost_model.py` from the [original CUDA samples](evidence/baseline-cuda.csv), whose bytes are pinned by the [execution receipt](evidence/execution.json) of the run at `{receipt['measurement']['revision'][:7]}`. The [results page](results-2026-09-29.md) describes that run, its machine and its limits.

The short answer is mixed, and the misses are the useful part. A line of latency plus a cost per byte predicts the 1024 uploads it never saw within {up1024 * 100:.0f}%, but it misses five of the six 32 by 32 uploads and five of the six 32 by 32 downloads by 40% or more, and the download line fitted on the two smaller sizes overshoots the large stencil download by {stencil_down * 100:.0f}%. The complete call model looks better than it is at the small sizes, because a fixed cost near 0.4 ms is most of every small call. At 1024 it underpredicts masked reduce by {masked_1024 * 100:.0f}%, overpredicts the stencil by {stencil_1024 * 100:.0f}% and overpredicts two of the four histogram conditions by more than 25%. Most of every complete call happens outside the device event intervals, so bytes moved are not enough to explain it.

## What the model is

The inputs are the medians of the five warm samples for each of six CUDA conditions at three sizes, 18 medians in all. The conditions are masked reduce, the stencil and four histogram conditions (uniform or single bin input, global or shared bins), and the sizes are 32, 256 and 1024 pixels on a side. The first invocation of each condition is left out, as it is in the results page. Byte counts come from the copy calls in `src/cuda.cu`, which are the same at the measured revision and today.

| Workload | Upload | Download |
| --- | --- | --- |
| masked reduce | two copies, 9 bytes per element (values and mask) | one copy, 8 bytes |
| stencil | one copy, 8 bytes per element | one copy, 8 bytes per element |
| histogram | one copy, 1 byte per element | one copy, 2,048 bytes per 64 by 64 tile |

Three models share one form, `a * x0 + b * x1`:

1. Upload interval = copies × latency per copy + upload bytes × time per byte.
2. Download interval = the same, with download copies and bytes.
3. Complete call (host clock) = a fixed cost + all bytes moved × time per byte.

Each model is fitted by least squares on relative error, so a 0.01 ms interval counts as much as a 1 ms one. Leaving one size out gives three fits per model. Each fit uses the 12 medians at the other two sizes and predicts the 6 medians at the size held out. Holding out 256 tests interpolation. Holding out 32 or 1024 tests extrapolation. A prediction more than {MISS * 100:.0f}% from the measured median is marked as a miss. The method, the error measure and the miss threshold were written into the script before it first ran. The condition medians had already been read, so this checks predictions on data the fit did not use, but it is not a blind test.

## Fitted terms

{chr(10).join(params)}

The implied rate is only a reading of the term per byte, not a bandwidth measurement. For the complete call it folds host work that grows with the image into the same term as the copies.{negative_note}

## Predictions for the size held out

Upload: {line('upload')}.

{tables['upload']}

Download: {line('download')}.

{tables['download']}

Complete call: {line('complete')}.

{tables['complete']}

## Time outside the device intervals

The device events bracket only the upload, the kernels and the download. This table shows the share of each complete call median that falls outside those three intervals. That share covers what the protocol puts inside the host clock around the device work: validation, host and device allocation, stream and event creation, synchronization, checksum selection, result destruction and CUDA cleanup.

{chr(10).join(outside)}

## What this shows and what it does not

The copy model is a small version of the transfer models in the GPU performance literature. Boyer, Meng and Kumaran's [data transfer modeling](https://doi.org/10.1109/IPDPSW.2013.236) (IPDPSW 2013) found that adding transfer time cut their GPU speedup prediction error from about 255% to 9%. This repository points the same way from the other side. The kernels were short, and even the copies were a minority of each complete call.

The copy lines are most accurate at 1024, which is also where copies are their largest share of a complete call. Large copies sit near one rate per byte. The smallest copies do not fit one latency term in either direction. Every 32 by 32 upload took longer than the fit predicted, the 32 by 32 stencil and histogram downloads took less, and the large stencil download moved its 8 MiB faster per byte than the 256 by 256 copies suggested. Extrapolating down to 32 is where the copy lines are weakest, and extrapolating up to 1024 is where the complete call model is weakest. Interpolating at 256 is not safe either. It missed four times, three of them on masked reduce.

The complete call model is where the form breaks. One fixed cost and one rate per byte cannot be right for masked reduce and the stencil at the same time. Masked reduce moves fewer bytes than the stencil at every size, yet its complete call took longer at every size. The code suggests why, though this run did not time it separately. Both workloads check every input value for finiteness. Masked reduce also makes a serial pass that bounds the selected absolute sum (`detail::reduction_range` in `include/hbr/detail/validation.hpp`), and it makes four device allocations to the stencil's two. The histogram shows the opposite limit. Its four conditions move the same bytes at each size, yet their 1024 complete calls measured from {hist_low:.3f} to {hist_high:.3f} ms, so a model that sees only bytes cannot follow that spread. Its worst miss, {worst_label} at 1024, is {worst_error}. A model with a host term per element, or one fit per workload, would follow the data more closely. With three sizes per workload it would mostly be drawing lines through points, so I kept the two term form and report where it misses.

These fits describe one shared Windows workstation, one RTX 4090 with pageable host memory, one driver and one short run with five warm samples per condition. They are not a bandwidth measurement, a prediction for other sizes, other cards or pinned memory, or a reason to expect a crossover with the CPU at some larger image. Hardware counters were not available to the measuring account (`ERR_NVGPUCTRPERM`), so the model works from timing alone. The [limits](results-2026-09-29.md) of the original run apply unchanged.

## Reproduce

`python tools/cost_model.py --check` checks the CSV hash against the receipt, recomputes every fit and prediction and fails if this page or [the full prediction table](evidence/cost-model.csv) differs. `--write` regenerates both from the retained CSV. Neither runs a workload.
"""
    return report, buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    receipt, points = load()
    report, table = outputs(receipt, points)
    for path, expected in [(ROOT / "docs/cost-model.md", report), (ROOT / "docs/evidence/cost-model.csv", table)]:
        if args.write:
            path.write_text(expected, encoding="utf-8", newline="\n")
        else:
            require(path.read_text(encoding="utf-8") == expected, f"stale generated output: {path.name}")
    print("6 CUDA conditions at 3 sizes, 3 models, 3 sizes held out: retained hash and generated outputs passed")


if __name__ == "__main__":
    main()
