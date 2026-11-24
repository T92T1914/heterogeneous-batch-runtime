import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("report_sycl_cuda", ROOT / "tools" / "report_sycl_cuda.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def synthetic_rows():
    """Rows in the order and shape that hbr_sycl_cuda_bench --run prints, with made up timings."""
    rows = []
    for condition, side in enumerate(report.SIDES):
        for iteration in range(report.SAMPLES):
            for order in range(6):
                workload = report.WORKLOADS[(iteration + condition + order // 2) % 3]
                backend = report.BACKENDS[(order + iteration) % 2]
                up, down = report.transfer_bytes(workload, side)
                checksum = report.histogram_checksum(side) if workload == "tile_histogram" else 0.25 * side
                cuda = backend == "cuda"
                rows.append({"side": str(side), "workload": workload, "backend": backend,
                             "iteration": str(iteration), "order": str(order),
                             "host_ms": repr(1.0 + iteration * 0.01 + (0.0 if cuda else 0.5)),
                             "upload_ms": repr(0.4) if cuda else "", "kernel_ms": repr(0.2) if cuda else "",
                             "download_ms": repr(0.3) if cuda else "", "checksum": repr(checksum),
                             "upload_bytes": str(up), "download_bytes": str(down),
                             "gpu_util_percent": str(20 + iteration % 7), "gpu_memory_mib": str(6000 + side)})
    return rows


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows()


def copy(rows):
    return [dict(r) for r in rows]


def find(rows, **match):
    return next(i for i, r in enumerate(rows) if all(r[k] == v for k, v in match.items()))


def test_synthetic_run_passes_and_has_every_protocol_row(rows):
    runs = report.validate(rows)
    assert sum(len(group) for group in runs.values()) == report.PROTOCOL["run_rows"] == 2178
    assert set(runs) == {(s, w, b) for s in report.SIDES for w in report.WORKLOADS for b in report.BACKENDS}


def test_the_first_rows_pair_the_backends_back_to_back():
    assert report.expected_sequence()[:13] == [
        (32, "masked_reduce", "cuda", 0, 0), (32, "masked_reduce", "sycl", 0, 1),
        (32, "tile_histogram", "cuda", 0, 2), (32, "tile_histogram", "sycl", 0, 3),
        (32, "stencil3x3", "cuda", 0, 4), (32, "stencil3x3", "sycl", 0, 5),
        (32, "tile_histogram", "sycl", 1, 0), (32, "tile_histogram", "cuda", 1, 1),
        (32, "stencil3x3", "sycl", 1, 2), (32, "stencil3x3", "cuda", 1, 3),
        (32, "masked_reduce", "sycl", 1, 4), (32, "masked_reduce", "cuda", 1, 5),
        (32, "stencil3x3", "cuda", 2, 0)]
    assert report.expected_sequence()[726] == (256, "tile_histogram", "cuda", 0, 0)


def test_the_warm_iterations_balance_place_and_first_backend():
    # Over the 120 warm iterations each workload holds each place in the round with each
    # backend first 20 times, on every side.
    counts = {}
    for side, workload, backend, iteration, order in report.expected_sequence():
        if iteration and order % 2 == 0:
            key = (side, workload, order // 2, backend)
            counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 3 * 3 * 3 * 2
    assert set(counts.values()) == {20}


def test_summary_uses_only_the_warm_samples(rows):
    stats = report.summarize(report.validate(rows))
    cuda, sycl = stats[(32, "masked_reduce", "cuda")], stats[(32, "masked_reduce", "sycl")]
    assert cuda["n"] == sycl["n"] == report.PROTOCOL["warm_samples"]
    assert cuda["host_ms"]["min"] == pytest.approx(1.01)
    assert cuda["first_host_ms"] == pytest.approx(1.0)
    assert sycl["host_ms"]["median"] == pytest.approx(2.105)
    assert cuda["kernel_ms"]["median"] == pytest.approx(0.2)
    assert "kernel_ms" not in sycl


def test_tables_put_both_backends_side_by_side_and_keep_a_slower_sycl(rows):
    text = report.report(rows)
    assert "| 1024 × 1024 | stencil3x3 | 1.60500 | 2.10500 | 1.31 | " in text
    assert "| 32 × 32 | tile_histogram | sycl | 120 | 2.10500 | 1.51000 to 2.70000 | 1.50000 | none | none | none |" in text
    assert "| yes |" in text


@pytest.mark.parametrize("field, value, message", [
    ("host_ms", "nan", "invalid host_ms"),
    ("host_ms", "-0.1", "invalid host_ms"),
    ("upload_ms", "", "could not convert"),
    ("kernel_ms", "-1", "invalid kernel_ms"),
    ("upload_bytes", "1", "transfer bytes differ"),
    ("checksum", "1.5", "output differs across samples"),
    ("checksum", "inf", "nonfinite checksum"),
    ("order", "4", "order the protocol fixes"),
    ("backend", "sycl", "order the protocol fixes"),
    ("gpu_util_percent", "101", "invalid gpu_util_percent"),
    ("gpu_util_percent", "-1", "invalid gpu_util_percent"),
    ("gpu_util_percent", "", "invalid gpu_util_percent"),
    ("gpu_memory_mib", "6000.5", "invalid gpu_memory_mib"),
])
def test_a_changed_cuda_row_is_rejected(rows, field, value, message):
    changed = copy(rows)
    target = find(changed, side="256", iteration="3", workload="masked_reduce", backend="cuda")
    changed[target][field] = value
    with pytest.raises(ValueError, match=message):
        report.validate(changed)


def test_a_sycl_row_with_device_timing_is_rejected(rows):
    changed = copy(rows)
    changed[find(changed, side="32", iteration="5", workload="stencil3x3", backend="sycl")]["kernel_ms"] = "0"
    with pytest.raises(ValueError, match="SYCL row carries device timing"):
        report.validate(changed)


def test_a_histogram_that_differs_from_the_expected_counts_is_rejected(rows):
    changed = copy(rows)
    for row in changed:
        if row["workload"] == "tile_histogram" and row["backend"] == "sycl" and row["side"] == "1024":
            row["checksum"] = repr(float(row["checksum"]) + 1)
    with pytest.raises(ValueError, match="histogram differs"):
        report.validate(changed)


def test_a_missing_extra_or_moved_row_is_rejected(rows):
    with pytest.raises(ValueError, match="expected 2178 run rows, found 2177"):
        report.validate(rows[:-1])
    with pytest.raises(ValueError, match="expected 2178 run rows, found 2179"):
        report.validate(rows + [dict(rows[-1])])
    moved = copy(rows)
    moved[10], moved[11] = moved[11], moved[10]
    with pytest.raises(ValueError, match="row 12 is not in the order"):
        report.validate(moved)


def test_a_csv_with_no_rows_is_rejected():
    with pytest.raises(ValueError, match="no run rows"):
        report.validate([])


def test_backends_with_different_checksums_are_reported(rows):
    changed = copy(rows)
    for row in changed:
        if row["workload"] == "masked_reduce" and row["backend"] == "sycl" and row["side"] == "256":
            row["checksum"] = repr(64.00000000000001)
    text = report.report(changed)
    assert "| 256 × 256 | masked_reduce | 1.60500 | 2.10500 | 1.31 | " in text
    line = next(l for l in text.splitlines() if l.startswith("| 256 × 256 | masked_reduce | 1.605"))
    assert line.endswith("| no |")


def test_a_short_row_is_a_value_error(rows):
    changed = copy(rows)
    changed[5]["download_bytes"] = None
    with pytest.raises(ValueError, match="wrong number of fields"):
        report.validate(changed)


def test_main_reads_the_benchmark_csv(rows, tmp_path, capsys):
    path = tmp_path / "sycl-cuda.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=report.HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"stderr": RECEIPT_STDERR}), encoding="utf-8")
    assert report.main([str(path), "--receipt", str(receipt)]) == 0
    out = capsys.readouterr().out
    assert "SYCL ÷ CUDA" in out
    assert "| 256 × 256 | masked_reduce | cuda | 120 | " in out
    assert "| after 1024 | 17 percent | 6650 MiB |" in out
    assert report.main([str(path)]) == 0
    assert "Idle reading" not in capsys.readouterr().out


RECEIPT_STDERR = ["CUDA: Stand-in GPU SM 8.9", "GPU idle before 32: utilization 21 percent, memory 6640 MiB",
                  "GPU idle before 256: utilization 19 percent, memory 6644 MiB",
                  "GPU idle before 1024: utilization 18 percent, memory 6648 MiB",
                  "GPU idle after 1024: utilization 17 percent, memory 6650 MiB"]


def test_idle_readings_come_from_the_receipt_stderr_in_run_order():
    readings = report.idle_readings({"stderr": RECEIPT_STDERR + ["GPU idle sideways: utilization 1 percent"]})
    assert [r["point"] for r in readings] == ["before 32", "before 256", "before 1024", "after 1024"]
    assert readings[0] == {"point": "before 32", "gpu_util_percent": 21, "gpu_memory_mib": 6640}
    assert report.idle_readings({}) == []


def test_pairs_divide_within_an_iteration_and_split_by_first_backend(rows):
    changed = copy(rows)
    # SYCL is twice CUDA in pairs where CUDA ran first and three times where SYCL ran first.
    for row in changed:
        if row["backend"] == "sycl":
            factor = 2.0 if int(row["iteration"]) % 2 == 0 else 3.0
            row["host_ms"] = repr(factor * (1.0 + int(row["iteration"]) * 0.01))
    pairs = report.paired(report.validate(changed))
    entry = pairs[(1024, "tile_histogram")]
    assert entry["n"] == 120 and entry["n_cuda_first"] == 60
    assert entry["cuda_first"] == pytest.approx(2.0) and entry["sycl_first"] == pytest.approx(3.0)
    assert entry["all"]["min"] == pytest.approx(2.0) and entry["all"]["max"] == pytest.approx(3.0)
    text = report.report(changed)
    assert "| 1024 × 1024 | tile_histogram | 120 | 2.500 | 2.000 to 3.000 | 2.000 to 3.000 | 2.000 | 3.000 |" in text


def test_load_summarizes_every_reading_of_a_side(rows):
    loads = report.load(report.validate(rows))
    assert loads[256]["n"] == 726
    assert loads[256]["gpu_memory_mib"]["median"] == 6256
    assert loads[32]["gpu_util_percent"]["min"] == 20 and loads[32]["gpu_util_percent"]["max"] == 26
    assert "| 32 × 32 | 726 | 23 percent | 20 to 26 percent | 6032 MiB | 6032 to 6032 MiB |" in report.report(rows)


def test_sequence_differences_names_the_first_moved_row():
    good = [dict(zip(report.SEQUENCE_HEADER, map(str, row))) for row in report.expected_sequence()]
    assert report.sequence_differences(good) is None
    moved = [dict(r) for r in good]
    moved[1]["backend"] = "cuda"
    assert report.sequence_differences(moved).startswith("sequence row 3 is")
    assert report.sequence_differences(good[:-1]) == "expected 2178 sequence rows, found 2177"
    assert report.sequence_differences([]) == "unexpected sequence header"


def test_the_csv_header_is_the_one_the_benchmark_prints():
    source = (ROOT / "tools" / "sycl_cuda_benchmark.cpp").read_text(encoding="utf-8")
    assert '"' + ",".join(report.HEADER) + '\\n"' in source


@pytest.mark.parametrize("side", [32, 256])
def test_expected_histogram_checksum_matches_the_runtime(side):
    np = pytest.importorskip("numpy")
    hbr = pytest.importorskip("heterogeneous_batch_runtime")
    image = ((np.arange(side * side, dtype=np.uint64) * 13) % 256).astype(np.uint8).reshape(side, side)
    counts = hbr.tile_histogram(image, report.TILE, report.TILE, "scalar").ravel()
    weights = np.arange(1, counts.size + 1, dtype=np.uint64)
    assert float(int((counts * weights).sum())) == report.histogram_checksum(side)
