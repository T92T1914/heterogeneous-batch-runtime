import csv
import importlib.util
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
                index = (order + iteration + condition) % 6
                workload, backend = report.WORKLOADS[index % 3], report.BACKENDS[index // 3]
                up, down = report.transfer_bytes(workload, side)
                checksum = report.histogram_checksum(side) if workload == "tile_histogram" else 0.25 * side
                cuda = backend == "cuda"
                rows.append({"side": str(side), "workload": workload, "backend": backend,
                             "iteration": str(iteration), "order": str(order),
                             "host_ms": repr(1.0 + iteration * 0.01 + (0.0 if cuda else 0.5)),
                             "upload_ms": repr(0.4) if cuda else "", "kernel_ms": repr(0.2) if cuda else "",
                             "download_ms": repr(0.3) if cuda else "", "checksum": repr(checksum),
                             "upload_bytes": str(up), "download_bytes": str(down)})
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
    assert sum(len(group) for group in runs.values()) == report.PROTOCOL["run_rows"] == 450
    assert set(runs) == {(s, w, b) for s in report.SIDES for w in report.WORKLOADS for b in report.BACKENDS}


def test_the_first_rows_follow_the_rotation():
    assert report.expected_sequence()[:7] == [
        (32, "masked_reduce", "cuda", 0, 0), (32, "tile_histogram", "cuda", 0, 1),
        (32, "stencil3x3", "cuda", 0, 2), (32, "masked_reduce", "sycl", 0, 3),
        (32, "tile_histogram", "sycl", 0, 4), (32, "stencil3x3", "sycl", 0, 5),
        (32, "tile_histogram", "cuda", 1, 0)]
    assert report.expected_sequence()[150] == (256, "tile_histogram", "cuda", 0, 0)


def test_summary_uses_only_the_warm_samples(rows):
    stats = report.summarize(report.validate(rows))
    cuda, sycl = stats[(32, "masked_reduce", "cuda")], stats[(32, "masked_reduce", "sycl")]
    assert cuda["n"] == sycl["n"] == report.PROTOCOL["warm_samples"]
    assert cuda["host_ms"]["min"] == pytest.approx(1.01)
    assert cuda["first_host_ms"] == pytest.approx(1.0)
    assert sycl["host_ms"]["median"] == pytest.approx(1.625)
    assert cuda["kernel_ms"]["median"] == pytest.approx(0.2)
    assert "kernel_ms" not in sycl


def test_tables_put_both_backends_side_by_side_and_keep_a_slower_sycl(rows):
    text = report.tables(report.summarize(report.validate(rows)))
    assert "| 1024 × 1024 | stencil3x3 | 1.12500 | 1.62500 | 1.44 | " in text
    assert "| 32 × 32 | tile_histogram | sycl | 24 | 1.62500 | 1.51000 to 1.74000 | 1.50000 | none | none | none |" in text
    assert "| yes |" in text


@pytest.mark.parametrize("field, value, message", [
    ("host_ms", "nan", "invalid host_ms"),
    ("host_ms", "-0.1", "invalid host_ms"),
    ("upload_ms", "", "could not convert"),
    ("kernel_ms", "-1", "invalid kernel_ms"),
    ("upload_bytes", "1", "transfer bytes differ"),
    ("checksum", "1.5", "output differs across samples"),
    ("checksum", "inf", "nonfinite checksum"),
    ("order", "4", "rotation"),
    ("backend", "sycl", "rotation"),
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
    with pytest.raises(ValueError, match="expected 450 run rows, found 449"):
        report.validate(rows[:-1])
    with pytest.raises(ValueError, match="expected 450 run rows, found 451"):
        report.validate(rows + [dict(rows[-1])])
    moved = copy(rows)
    moved[10], moved[11] = moved[11], moved[10]
    with pytest.raises(ValueError, match="row 12 is not in the rotation"):
        report.validate(moved)


def test_a_csv_with_no_rows_is_rejected():
    with pytest.raises(ValueError, match="no run rows"):
        report.validate([])


def test_backends_with_different_checksums_are_reported(rows):
    changed = copy(rows)
    for row in changed:
        if row["workload"] == "masked_reduce" and row["backend"] == "sycl" and row["side"] == "256":
            row["checksum"] = repr(64.00000000000001)
    text = report.tables(report.summarize(report.validate(changed)))
    assert "| 256 × 256 | masked_reduce | 1.12500 | 1.62500 | 1.44 | " in text
    line = next(l for l in text.splitlines() if l.startswith("| 256 × 256 | masked_reduce | 1.125"))
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
    assert report.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "SYCL ÷ CUDA" in out
    assert "| 256 × 256 | masked_reduce | cuda | 24 | " in out


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
