import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("report_pinned", ROOT / "tools" / "report_pinned.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def synthetic_rows():
    """Rows in the order and shape that hbr_cuda_pinned_bench --run prints, with made up timings."""
    rows = []

    def add(phase, side, workload, memory, iteration, order, host, upload=0.0, kernel=0.0, download=0.0,
            checksum=0.0, up=0, down=0):
        rows.append({"phase": phase, "side": str(side), "workload": workload, "host_memory": memory,
                     "iteration": str(iteration), "order": str(order), "host_ms": repr(host),
                     "upload_ms": repr(upload), "kernel_ms": repr(kernel), "download_ms": repr(download),
                     "checksum": repr(checksum), "upload_bytes": str(up), "download_bytes": str(down)})

    for condition, side in enumerate(report.SIDES):
        add("setup", side, "all", "pinned", -1, -1, 0.5, up=side * side * 10)
        for iteration in range(report.SAMPLES):
            for order in range(6):
                index = (order + iteration + condition) % 6
                workload, memory = report.WORKLOADS[index % 3], report.MEMORIES[index // 3]
                up, down = report.transfer_bytes(workload, side)
                checksum = report.histogram_checksum(side) if workload == "tile_histogram" else 0.25 * side
                pinned = memory == "pinned"
                add("run", side, workload, memory, iteration, order, 1.0 + iteration * 0.01 - 0.2 * pinned,
                    0.4 - 0.1 * pinned, 0.2, 0.3, checksum, up, down)
        add("retire", side, "all", "pinned", -1, -1, 0.25, up=side * side * 10)
    return rows


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows()


def copy(rows):
    return [dict(r) for r in rows]


def test_synthetic_run_passes_and_has_every_protocol_row(rows):
    runs, lifecycle = report.validate(rows)
    assert sum(len(group) for group in runs.values()) == report.PROTOCOL["run_rows"]
    assert len(lifecycle) == 2 * len(report.SIDES)


def test_summary_uses_only_the_warm_samples(rows):
    stats = report.summarize(report.validate(rows)[0])
    entry = stats[(32, "masked_reduce", "pageable")]
    assert entry["n"] == report.PROTOCOL["warm_samples"]
    assert entry["host_ms"]["min"] == pytest.approx(1.01)
    assert entry["first_host_ms"] == pytest.approx(1.0)
    assert stats[(32, "masked_reduce", "pinned")]["upload_ms"]["median"] == pytest.approx(0.3)


def test_tables_show_both_memory_kinds_side_by_side(rows):
    runs, lifecycle = report.validate(rows)
    text = report.tables(report.summarize(runs), lifecycle)
    assert "| 1024 × 1024 | stencil3x3 | 8388608 | " in text
    assert "| 1024 × 1024 | 10485760 | 0.50000 | 0.25000 |" in text


@pytest.mark.parametrize("field, value, message", [
    ("upload_ms", "nan", "invalid upload_ms"),
    ("host_ms", "-0.1", "invalid host_ms"),
    ("order", "5", "order differs"),
    ("upload_bytes", "1", "transfer bytes differ"),
    ("checksum", "1.5", "output differs"),
    ("checksum", "inf", "nonfinite checksum"),
    ("kernel_ms", "-nan(ind)", "could not convert"),
])
def test_a_changed_run_row_is_rejected(rows, field, value, message):
    changed = copy(rows)
    target = next(i for i, r in enumerate(changed) if r["phase"] == "run" and r["iteration"] == "3"
                  and r["workload"] == "masked_reduce" and r["host_memory"] == "pinned")
    if field == "order":
        value = str((int(changed[target]["order"]) + 1) % 6)
    changed[target][field] = value
    with pytest.raises(ValueError, match=message):
        report.validate(changed)


def test_a_pinned_histogram_that_differs_from_the_oracle_is_rejected(rows):
    changed = copy(rows)
    target = next(r for r in changed if r["phase"] == "run" and r["workload"] == "tile_histogram"
                  and r["host_memory"] == "pinned")
    target["checksum"] = repr(float(target["checksum"]) + 1)
    with pytest.raises(ValueError, match="output differs"):
        report.validate(changed)


def test_a_missing_sample_or_lifecycle_row_is_rejected(rows):
    without_run = [r for r in rows if not (r["phase"] == "run" and r["iteration"] == "24"
                                           and r["side"] == "256" and r["workload"] == "stencil3x3"
                                           and r["host_memory"] == "pageable")]
    with pytest.raises(ValueError, match="iterations"):
        report.validate(without_run)
    without_retire = [r for r in rows if not (r["phase"] == "retire" and r["side"] == "32")]
    with pytest.raises(ValueError, match="setup or retire"):
        report.validate(without_retire)


def test_lifecycle_rows_and_file_order_are_checked(rows):
    changed = copy(rows)
    changed[0]["download_bytes"] = "8"
    with pytest.raises(ValueError, match="malformed setup"):
        report.validate(changed)
    moved = copy(rows)
    moved.append(moved.pop(0))
    with pytest.raises(ValueError, match="order the benchmark prints"):
        report.validate(moved)


def test_a_short_row_is_a_value_error(rows):
    changed = copy(rows)
    changed[5]["download_bytes"] = None
    with pytest.raises(ValueError, match="wrong number of fields"):
        report.validate(changed)


def test_main_reads_the_benchmark_csv(rows, tmp_path, capsys):
    path = tmp_path / "pinned.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=report.HEADER)
        writer.writeheader()
        writer.writerows(rows)
    assert report.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "Complete call, pinned" in out
    assert "| 32 × 32 | masked_reduce | pinned | 24 | " in out


@pytest.mark.parametrize("side", [32, 256])
def test_expected_histogram_checksum_matches_the_runtime(side):
    np = pytest.importorskip("numpy")
    hbr = pytest.importorskip("heterogeneous_batch_runtime")
    image = ((np.arange(side * side, dtype=np.uint64) * 13) % 256).astype(np.uint8).reshape(side, side)
    counts = hbr.tile_histogram(image, report.TILE, report.TILE, "scalar").ravel()
    weights = np.arange(1, counts.size + 1, dtype=np.uint64)
    assert float(int((counts * weights).sum())) == report.histogram_checksum(side)
