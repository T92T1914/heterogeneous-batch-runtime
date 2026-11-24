import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("report_pinned", ROOT / "tools" / "report_pinned.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

IDLE_STDERR = ["NVIDIA GeForce RTX 4090 SM 8.9", "NVML: the CUDA device found by its UUID",
               "GPU idle before 32: utilization 21 percent, memory 6100 MiB",
               "GPU idle before 256: utilization 18 percent, memory 6120 MiB",
               "GPU idle before 1024: utilization 25 percent, memory 6130 MiB",
               "GPU idle after 1024: utilization 30 percent, memory 6090 MiB"]


def synthetic_rows():
    """Rows in the order and shape that hbr_cuda_pinned_bench --run prints, with made up timings.

    Pinned calls take 0.8 of the pageable call of the same iteration, and the load reading
    rises with the iteration, so the paired ratios and the load table have known values.
    """
    rows = []

    def add(phase, side, workload, memory, iteration, order, host, upload=0.0, kernel=0.0, download=0.0,
            checksum=0.0, up=0, down=0, util=20, memory_mib=6000):
        rows.append({"phase": phase, "side": str(side), "workload": workload, "host_memory": memory,
                     "iteration": str(iteration), "order": str(order), "host_ms": repr(host),
                     "upload_ms": repr(upload), "kernel_ms": repr(kernel), "download_ms": repr(download),
                     "checksum": repr(checksum), "upload_bytes": str(up), "download_bytes": str(down),
                     "gpu_util_percent": str(util), "gpu_memory_mib": str(memory_mib)})

    for side, workload, memory, iteration, order in report.run_sequence():
        if iteration == 0 and order == 0:
            add("setup", side, "all", "pinned", -1, -1, 0.5, up=side * side * 10)
        up, down = report.transfer_bytes(workload, side)
        checksum = report.histogram_checksum(side) if workload == "tile_histogram" else 0.25 * side
        pinned = memory == "pinned"
        host = (1.0 + iteration * 0.01) * (0.8 if pinned else 1.0)
        add("run", side, workload, memory, iteration, order, host, 0.4 - 0.1 * pinned, 0.2, 0.3,
            checksum, up, down, util=10 + iteration % 50, memory_mib=6000 + side)
        if iteration == report.SAMPLES - 1 and order == report.COMBINATIONS - 1:
            add("retire", side, "all", "pinned", -1, -1, 0.25, up=side * side * 10)
    return rows


@pytest.fixture(scope="module")
def rows():
    return synthetic_rows()


def copy(rows):
    return [dict(r) for r in rows]


def find(rows, **want):
    return next(i for i, r in enumerate(rows) if all(r[k] == v for k, v in want.items()))


def test_synthetic_run_passes_and_has_every_protocol_row(rows):
    runs, lifecycle = report.validate(rows)
    assert sum(len(group) for group in runs.values()) == report.PROTOCOL["run_rows"] == 2178
    assert len(lifecycle) == 2 * len(report.SIDES)


def test_summary_uses_only_the_warm_samples(rows):
    stats = report.summarize(report.validate(rows)[0])
    entry = stats[(32, "masked_reduce", "pageable")]
    assert entry["n"] == report.PROTOCOL["warm_samples"] == 120
    assert entry["host_ms"]["min"] == pytest.approx(1.01)
    assert entry["first_host_ms"] == pytest.approx(1.0)
    assert stats[(32, "masked_reduce", "pinned")]["upload_ms"]["median"] == pytest.approx(0.3)


def test_pairs_take_the_ratio_within_each_iteration(rows):
    pairs = report.paired(report.validate(rows)[0])
    pair = pairs[(256, "stencil3x3")]
    assert pair["n"] == 120
    assert pair["all"]["median"] == pytest.approx(0.8)
    assert pair["pageable_first"] == pytest.approx(0.8) and pair["pinned_first"] == pytest.approx(0.8)


def test_the_first_memory_kind_alternates_by_iteration(rows):
    runs = report.validate(rows)[0]
    pageable, pinned = runs[(32, "tile_histogram", "pageable")], runs[(32, "tile_histogram", "pinned")]
    for a, b in zip(pageable, pinned):
        assert (int(a["order"]) < int(b["order"])) == (int(a["iteration"]) % 2 == 0)
        assert abs(int(a["order"]) - int(b["order"])) == 1


def test_load_summarizes_every_run_row_of_a_side(rows):
    loads = report.load(report.validate(rows)[0])
    assert loads[1024]["n"] == 726
    assert loads[1024]["gpu_memory_mib"]["median"] == 7024
    assert loads[32]["gpu_util_percent"]["min"] == 10 and loads[32]["gpu_util_percent"]["max"] == 59


def test_tables_show_both_memory_kinds_the_pairs_and_the_load(rows):
    text = report.report(rows, {"status": "passed", "stderr": IDLE_STDERR})
    assert "| 1024 × 1024 | stencil3x3 | 8388608 | " in text
    assert "| 1024 × 1024 | 10485760 | 0.50000 | 0.25000 |" in text
    assert "| 256 × 256 | stencil3x3 | 120 | 0.800 | " in text
    assert "| 1024 × 1024 | 726 | " in text
    assert "| after 1024 | 30 percent | 6090 MiB |" in text


def test_idle_readings_come_from_the_receipt_in_order():
    readings = report.idle_readings({"stderr": IDLE_STDERR})
    assert [r["point"] for r in readings] == report.IDLE_POINTS
    assert readings[0] == {"point": "before 32", "gpu_util_percent": 21, "gpu_memory_mib": 6100}


@pytest.mark.parametrize("field, value, message", [
    ("upload_ms", "nan", "invalid upload_ms"),
    ("host_ms", "-0.1", "invalid host_ms"),
    ("order", "5", "order the protocol fixes"),
    ("host_memory", "pageable", "order the protocol fixes"),
    ("upload_bytes", "1", "transfer bytes differ"),
    ("checksum", "1.5", "output differs"),
    ("checksum", "inf", "nonfinite checksum"),
    ("kernel_ms", "-nan(ind)", "could not convert"),
    ("gpu_util_percent", "101", "invalid gpu_util_percent"),
    ("gpu_util_percent", "", "invalid gpu_util_percent"),
    ("gpu_memory_mib", "-1", "invalid gpu_memory_mib"),
])
def test_a_changed_run_row_is_rejected(rows, field, value, message):
    changed = copy(rows)
    target = find(changed, phase="run", iteration="3", workload="masked_reduce", host_memory="pinned")
    if field == "order":
        value = str((int(changed[target]["order"]) + 2) % 6)
    changed[target][field] = value
    with pytest.raises(ValueError, match=message):
        report.validate(changed)


def test_a_complete_call_of_zero_time_is_rejected(rows):
    changed = copy(rows)
    changed[find(changed, phase="run", iteration="7", workload="stencil3x3", host_memory="pageable")]["host_ms"] = "0"
    with pytest.raises(ValueError, match="zero time"):
        report.validate(changed)


@pytest.mark.parametrize("receipt, message", [
    ({"status": "passed", "stderr": IDLE_STDERR[:-1]}, "idle readings"),
    ({"status": "passed", "stderr": IDLE_STDERR[:2] + IDLE_STDERR[3:] + IDLE_STDERR[2:3]}, "idle readings"),
    ({"status": "unqualified", "stderr": IDLE_STDERR}, "not 'passed'"),
])
def test_a_receipt_without_four_idle_readings_from_a_passed_run_is_refused(rows, receipt, message):
    with pytest.raises(ValueError, match=message):
        report.report(rows, receipt)


def test_main_refuses_a_receipt_of_another_attempt(rows, tmp_path):
    path = tmp_path / "pinned.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=report.HEADER)
        writer.writeheader()
        writer.writerows(rows)
    receipt = tmp_path / "receipt.json"
    good = {"status": "passed", "stderr": IDLE_STDERR, "csv_sha256": report.hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt.write_text(report.json.dumps(good), encoding="utf-8")
    assert report.main([str(path), "--receipt", str(receipt)]) == 0
    receipt.write_text(report.json.dumps(dict(good, csv_sha256="0" * 64)), encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256"):
        report.main([str(path), "--receipt", str(receipt)])


def test_a_pinned_histogram_that_differs_from_the_oracle_is_rejected(rows):
    changed = copy(rows)
    target = changed[find(changed, phase="run", workload="tile_histogram", host_memory="pinned")]
    target["checksum"] = repr(float(target["checksum"]) + 1)
    with pytest.raises(ValueError, match="expected histogram"):
        report.validate(changed)


def test_a_missing_sample_or_lifecycle_row_is_rejected(rows):
    without_run = copy(rows)
    del without_run[find(without_run, phase="run", iteration="120", side="256", workload="stencil3x3",
                         host_memory="pageable")]
    with pytest.raises(ValueError, match="expected 2184 rows"):
        report.validate(without_run)
    without_retire = copy(rows)
    del without_retire[find(without_retire, phase="retire", side="32")]
    with pytest.raises(ValueError, match="expected 2184 rows"):
        report.validate(without_retire)


def test_lifecycle_rows_and_file_order_are_checked(rows):
    changed = copy(rows)
    changed[0]["download_bytes"] = "8"
    with pytest.raises(ValueError, match="malformed setup"):
        report.validate(changed)
    moved = copy(rows)
    moved.append(moved.pop(0))
    with pytest.raises(ValueError, match="order the protocol fixes"):
        report.validate(moved)


def test_a_short_row_is_a_value_error(rows):
    changed = copy(rows)
    changed[5]["download_bytes"] = None
    with pytest.raises(ValueError, match="wrong number of fields"):
        report.validate(changed)


def test_sequence_differences_names_the_first_wrong_row():
    rows = [dict(zip(report.SEQUENCE_HEADER, map(str, entry))) for entry in report.run_sequence()]
    assert report.sequence_differences(rows) is None
    rows[7]["workload"] = "stencil3x3" if rows[7]["workload"] != "stencil3x3" else "masked_reduce"
    assert report.sequence_differences(rows).startswith("sequence row 9 is")
    assert report.sequence_differences(rows[:-1]) == "expected 2178 sequence rows, found 2177"


def test_main_reads_the_benchmark_csv(rows, tmp_path, capsys):
    path = tmp_path / "pinned.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=report.HEADER)
        writer.writeheader()
        writer.writerows(rows)
    assert report.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "Complete call, pinned" in out
    assert "| 32 × 32 | masked_reduce | pinned | 120 | " in out
    assert "Idle reading" not in out


@pytest.mark.parametrize("side", [32, 256])
def test_expected_histogram_checksum_matches_the_runtime(side):
    np = pytest.importorskip("numpy")
    hbr = pytest.importorskip("heterogeneous_batch_runtime")
    image = ((np.arange(side * side, dtype=np.uint64) * 13) % 256).astype(np.uint8).reshape(side, side)
    counts = hbr.tile_histogram(image, report.TILE, report.TILE, "scalar").ravel()
    weights = np.arange(1, counts.size + 1, dtype=np.uint64)
    assert float(int((counts * weights).sum())) == report.histogram_checksum(side)
