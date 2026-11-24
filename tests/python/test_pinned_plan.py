import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_pinned_plan", ROOT / "tools" / "check_pinned_plan.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
report = check.report_pinned

# What tools/cuda_pinned_benchmark.cpp prints for --plan at this revision.
PLAN = {"square_sides": [32, 256, 1024], "workloads": ["masked_reduce", "tile_histogram", "stencil3x3"],
        "host_memory": ["pageable", "pinned"], "tile_shape": [64, 64], "samples_per_condition": 121,
        "additional_unrecorded_warmups": 2, "run_rows": 2178}


def protocol():
    return json.loads(check.PROTOCOL.read_text(encoding="utf-8"))


def benchmark_sequence():
    """What --sequence prints, computed the way the benchmark's slot() computes it."""
    rows = []
    for condition, side in enumerate(PLAN["square_sides"]):
        for iteration in range(PLAN["samples_per_condition"]):
            for order in range(6):
                workload = (iteration + condition + order // 2) % 3
                memory = (order % 2 + iteration % 2) % 2
                rows.append({"side": str(side), "workload": PLAN["workloads"][workload],
                             "host_memory": PLAN["host_memory"][memory], "iteration": str(iteration),
                             "order": str(order)})
    return rows


def write_sequence(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=report.SEQUENCE_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def test_committed_protocol_matches_the_plan():
    assert check.differences(PLAN, protocol()) == []


def test_a_changed_sample_count_is_named():
    plan = dict(PLAN, samples_per_condition=120, run_rows=2160)
    assert check.differences(plan, protocol()) == ["samples_per_condition", "run_rows"]


def test_protocol_row_count_must_follow_from_its_conditions():
    changed = dict(protocol(), run_rows=2179)
    assert "run_rows arithmetic" in check.differences(dict(PLAN, run_rows=2179), changed)


def test_protocol_warm_samples_must_follow_from_its_samples():
    changed = dict(protocol(), warm_samples=121)
    assert check.differences(PLAN, changed) == ["warm_samples arithmetic"]


def test_plan_constants_appear_in_the_benchmark_source():
    source = (ROOT / "tools" / "cuda_pinned_benchmark.cpp").read_text(encoding="utf-8")
    assert "sides{32,256,1024}" in source
    assert "samples=121, warmups=2" in source
    assert "tile=64" in source
    assert "return {(iteration+condition+order/2)%3, (order%2+iteration%2)%2};" in source


def test_the_benchmark_reads_nvml_only_from_system32_on_windows():
    source = (ROOT / "tools" / "cuda_pinned_benchmark.cpp").read_text(encoding="utf-8")
    assert 'LoadLibraryExW(L"nvml.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32)' in source
    assert '"nvmlDeviceGetHandleByUUID"' in source
    # One reading before the setup, each run and the retire row, each before its clock starts.
    assert source.count("gpu=monitor.read();") == 3 and "auto gpu=monitor.read();" in source


def test_the_benchmark_order_is_the_order_the_report_expects():
    assert report.sequence_differences(benchmark_sequence()) is None
    assert len(benchmark_sequence()) == protocol()["run_rows"]


def test_each_pair_runs_back_to_back_with_alternating_first_memory():
    rows = benchmark_sequence()
    for first, second in zip(rows[::2], rows[1::2]):
        assert (first["side"], first["workload"], first["iteration"]) == \
            (second["side"], second["workload"], second["iteration"])
        assert {first["host_memory"], second["host_memory"]} == {"pageable", "pinned"}
        assert (first["host_memory"] == "pageable") == (int(first["iteration"]) % 2 == 0)


def test_main_reports_a_mismatch(tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(PLAN, host_memory=["pageable"])), encoding="utf-8")
    assert check.main(["check_pinned_plan.py", str(path)]) == 1
    assert "host_memory" in capsys.readouterr().err


def test_main_checks_the_run_order(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(PLAN), encoding="utf-8")
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    rows = benchmark_sequence()
    write_sequence(good, rows)
    swapped = [dict(r) for r in rows]
    swapped[0]["host_memory"], swapped[1]["host_memory"] = swapped[1]["host_memory"], swapped[0]["host_memory"]
    write_sequence(bad, swapped)
    assert check.main(["check_pinned_plan.py", str(plan), str(good)]) == 0
    assert "Run order matches" in capsys.readouterr().out
    assert check.main(["check_pinned_plan.py", str(plan), str(bad)]) == 1
    assert "sequence row 2" in capsys.readouterr().err
