import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_pinned_plan", ROOT / "tools" / "check_pinned_plan.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

# What tools/cuda_pinned_benchmark.cpp prints for --plan at this revision.
PLAN = {"square_sides": [32, 256, 1024], "workloads": ["masked_reduce", "tile_histogram", "stencil3x3"],
        "host_memory": ["pageable", "pinned"], "tile_shape": [64, 64], "samples_per_condition": 25,
        "additional_unrecorded_warmups": 2, "run_rows": 450}


def protocol():
    return json.loads(check.PROTOCOL.read_text(encoding="utf-8"))


def test_committed_protocol_matches_the_plan():
    assert check.differences(PLAN, protocol()) == []


def test_a_changed_sample_count_is_named():
    plan = dict(PLAN, samples_per_condition=24, run_rows=432)
    assert check.differences(plan, protocol()) == ["samples_per_condition", "run_rows"]


def test_protocol_row_count_must_follow_from_its_conditions():
    changed = dict(protocol(), run_rows=451)
    assert "run_rows arithmetic" in check.differences(dict(PLAN, run_rows=451), changed)


def test_plan_constants_appear_in_the_benchmark_source():
    source = (ROOT / "tools" / "cuda_pinned_benchmark.cpp").read_text(encoding="utf-8")
    assert "sides{32,256,1024}" in source
    assert "samples=25, warmups=2" in source
    assert "tile=64" in source


def test_main_reports_a_mismatch(tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(PLAN, host_memory=["pageable"])), encoding="utf-8")
    assert check.main(["check_pinned_plan.py", str(path)]) == 1
    assert "host_memory" in capsys.readouterr().err
