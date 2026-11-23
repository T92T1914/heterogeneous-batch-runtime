import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_sycl_cuda_plan", ROOT / "tools" / "check_sycl_cuda_plan.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

# What tools/sycl_cuda_benchmark.cpp prints for --plan at this revision.
PLAN = {"square_sides": [32, 256, 1024], "workloads": ["masked_reduce", "tile_histogram", "stencil3x3"],
        "backends": ["cuda", "sycl"], "tile_shape": [64, 64], "samples_per_condition": 121,
        "additional_unrecorded_warmups": 2, "run_rows": 2178}


def protocol():
    return json.loads(check.PROTOCOL.read_text(encoding="utf-8"))


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_committed_protocol_matches_the_plan():
    assert check.differences(PLAN, protocol()) == []


def test_a_changed_sample_count_is_named():
    plan = dict(PLAN, samples_per_condition=24, run_rows=432)
    assert check.differences(plan, protocol()) == ["samples_per_condition", "run_rows"]


def test_a_dropped_backend_is_named():
    assert check.differences(dict(PLAN, backends=["cuda"], run_rows=1089), protocol()) == ["backends", "run_rows"]


def test_protocol_counts_must_follow_from_its_conditions():
    changed = dict(protocol(), run_rows=451, warm_samples=25)
    found = check.differences(dict(PLAN, run_rows=451), changed)
    assert "run_rows arithmetic" in found
    assert "warm_samples arithmetic" in found


def test_plan_constants_appear_in_the_benchmark_source():
    text = source("tools/sycl_cuda_benchmark.cpp")
    assert "sides{32,256,1024}" in text
    assert 'workloads{"masked_reduce","tile_histogram","stencil3x3"}' in text
    assert 'backends{"cuda","sycl"}' in text
    assert "samples=121, warmups=2" in text
    assert "tile=64" in text
    assert '"run_rows\\": "<<sides.size()*combinations*samples' in text


def test_the_inputs_are_those_of_the_pinned_benchmark():
    generator = "values_[i]=static_cast<double>(i%1009)/1009-0.5; mask_[i]=i%3!=0; image_[i]=static_cast<std::uint8_t>(i*13);"
    pinned = "values[i]=static_cast<double>(i%1009)/1009-0.5; mask[i]=i%3!=0; image[i]=static_cast<std::uint8_t>(i*13);"
    assert generator in source("tools/sycl_cuda_benchmark.cpp")
    assert pinned in source("tools/cuda_pinned_benchmark.cpp")


def test_the_launch_shape_the_protocol_names_is_in_both_libraries():
    cuda, sycl = source("src/cuda.cu"), source("src/sycl.cpp")
    # CUDA: at most 4096 blocks of 256 for the stencil and the first reduction pass, one block
    # of 256 for the second pass and one block of 256 for each histogram tile.
    assert "std::min<std::size_t>(4096, n / 256 + (n % 256 != 0))" in cuda
    assert "const auto nb = blocks(values.size());" in cuda
    assert "reduce_kernel<<<nb, 256, 0, op.stream>>>(x, m, values.size(), partial)" in cuda
    assert "reduce_kernel<<<1, 256, 0, op.stream>>>(partial, nullptr, nb, result)" in cuda
    assert "stencil_kernel<<<blocks(image.size()),256,0,op.stream>>>" in cuda
    assert cuda.count("<<<static_cast<unsigned>(tiles),256,0,op.stream>>>") == 2
    # SYCL: the same counts as work groups of 256.
    assert "constexpr std::size_t lanes = 256;" in sycl
    assert "std::min<std::size_t>(4096, n / lanes + (n % lanes != 0))" in sycl
    assert "const auto groups = grid_groups(values.size());" in sycl
    assert "reduce(op.queue(), x, m, values.size(), groups, partial);" in sycl
    assert "reduce(op.queue(), partial, nullptr, groups, 1, result);" in sycl
    assert "stencil(op.queue(), x, output, rows, cols, grid_groups(image.size()));" in sycl
    assert sycl.count("sycl::nd_range<1>(groups * lanes, lanes)") == 2
    assert "sycl::nd_range<1>(tiles * lanes, lanes)" in sycl


def test_both_calls_use_the_default_histogram_bins():
    assert "bool shared_bins = true" in source("include/hbr/cuda.hpp")
    assert "bool local_bins = true" in source("include/hbr/sycl.hpp")
    text = source("tools/sycl_cuda_benchmark.cpp")
    assert "hbr::cuda_tile_histogram(in.image,side,side,tile,tile)" in text
    assert "hbr::sycl_tile_histogram(in.image,side,side,tile,tile)" in text


def test_main_reports_a_mismatch(tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(PLAN, backends=["sycl", "cuda"])), encoding="utf-8")
    assert check.main(["check_sycl_cuda_plan.py", str(path)]) == 1
    assert "backends" in capsys.readouterr().err


def test_main_accepts_the_plan_and_rejects_a_missing_argument(tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(PLAN), encoding="utf-8")
    assert check.main(["check_sycl_cuda_plan.py", str(path)]) == 0
    assert "2178 run rows" in capsys.readouterr().out
    assert check.main(["check_sycl_cuda_plan.py"]) == 2


def test_the_benchmark_stops_when_the_backends_report_different_gpus():
    assert 'if (cuda.name!=sycl.name) throw std::runtime_error("CUDA and SYCL report different GPUs");' \
        in source("tools/sycl_cuda_benchmark.cpp")


def write_sequence(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write("side,workload,backend,iteration,order\n")
        for row in rows:
            stream.write(",".join(map(str, row)) + "\n")


def test_main_checks_the_run_order_when_given_one(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(PLAN), encoding="utf-8")
    sequence = check.report_sycl_cuda.expected_sequence()
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    write_sequence(good, sequence)
    write_sequence(bad, sequence[:6] + [sequence[7], sequence[6]] + sequence[8:])
    assert check.main(["check_sycl_cuda_plan.py", str(plan), str(good)]) == 0
    assert "Run order matches" in capsys.readouterr().out
    assert check.main(["check_sycl_cuda_plan.py", str(plan), str(bad)]) == 1
    assert "sequence row 8 is" in capsys.readouterr().err


def test_the_benchmark_and_the_report_share_one_order_formula():
    text = source("tools/sycl_cuda_benchmark.cpp")
    assert "return {(iteration+condition+order/2)%3, (order%2+iteration%2)%2};" in text
    assert "const auto [workload,backend]=slot(condition,iteration,order);" in text
    report = source("tools/report_sycl_cuda.py")
    assert "WORKLOADS[(iteration + condition + order // 2) % len(WORKLOADS)]" in report
    assert "BACKENDS[(order + iteration) % len(BACKENDS)]" in report


def test_the_gpu_reading_comes_before_the_clock_and_the_uuid_is_not_printed():
    text = source("tools/sycl_cuda_benchmark.cpp")
    run = text[text.index("const auto gpu=monitor.read();"):]
    assert run.index("const auto gpu=monitor.read();") < run.index("execute(workload,backend,data.inputs(),side)")
    assert 'dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL)' in text
    assert '"nvmlDeviceGetHandleByUUID"' in text
    assert "std::cerr<<uuid" not in text and "<<monitor" not in text
    assert "constexpr auto idle_pause=std::chrono::seconds(2);" in text
