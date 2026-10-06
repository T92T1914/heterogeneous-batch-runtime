"""Build the static evidence page from pinned retained records, without execution."""
import argparse
from collections import defaultdict
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "34e2b1151681ba231eb38713bde1a2f21cf8444f"
REPO = "https://github.com/T92T1914/heterogeneous-batch-runtime"
LABELS = {
    "scalar_1": "Scalar CPU, 1 thread", "optimized_1": "Optimized CPU, 1 thread",
    "optimized_2": "Optimized CPU, 2 threads", "cpu_optimized_1": "Optimized CPU, 1 thread",
    "cpu_optimized_2": "Optimized CPU, 2 threads", "cuda_global": "CUDA, global bins",
    "cuda_shared": "CUDA, shared bins", "cuda": "CUDA",
    "cuda_simple_global": "Simple CUDA, global bins", "cuda_simple_shared": "Simple CUDA, shared bins",
    "cuda_reuse_global": "Reused CUDA, global bins", "cuda_reuse_shared": "Reused CUDA, shared bins",
    "original_1": "Original CPU, 1 thread", "followup_1": "Dispatch once, 1 thread",
    "original_2": "Original CPU, 2 threads", "followup_2": "Dispatch once, 2 threads",
    "cpp-fresh": "Fresh C++ owner/session", "cpp-reused": "Reused C++ owner/session",
    "python-cpu-reused": "Direct Python CPU session",
}
WORKLOADS = {"masked_reduce": "Masked reduction", "tile_histogram": "Tiled histogram", "stencil3x3": "3 by 3 stencil"}
METHOD_ORDER = list(LABELS)


def link(path, revision=SOURCE):
    return f"{REPO}/blob/{revision}/{path}"


def read_json(root, path):
    return json.loads((root / path).read_text(encoding="utf-8"))


def read_csv(root, path):
    with (root / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_inputs(root):
    pins = read_json(root, "web/source-pins.json")
    if pins["source_revision"] != SOURCE:
        raise ValueError("site inputs do not name the frozen source")
    for path, digest in pins["files"].items():
        target = (root / path).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("input path escapes repository")
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("retained site input changed: " + path)
    return pins


def finite(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("invalid elapsed interval")
    return number


def summary(condition, label, method, rows, field, first, events=False):
    values = [finite(r[field]) for r in rows]
    if not values:
        raise ValueError("empty timing condition")
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    item = {"condition": condition, "condition_label": label, "method": method,
            "method_label": LABELS[method], "samples": len(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values), "p25_ms": quartiles[0], "p75_ms": quartiles[2],
            "first_ms": finite(first[field]), "upload_ms": None, "kernel_ms": None, "download_ms": None}
    if events:
        for name in ("upload", "kernel", "download"):
            item[name + "_ms"] = statistics.median(finite(r[name + "_ms"]) for r in rows)
    return item


def group_native(rows, expected, field, key, method, label, samples):
    if len(rows) != expected:
        raise ValueError("incomplete native record")
    groups = defaultdict(list)
    for row in rows:
        groups[(key(row), method(row))].append(row)
    result = []
    for (condition, backend), group in groups.items():
        ordered = sorted(group, key=lambda r: int(r["iteration"]))
        if [int(r["iteration"]) for r in ordered] != list(range(samples + 1)):
            raise ValueError("missing or duplicate native invocation")
        events = "kernel_ms" in ordered[0] and backend.startswith("cuda")
        result.append(summary(condition, label(ordered[0]), backend, ordered[1:], field, ordered[0], events))
    conditions = list(dict.fromkeys(r["condition"] for r in result))
    return sorted(result, key=lambda r: (conditions.index(r["condition"]), METHOD_ORDER.index(r["method"])))


def inference_rows(record):
    if len(record["rows"]) != 240 or (record["samples"], record["warmups"]) != (20, 5):
        raise ValueError("incomplete inference comparison")
    result = []
    lifecycle = []
    for batch in (1, 2, 8):
        for mode in ("cpp-fresh", "cpp-reused", "python-cpu-reused"):
            group = [r for r in record["rows"] if r["batch"] == batch and r["mode"] == mode]
            samples = [r for r in group if r["phase"] == "sample"]
            first = [r for r in group if r["phase"] == "first_invocation"]
            warmups = [r for r in group if r["phase"] == "warmup"]
            if len(first) != 1 or len(warmups) != 5 or {r["sample"] for r in samples} != set(range(20)) or len(samples) != 20:
                raise ValueError("missing or duplicate inference invocation")
            converted = [dict(r, host_ms=finite(r["seconds"]) * 1000) for r in samples]
            converted_first = dict(first[0], host_ms=finite(first[0]["seconds"]) * 1000)
            result.append(summary(str(batch), f"{batch} image" + ("s" if batch != 1 else ""), mode, converted, "host_ms", converted_first))
        for phase in ("setup", "retirement"):
            rows = [r for r in record["rows"] if r["batch"] == batch and r["phase"] == phase]
            if len(rows) != 1 or rows[0]["mode"] != "cpp-reused":
                raise ValueError("missing session boundary")
            lifecycle.append({"condition_label": f"{batch} image" + ("s" if batch != 1 else ""),
                              "method_label": "Reused C++ owner/session", "phase": phase,
                              "milliseconds": finite(rows[0]["seconds"]) * 1000, "device_bytes": None})
    return result, lifecycle


def experiment(identity, title, description, finding, boundary, excluded, limitations, source, raw, protocol, report, rows, lifecycle=()):
    conditions = list(dict.fromkeys((r["condition"], r["condition_label"]) for r in rows))
    return {"id": identity, "title": title, "description": description, "finding": finding,
            "boundary": boundary, "excluded": excluded, "limitations": limitations,
            "source_revision": source, "source_url": f"{REPO}/tree/{source}",
            "raw_url": link(raw), "protocol_url": link(protocol, source), "report_url": link(report),
            "conditions": [{"id": key, "label": value} for key, value in conditions],
            "rows": rows, "lifecycle": list(lifecycle)}


def build_data(root=ROOT):
    pins = verify_inputs(root)
    execution = read_json(root, "docs/evidence/execution.json")
    cpu = read_csv(root, "docs/evidence/baseline-cpu.csv")
    followup = read_csv(root, "docs/evidence/followup-cpu.csv")
    gpu = read_csv(root, "docs/evidence/baseline-cuda.csv")
    reuse = read_csv(root, "docs/evidence/reuse.csv")
    reuse_receipt = read_json(root, "docs/evidence/reuse-execution.json")
    cuda_inference = read_json(root, "docs/inference-cuda-evidence.json")
    cpu_inference = read_json(root, "docs/inference-cpu-reuse.json")
    native_key = lambda r: r["workload"] + ":" + r["side"]
    native_label = lambda r: WORKLOADS[r["workload"]] + ", " + r["side"] + " by " + r["side"]
    cpu_rows = group_native(cpu, 162, "elapsed_ms", native_key,
                            lambda r: r["backend"] + "_" + r["threads"], native_label, 5)
    gpu_key = lambda r: r["workload"] + ":" + r["side"] + ":" + r["distribution"]
    gpu_label = lambda r: native_label(r) + (", " + ("uniform bytes" if r["distribution"] == "uniform" else "one occupied bin") if r["workload"] == "tile_histogram" else "")
    gpu_method = lambda r: ("cuda_global" if r["workload"] == "tile_histogram" else "cuda") if r["backend"] == "cuda_baseline" else r["backend"]
    gpu_rows = group_native(gpu, 180, "host_total_ms", gpu_key, gpu_method, gpu_label, 5)
    dispatch_rows = []
    for records, prefix in ((cpu, "original"), (followup, "followup")):
        selected = [r for r in records if r["workload"] == "stencil3x3" and r["backend"] == "optimized"]
        dispatch_rows.extend(group_native(selected, 36, "elapsed_ms", lambda r: r["side"],
                             lambda r: prefix + "_" + r["threads"], lambda r: r["side"] + " by " + r["side"], 5))
    if len(reuse) != 924:
        raise ValueError("incomplete reuse record")
    reuse_rows = group_native([r for r in reuse if r["phase"] == "run"], 900, "host_ms",
                             lambda r: r["side"] + ":" + r["distribution"], lambda r: r["backend"],
                             lambda r: r["side"] + " by " + r["side"] + ", " + ("uniform bytes" if r["distribution"] == "uniform" else "one occupied bin"), 24)
    lifecycle = [{"condition_label": r["side"] + " by " + r["side"] + ", " + r["distribution"],
                  "method_label": LABELS[r["backend"]], "phase": r["phase"], "milliseconds": finite(r["host_ms"]),
                  "device_bytes": int(r["device_bytes"])} for r in reuse if r["phase"] != "run"]
    cpu_inference_rows, cpu_lifecycle = inference_rows(cpu_inference)
    cuda_inference_rows, cuda_lifecycle = inference_rows(cuda_inference["comparison"])
    numerical_limits = "Five warm samples, fixed condition order and a shared Windows workstation. First invocation is retained, not a cold-process measurement."
    inference_boundary = "Prepared tensors to completed host logits. Fresh C++ calls include session creation, admission, reconciliation and close. Reused setup and retirement are separate."
    inference_excluded = "PNG decoding/normalization, reference validation and journal writes. The direct Python baseline lacks the wrapper's request identity and generation accounting."
    studies = [
        experiment("cpu", "CPU numerical paths", "Scalar references and one- or two-thread optimized calls across three workloads.",
                   "Extra threads and an optimized path can cost more than they save on small inputs. Inspect each workload and size.",
                   "Completed synchronous native call, including validation, allocation and output disposal.",
                   "Input generation, Python snapshots, runtime queueing and GPU transfers.", numerical_limits,
                   execution["measurement"]["revision"], "docs/evidence/baseline-cpu.csv", "docs/cpu-protocol.json", "docs/results-2026-09-29.md", cpu_rows),
        experiment("cuda-baseline", "Original CPU / CUDA comparison", "The same original complete-call boundary with separate CUDA event intervals.",
                   "CUDA complete-call medians were slower than optimized CPU in every tested condition. Shared bins helped uniform input but hurt concentrated input at the largest size.",
                   "Validation, allocations, checksum selection and result destruction. CUDA includes upload, launch, download, synchronization and resource cleanup.",
                   "Input generation, Python conversion and queueing. Device events are separate intervals and are never added to host elapsed time.", numerical_limits,
                   execution["measurement"]["revision"], "docs/evidence/baseline-cuda.csv", "docs/cuda-protocol.json", "docs/results-2026-09-29.md", gpu_rows),
        experiment("dispatch", "CPU stencil dispatch correction", "The original and follow-up collections keep the same stencil arithmetic and thread counts.",
                   "Moving AVX2 support detection out of the row loop lowered every listed stencil median in this follow-up. Thread startup still outweighed small work.",
                   "Same completed native CPU boundary in two separate collections. Original CPU results are not relabelled as a new GPU baseline.",
                   "Input generation, Python conversion and queueing.", numerical_limits + " Shared-host timing alone does not establish causality.",
                   execution["followup"]["revision"], "docs/evidence/followup-cpu.csv", "docs/cpu-followup-protocol.json", "docs/results-2026-09-29.md", dispatch_rows),
        experiment("reuse", "Histogram resource reuse", "Six paths retain both CPU thread counts and global/shared bin methods. Each histogram uses 64 by 64 tiles.",
                   "Small inputs favored CPU. At 1024 by 1024, reused shared bins for uniform input and reused global bins for concentrated input beat the faster tested CPU median.",
                   "Call entry to an independently owned, usable host result. Both CUDA paths upload current input and synchronize. Simple calls retire resources inside the call.",
                   "Setup, final retirement, output comparison, checksum and host result destruction. Reused resources survive between calls. This boundary differs from the original study.",
                   "24 warm samples per condition, rotated backend order and a shared Windows workstation. Two extra warmups are outside the retained rows. No pinned input, zero-copy or transfer overlap.",
                   reuse_receipt["source"], "docs/evidence/reuse.csv", "docs/reuse-protocol.json", "docs/reuse-results.md", reuse_rows, lifecycle),
        experiment("inference-cpu", "CPU inference session reuse", "Serial one-image calls through one ONNX session, at application batch sizes 1, 2 and 8.",
                   "Reusing the owner/session lowered repeated call cost. The direct Python CPU session was faster than the C++ wrapper at every tested batch size.",
                   inference_boundary, inference_excluded,
                   "20 samples after five warmups. First invocation is separate. One-thread CPU configuration on a shared Windows workstation. This is also a wrapper comparison, not a language ranking.",
                   cpu_inference["source_revision"], "docs/inference-cpu-reuse.json", "docs/inference-protocol.md", "docs/inference-results.md", cpu_inference_rows, cpu_lifecycle),
        experiment("inference-cuda", "Required CUDA inference", "Required CUDA owner/session paths compared with a direct CPU session in the same provider installation.",
                   "The direct Python CPU baseline was faster at all three tested batches. Reuse helped repeated CUDA calls, but setup and retirement still cost time.",
                   inference_boundary, inference_excluded,
                   "20 samples after five warmups on a shared Windows workstation. Application batches are serial one-image calls, not fused GPU batching. Operator events prove placement, not kernel timing or overlap.",
                   cuda_inference["comparison_source_revision"], "docs/inference-cuda-evidence.json", "docs/inference-cuda-protocol.md", "docs/inference-cuda-results.md", cuda_inference_rows, cuda_lifecycle),
    ]
    studies[2]["comparative_source_revision"] = execution["measurement"]["revision"]
    studies[2]["comparative_source_url"] = f'{REPO}/tree/{execution["measurement"]["revision"]}'
    studies[2]["secondary_raw_url"] = link("docs/evidence/baseline-cpu.csv")
    original_cpu = read_json(root, "docs/inference-cpu-evidence.json")
    outputs = {"predictions": cuda_inference["application"]["predictions"], "logits": cuda_inference["application"]["logits"],
               "reference_logits": cuda_inference["reference"]["expected_logits"], "rtol": cuda_inference["reference"]["rtol"],
               "atol": cuda_inference["reference"]["atol"], "max_absolute_difference": cuda_inference["reference"]["max_absolute_logit_difference"],
               "cuda_provider_events": cuda_inference["application"]["placement"]["provider_events"],
               "cpu_provider_events": original_cpu["application"]["placement"]["provider_events"]}
    memory = {"cpu": read_json(root, "docs/inference-cpu-memory.json")["snapshots"],
              "cuda": cuda_inference["corrected_memory"]["snapshots"],
              "cuda_source_revision": cuda_inference["corrected_memory_source_revision"]}
    return {"schema_version": 1, "source_revision": SOURCE, "theme_revision": pins["theme_revision"],
            "input_sha256": pins["files"], "experiments": studies, "inference_outputs": outputs, "memory": memory,
            "hardware": cuda_inference["hardware"], "default_selection": {"experiment": "reuse", "condition": "1024:uniform", "metric": "median_ms"}}


def number(value):
    return "Not measured" if value is None else f"{value:.5f}"


def table(headers, rows, caption):
    head = "".join(f'<th scope="col">{escape(text)}</th>' for text in headers)
    body_rows = []
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            tag = 'th scope="row"' if i == 0 else "td"
            end = "th" if i == 0 else "td"
            cells.append(f'<{tag}>{escape(str(value))}</{end}>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(body_rows)
    return f'<div class="table-wrap" role="region" tabindex="0" aria-label="{escape(caption)}"><table><caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def static_tables(data):
    blocks = []
    for study in data["experiments"]:
        rows = [[r["condition_label"], r["method_label"], r["samples"], *[number(r[f]) for f in ("median_ms", "p25_ms", "p75_ms", "min_ms", "max_ms", "first_ms", "upload_ms", "kernel_ms", "download_ms")]] for r in study["rows"]]
        content = table(["Condition", "Path", "Warm samples", "Median ms", "25% ms", "75% ms", "Minimum ms", "Maximum ms", "First ms", "Upload ms", "Kernel ms", "Download ms"], rows, study["title"] + ": all retained conditions")
        if study["lifecycle"]:
            content += table(["Condition", "Path", "Phase", "ms", "Explicit device bytes"], [[r["condition_label"], r["method_label"], r["phase"], number(r["milliseconds"]), r["device_bytes"] if r["device_bytes"] is not None else "Not a process/device total"] for r in study["lifecycle"]], study["title"] + ": separate setup and retirement observations")
        content += f'<p class="fine">Executed source <a href="{study["source_url"]}"><code>{study["source_revision"]}</code></a>. <a href="{study["raw_url"]}">Raw record</a>, <a href="{study["protocol_url"]}">protocol</a>, <a href="{study["report_url"]}">report</a>.</p>'
        if "comparative_source_revision" in study:
            content += f'<p class="fine">Original source <a href="{study["comparative_source_url"]}"><code>{study["comparative_source_revision"]}</code></a> and <a href="{study["secondary_raw_url"]}">original CPU rows</a> remain separate from the follow-up.</p>'
        blocks.append(f'<details id="table-{study["id"]}"><summary>{escape(study["title"])}</summary><p>{escape(study["finding"])}</p><p><strong>Boundary:</strong> {escape(study["boundary"])}</p><p><strong>Outside:</strong> {escape(study["excluded"])}</p><p class="muted">{escape(study["limitations"])}</p>{content}</details>')
    return "\n".join(blocks)


def output_tables(data):
    output = data["inference_outputs"]
    rows = [[i, predicted, "Matches label" if i == predicted else "Misclassified"] for i, predicted in enumerate(output["predictions"])]
    result = table(["Drawing label", "Predicted class", "Observed outcome"], rows, "Eight original geometric drawings, not an accuracy dataset")
    logit_rows = [[i, digit, f"{actual:.8g}", f"{expected:.8g}"] for i, (a, b) in enumerate(zip(output["logits"], output["reference_logits"], strict=True)) for digit, (actual, expected) in enumerate(zip(a, b, strict=True))]
    return result + '<details><summary>Inspect all 80 CUDA logits and their reference</summary>' + table(["Drawing label", "Class", "CUDA logit", "Reference logit"], logit_rows, "Complete CUDA outputs versus the independent ONNX reference") + "</details>"


def memory_tables(data):
    result = []
    for backend in ("cpu", "cuda"):
        rows = []
        for r in data["memory"][backend]:
            row = [r["phase"].replace("_", " "), r["completed_requests"], f'{r["working_set_bytes"]/2**20:.3f}', f'{r["private_usage_bytes"]/2**20:.3f}']
            if backend == "cuda":
                row.append(f'{r["device_free_bytes"]/2**20:.3f}')
            rows.append(row)
        headers = ["Phase", "Completed requests", "Process working set MiB", "Process private commit MiB"]
        if backend == "cuda":
            headers.append("Shared device free MiB")
        result.append(table(headers, rows, backend.upper() + ": separate fresh-process memory observation"))
    return "\n".join(result)


def outputs(root=ROOT):
    data = build_data(root)
    template = (root / "web/index.template.html").read_text(encoding="utf-8")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    safe = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = template.replace("@@TABLES@@", static_tables(data)).replace("@@OUTPUTS@@", output_tables(data)).replace("@@MEMORY@@", memory_tables(data)).replace("@@DATA@@", safe).replace("@@SOURCE@@", SOURCE)
    if "@@" in html:
        raise ValueError("unresolved site template marker")
    tokens = read_json(root, "web/theme-tokens.json")
    def rules(theme):
        return ";".join(f"--{name}:{value}" for name, value in tokens["themes"][theme].items())
    css = "/* Adapter for pinned Clair/Obscur tokens at " + data["theme_revision"] + ". */\n"
    css += ':root{color-scheme:light;' + rules("Clair") + '}\n'
    css += ':root[data-appearance=obscur]{color-scheme:dark;' + rules("Obscur") + '}\n'
    css += '@media(prefers-color-scheme:dark){:root:not([data-appearance=clair]):not([data-appearance=obscur]){color-scheme:dark;' + rules("Obscur") + '}}\n'
    css += '@media(forced-colors:active){:root,:root[data-appearance]{--canvas:Canvas;--panel:Canvas;--text:CanvasText;--muted:CanvasText;--divider:CanvasText;--control:ButtonFace;--hover:ButtonFace;--selected:Highlight;--border:ButtonText;--accent:LinkText;--on_accent:Canvas;--focus:Highlight;--code:Canvas;--success:CanvasText;--error:CanvasText}}\n'
    return {"web/index.html": html, "web/evidence.json": json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "web/appearance.css": css}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate only owned site outputs")
    args = parser.parse_args()
    generated = outputs()
    for path, text in generated.items():
        if args.write:
            (ROOT / path).write_text(text, encoding="utf-8", newline="\n")
        elif not (ROOT / path).exists() or (ROOT / path).read_bytes() != text.encode("utf-8"):
            raise ValueError("stale site output: " + path)
    print("Pinned retained records and all three generated site outputs match; no workload executed")


if __name__ == "__main__":
    main()
