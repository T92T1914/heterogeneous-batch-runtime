"""Retained-data and static fallback checks. No model, native or browser work."""
import copy
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_site", ROOT / "tools/build_site.py")
site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site)


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.anchors = []
        self.assets = []
        self.rows = 0
        self.row_headings = 0
        self.captions = 0
        self.embedded = ""
        self.in_data = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                raise ValueError("duplicate page identity")
            self.ids.add(attrs["id"])
        if tag == "a":
            self.anchors.append(attrs["href"])
        if "src" in attrs:
            self.assets.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.assets.append(attrs["href"])
        if tag == "tr":
            self.rows += 1
        if tag == "th" and attrs.get("scope") == "row":
            self.row_headings += 1
        if tag == "caption":
            self.captions += 1
        if tag == "script" and attrs.get("id") == "evidence-data":
            self.in_data = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, text):
        if self.in_data:
            self.embedded += text


class SiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = site.build_data()
        cls.generated = site.outputs()
        cls.studies = {s["id"]: s for s in cls.data["experiments"]}

    def test_all_retained_conditions_and_sample_counts(self):
        expected = {"cpu": (27, 5), "cuda-baseline": (30, 5), "dispatch": (12, 5),
                    "reuse": (36, 24), "inference-cpu": (9, 20), "inference-cuda": (9, 20)}
        self.assertEqual(set(self.studies), set(expected))
        for name, (count, n) in expected.items():
            rows = self.studies[name]["rows"]
            self.assertEqual(len(rows), count)
            self.assertEqual({r["samples"] for r in rows}, {n})
            self.assertEqual(len({(r["condition"], r["method"]) for r in rows}), count)

    def test_reuse_actual_reference_values_and_cpu_small_win(self):
        rows = self.studies["reuse"]["rows"]
        large = {r["method"]: r for r in rows if r["condition"] == "1024:uniform"}
        self.assertAlmostEqual(large["cpu_optimized_2"]["median_ms"], .31800, places=8)
        self.assertAlmostEqual(large["cuda_simple_shared"]["median_ms"], .65315, places=8)
        self.assertAlmostEqual(large["cuda_reuse_shared"]["median_ms"], .27150, places=8)
        self.assertAlmostEqual(large["cuda_reuse_shared"]["kernel_ms"], .036816, places=5)
        small = {r["method"]: r["median_ms"] for r in rows if r["condition"] == "256:uniform"}
        self.assertLess(small["cpu_optimized_1"], min(v for k, v in small.items() if k.startswith("cuda")))
        self.assertEqual(len(self.studies["reuse"]["lifecycle"]), 24)

    def test_original_complete_call_loses_to_cpu_in_every_condition(self):
        study = self.studies["cuda-baseline"]
        for condition in study["conditions"]:
            rows = [r for r in study["rows"] if r["condition"] == condition["id"]]
            cpu = next(r["median_ms"] for r in rows if r["method"] == "cpu_optimized_2")
            self.assertTrue(all(cpu < r["median_ms"] for r in rows if r["method"].startswith("cuda")))

    def test_inference_negative_values_and_unmeasured_device_intervals(self):
        expected = {1: (.39140, .08870), 2: (.84625, .12975), 8: (2.22550, .33425)}
        for count, (cuda, cpu) in expected.items():
            rows = {r["method"]: r for r in self.studies["inference-cuda"]["rows"] if r["condition"] == str(count)}
            self.assertAlmostEqual(rows["cpp-reused"]["median_ms"], cuda, places=7)
            self.assertAlmostEqual(rows["python-cpu-reused"]["median_ms"], cpu, places=7)
            self.assertTrue(all(r["kernel_ms"] is None for r in rows.values()))
            self.assertLess(cpu, cuda)
        for study in self.studies.values():
            for row in study["rows"]:
                if row["method"].startswith("cpu") or study["id"] in ("cpu", "dispatch"):
                    self.assertIsNone(row["kernel_ms"])

    def test_dispatch_keeps_both_sources_and_original_gpu_identity(self):
        study = self.studies["dispatch"]
        self.assertEqual(study["source_revision"], "b2cb6a8370646bb9a044ad677c8d4551bb8d1161")
        self.assertEqual(study["comparative_source_revision"], "675bd82ad9f2ed25cda41835835a1074ebc88ace")
        self.assertEqual(self.studies["cuda-baseline"]["source_revision"], study["comparative_source_revision"])
        for condition in study["conditions"]:
            values = {r["method"]: r["median_ms"] for r in study["rows"] if r["condition"] == condition["id"]}
            for threads in (1, 2):
                self.assertLess(values[f"followup_{threads}"], values[f"original_{threads}"])

    def test_inference_full_outputs_and_misclassifications_are_retained(self):
        outputs = self.data["inference_outputs"]
        self.assertEqual(outputs["predictions"], [2,3,2,3,4,5,5,2])
        self.assertEqual(outputs["cuda_provider_events"], {"CUDAExecutionProvider":64})
        self.assertEqual(outputs["cpu_provider_events"], {"CPUExecutionProvider":64})
        self.assertEqual((outputs["rtol"], outputs["atol"]), (1e-5, 2e-5))
        for actual, expected in zip(outputs["logits"], outputs["reference_logits"], strict=True):
            self.assertEqual(len(actual), 10)
            for a, b in zip(actual, expected, strict=True):
                self.assertLessEqual(abs(a-b), 2e-5 + 1e-5*abs(b))
        self.assertEqual(len(self.data["memory"]["cuda"]), 8)

    def test_generated_outputs_are_exact_and_do_not_reacquire(self):
        for name, text in self.generated.items():
            self.assertEqual((ROOT / name).read_bytes(), text.encode("utf-8"))
        page = Page()
        page.feed(self.generated["web/index.html"])
        self.assertEqual(json.loads(page.embedded), self.data)
        self.assertTrue(all("/" not in asset and "http" not in asset for asset in page.assets))
        for anchor in page.anchors:
            if anchor.startswith("#"):
                self.assertIn(anchor[1:], page.ids)
            elif anchor.startswith("https:"):
                self.assertTrue(anchor == "https://github.com/T92T1914" or anchor.startswith("https://github.com/T92T1914/"))
            else:
                self.assertTrue((ROOT / "web" / anchor).is_file())
        self.assertEqual(page.row_headings, 123 + 36 + 8 + 80 + 16)
        self.assertEqual(page.captions, 6 + 3 + 1 + 1 + 2)

    def test_retained_mutation_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = json.loads((ROOT / "web/source-pins.json").read_text())
            for path in ["web/source-pins.json", *pins["files"]]:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, root / path)
            path = root / "docs/evidence/reuse.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "retained site input changed: docs/evidence/reuse.csv"):
                site.build_data(root)

    def test_incomplete_or_duplicate_inference_samples_reject(self):
        original = json.loads((ROOT / "docs/inference-cpu-reuse.json").read_text())
        changed = copy.deepcopy(original)
        first = next(i for i, r in enumerate(changed["rows"]) if r["phase"] == "sample")
        second = next(i for i, r in enumerate(changed["rows"]) if r["phase"] == "sample" and i != first and r["mode"] == changed["rows"][first]["mode"])
        changed["rows"][second]["sample"] = changed["rows"][first]["sample"]
        with self.assertRaisesRegex(ValueError, "missing or duplicate inference invocation"):
            site.inference_rows(changed)

    def test_theme_roles_are_exact_pinned_adapter(self):
        tokens = json.loads((ROOT / "web/theme-tokens.json").read_text())
        self.assertEqual(hashlib.sha256((ROOT / "web/theme-tokens.json").read_bytes()).hexdigest(), "889df65e0f4f4eea99a81c535d542e7332b537c000c332402a3ffa6a6cbb15b7")
        css = self.generated["web/appearance.css"]
        for theme in tokens["themes"].values():
            for role, value in theme.items():
                self.assertIn(f"--{role}:{value}", css)


if __name__ == "__main__":
    unittest.main()
