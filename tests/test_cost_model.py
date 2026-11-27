import csv
import importlib.util
from pathlib import Path
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cost_model", ROOT / "tools" / "cost_model.py")
cost_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost_model)


class CostModelTest(unittest.TestCase):
    def test_fit_recovers_exact_terms(self):
        a, b = 0.004, 1e-7
        samples = [(copies, size, a * copies + b * size) for copies, size in [(1, 1024), (2, 9216), (1, 2 ** 20), (2, 9 * 2 ** 20)]]
        fitted = cost_model.fit(samples)
        self.assertAlmostEqual(fitted[0], a, places=12)
        self.assertAlmostEqual(fitted[1], b, places=18)

    def test_byte_counts_follow_the_copy_calls(self):
        self.assertEqual(cost_model.transfers("masked_reduce", 32), {"upload": (2, 9 * 1024), "download": (1, 8)})
        self.assertEqual(cost_model.transfers("stencil3x3", 256), {"upload": (1, 8 * 65536), "download": (1, 8 * 65536)})
        # 32 by 32 is one partial tile, 1024 by 1024 is 16 by 16 full tiles of 256 bins at 8 bytes.
        self.assertEqual(cost_model.transfers("tile_histogram", 32)["download"], (1, 2048))
        self.assertEqual(cost_model.transfers("tile_histogram", 1024)["download"], (1, 256 * 2048))

    def test_every_prediction_is_held_out(self):
        _, points = cost_model.load()
        folds = cost_model.evaluate(points)
        self.assertEqual(len(folds), 3 * 3 * 6)
        for name, view in cost_model.MODELS.items():
            for held in cost_model.SIDES:
                # Refit from the other two sizes only. The tool's fold terms and predictions must match exactly,
                # and moving a held-out median must not change them.
                a, b = cost_model.fit([view(p) for p in points if p["side"] != held])
                shifted = [dict(p, upload_ms=p["upload_ms"] * 9, download_ms=p["download_ms"] * 9,
                                host_total_ms=p["host_total_ms"] * 9) if p["side"] == held else p for p in points]
                self.assertEqual(cost_model.fit([view(p) for p in shifted if p["side"] != held]), (a, b))
                tested = [f for f in folds if f["model"] == name and f["held_out"] == held]
                self.assertEqual(len(tested), 6)
                for f, p in zip(tested, [p for p in points if p["side"] == held]):
                    x0, x1, y = view(p)
                    self.assertAlmostEqual(f["fixed_ms"], a, places=12)
                    self.assertAlmostEqual(f["ms_per_mib"], b * 2 ** 20, places=12)
                    self.assertAlmostEqual(f["predicted_ms"], a * x0 + b * x1, places=12)
                    self.assertEqual(f["measured_ms"], y)

    def test_committed_outputs_match(self):
        receipt, points = cost_model.load()
        report, table = cost_model.outputs(receipt, points)
        self.assertEqual((ROOT / "docs" / "cost-model.md").read_text(encoding="utf-8"), report)
        self.assertEqual((ROOT / "docs" / "evidence" / "cost-model.csv").read_text(encoding="utf-8"), table)

    def test_committed_forecast_matches(self):
        # The forecast is committed before the generation 4 run. If the file and the script disagree,
        # the prediction on record is not the one the script makes.
        receipt, points = cost_model.load()
        self.assertEqual(cost_model.FORECAST.read_text(encoding="utf-8"), cost_model.forecast_output(receipt, points))

    def test_forecast_fits_all_sizes_and_scales_only_the_term_per_byte(self):
        _, points = cost_model.load()
        terms, same = cost_model.forecast(points, 1.0)
        for entry in same:
            for field in cost_model.FORECAST_FIELDS.values():
                self.assertEqual(entry[field]["scaled"], entry[field]["unscaled"])
        _, halved = cost_model.forecast(points, 2.0)
        self.assertEqual(len(halved), 18)
        for name, view in cost_model.MODELS.items():
            a, b = cost_model.fit([view(p) for p in points])
            self.assertEqual(terms[name], (a, b))
            for entry, p in zip(halved, points):
                x0, x1, y = view(p)
                cell = entry[cost_model.FORECAST_FIELDS[name]]
                self.assertAlmostEqual(cell["unscaled"], a * x0 + b * x1, places=9)
                self.assertAlmostEqual(cell["scaled"], a * x0 + b / 2 * x1, places=9)
                self.assertAlmostEqual(cell["september_median"], y, places=9)

    def test_upload_rate_ratio_comes_from_the_largest_pageable_uploads(self):
        ratio, binary, runs = cost_model.upload_rate_ratio()
        rows = {}
        for label, name in cost_model.PINNED_RUNS:
            text = (ROOT / "docs" / "evidence" / f"{name}.csv").read_bytes().decode("utf-8-sig")
            rows[label] = [r for r in csv.DictReader(text.splitlines())
                           if r["phase"] == "run" and r["host_memory"] == "pageable" and r["side"] == "1024"
                           and r["workload"] in ("masked_reduce", "stencil3x3") and r["iteration"] != "0"]
            self.assertEqual(len(rows[label]), 240)
        total = {label: sum(statistics.median(float(r["upload_ms"]) for r in rows[label] if r["workload"] == w)
                            for w in ("masked_reduce", "stencil3x3")) for label in rows}
        self.assertAlmostEqual(ratio, total["generation 3"] / total["generation 4"], places=12)
        self.assertGreater(ratio, 1)
        self.assertEqual(binary, runs["generation 3"][0]["binary_sha256"])


if __name__ == "__main__":
    unittest.main()
