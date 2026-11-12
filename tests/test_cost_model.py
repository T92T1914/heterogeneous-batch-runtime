import importlib.util
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
