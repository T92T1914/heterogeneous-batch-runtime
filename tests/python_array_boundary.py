"""Exercise real array marshalling and binding code with an explicit host CUDA stand-in."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arrays = load("_array_boundary_test", sys.argv.pop(1))
native = load("_host_cuda_binding_test", sys.argv.pop(1))


class ArrayBoundary(unittest.TestCase):
    def test_matching_empty_and_independent_outputs(self):
        for shape, values in [([2, 2], [1, 2, 3, 4]), ([0, 2], []), ([], [7])]:
            first = arrays.copy(values, shape)
            second = arrays.copy(values, shape)
            self.assertEqual(first.shape, tuple(shape))
            self.assertEqual(first.dtype, np.uint64)
            np.testing.assert_array_equal(first.ravel(), values)
            self.assertTrue(first.flags.owndata)
            self.assertFalse(np.shares_memory(first, second))

    def test_mismatched_lengths_and_invalid_products(self):
        for values, shape in [([1, 2], [1]), ([1], [2]), ([1], [0]), ([], [1]), ([], [-1]),
                              ([], [sys.maxsize, 2]), ([], [sys.maxsize]),
                              ([], [0, sys.maxsize//8 + 1]), ([], [0, 2, sys.maxsize//8])]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                arrays.copy(values, shape)

    def test_private_geometry_rejected_before_native_run(self):
        self.assertIn("HOST STAND-IN", native._cuda_select_device(0))
        context = native._CudaHistogramContext(2, 4, 1, 2)
        image = np.arange(8, dtype=np.uint8).reshape(2, 4)
        first = context.run(image, 2, 4, 1, 2)
        self.assertEqual(first["request_id"], 1)
        self.assertEqual(first["value"].shape, (2, 2, 256))
        saved = first["value"].copy()
        for rows, cols, tr, tc in [(2,4,2,4), (2,4,1,1), (2,4,2,1), (1,8,1,2), (2,4,0,2)]:
            with self.subTest(config=(rows,cols,tr,tc)), self.assertRaisesRegex(ValueError, "fixed image and tile"):
                context.run(image.reshape(rows, cols), rows, cols, tr, tc)
        for bad in [image.astype(np.int16), image[:, ::-1]]:
            with self.assertRaises((TypeError, ValueError)):
                context.run(bad, 2, 4, 1, 2)
        second = context.run(image + 1, 2, 4, 1, 2)
        self.assertEqual(second["request_id"], 2)
        self.assertFalse(np.shares_memory(first["value"], second["value"]))
        np.testing.assert_array_equal(first["value"], saved)
        self.assertEqual(int(first["value"].sum()), image.size)
        context.close()

    def test_private_empty_output(self):
        context = native._CudaHistogramContext(0, 3, 1, 2)
        result = context.run(np.empty((0, 3), np.uint8), 0, 3, 1, 2)
        self.assertEqual(result["value"].shape, (0, 2, 256))
        context.close()


if __name__ == "__main__":
    unittest.main()
