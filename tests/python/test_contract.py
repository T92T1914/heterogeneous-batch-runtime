import concurrent.futures
import math

import numpy as np
import pytest

import heterogeneous_batch_runtime as hbr


@pytest.mark.parametrize("n", [0, 1, 3, 7, 32771, 65537])
def test_reduce_against_fsum(n):
    rng = np.random.default_rng(71)
    values = rng.uniform(-100, 100, n)
    mask = rng.integers(0, 3, n, dtype=np.uint8)
    oracle = math.fsum(values[mask != 0])
    tolerance = 8 * np.finfo(float).eps * max(1, n) * np.abs(values[mask != 0]).sum()
    for backend in ("scalar", "optimized"):
        for threads in (1, 2, 4):
            assert abs(hbr.masked_reduce(values, mask, backend, threads) - oracle) <= max(1e-12, tolerance)


@pytest.mark.parametrize("shape", [(0, 0), (0, 7), (1, 9), (2, 2), (3, 3), (17, 19)])
def test_images_against_independent_oracles(shape):
    rng = np.random.default_rng(81)
    image = rng.uniform(-10, 10, shape)
    before = image.copy()
    expected = image.copy()
    for r in range(1, shape[0] - 1):
        for c in range(1, shape[1] - 1):
            expected[r, c] = math.fsum((image[r-1:r+2, c-1:c+2] / 9).flat)
    for backend in ("scalar", "optimized"):
        actual = hbr.stencil3x3(image, backend)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(image, before)
        assert not np.shares_memory(actual, image)
    values = rng.integers(0, 256, shape, dtype=np.uint8)
    for tile in ((1, 1), (4, 5), (100, 100)):
        expected_hist = np.zeros(((shape[0]+tile[0]-1)//tile[0], (shape[1]+tile[1]-1)//tile[1], 256), dtype=np.uint64)
        for r in range(expected_hist.shape[0]):
            for c in range(expected_hist.shape[1]):
                region = values[r*tile[0]:(r+1)*tile[0], c*tile[1]:(c+1)*tile[1]]
                expected_hist[r, c] = np.bincount(region.ravel(), minlength=256)
        for backend in ("scalar", "optimized"):
            np.testing.assert_array_equal(hbr.tile_histogram(values, *tile, backend), expected_hist)


def test_invalid_inputs():
    with pytest.raises(TypeError):
        hbr.masked_reduce([1.0], np.array([1], dtype=np.uint8))
    with pytest.raises(TypeError):
        hbr.stencil3x3(np.ones((3, 3), dtype=">f8"))
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            hbr.masked_reduce(np.array([bad]), np.array([0], dtype=np.uint8))
    with pytest.raises(TypeError):
        hbr.masked_reduce(np.array([1], dtype=np.float32), np.array([1], dtype=np.uint8))
    with pytest.raises(ValueError):
        hbr.stencil3x3(np.zeros((4, 4))[:, ::2])
    with pytest.raises(ValueError):
        hbr.tile_histogram(np.zeros((1, 1), dtype=np.uint8), 0, 1)
    with pytest.raises(ValueError):
        hbr.masked_reduce(np.ones(2), np.ones(1, dtype=np.uint8))
    with pytest.raises(ValueError):
        hbr.stencil3x3(np.zeros((1, 1)), "unknown")
    with pytest.raises(ValueError):
        hbr.stencil3x3(np.zeros((1, 1)), threads=0)


def test_overflow_guard_does_not_round_away_small_terms():
    maximum = np.finfo(float).max
    values = np.array([math.nextafter(maximum, 0)] + [math.ldexp(1, 969)] * 12)
    for backend in ("scalar", "optimized"):
        with pytest.raises(OverflowError):
            hbr.masked_reduce(values, np.ones(len(values), dtype=np.uint8), backend)
        assert hbr.masked_reduce(np.array([maximum]), np.ones(1, dtype=np.uint8), backend) == maximum
        assert np.isfinite(hbr.stencil3x3(np.full((5, 5), maximum), backend)).all()


def test_concurrent_calls_and_lifetimes():
    def run(seed):
        values = np.random.default_rng(seed).uniform(-10, 10, 40001)
        mask = np.ones(len(values), dtype=np.uint8)
        return hbr.masked_reduce(values, mask, threads=1), math.fsum(values)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for actual, expected in pool.map(run, range(8)):
            assert actual == pytest.approx(expected, abs=1e-9)
