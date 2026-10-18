"""Inert Python preparation and input-admission contract checks."""
import importlib.util
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest


class _UnusedDependency(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError(f"unused optional dependency was used: {self.__name__}.{name}")


class _ConstructionForbidden:
    def __new__(cls, *args, **kwargs):
        raise AssertionError("session/executor construction is outside the inert check")


def _module(name):
    value = _UnusedDependency(name)
    value.__path__ = []
    return value


@pytest.fixture
def boundaries(monkeypatch):
    """Load the actual two source files with real NumPy and inert unused imports."""
    root = Path(os.environ.get("RUNTIME_SOURCE_ROOT", str(Path(__file__).resolve().parents[3]))).resolve(strict=True)
    package_dir = root / "examples/image_classifier/python/hbr_image_classifier"
    package_name = "_hbr_admission_inert"
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_dir)]
    pil = _module("PIL")
    pil.Image = _module("PIL.Image")
    pil.ImageDraw = _module("PIL.ImageDraw")
    adaptive = _module("adaptive_timing")
    execution = _module("adaptive_timing.execution")
    execution.OwnerExecutor = _ConstructionForbidden
    session = _module(package_name + ".session")
    session.Session = _ConstructionForbidden
    replacements = {
        package_name: package,
        "PIL": pil,
        "PIL.Image": pil.Image,
        "PIL.ImageDraw": pil.ImageDraw,
        "onnx": _module("onnx"),
        "adaptive_timing": adaptive,
        "adaptive_timing.execution": execution,
        package_name + ".session": session,
    }
    with monkeypatch.context() as scoped:
        for name, value in replacements.items():
            scoped.setitem(sys.modules, name, value)
        loaded = []
        for leaf in ("model", "executor"):
            name = package_name + "." + leaf
            spec = importlib.util.spec_from_file_location(name, package_dir / (leaf + ".py"))
            module = importlib.util.module_from_spec(spec)
            scoped.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
            loaded.append(module)
        yield tuple(loaded)


class _FakeOwner:
    def __init__(self):
        self._accepted = 0
        self.owner_checks = 0
        self.submissions = []

    def _assert_owner(self):
        self.owner_checks += 1

    def submit_owned(self, dispatch, generation, callback):
        self.submissions.append((dispatch, generation, callback))


def _invoke(boundaries, boundary, value, owner=None):
    model, executor = boundaries
    if boundary == "prepare":
        return model.prepare_arrays(value), None
    owner = owner if owner is not None else _FakeOwner()
    executor.InferenceExecutor.submit_batch(owner, object(), 7, value)
    return None, owner


def _ordinary(boundary, count=1, layout="contiguous"):
    shape = (count, 28, 28) if boundary == "prepare" else (count, 1, 28, 28)
    dtype = np.uint8 if boundary == "prepare" else np.float32
    if layout == "unaligned":
        storage = np.empty(int(np.prod(shape)) * np.dtype(dtype).itemsize + 1, np.uint8)
        value = np.ndarray(shape, dtype=dtype, buffer=storage, offset=1)
        value.fill(1)
        if boundary == "submit" and count:
            assert not value.flags.aligned
        return value
    value = np.ones(shape, dtype=dtype)
    return value[..., ::-1] if layout == "strided" else value


def _subclass(value, **lies):
    attributes = {}
    for name, answer in lies.items():
        if name == "length":
            attributes["__len__"] = lambda self, answer=answer: answer
        else:
            attributes[name] = property(lambda self, answer=answer: answer)
    return value.view(type("MisleadingArray", (np.ndarray,), attributes))


def _snapshot_spy(monkeypatch, *, refuse=False):
    calls = []
    original = np.array

    def snapshot(*args, **kwargs):
        if refuse:
            raise AssertionError("owned snapshot reached before input refusal")
        owned = original(*args, **kwargs)
        calls.append(owned)
        return owned

    monkeypatch.setattr(np, "array", snapshot)
    return calls


    # The captured callback is never invoked.


@pytest.mark.parametrize("boundary", ["prepare", "submit"])
@pytest.mark.parametrize("case", ["len", "shape", "ndim_and_shape", "dtype"])
def test_rejects_misleading_actual_metadata_before_snapshot(boundaries, monkeypatch, boundary, case):
    valid_shape = (1, 28, 28) if boundary == "prepare" else (1, 1, 28, 28)
    dtype = np.uint8 if boundary == "prepare" else np.float32
    expected = ValueError
    if case == "len":
        value = _subclass(_ordinary(boundary, 9), length=1)
    elif case == "shape":
        bad_shape = (1, 27, 28) if boundary == "prepare" else (1, 1, 27, 28)
        value = _subclass(np.ones(bad_shape, dtype), shape=valid_shape)
    elif case == "ndim_and_shape":
        value = _subclass(np.ones((1, 28), dtype), ndim=len(valid_shape), shape=valid_shape)
    else:
        value = _subclass(np.ones(valid_shape, np.float64), dtype=np.dtype(dtype))
        expected = TypeError
    owner = _FakeOwner()
    with monkeypatch.context() as scoped:
        _snapshot_spy(scoped, refuse=True)
        with pytest.raises(expected):
            _invoke(boundaries, boundary, value, owner)
    assert owner._accepted == 0 and not owner.submissions


@pytest.mark.parametrize("boundary", ["prepare", "submit"])
@pytest.mark.parametrize("case", ["len", "shape", "ndim", "dtype"])
def test_valid_actual_metadata_wins_over_subclass_properties(boundaries, monkeypatch, boundary, case):
    ordinary = _ordinary(boundary)
    lies = {
        "len": {"length": 9},
        "shape": {"shape": (1, 99)},
        "ndim": {"ndim": 2},
        "dtype": {"dtype": np.dtype("float64")},
    }
    value = _subclass(ordinary, **lies[case])
    with monkeypatch.context() as scoped:
        copies = _snapshot_spy(scoped)
        result, owner = _invoke(boundaries, boundary, value)
    assert len(copies) == 1
    if boundary == "prepare":
        assert result.shape == (1, 1, 28, 28) and result.dtype == np.float32
    else:
        assert owner._accepted == 1 and len(owner.submissions) == 1


@pytest.mark.parametrize("boundary", ["prepare", "submit"])
@pytest.mark.parametrize("count", [0, 1, 8])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "unaligned"])
def test_zero_one_eight_layout_and_independent_snapshot(boundaries, monkeypatch, boundary, count, layout):
    value = _ordinary(boundary, count, layout)
    with monkeypatch.context() as scoped:
        copies = _snapshot_spy(scoped)
        result, owner = _invoke(boundaries, boundary, value)
    assert len(copies) == 1
    owned = copies[0]
    assert type(owned) is np.ndarray and owned.flags.c_contiguous
    assert not np.shares_memory(value, owned)
    before = owned.copy()
    value.fill(0)
    np.testing.assert_array_equal(owned, before)
    if boundary == "prepare":
        assert result.shape == (count, 1, 28, 28)
        assert result.dtype == np.float32 and result.flags.c_contiguous
        assert not np.shares_memory(result, value)
        np.testing.assert_array_equal(result, np.full(result.shape, np.float32(1) / np.float32(255), np.float32))
    else:
        assert not owned.flags.writeable
        assert owner._accepted == 1 and len(owner.submissions) == 1
        callback = owner.submissions[0][2]
        captured = dict(zip(callback.__code__.co_freevars, (cell.cell_contents for cell in callback.__closure__)))
        assert captured["owned"] is owned


@pytest.mark.parametrize("boundary", ["prepare", "submit"])
def test_non_ndarray_refused_before_snapshot(boundaries, monkeypatch, boundary):
    with monkeypatch.context() as scoped:
        _snapshot_spy(scoped, refuse=True)
        with pytest.raises(TypeError):
            _invoke(boundaries, boundary, [])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_submit_normalization_refused_before_snapshot(boundaries, monkeypatch, bad):
    value = _ordinary("submit")
    value.fill(bad)
    owner = _FakeOwner()
    with monkeypatch.context() as scoped:
        _snapshot_spy(scoped, refuse=True)
        with pytest.raises(ValueError):
            _invoke(boundaries, "submit", value, owner)
    assert owner._accepted == 0 and not owner.submissions


def test_submit_request_limit_precedes_snapshot(boundaries, monkeypatch):
    value = _ordinary("submit")
    owner = _FakeOwner()
    owner._accepted = 256
    with monkeypatch.context() as scoped:
        _snapshot_spy(scoped, refuse=True)
        with pytest.raises(RuntimeError, match="request limit"):
            _invoke(boundaries, "submit", value, owner)
    assert owner._accepted == 256 and not owner.submissions


def test_submit_owner_check_precedes_snapshot(boundaries, monkeypatch):
    value = _ordinary("submit")
    owner = _FakeOwner()

    def fail_owner():
        raise RuntimeError("owner tripwire")

    owner._assert_owner = fail_owner
    with monkeypatch.context() as scoped:
        _snapshot_spy(scoped, refuse=True)
        with pytest.raises(RuntimeError, match="owner tripwire"):
            _invoke(boundaries, "submit", value, owner)
    assert owner._accepted == 0 and not owner.submissions
