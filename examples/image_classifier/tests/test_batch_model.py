from hashlib import sha256
import os

import onnx
import pytest

from hbr_image_classifier.batching import BATCH_RECIPE, execution_model_data
from hbr_image_classifier.model import Model, MODEL_SHA256, load_model


@pytest.fixture(scope="module")
def model():
    path = os.environ.get("HBR_MNIST_MODEL")
    if not path:
        pytest.fail("HBR_MNIST_MODEL must name the separately acquired reviewed model")
    return load_model(path)


def test_serial_preserves_original_admission_and_bytes(model):
    assert execution_model_data(model, "serial") is model.data
    assert sha256(model.data).hexdigest() == model.sha256 == MODEL_SHA256


def test_batch_recipe_changes_only_flatten_shape_and_batch_annotations(model):
    source = onnx.load_model_from_string(model.data)
    data = execution_model_data(model, "batched")
    candidate = onnx.load_model_from_string(data)
    assert BATCH_RECIPE == "mnist-opset8-batch-preserving-reshape-v1"
    assert sha256(data).hexdigest() == "cfba8152caa873db9563fef0daa1bebe97d2f3dfc3ec23cf72cc5db31a4ee539"
    assert execution_model_data(model, "batched") == data
    assert sha256(model.data).hexdigest() == MODEL_SHA256
    assert source.ir_version == candidate.ir_version == 3
    assert [(x.domain, x.version) for x in candidate.opset_import] == [("", 8)]
    assert [x.SerializeToString() for x in source.graph.node] == [x.SerializeToString() for x in candidate.graph.node]

    before = {x.name: x for x in source.graph.initializer}
    after = {x.name: x for x in candidate.graph.initializer}
    assert list(before) == list(after)
    changed = [name for name in before if before[name].SerializeToString() != after[name].SerializeToString()]
    assert changed == ["Pooling160_Output_0_reshape0_shape"]
    assert list(after[changed[0]].int64_data) == [0, 256]
    assert list(after["Parameter193_reshape1_shape"].int64_data) == [256, 10]
    assert all(before[name].SerializeToString() == after[name].SerializeToString()
               for name in before if before[name].data_type == onnx.TensorProto.FLOAT)

    original_values = {x.name: x for x in [*source.graph.input, *source.graph.output, *source.graph.value_info]}
    candidate_values = {x.name: x for x in [*candidate.graph.input, *candidate.graph.output, *candidate.graph.value_info]}
    changed_values = {name for name in original_values
                      if original_values[name].SerializeToString() != candidate_values[name].SerializeToString()}
    assert changed_values == {
        "Input3", "Convolution28_Output_0", "Plus30_Output_0", "ReLU32_Output_0",
        "Pooling66_Output_0", "Convolution110_Output_0", "Plus112_Output_0",
        "ReLU114_Output_0", "Pooling160_Output_0", "Pooling160_Output_0_reshape0",
        "Times212_Output_0", "Plus214_Output_0",
    }
    for name in changed_values:
        dimension = candidate_values[name].type.tensor_type.shape.dim[0]
        assert dimension.dim_param == "N" and not dimension.HasField("dim_value")
        dimension.CopyFrom(original_values[name].type.tensor_type.shape.dim[0])
    after[changed[0]].CopyFrom(before[changed[0]])
    assert candidate.SerializeToString(deterministic=True) == source.SerializeToString(deterministic=True)


def test_batch_graph_checker_and_inferred_shapes(model):
    candidate = onnx.load_model_from_string(execution_model_data(model, "batched"))
    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    assert len(inferred.graph.output) == 1
    output = inferred.graph.output[0]
    assert output.name == "Plus214_Output_0"
    assert output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert output.type.tensor_type.shape.dim[0].dim_param == "N"
    assert output.type.tensor_type.shape.dim[1].dim_value == 10
    values = {x.name: x for x in inferred.graph.value_info}
    assert values["Pooling160_Output_0_reshape0"].type.tensor_type.shape.dim[0].dim_param == "N"
    assert values["Pooling160_Output_0_reshape0"].type.tensor_type.shape.dim[1].dim_value == 256


@pytest.mark.parametrize("execution_mode", [None, True, [], "auto", "batch", "serial "])
def test_execution_mode_is_explicit_before_graph_parsing(model, monkeypatch, execution_mode):
    def unexpected_parse(*args, **kwargs):
        pytest.fail("an unsupported execution mode must not parse or construct a graph")
    monkeypatch.setattr(onnx, "load_model_from_string", unexpected_parse)
    with pytest.raises(ValueError, match="serial or batched"):
        execution_model_data(model, execution_mode)


@pytest.mark.parametrize("execution_mode", ["serial", "batched"])
def test_batch_derivation_refuses_unreviewed_sources(model, execution_mode):
    with pytest.raises(TypeError, match="reviewed Model"):
        execution_model_data(model.data, execution_mode)
    with pytest.raises(TypeError, match="immutable"):
        execution_model_data(Model(bytearray(model.data), MODEL_SHA256), execution_mode)
    with pytest.raises(ValueError, match="reviewed artifact"):
        execution_model_data(Model(model.data, "0" * 64), execution_mode)
    modified = model.data[:-1] + bytes([model.data[-1] ^ 1])
    for digest in (MODEL_SHA256, sha256(modified).hexdigest()):
        with pytest.raises(ValueError, match="reviewed artifact"):
            execution_model_data(Model(modified, digest), execution_mode)
