"""Derive one reviewed batch graph locally without redistributing model bytes."""
from hashlib import sha256

import onnx

from .model import Model, MODEL_SHA256


BATCH_RECIPE = "mnist-opset8-batch-preserving-reshape-v1"
_BATCH_MODEL_SHA256 = "cfba8152caa873db9563fef0daa1bebe97d2f3dfc3ec23cf72cc5db31a4ee539"
_SHAPE_INITIALIZER = "Pooling160_Output_0_reshape0_shape"
_BATCH_VALUES = {
    "Input3": (1, 1, 28, 28),
    "Convolution28_Output_0": (1, 8, 28, 28),
    "Plus30_Output_0": (1, 8, 28, 28),
    "ReLU32_Output_0": (1, 8, 28, 28),
    "Pooling66_Output_0": (1, 8, 14, 14),
    "Convolution110_Output_0": (1, 16, 14, 14),
    "Plus112_Output_0": (1, 16, 14, 14),
    "ReLU114_Output_0": (1, 16, 14, 14),
    "Pooling160_Output_0": (1, 16, 4, 4),
    "Pooling160_Output_0_reshape0": (1, 256),
    "Times212_Output_0": (1, 10),
    "Plus214_Output_0": (1, 10),
}


def execution_model_data(model, execution_mode):
    """Return original serial bytes or the exact reviewed in-memory batch graph."""
    if not isinstance(execution_mode, str) or execution_mode not in ("serial", "batched"):
        raise ValueError("execution_mode must be serial or batched")
    if not isinstance(model, Model):
        raise TypeError("a reviewed Model is required")
    if not isinstance(model.data, bytes):
        raise TypeError("immutable reviewed model bytes required")
    if len(model.data) != 26454 or model.sha256 != MODEL_SHA256 or sha256(model.data).hexdigest() != MODEL_SHA256:
        raise ValueError("model does not match the reviewed artifact")
    if execution_mode == "serial":
        return model.data

    original = onnx.load_model_from_string(model.data)
    if original.ir_version != 3 or [(x.domain, x.version) for x in original.opset_import] != [("", 8)]:
        raise ValueError("unexpected model IR or opset for batch recipe")
    if original.functions or any(x.domain for x in original.graph.node):
        raise ValueError("custom operators are not admitted")
    if any(x.data_location == onnx.TensorProto.EXTERNAL or x.external_data for x in original.graph.initializer):
        raise ValueError("external tensor files are not admitted")
    candidate = onnx.ModelProto()
    candidate.CopyFrom(original)
    initializers = {x.name: x for x in candidate.graph.initializer}
    tensor = initializers.get(_SHAPE_INITIALIZER)
    if (tensor is None or tensor.data_type != onnx.TensorProto.INT64
            or list(tensor.dims) != [2] or tensor.raw_data
            or list(tensor.int64_data) != [1, 256]):
        raise ValueError("reviewed flattening shape required for batch recipe")
    flatten = next((x for x in candidate.graph.node if x.name == "Times212_reshape0"), None)
    if (flatten is None or flatten.domain or flatten.op_type != "Reshape"
            or list(flatten.input) != ["Pooling160_Output_0", _SHAPE_INITIALIZER]
            or list(flatten.output) != ["Pooling160_Output_0_reshape0"] or flatten.attribute):
        raise ValueError("reviewed flattening node required for batch recipe")
    values = {x.name: x for x in [*candidate.graph.input, *candidate.graph.output, *candidate.graph.value_info]}
    for name, expected in _BATCH_VALUES.items():
        value = values.get(name)
        if (value is None or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                or tuple(d.dim_value for d in value.type.tensor_type.shape.dim) != expected
                or any(not d.HasField("dim_value") for d in value.type.tensor_type.shape.dim)):
            raise ValueError(f"reviewed value shape required for batch recipe: {name}")
        dimension = value.type.tensor_type.shape.dim[0]
        dimension.ClearField("dim_value")
        dimension.dim_param = "N"
    # Reshape-5, used by opset 8, copies the incoming batch dimension for zero.
    # The weight reshape and every trained weight/operator remain unchanged.
    tensor.int64_data[:] = [0, 256]
    onnx.checker.check_model(candidate, full_check=True)
    data = candidate.SerializeToString(deterministic=True)
    if sha256(data).hexdigest() != _BATCH_MODEL_SHA256:
        raise ValueError("derived graph does not match the reviewed batch recipe")
    return data
