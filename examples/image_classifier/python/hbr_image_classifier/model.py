"""Pinned public artifact and bounded image preparation."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import warnings
import numpy as np
from PIL import Image, ImageDraw
import onnx

MODEL_SHA256 = "2f06e72de813a8635c9bc0397ac447a601bdbfa7df4bebc278723b958831c9bf"
MODEL_URL = "https://huggingface.co/onnxmodelzoo/mnist-8/resolve/a19f9a8c2333de1df9b03f10f5739f468b699a1a/mnist-8.onnx"
MAX_BATCH = 8
MAX_FILE_BYTES = 1048576

@dataclass(frozen=True)
class Model:
    data: bytes
    sha256: str

def load_model(path):
    path = Path(path)
    with path.open("rb") as source:
        data = source.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("model exceeds the byte limit")
    digest = sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("model does not match the reviewed artifact")
    model = onnx.load_model_from_string(data)
    if model.ir_version != 3 or [(o.domain, o.version) for o in model.opset_import] != [("", 8)]:
        raise ValueError("unexpected model IR or opset")
    if model.functions or any(n.domain for n in model.graph.node):
        raise ValueError("custom operators are not admitted")
    if any(x.data_location == onnx.TensorProto.EXTERNAL or x.external_data for x in model.graph.initializer):
        raise ValueError("external tensor files are not admitted")
    onnx.checker.check_model(model, full_check=True)
    return Model(data, digest)

def prepare_arrays(images):
    """Caller excludes external native writers during the owned snapshot copy."""
    if not isinstance(images, np.ndarray) or images.dtype != np.dtype("uint8"):
        raise TypeError("exact uint8 NumPy input required")
    if images.ndim != 3 or images.shape[1:] != (28, 28) or len(images) > MAX_BATCH:
        raise ValueError("images must have shape (N,28,28), N <= 8")
    owned = np.array(images, copy=True, order="C")
    return np.ascontiguousarray(owned[:, None], dtype=np.float32) / np.float32(255)

def read_images(paths):
    paths = tuple(paths)
    if len(paths) > MAX_BATCH:
        raise ValueError("at most eight images per batch")
    images = []
    for path in paths:
        with Path(path).open("rb") as source:
            data = source.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("image exceeds the byte limit")
        from io import BytesIO
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("only single-frame PNG images are admitted")
                if min(image.size) < 1 or max(image.size) > 256:
                    raise ValueError("image dimensions must be in [1,256]")
                # Alpha is composited over black before grayscale conversion.
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", image.size, (0, 0, 0, 255))
                gray = Image.alpha_composite(background, rgba).convert("L")
                images.append(np.array(gray.resize((28,28), Image.Resampling.BILINEAR), dtype=np.uint8))
    return prepare_arrays(np.stack(images) if images else np.empty((0,28,28), dtype=np.uint8))

def example_images():
    """Original geometric digits, not a held-out accuracy dataset."""
    segments = (((8,4),(19,4)), ((20,5),(20,12)), ((20,15),(20,23)),
                ((8,24),(19,24)), ((7,15),(7,23)), ((7,5),(7,12)), ((8,14),(19,14)))
    patterns = ("012345", "12", "01346", "01236", "1256", "02356", "023456", "012", "0123456", "012356")
    output = []
    for pattern in patterns:
        image = Image.new("L", (28,28), 0)
        draw = ImageDraw.Draw(image)
        for n in pattern:
            draw.line(segments[int(n)], fill=255, width=3)
        output.append(np.array(image))
    return np.stack(output)

def probabilities(logits):
    if logits.ndim != 2 or logits.shape[1] != 10 or not np.isfinite(logits).all():
        raise ValueError("finite (N,10) logits required")
    values = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)
