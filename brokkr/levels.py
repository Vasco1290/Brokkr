"""How many of the 256 levels does each quantized layer actually use? (task 4.0, the mechanism test)

An INT8 model stores every activation as one of 256 levels (8 bits). If a layer's range was stretched
to cover rare extreme values, ordinary values may use only a few of those levels. This module reads
the levels an image really uses in each quantized activation of a QDQ-format ONNX model.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def activation_quantize_outputs(model: onnx.ModelProto) -> list:
    """Output names of the QuantizeLinear nodes that quantize activations (not stored weights)."""
    stored = {i.name for i in model.graph.initializer}
    return [n.output[0] for n in model.graph.node
            if n.op_type == "QuantizeLinear" and n.input[0] not in stored]


def expose(model_path, out_path, names: list) -> Path:
    """Save a copy of the model that also outputs the given tensors (after its normal outputs)."""
    model = onnx.load(str(model_path))
    existing = {o.name for o in model.graph.output}
    for name in names:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.UINT8, None))
    onnx.save(model, str(out_path))
    return Path(out_path)


def distinct_levels(values: np.ndarray) -> np.ndarray:
    """Distinct 8-bit levels used per image. values: (images, ...) uint8 or int8 -> (images,) ints."""
    n = values.shape[0]
    flat = values.reshape(n, -1).astype(np.int64)
    if values.dtype == np.int8:
        flat += 128  # map -128..127 to 0..255
    offsets = (np.arange(n) * 256)[:, None]
    counts = np.bincount((flat + offsets).ravel(), minlength=n * 256).reshape(n, 256)
    return (counts > 0).sum(axis=1)
