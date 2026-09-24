"""Make smaller versions of an FP32 ONNX model: FP16 and INT8.

FP16 ("half precision"): every weight is stored in 16 bits instead of 32. Half the size,
    tiny rounding changes. Speed depends on whether the CPU can compute in FP16 directly.
INT8: weights and activations become 8-bit integers. About 4x smaller than FP32. To pick
    good integer ranges, the model is first run on a few real "calibration" images to see
    what values flow through it. Those images must not be the ones we later test on.
"""

from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quant_pre_process, quantize_static)

# Settings are kept in one place so they can be saved alongside the results.
INT8_SETTINGS = {
    "method": "static post-training quantization (onnxruntime)",
    "format": "QDQ",
    "weights": "int8, per-channel",
    "activations": "uint8, per-tensor",
    "calibration_method": "MinMax",
}


def to_fp16(fp32_path, out_path) -> Path:
    """Convert weights to FP16. Inputs and outputs stay FP32 so callers don't need to change."""
    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(out_path))
    return Path(out_path)


class ImageBatches(CalibrationDataReader):
    """Feeds preprocessed calibration images to the quantizer, one batch at a time."""

    def __init__(self, batches, input_name: str = "images"):
        self._batches = iter(batches)
        self._input_name = input_name

    def get_next(self):
        batch = next(self._batches, None)
        return None if batch is None else {self._input_name: batch.astype(np.float32)}


def to_int8(fp32_path, out_path, calibration_batches) -> Path:
    """Static INT8 quantization, using `calibration_batches` (arrays of shape (N, 3, H, W))."""
    out_path = Path(out_path)
    prepared = out_path.with_name(out_path.stem + "_prep.onnx")
    # Recommended first step: shape inference and graph clean-up so more operations get quantized.
    quant_pre_process(str(fp32_path), str(prepared))
    quantize_static(
        str(prepared),
        str(out_path),
        ImageBatches(calibration_batches),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prepared.unlink()
    return out_path
