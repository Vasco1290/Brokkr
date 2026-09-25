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
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

# Settings are kept in one place so they can be saved alongside the results.
INT8_SETTINGS = {
    "method": "static post-training quantization (onnxruntime)",
    "format": "QDQ",
    "weights": "int8, per-channel",
    "activations": "uint8, per-tensor",
    "calibration_method": "MinMax",
}

# Calibration methods compared in Stage 3 (task 3.2). The method only decides the ACTIVATION ranges,
# from the values seen on the calibration images; weights always use their own exact min/max.
# Bin counts are onnxruntime's defaults (quantize_static cannot change them); they are written down
# here so the record says exactly what ran.
INT8_METHODS = {
    "minmax": {"calibrate_method": CalibrationMethod.MinMax, "extra_options": {},
               "description": "range = smallest to largest value seen"},
    "percentile99.99": {"calibrate_method": CalibrationMethod.Percentile,
                        "extra_options": {"CalibPercentile": 99.99},
                        "description": "range = 99.99th percentile of absolute values (2048 bins), "
                                       "clipped to the smallest/largest value seen"},
    "percentile99.999": {"calibrate_method": CalibrationMethod.Percentile,
                         "extra_options": {"CalibPercentile": 99.999},
                         "description": "range = 99.999th percentile of absolute values (2048 bins), "
                                        "clipped to the smallest/largest value seen"},
    "entropy": {"calibrate_method": CalibrationMethod.Entropy, "extra_options": {},
                "description": "range that loses least information (KL divergence), "
                               "128-bin histogram, 128 quantized bins"},
}


def to_fp16(fp32_path, out_path) -> Path:
    """Convert weights to FP16. Inputs and outputs stay FP32 so callers don't need to change."""
    model = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(out_path))
    return Path(out_path)


class ImageBatches(CalibrationDataReader):
    """Feeds preprocessed calibration images to the quantizer, one batch at a time.

    set_range lets onnxruntime read the batches one group at a time (see `group_batches` in to_int8).
    """

    def __init__(self, batches, input_name: str = "images"):
        self._batches = [batch.astype(np.float32) for batch in batches]
        self._input_name = input_name
        self.set_range(0, len(self._batches))

    def __len__(self):
        return len(self._batches)

    def set_range(self, start_index: int, end_index: int):
        self._next, self._end = start_index, min(end_index, len(self._batches))

    def get_next(self):
        if self._next >= self._end:
            return None
        self._next += 1
        return {self._input_name: self._batches[self._next - 1]}


def to_int8(fp32_path, out_path, calibration_batches, method: str = "minmax",
            group_batches: int | None = None) -> Path:
    """Static INT8 quantization, using `calibration_batches` (arrays of shape (N, 3, H, W)).

    method: a key of INT8_METHODS.
    group_batches: feed the calibration batches this many at a time (onnxruntime's
        CalibStridedMinMax option, which works for every method). Percentile and Entropy otherwise
        keep every layer's output for all images in memory at once (about 22 GB for 512 images of
        MobileNetV3-Large). MinMax gives identical ranges either way; for the histogram methods the
        first group sets the bin width, so the group size is recorded with the model.
    """
    out_path = Path(out_path)
    prepared = out_path.with_name(out_path.stem + "_prep.onnx")
    # Recommended first step: shape inference and graph clean-up so more operations get quantized.
    quant_pre_process(str(fp32_path), str(prepared))
    extra_options = dict(INT8_METHODS[method]["extra_options"])
    if group_batches:
        extra_options["CalibStridedMinMax"] = group_batches
    quantize_static(
        str(prepared),
        str(out_path),
        ImageBatches(calibration_batches),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=INT8_METHODS[method]["calibrate_method"],
        extra_options=extra_options,
    )
    prepared.unlink()
    return out_path
