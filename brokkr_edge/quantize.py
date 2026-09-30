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
from onnx import numpy_helper
from onnxconverter_common import float16
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

from brokkr_edge.benchmark import make_session

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


# Stage 3 feeds calibration images in batches of 32, 4 batches (128 images) per group. The group size
# is a fixed part of the INT8 method: it can shift Percentile/Entropy ranges (see to_int8).
CALIBRATION_BATCH = 32
CALIBRATION_GROUP_BATCHES = 4


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
            group_batches: int | None = None, unrounded_output_ops: list | None = None,
            skip_symbolic_shape: bool = False, keep_float_outputs: list | None = None,
            report: dict | None = None) -> Path:
    """Static INT8 quantization, using `calibration_batches` (arrays of shape (N, 3, H, W)).

    method: a key of INT8_METHODS.
    group_batches: feed the calibration batches this many at a time (onnxruntime's
        CalibStridedMinMax option, which works for every method). Percentile and Entropy otherwise
        keep every layer's output for all images in memory at once (about 22 GB for 512 images of
        MobileNetV3-Large). MinMax gives identical ranges either way; for the histogram methods the
        first group sets the bin width, so the group size is recorded with the model.
    unrounded_output_ops: operation types whose OUTPUT stays in float (their weights stay int8),
        e.g. ["Gemm"] for the final layer (task 3.4; onnxruntime's OpTypesToExcludeOutputQuantization).
    skip_symbolic_shape: skip the symbolic shape inference in onnxruntime's preparation step
        (quant_pre_process). Off by default; used only where that step crashes (ConvNeXt-Tiny, task 4.1),
        after checking it leaves MobileNetV3-Large's INT8 model unchanged. Recorded with the model.
    keep_float_outputs: output tensor names of nodes that stay in float (not quantized): the nodes of
        the prepared model that produce them are passed to onnxruntime as nodes_to_exclude (SE1,
        brokkr_edge.se_float). Every name must be found, on a named node, or the build stops.
    report: if given, filled with what was done ("nodes_kept_float": the excluded node names).
    """
    out_path = Path(out_path)
    prepared = out_path.with_name(out_path.stem + "_prep.onnx")
    # Recommended first step: shape inference and graph clean-up so more operations get quantized.
    quant_pre_process(str(fp32_path), str(prepared), skip_symbolic_shape=skip_symbolic_shape)
    extra_options = dict(INT8_METHODS[method]["extra_options"])
    if group_batches:
        extra_options["CalibStridedMinMax"] = group_batches
    if unrounded_output_ops:
        extra_options["OpTypesToExcludeOutputQuantization"] = list(unrounded_output_ops)
    nodes_to_exclude = []
    if keep_float_outputs:
        wanted = set(keep_float_outputs)
        nodes = [n for n in onnx.load(str(prepared)).graph.node if wanted & set(n.output)]
        found = {o for n in nodes for o in n.output} & wanted
        names = [n.name for n in nodes]
        if found != wanted or not all(names) or len(set(names)) != len(names):
            prepared.unlink()
            raise ValueError(f"keep_float_outputs: {len(wanted - found)} outputs not found in the prepared "
                             "model, or their nodes are unnamed or share a name")
        nodes_to_exclude = names
    if report is not None:
        report["nodes_kept_float"] = nodes_to_exclude
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
        nodes_to_exclude=nodes_to_exclude or None,
    )
    prepared.unlink()
    return out_path


def weight_quantization(path) -> dict:
    """Count the int8 weight tensors stored with one scale per output channel vs one scale in total.

    Read from the model file itself, so it shows what was actually built, not what was requested.
    """
    model = onnx.load(str(path))
    stored = {init.name: init for init in model.graph.initializer}
    counts = {"per_channel": 0, "per_tensor": 0}
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear" and node.input[0] in stored:
            values = numpy_helper.to_array(stored[node.input[0]])
            if values.dtype == np.int8 and values.ndim >= 2:  # a weight matrix or filter bank
                scale = numpy_helper.to_array(stored[node.input[1]])
                counts["per_channel" if scale.size > 1 else "per_tensor"] += 1
    return counts


def damaged_calibration_plan(n_images: int, allowed: list, seed: int = 5) -> list:
    """Which calibration images get damaged, and how (task 3.3).

    Exactly half the images, chosen at random, get a random corruption from `allowed` at a random
    severity 1-5; the rest stay clean. Returns (corruption name or None, severity) per image, in the
    calibration split's order. The random draws depend only on n_images, len(allowed) and seed, so
    the five leave-one-out models damage the same images at the same severities; only the list of
    allowed corruptions differs.
    """
    rng = np.random.default_rng(seed)
    damaged = rng.permutation(n_images)[: n_images // 2]
    kinds = rng.integers(0, len(allowed), n_images)
    severities = rng.integers(1, 6, n_images)  # 1 to 5
    plan = [(None, 0)] * n_images
    for i in damaged:
        plan[i] = (allowed[kinds[i]], int(severities[i]))
    return plan


def check_int8_build(path, images: np.ndarray, fp32_top1: np.ndarray, expected_weights: dict,
                     min_agreement: float = 0.20) -> dict:
    """Safety checks for a freshly built INT8 model. Returns {description: passed}.

    It loads and gives finite scores of the right shape on `images`; its top answer agrees with
    FP32's (`fp32_top1`) on at least `min_agreement` of them; its weights are stored as expected.
    """
    session = make_session(path, num_threads=4)
    scores = np.concatenate([session.run(None, {"images": images[i:i + CALIBRATION_BATCH]})[0]
                             for i in range(0, len(images), CALIBRATION_BATCH)])
    agreement = float(np.mean(scores.argmax(1) == fp32_top1))
    weights = weight_quantization(path)
    return {f"loads and gives {(len(images), 1000)} scores": scores.shape == (len(images), 1000),
            "all scores finite": bool(np.isfinite(scores).all()),
            f"top-1 agreement with FP32 >= {min_agreement:.0%} (got {agreement:.1%})":
                agreement >= min_agreement,
            f"weights stored as expected {expected_weights}": weights == expected_weights}
