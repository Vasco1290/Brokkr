"""Checks on a user's models and on what they output (docs/user_models.md, sections 1 and 3).

- inspect_model / check_model: the model loads, has one float32 input of the stated shape and layout and one
  output with one score per class; an "FP32" model holds no quantized operations and no FP16 weights; a
  declared INT8 build holds quantized operations.
- check_pair: a supplied shrunk build has the FP32 model's input and output, and is a different file.
- prepare: the user's stated preprocessing (resize, centre crop, mean and std, channel order, layout).
- check_outputs / as_logits: logits or probabilities, as declared (softmax applied twice would quietly spoil
  coverage and ECE).
- agreement: how often the shrunk build's top answer equals FP32's (study lines: failed below 20%, warning
  below 90%).
- expected_accuracy: Brokkr's clean FP32 top-1 must be within 5 points of the user's own figure, and above
  chance; a large preprocessing or class-order mistake shows up here instead of in a quietly wrong label.
"""

import numpy as np
import onnx
from onnx import numpy_helper

from brokkr_edge.accuracy import bootstrap_ci, open_image, resize_and_crop
from brokkr_edge.benchmark import make_session
from brokkr_edge.schema import MIN_AGREEMENT_WITH_FP32

WARN_AGREEMENT_BELOW = 0.90  # the study's warning line for INT8 agreement with FP32
EXPECTED_ACCURACY_TOLERANCE = 0.05  # H, D6: 5 points
SUMS_TO_ONE = 1e-3  # a probability row sums to 1 within this
BATCH = 32
CHECK_THREADS = 4
QUANTIZED_OPS = ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "ConvInteger",
                 "MatMulInteger", "DynamicQuantizeMatMul")


def _is_quantized_op(op_type: str) -> bool:
    return op_type in QUANTIZED_OPS or op_type.startswith("QLinear")


def inspect_model(path) -> dict:
    """What a model file holds: its input and output (name, type, shape) and what it is made of.

    Shapes keep ONNX Runtime's view: a whole number for a fixed dimension, a name or None for a free one.
    Raises ValueError if the file cannot be loaded.
    """
    try:
        model = onnx.load(str(path))
        session = make_session(path, CHECK_THREADS)
    except Exception as e:  # onnx and ONNX Runtime raise their own error types for a bad file
        raise ValueError(f"cannot be loaded by ONNX Runtime ({type(e).__name__}: {e})") from e
    weights = [numpy_helper.to_array(t) for t in model.graph.initializer]
    return {
        "inputs": [{"name": x.name, "type": x.type, "shape": list(x.shape)} for x in session.get_inputs()],
        "outputs": [{"name": x.name, "type": x.type, "shape": list(x.shape)} for x in session.get_outputs()],
        "found": {
            "quantized_operations": sum(_is_quantized_op(n.op_type) for n in model.graph.node),
            "eight_bit_weight_tensors": sum(w.dtype in (np.int8, np.uint8) and w.ndim >= 2 for w in weights),
            "fp16_weight_tensors": sum(w.dtype == np.float16 for w in weights),
        },
    }


def _fixed(dim) -> bool:
    return isinstance(dim, int)


def check_model(info: dict, prep: dict, n_classes: int, role: str) -> list:
    """Every way the model cannot be tested with these settings, as sentences. role: "fp32" or "shrunk"."""
    what = "the FP32 model" if role == "fp32" else "the shrunk build"
    problems = []
    if len(info["inputs"]) != 1 or len(info["outputs"]) != 1:
        return [f"{what}: needs exactly one input and one output (it has {len(info['inputs'])} and "
                f"{len(info['outputs'])})"]
    x, y = info["inputs"][0], info["outputs"][0]
    if x["type"] != "tensor(float)":
        problems.append(f"{what}: its input must be float32 (it is {x['type']})")
    crop = prep["crop"]
    expected = [None, 3, crop, crop] if prep["layout"] == "NCHW" else [None, crop, crop, 3]
    if len(x["shape"]) != 4:
        problems.append(f"{what}: its input must have 4 dimensions (it has {len(x['shape'])})")
    else:
        for i, (got, want) in enumerate(zip(x["shape"], expected, strict=True)):
            if i == 0:
                if _fixed(got) and got != 1:
                    problems.append(f"{what}: a fixed batch size must be 1 (it is {got}); free is also fine")
            elif _fixed(got) and got != want:
                problems.append(f"{what}: input shape {x['shape']} does not match the stated layout "
                                f"{prep['layout']} and crop {crop} (expected {['batch', *expected[1:]]})")
                break
    if y["type"] != "tensor(float)":
        problems.append(f"{what}: its output must be float32 (it is {y['type']})")
    if len(y["shape"]) != 2:
        problems.append(f"{what}: its output must have 2 dimensions, (batch, classes) (it has {y['shape']})")
    elif _fixed(y["shape"][1]) and y["shape"][1] != n_classes:
        problems.append(f"{what}: it gives {y['shape'][1]} scores per image, but the settings file lists "
                        f"{n_classes} classes")
    found = info["found"]
    if role == "fp32" and (found["quantized_operations"] or found["fp16_weight_tensors"]):
        problems.append(f"{what}: it holds {found['quantized_operations']} quantized operations and "
                        f"{found['fp16_weight_tensors']} FP16 weight tensors, so it is not an FP32 model")
    if role == "shrunk" and not found["quantized_operations"]:
        problems.append(f"{what}: declared INT8, but the file holds no quantized operations")
    return problems


def _same_dims(a: list, b: list) -> bool:
    """Same length, and every pair of fixed dimensions equal (a free dimension matches any size)."""
    return len(a) == len(b) and all(not (_fixed(p) and _fixed(q)) or p == q
                                    for p, q in zip(a, b, strict=True))


def check_pair(fp32: dict, shrunk: dict, fp32_sha256: str, shrunk_sha256: str) -> list:
    """Problems with a supplied FP32 + shrunk pair: same input and output, and different files."""
    problems = []
    if fp32_sha256 == shrunk_sha256:
        problems.append("the shrunk build is the same file as the FP32 model (identical SHA-256)")
    a, b = fp32["inputs"][0], shrunk["inputs"][0]
    if a["name"] != b["name"] or a["type"] != b["type"] or not _same_dims(a["shape"], b["shape"]):
        problems.append(f"the two builds take different inputs: FP32 {a['name']} {a['type']} {a['shape']}, "
                        f"shrunk {b['name']} {b['type']} {b['shape']}")
    if not _same_dims(fp32["outputs"][0]["shape"], shrunk["outputs"][0]["shape"]):
        problems.append(f"the two builds give different outputs: FP32 {fp32['outputs'][0]['shape']}, "
                        f"shrunk {shrunk['outputs'][0]['shape']}")
    return problems


def prepare(path, prep: dict) -> np.ndarray:
    """One image as the model takes it: resize the shorter side, centre crop, scale to 0-1, subtract the mean,
    divide by the std (RGB order), then reorder channels and dimensions as stated. Photo-rotation tags are not
    applied (docs/user_models.md, section 3)."""
    pixels = resize_and_crop(open_image(path), prep["resize"], prep["crop"], prep["interpolation"])
    x = (pixels.astype(np.float32) / 255.0 - np.asarray(prep["mean"], np.float32)) / np.asarray(prep["std"],
                                                                                                np.float32)
    if prep["channel_order"] == "BGR":
        x = x[..., ::-1]
    if prep["layout"] == "NCHW":
        x = x.transpose(2, 0, 1)
    return np.ascontiguousarray(x, dtype=np.float32)


def run(session, paths: list, prep: dict, batch: int = BATCH) -> np.ndarray:
    """The model's outputs for these image files, as float32 (batch 1 for a model with a fixed batch of 1)."""
    name = session.get_inputs()[0].name
    outputs = []
    for start in range(0, len(paths), batch):
        x = np.stack([prepare(p, prep) for p in paths[start:start + batch]])
        outputs.append(session.run(None, {name: x})[0])
    return np.concatenate(outputs).astype(np.float32)


def _rows_are_probabilities(out: np.ndarray) -> bool:
    return bool((out >= 0).all() and np.all(np.abs(out.sum(axis=1) - 1.0) <= SUMS_TO_ONE))


def check_outputs(out: np.ndarray, declared: str, n_classes: int, role: str = "fp32") -> list:
    """Do the outputs look like what the settings file declares? role "shrunk" checks less (its rounded
    probabilities need not sum to exactly 1)."""
    what = "the FP32 model" if role == "fp32" else "the shrunk build"
    if out.ndim != 2 or out.shape[1] != n_classes:
        return [f"{what}: gave outputs of shape {list(out.shape)}, expected (images, {n_classes})"]
    if not np.isfinite(out).all():
        return [f"{what}: gave scores that are not finite numbers"]
    if declared == "probabilities":
        if role == "fp32" and not _rows_are_probabilities(out):
            return [f"{what}: declared probabilities, but its outputs are not all non-negative rows "
                    "that sum to "
                    f"1 (within {SUMS_TO_ONE})"]
        if role == "shrunk" and (out < 0).any():
            return [f"{what}: declared probabilities, but some outputs are negative"]
    elif role == "fp32" and _rows_are_probabilities(out):
        return [f"{what}: declared logits, but every output row is non-negative and sums to 1: "
                "these look like "
                'probabilities (set "outputs": "probabilities")']
    return []


def as_logits(out: np.ndarray, declared: str) -> np.ndarray:
    """Scores that softmax turns back into the model's own probabilities: log(p) for probabilities (zero is
    clipped to the smallest float32 number), unchanged for logits."""
    if declared == "logits":
        return out
    return np.log(np.clip(out, np.finfo(np.float32).tiny, None)).astype(np.float32)


def agreement(fp32_out: np.ndarray, shrunk_out: np.ndarray) -> dict:
    """Top-1 agreement (ties to the lower class number) and what it means for the shrunk build."""
    value = float(np.mean(fp32_out.argmax(axis=1) == shrunk_out.argmax(axis=1)))
    return {"agreement": value, "n_items": int(len(fp32_out)),
            "failed_below": MIN_AGREEMENT_WITH_FP32, "warn_below": WARN_AGREEMENT_BELOW,
            "status": "usable" if value >= MIN_AGREEMENT_WITH_FP32 else "failed"}


def expected_accuracy(correct: np.ndarray, stated: dict, n_classes: int) -> dict:
    """Brokkr's clean FP32 top-1 on the test images against the user's own figure (H, D6).

    Pass = |correct - stated x n| <= 0.05 x n (in whole images) and the 95% interval's lower end above chance.
    """
    n, count = len(correct), int(correct.sum())
    ci = bootstrap_ci(correct.astype(np.float64))
    within = abs(count - stated["top1"] * n) <= EXPECTED_ACCURACY_TOLERANCE * n
    above_chance = ci[0] > 1.0 / n_classes
    return {"measured": count / n, "ci95": list(ci), "n_items": n, "correct": count,
            "stated": stated["top1"], "stated_n_images": stated["n_images"],
            "measured_on": stated["measured_on"],
            "tolerance": EXPECTED_ACCURACY_TOLERANCE, "chance": 1.0 / n_classes,
            "within_tolerance": bool(within), "above_chance": bool(above_chance),
            "pass": bool(within and above_chance)}
