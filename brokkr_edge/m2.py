"""M2 (docs/hypotheses_stage4.md): where does INT8's extra rounding error under darkness and fog arise?

Local rounding error at one quantized tensor: take the FP32 model's values there, round them with that
tensor's INT8 scale and zero-point (brokkr_edge.levels.fake_quantize: divide, round halves to even, add the
zero-point, clip to 8 bits, turn back into numbers), and compare with the unrounded values as a
signal-to-noise ratio, SQNR = 20·log10(‖x‖ / ‖x − x̂‖) in dB (ONNX Runtime's formula, with its guard:
each norm at least the machine epsilon). It is the error INT8 adds at that tensor alone, given perfect
inputs. Everything is computed per image first.

Extra error at tensor t: E(t) = mean over images of [SQNR_clean(t) - SQNR_damaged(t)] (positive = the
damage makes INT8 round that tensor worse). E_early = median of E(t) over the first 10% of the matched
tensors after the input (#1 .. #ceil(N/10)); E_rest = median over the tensors after that block. 95%
intervals: 1,000 resamples of the images, seed 0, recomputing E(t) and both medians.

Verdict rules: the committed M2 section and the dated notes of 27 September 2026 (per-model "or" for
rejects; "interval includes 0" read as "interval not above zero"; M2b's -0.05..+0.05 band inclusive,
"below -0.05" strict).
"""

import math

import numpy as np
import onnx
from onnx import numpy_helper

from brokkr_edge.levels import SATURATION, fake_quantize

N_RESAMPLES, SEED = 1000, 0
EPS = np.finfo(float).eps  # ONNX Runtime's guard in compute_signal_to_quantization_noice_ratio
EARLY_FRACTION = 0.10
REPORTED_FRACTIONS = (0.05, 0.20)
LARGE_DB = 3.0        # M2a: E_early at least this
NO_EXTRA_DB = 1.0     # "no extra error": E(t) never reaches this at any matched tensor
BAND = 0.05           # M2b: the interval of mean R(damaged) - R(clean) inside [-0.05, +0.05]
SUPPORTS_NEEDED = 6   # M2a and M2b: at least 6 models


def activation_quantizers(model: onnx.ModelProto) -> list:
    """Activation QuantizeLinear nodes in graph order: tensor name, scale, zero-point, and the output of
    the DequantizeLinear that reads it (what the INT8 model computes with). Each tensor listed once."""
    stored = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    dequantized = {}
    for n in model.graph.node:
        if n.op_type == "DequantizeLinear":
            dequantized.setdefault(n.input[0], n.output[0])
    found, seen = [], set()
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and n.input[0] not in stored and n.input[0] not in seen:
            seen.add(n.input[0])
            found.append({"tensor": n.input[0], "scale": stored[n.input[1]], "zero_point": stored[n.input[2]],
                          "dequantized": dequantized.get(n.output[0])})
    return found


def squared_norms(reference: np.ndarray, approximation: np.ndarray) -> tuple:
    """Per image: ‖x‖² and ‖x − x̂‖² (float64). Kept separate so SQNR can be per image or pooled."""
    n = reference.shape[0]
    x = reference.reshape(n, -1).astype(np.float64)
    d = x - approximation.reshape(n, -1).astype(np.float64)
    return (x * x).sum(axis=1), (d * d).sum(axis=1)


def sqnr_db(signal2, noise2) -> np.ndarray:
    """SQNR in dB from squared norms, with ONNX Runtime's guard (each norm at least EPS)."""
    signal = np.maximum(np.sqrt(np.asarray(signal2, dtype=np.float64)), EPS)
    noise = np.maximum(np.sqrt(np.asarray(noise2, dtype=np.float64)), EPS)
    return 20 * np.log10(signal / noise)


def pooled_sqnr_db(signal2: np.ndarray, noise2: np.ndarray) -> float:
    """SQNR over all images at once, as onnxruntime.quantization.qdq_loss_debug pools them."""
    return float(sqnr_db(signal2.sum(), noise2.sum()))


def local_rounding(x: np.ndarray, scale, zero_point) -> tuple:
    """Per image: squared norms of the FP32 values and of the error INT8 adds at this tensor alone."""
    return squared_norms(x, fake_quantize(x, scale, zero_point))


def levels(x: np.ndarray, scale, zero_point) -> np.ndarray:
    """The 8-bit levels QuantizeLinear gives x (round halves to even, add the zero-point, clip)."""
    low, high = SATURATION[np.asarray(zero_point).dtype]
    return np.clip(np.rint(np.asarray(x, dtype=np.float32) / np.float32(scale)) + np.float32(zero_point),
                   low, high)


def input_level_ratio(images: np.ndarray, scale, zero_point) -> np.ndarray:
    """M2b, per image: the mean over the colour channels of V_q / min(V_pre, 256), where V_pre = distinct
    values in the channel of the normalised input and V_q = distinct 8-bit levels after quantizing it."""
    q = levels(images, scale, zero_point)
    ratios = np.empty(images.shape[:2])
    for i in range(images.shape[0]):
        for c in range(images.shape[1]):
            v_pre = np.unique(images[i, c]).size
            ratios[i, c] = np.unique(q[i, c]).size / min(v_pre, 256)
    return ratios.mean(axis=1)


def early_count(n_after_input: int, fraction: float = EARLY_FRACTION) -> int:
    """How many tensors after the input form the early block: ceil(N x fraction)."""
    return math.ceil(n_after_input * fraction)


def _medians(e: np.ndarray, k: int) -> tuple:
    return float(np.median(e[1:1 + k])), float(np.median(e[1 + k:]))


def extra_error(clean_db: np.ndarray, damaged_db: np.ndarray, fraction: float = EARLY_FRACTION) -> dict:
    """E(t), E_early, E_rest and E_early - E_rest, with bootstrap 95% intervals over the images.

    clean_db, damaged_db: (images, tensors) local SQNR in dB, the same images in the same order, tensor
    #0 the input image (belongs to neither block).
    """
    diff = clean_db - damaged_db
    n_images, n_tensors = diff.shape
    k = early_count(n_tensors - 1, fraction)
    if n_tensors - 1 - k < 1:
        raise ValueError("no tensors left after the early block")
    e = diff.mean(axis=0)
    early, rest = _medians(e, k)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_RESAMPLES):
        b_early, b_rest = _medians(diff[rng.integers(0, n_images, n_images)].mean(axis=0), k)
        boot.append((b_early, b_rest, b_early - b_rest))
    boot = np.array(boot)
    ci = [[float(v) for v in np.percentile(boot[:, j], [2.5, 97.5])] for j in range(3)]
    return {"E": e, "early_tensors": k, "rest_tensors": n_tensors - 1 - k,
            "E_early": early, "E_early_ci95": ci[0], "E_rest": rest, "E_rest_ci95": ci[1],
            "early_minus_rest": early - rest, "early_minus_rest_ci95": ci[2],
            "max_E": float(e.max()), "no_extra_error": bool(e.max() < NO_EXTRA_DB)}


def m2a_model(s: dict) -> dict:
    """One model's part in an M2a verdict (s from extra_error with the 10% block)."""
    early_above = s["E_early_ci95"][0] > 0
    supports = (not s["no_extra_error"] and s["E_early"] >= LARGE_DB and early_above
                and s["E_early"] > s["E_rest"] and s["early_minus_rest_ci95"][0] > 0)
    # Rejects (per model, dated note): E_early <= E_rest, or E_early's interval not above zero.
    rejects = s["E_early"] <= s["E_rest"] or not early_above
    return {"supports": bool(supports), "rejects": bool(rejects)}


def verdict(parts: dict, rejects_needed: int) -> dict:
    """SUPPORTS if at least 6 models support; else REJECTS if at least `rejects_needed` reject; else
    INCONCLUSIVE. parts: model -> {"supports": bool, "rejects": bool}."""
    n_support = sum(p["supports"] for p in parts.values())
    n_reject = sum(p["rejects"] for p in parts.values())
    if n_support >= SUPPORTS_NEEDED:
        v = "SUPPORTS"
    elif n_reject >= rejects_needed:
        v = "REJECTS"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "supporting": n_support, "supports_needed": SUPPORTS_NEEDED,
            "rejecting": n_reject, "rejects_needed": rejects_needed, "models": len(parts)}


def rejects_needed(n_models: int) -> int:
    """More than half of the models: 5 of 8, 4 of 7 (the committed counts)."""
    return n_models // 2 + 1


def ratio_change(r_clean: np.ndarray, r_damaged: np.ndarray) -> dict:
    """Mean of R(damaged) - R(clean) over the same images, with a paired bootstrap 95% interval."""
    d = r_damaged - r_clean
    rng = np.random.default_rng(SEED)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_RESAMPLES)]
    ci = [float(v) for v in np.percentile(boot, [2.5, 97.5])]
    return {"mean": float(d.mean()), "ci95": ci,
            "supports": bool(ci[0] >= -BAND and ci[1] <= BAND),  # inside the band, ends included
            "rejects": bool(ci[1] < -BAND)}                       # entirely below -0.05
