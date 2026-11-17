"""Brokkr's INT8 build of a user's model (step 3; docs/user_models.md, plan of 10 October 2026, section a, and
D2; H's decisions of 10 October 2026, 10 and 13).

The study's recipe, unchanged: Percentile 99.99 calibration on the 512 INT8-calibration images in canonical
order, fed 32 at a time and 128 images per group, int8 weights per channel and uint8 activations per tensor.
Only the preprocessing and the input name are the user's. A model with a fixed batch of 1 is fed one image at
a time, still 128 images per group.

If the build crashes (ONNX Runtime's preparation step can, as it did for ConvNeXt-Tiny in task 4.1), it is
tried once more with skip_symbolic_shape, and the retry is recorded (D2). If it crashes again, the build is
"failed" with no number (decision 10). Only an error's type is recorded, never its message (it can hold
paths).

A finished build is checked on the first 256 conformal-calibration images (the same check as for a supplied
build; never test images): its outputs are finite, its int8 weights are stored per channel, and its top answer
agrees with FP32's on at least 20% of them (below that: "failed"; below 90%: a warning on the label).

Everything is written to --out: int8.onnx and int8_build.json (decision 13), with no image file name and no
path. A run started again reuses a finished build of the same FP32 model.
"""

import json
from pathlib import Path

import numpy as np

from brokkr_edge import __version__
from brokkr_edge.benchmark import make_session
from brokkr_edge.quantize import (
    CALIBRATION_BATCH,
    CALIBRATION_GROUP_BATCHES,
    INT8_METHODS,
    to_int8,
    weight_quantization,
)
from brokkr_edge.results import sha256_of, written_atomically
from brokkr_edge.schema import MIN_AGREEMENT_WITH_FP32
from brokkr_edge.user_checks import CHECK_THREADS, agreement, check_outputs, prepare, run

METHOD = "percentile99.99"
GROUP_IMAGES = CALIBRATION_BATCH * CALIBRATION_GROUP_BATCHES  # 128, the study's group
AGREEMENT_IMAGES = 256
INT8_FILE, RECORD_FILE = "int8.onnx", "int8_build.json"
CRASHED = "the INT8 build crashed, also with skip_symbolic_shape"


def recipe(batch: int, skip_symbolic_shape: bool) -> dict:
    """The recipe as the label records it: the study's fields, and how the images were fed."""
    method = INT8_METHODS[METHOD]
    return {"method": "static post-training quantization (onnxruntime)", "calibration": METHOD,
            "display_name": method["display_name"], "description": method["description"], "format": "QDQ",
            "weights": "int8, per-channel", "activations": "uint8, per-tensor", "calibration_images": 512,
            "batch": batch, "group_images": GROUP_IMAGES, "skip_symbolic_shape": skip_symbolic_shape}


def calibration_batches(paths: list, prep: dict, batch: int) -> list:
    """The calibration images as the model takes them, `batch` at a time, in the order given."""
    return [np.stack([prepare(p, prep) for p in paths[i:i + batch]]) for i in range(0, len(paths), batch)]


def _failure(check: str, value=None, limit=None, n_items=None) -> dict:
    return {"check": check, "value": value, "limit": limit, "n_items": n_items,
            "split": "conformal_calibration" if n_items else None}


def _attempt(fp32_path, out_file: Path, batches: list, batch: int, input_name: str) -> tuple:
    """(skip_symbolic_shape used, attempts made), or skip_symbolic_shape None if both attempts crashed."""
    attempts = []
    for skip in (False, True):
        try:
            to_int8(fp32_path, out_file, batches, method=METHOD, group_batches=GROUP_IMAGES // batch,
                    skip_symbolic_shape=skip, input_name=input_name)
            attempts.append({"skip_symbolic_shape": skip, "error": None})
            return skip, attempts
        except Exception as e:  # ONNX Runtime and onnx raise many kinds of error; only the type is kept
            attempts.append({"skip_symbolic_shape": skip, "error": type(e).__name__})
            for leftover in (out_file, out_file.with_name(out_file.stem + "_prep.onnx")):
                leftover.unlink(missing_ok=True)
    return None, attempts


def _checks(int8_file: Path, fp32_path, sample: list, prep: dict, batch: int, declared: str,
            n_classes: int) -> tuple:
    """(checks, failure or None) of a finished build, on the sample images."""
    fp32_out = run(make_session(fp32_path, CHECK_THREADS), sample, prep, batch)
    int8_out = run(make_session(int8_file, CHECK_THREADS), sample, prep, batch)
    outputs = check_outputs(int8_out, declared, n_classes, "shrunk")
    weights = weight_quantization(int8_file)
    agree = agreement(fp32_out, int8_out)
    checks = {"outputs": {"pass": not outputs, "n_items": len(sample)},
              "weights": {**weights, "pass": weights["per_tensor"] == 0 and weights["per_channel"] > 0},
              "agreement_with_fp32": agree}
    if outputs:
        return checks, _failure("the INT8 build's outputs are not usable scores")
    if not checks["weights"]["pass"]:
        return checks, _failure("the INT8 build's weights are not stored per channel")
    if agree["status"] == "failed":
        return checks, _failure("top-1 agreement with FP32", agree["agreement"], MIN_AGREEMENT_WITH_FP32,
                                agree["n_items"])
    return checks, None


def _reusable(record_path: Path, fp32_sha256: str) -> dict | None:
    """A finished build of the same FP32 model, with its file unchanged (a resumed run), else None."""
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("fp32_sha256") != fp32_sha256:
        return None
    file = record.get("file")
    if file and not ((record_path.parent / file["name"]).is_file()
                     and sha256_of(record_path.parent / file["name"]) == file["sha256"]):
        return None
    return record


def build_int8(plan: dict, fp32_path, parts: dict, out, log=print) -> dict:
    """Build, check and record the INT8 build of the plan's FP32 model; returns the record (int8_build.json).

    parts: brokkr_edge.user_plan.rebuild_parts(plan, ...), the images of each part.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)  # before the build: to_int8 writes its files into it
    fp32_sha256 = plan["models"]["fp32"]["sha256"]
    record_path, int8_file = out / RECORD_FILE, out / INT8_FILE
    found = _reusable(record_path, fp32_sha256)
    if found:
        log(f"INT8 build: reused from an earlier run ({found['status']})")
        return found

    prep, batch = plan["preprocessing"], plan["runtime"]["batch"]
    calibration = [e["path"] for e in parts["int8_calibration"]]
    batches = calibration_batches(calibration, prep, batch)
    skip, attempts = _attempt(fp32_path, int8_file, batches, batch, plan["models"]["fp32"]["input"]["name"])
    record = {"kind": "brokkr-edge user int8 build", "made_by": f"brokkr-edge {__version__}",
              "fp32_sha256": fp32_sha256, "attempts": attempts,
              "recipe": recipe(batch, bool(skip)),
              "calibration": {"part": "int8_calibration", "n_items": len(calibration),
                              "source": plan["split"]["parts"]["int8_calibration"]["source"],
                              "fingerprint": plan["split"]["parts"]["int8_calibration"]["fingerprint"]}}
    if skip is None:
        record.update(file=None, checks=None, status="failed", failure=_failure(CRASHED))
        log(f"INT8 build: FAILED ({CRASHED})")
    else:
        sample = [e["path"] for e in parts["conformal_calibration"][:AGREEMENT_IMAGES]]
        checks, failure = _checks(int8_file, fp32_path, sample, prep, batch,
                                  plan["checks"]["outputs"]["declared"], len(plan["classes"]))
        record.update(file={"name": INT8_FILE, "sha256": sha256_of(int8_file),
                            "size_bytes": int8_file.stat().st_size},
                      checks=checks, status="failed" if failure else "usable", failure=failure)
        value = checks["agreement_with_fp32"]["agreement"]
        log(f"INT8 build: {record['status']}; top-1 agreement with FP32 {value:.1%} on {len(sample)} images"
            + (" (made with skip_symbolic_shape after a crash)" if skip else ""))
    with written_atomically(record_path) as tmp:
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8", newline="\n")
    return record
