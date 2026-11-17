"""Made-up models, images and settings files for the tests of a user's own model (tests only; never results).

The model is tiny: average each colour channel of the picture, then one linear layer. With the identity
weights, its top answer is the picture's strongest channel, so on the made-up pictures (class 0 red, 1 green,
2 blue, each pixel drawn at random around its colour) it is right on every image by construction. Other
weights give the "wrong" builds the tests need.
"""

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from brokkr_edge import user_label
from brokkr_edge.quantize import to_int8
from brokkr_edge.user_plan import REPOSITORY

CLASSES = ["red", "green", "blue"]
# The model's input: 32 x 32 pixels (pictures are 48 x 48, resized to 32 and cropped to 32). Raised from 8 on
# 17 November 2026: the ImageNet-C damage code refuses pictures smaller than 32 x 32, and every user run
# applies all 13 conditions (docs/user_models.md, section 4 and the note of 17 November 2026;
# tests/test_user_damage_sizes.py).
SIZE = 32
PREP = {"resize": SIZE, "crop": SIZE, "interpolation": "bilinear", "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5], "channel_order": "RGB", "layout": "NCHW"}
ROLLED = np.roll(np.eye(3), 1, axis=0)  # answers the next class: agrees with FP32 on no image
TWO_THIRDS = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])  # blue pictures get "red": agrees on 2 of 3 classes


def tiny_model(path, weights=None, probabilities=False, size=SIZE, batch="N", extra_output=False,
               input_name="images") -> Path:
    weights = np.eye(3) if weights is None else np.asarray(weights)
    n_out = len(weights)
    nodes = [helper.make_node("GlobalAveragePool", [input_name], ["pooled"]),
             helper.make_node("Flatten", ["pooled"], ["flat"]),
             helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1)]
    out = "logits"
    if probabilities:
        nodes.append(helper.make_node("Softmax", ["logits"], ["probs"], axis=1))
        out = "probs"
    outputs = [helper.make_tensor_value_info(out, TensorProto.FLOAT, [batch, n_out])]
    if extra_output:
        outputs.append(helper.make_tensor_value_info("flat", TensorProto.FLOAT, [batch, 3]))
    graph = helper.make_graph(
        nodes, "tiny", [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [batch, 3, size, size])],
        outputs, initializer=[numpy_helper.from_array(weights.astype(np.float32), "W"),
                              numpy_helper.from_array(np.zeros(n_out, np.float32), "B")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    Path(path).write_bytes(model.SerializeToString())
    return Path(path)


def quantized(fp32_path, out_path, size=SIZE) -> Path:
    """An INT8 build of a tiny model, made with Brokkr's INT8 code on random made-up pictures."""
    rng = np.random.default_rng(0)
    batches = [rng.uniform(-1, 1, (8, 3, size, size)).astype(np.float32) for _ in range(2)]
    return to_int8(fp32_path, out_path, batches, method="minmax")


def picture(label, rng, size=48) -> Image.Image:
    pixels = rng.integers(0, 70, (size, size, 3))
    if label is None:
        label = int(rng.integers(0, 3))
    pixels[..., label] = rng.integers(180, 256, (size, size))
    return Image.fromarray(pixels.astype(np.uint8))


def write_images(folder, per_class: dict, seed=0, mislabelled=False) -> Path:
    """folder/<class>/<n>.png, per_class = {class name: count}. Every file is different. mislabelled: each
    folder holds pictures of the next class's colour (so the tiny model is wrong on every one of them)."""
    rng = np.random.default_rng(seed)
    for name, count in per_class.items():
        (Path(folder) / name).mkdir(parents=True, exist_ok=True)
        colour = (CLASSES.index(name) + 1) % 3 if mislabelled else CLASSES.index(name)
        for i in range(count):
            picture(colour, rng).save(Path(folder) / name / f"{i:05d}.png")
    return Path(folder)


def write_unlabelled(folder, count, seed=99) -> Path:
    rng = np.random.default_rng(seed)
    Path(folder).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        picture(None, rng).save(Path(folder) / f"u{i:05d}.png")
    return Path(folder)


def settings(fp32="fp32.onnx", **changes) -> dict:
    data = {
        "settings_version": 1,
        "name": "made-up-colours",
        "fp32_model": fp32,
        "shrunk_model": None,
        "classes": list(CLASSES),
        "outputs": "logits",
        "preprocessing": dict(PREP),
        "split": "brokkr",
        "expected_accuracy": {"top1": 0.99, "n_images": 300, "measured_on": "made-up validation pictures"},
        "licences": {"model_code": "Apache-2.0", "model_weights": "CC0-1.0", "images": "CC0-1.0",
                     "images_source": "made up by the tests"},
        "declarations": {"images_not_used_to_train_the_model": True},
        "device_kind": "laptop",
    }
    data.update(changes)
    return data


_numbers = itertools.count()


def write_settings(folder, **changes) -> Path:
    path = Path(folder) / f"settings_{next(_numbers)}.json"
    path.write_text(json.dumps(settings(**changes)), encoding="utf-8")
    return path


def made_up_seed_sensitivity(path, differences=(0.01,)) -> Path:
    """A made-up stand-in for Brokkr's seed-sensitivity record (results/checks/seed_sensitivity.json, which is
    gitignored, so a fresh clone does not have it). tests/conftest.py points every test at one, so no test
    depends on that file being on this machine. Made up: nothing here is a result."""
    cells = [{"difference": d} for d in differences]
    Path(path).write_text(json.dumps({"model": "mobilenet_v3_large", "settings": {"n_images": 1234},
                                      "raw": {"cells": cells}}), encoding="utf-8")
    return Path(path)


# `brokkr-edge test` in a separate Python, with the seed-sensitivity record this test session uses.
_CLI = ("import sys; from pathlib import Path; import brokkr_edge.user_label as u; "
        "u.SEED_SENSITIVITY = Path(sys.argv[1]); from brokkr_edge import cli; "
        "sys.exit(cli.main(sys.argv[2:]))")


def command(*args, seed_sensitivity=None):
    """`brokkr-edge test ...` as a separate process; seed_sensitivity None: the record this session uses."""
    record = seed_sensitivity or user_label.SEED_SENSITIVITY
    return subprocess.run([sys.executable, "-c", _CLI, str(record), "test", *map(str, args)],
                          capture_output=True, text=True, cwd=REPOSITORY)
