"""`brokkr-edge test` (brokkr_edge.test_run, brokkr_edge.cli): the parts that need no model or dataset.

The numbers here are made up for the tests; they are never results. The command itself is checked
against real records by scripts/43_check_reproduction.py.
"""

import json
import subprocess
import sys

import numpy as np

from brokkr_edge.test_run import CONDITIONS, compare_scores, usable_precisions


def run(n_images=2000, n_classes=5, seed=0) -> dict:
    """Made-up scores for one record: every image's true class gets the highest score."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, n_images)
    logits = rng.random((n_images, n_classes)).astype(np.float32)
    logits[np.arange(n_images), labels] += 2.0
    return {"logits": logits, "labels": labels, "positions": np.arange(n_images)}


def with_wrong_answers(scores: dict, n: int) -> dict:
    """A copy whose first n images now get a wrong top-1 answer."""
    changed = {k: v.copy() for k, v in scores.items()}
    for i in range(n):
        changed["logits"][i, (changed["labels"][i] + 1) % changed["logits"].shape[1]] += 10.0
    return changed


def test_the_command_runs_the_13_conditions_of_task_4_1():
    assert len(CONDITIONS) == 13 and CONDITIONS[0]["corruption"] == "clean"
    assert sum(c["suite"] == "brokkr" for c in CONDITIONS) == 4
    assert sum(c["suite"] == "imagenet-c" for c in CONDITIONS) == 8


def test_identical_scores_are_reproduced_under_both_rules():
    old = run()
    for repeatable in (True, False):
        row = compare_scores(old, run(), repeatable)
        assert row["reproduced"] and row["scores_identical"] and row["top1_differs_on"] == 0


def test_the_repeatable_rule_needs_every_top1_answer_identical():
    old = run()
    nudged = {**old, "logits": old["logits"] + np.float32(1e-4)}  # scores differ, answers do not
    assert compare_scores(old, nudged, repeatable=True)["reproduced"]
    assert not compare_scores(old, nudged, repeatable=True)["scores_identical"]
    assert not compare_scores(old, with_wrong_answers(old, 1), repeatable=True)["reproduced"]


def test_the_other_rule_allows_one_image_in_a_thousand():
    old = run(n_images=2000)  # 0.1 points of 2,000 images = 2 images
    assert compare_scores(old, with_wrong_answers(old, 2), repeatable=False)["reproduced"]
    assert not compare_scores(old, with_wrong_answers(old, 3), repeatable=False)["reproduced"]


def test_other_images_are_never_a_reproduction():
    old, new = run(), run()
    new["positions"] = new["positions"] + 1
    assert not compare_scores(old, new, repeatable=True)["reproduced"]


def test_only_builds_with_a_usable_record_and_a_file_are_tested(tmp_path):
    def build(precision, agreement_ok, with_file=True):
        record = {"model": "tiny", "precision": precision, "licence": "MIT (test)",
                  "settings": {"calibration_method": "percentile99.99"} if precision != "fp32" else {},
                  "build": {"checks": {"agreement >= 20%": agreement_ok}},
                  "machine": {"git": {"commit": "a" * 40, "dirty": False}}}
        (tmp_path / f"tiny_{precision}.json").write_text(json.dumps(record), encoding="utf-8")
        if with_file:
            (tmp_path / f"tiny_{precision}.onnx").write_bytes(b"not a real model")

    build("fp32", True)
    build("int8_percentile99.99", False)  # failed its build check, like MobileNetV3-Small's INT8
    assert usable_precisions("tiny", tmp_path) == ["fp32"]
    build("int8_percentile99.99", True)
    assert usable_precisions("tiny", tmp_path) == ["fp32", "int8_percentile99.99"]
    (tmp_path / "tiny_int8_percentile99.99.onnx").unlink()
    assert usable_precisions("tiny", tmp_path) == ["fp32"]


def test_the_command_starts_without_pytorch():
    code = ("import sys; import brokkr_edge.cli, brokkr_edge.test_run; "
            "assert 'torch' not in sys.modules and 'torchvision' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
    help_text = subprocess.run([sys.executable, "-m", "brokkr_edge.cli", "test", "--help"],
                               capture_output=True, text=True, check=True).stdout
    assert "--model" in help_text and "brokkr-edge test" in help_text
