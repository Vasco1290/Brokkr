"""Checking a user's model and images, end to end on made-up data (brokkr_edge.user_plan, user_checks,
no_network and the --config command; docs/user_models.md, sections 1-7, slice 1).

One folder of made-up pictures (390 per class) and a few tiny models are made once for the module. They are
tests only: no number here is a result.
"""

import json
import shutil
import socket
import subprocess
import sys

import numpy as np
import pytest
from user_made_up import (
    CLASSES,
    ROLLED,
    TWO_THIRDS,
    quantized,
    settings,
    tiny_model,
    write_images,
    write_settings,
    write_unlabelled,
)

from brokkr_edge.cli import final_line
from brokkr_edge.no_network import NetworkBlocked, no_network
from brokkr_edge.user_checks import as_logits, check_outputs, check_pair, expected_accuracy, inspect_model
from brokkr_edge.user_plan import NETWORK_WORDING, REPOSITORY, check_out_folder, check_user_inputs
from brokkr_edge.user_settings import UserInputError

PER_CLASS = 390  # enough for 512 INT8 images, then at least 200 conformal-calibration and 200 test images


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = tmp_path_factory.mktemp("user")
    write_images(d / "images", {c: PER_CLASS for c in CLASSES})
    write_unlabelled(d / "calib", 520)
    fp32 = tiny_model(d / "fp32.onnx")
    tiny_model(d / "fp32_copy_with_other_weights.onnx", weights=np.eye(3) * 2)
    tiny_model(d / "probabilities.onnx", probabilities=True)
    tiny_model(d / "four_classes.onnx", weights=np.vstack([np.eye(3), np.zeros((1, 3))]))
    tiny_model(d / "two_outputs.onnx", extra_output=True)
    tiny_model(d / "batch_two.onnx", batch=2)
    quantized(fp32, d / "int8.onnx")
    quantized(tiny_model(d / "rolled.onnx", weights=ROLLED), d / "int8_rolled.onnx")
    quantized(tiny_model(d / "two_thirds.onnx", weights=TWO_THIRDS), d / "int8_two_thirds.onnx")
    quantized(tiny_model(d / "big.onnx", size=16), d / "int8_big.onnx", size=16)
    return d


def check(made, calib=False, images=None, **changes) -> dict:
    return check_user_inputs(write_settings(made, **changes), images or made / "images",
                             made / "calib" if calib else None, log=lambda *_: None)


def stops(made, *words, calib=False, images=None, **changes) -> list:
    with pytest.raises(UserInputError) as raised:
        check(made, calib, images, **changes)
    text = "\n".join(raised.value.problems)
    for w in words:
        assert w in text
    return raised.value.problems


def shrunk(file="int8.onnx") -> dict:
    return {"shrunk_model": {"file": file, "precision": "int8", "made_by": "made-up test build"}}


def test_brokkr_builds_int8_from_the_labelled_images(made):
    plan = check(made)
    parts = plan["split"]["parts"]
    assert parts["int8_calibration"]["source"] == "labelled" and parts["int8_calibration"]["n_items"] == 512
    n_rest = 3 * PER_CLASS - 512
    assert parts["conformal_calibration"]["n_items"] + parts["test"]["n_items"] == n_rest
    assert parts["conformal_calibration"]["n_items"] >= 200 and parts["test"]["n_items"] >= 200
    assert plan["checks"]["expected_accuracy"]["pass"] and plan["checks"]["outputs"]["pass"]
    assert plan["int8_build"].startswith("made by Brokkr") and "shrunk" not in plan["models"]
    assert {(w["kind"], w["part"]) for w in plan["warnings"]} == {
        ("few images", "conformal_calibration"), ("few images", "test")}
    assert plan["model_id"] == f"user/made-up-colours@{plan['models']['fp32']['sha256'][:12]}"


def test_the_unlabelled_calibration_folder_is_used_and_checked_for_overlap(made, tmp_path):
    plan = check(made, calib=True)
    assert plan["split"]["parts"]["int8_calibration"]["source"] == "calib-images"
    assert plan["split"]["calibration_folder"]["n_items"] == 520
    parts = plan["split"]["parts"]
    assert parts["conformal_calibration"]["n_items"] + parts["test"]["n_items"] == 3 * PER_CLASS
    calib = shutil.copytree(made / "calib", tmp_path / "calib")
    shutil.copy(made / "images" / "blue" / "00007.png", calib / "also_labelled.png")
    with pytest.raises(UserInputError, match="also_labelled.png is also in the labelled folder"):
        check_user_inputs(write_settings(made), made / "images", calib, log=lambda *_: None)


def test_a_good_supplied_pair_passes(made):
    plan = check(made, **shrunk())
    assert plan["int8_build"] == "supplied by the submitter"
    assert plan["split"]["parts"]["int8_calibration"] is None
    a = plan["checks"]["agreement_with_fp32"]
    assert a["status"] == "usable" and a["agreement"] >= 0.9 and a["n_items"] == 256
    assert plan["models"]["shrunk"]["found"]["quantized_operations"] > 0
    assert not any(w["kind"] == "low agreement with FP32" for w in plan["warnings"])


def test_agreement_below_20_percent_marks_the_build_failed_and_below_90_warns(made):
    failed = check(made, **shrunk("int8_rolled.onnx"))
    assert failed["models"]["shrunk"]["status"] == "failed"
    assert failed["checks"]["agreement_with_fp32"]["agreement"] < 0.2
    assert any(w["kind"] == "low agreement with FP32" for w in failed["warnings"])
    warned = check(made, **shrunk("int8_two_thirds.onnx"))
    a = warned["checks"]["agreement_with_fp32"]
    assert warned["models"]["shrunk"]["status"] == "usable" and 0.2 <= a["agreement"] < 0.9
    assert any(w["kind"] == "low agreement with FP32" and w["agreement"] == a["agreement"]
               for w in warned["warnings"])


def test_pair_mismatches_stop_the_run(made):
    stops(made, "declared INT8, but the file holds no quantized operations", **shrunk("fp32.onnx"))
    stops(made, "declared INT8, but the file holds no quantized operations",
          **shrunk("fp32_copy_with_other_weights.onnx"))
    stops(made, "does not match the stated layout", **shrunk("int8_big.onnx"))
    stops(made, "would not be used", calib=True, **shrunk())


def test_the_same_file_twice_is_not_a_pair(made):
    info = inspect_model(made / "int8.onnx")
    assert check_pair(info, info, "a" * 64, "a" * 64) == [
        "the shrunk build is the same file as the FP32 model (identical SHA-256)"]
    assert check_pair(info, info, "a" * 64, "b" * 64) == []


def test_model_checks_stop_the_run(made):
    stops(made, "not an FP32 model", fp32="int8.onnx")
    stops(made, "4 scores per image", fp32="four_classes.onnx")
    stops(made, "exactly one input and one output", fp32="two_outputs.onnx")
    stops(made, "a fixed batch size must be 1", fp32="batch_two.onnx")
    stops(made, "does not match the stated layout NHWC",
          preprocessing={**settings()["preprocessing"], "layout": "NHWC"})
    prep = {**settings()["preprocessing"], "resize": 16, "crop": 16}
    stops(made, "does not match the stated layout NCHW and crop 16", preprocessing=prep)


def test_logits_and_probabilities_must_be_declared_as_they_are(made):
    stops(made, "these look like probabilities", fp32="probabilities.onnx")
    assert check(made, fp32="probabilities.onnx", outputs="probabilities")["checks"]["outputs"]["pass"]
    stops(made, "declared probabilities, but its outputs are not", outputs="probabilities")


def test_outputs_check_and_log_of_probabilities():
    p = np.array([[0.7, 0.3, 0.0], [0.2, 0.5, 0.3]], np.float32)
    back = np.exp(as_logits(p, "probabilities"))
    assert np.allclose(back / back.sum(axis=1, keepdims=True), p, atol=1e-6)
    assert check_outputs(p, "probabilities", 3, "shrunk") == []
    assert "not finite" in check_outputs(np.array([[np.nan, 0, 1]], np.float32), "logits", 3)[0]
    assert "shape" in check_outputs(p, "logits", 4)[0]


def test_a_wrong_class_order_or_channel_order_is_caught_by_the_expected_accuracy(made, tmp_path):
    reordered = tmp_path / "images"
    for c in CLASSES:  # the same pictures, classes listed in another order than the model's outputs
        shutil.copytree(made / "images" / c, reordered / c)
    problems = stops(made, "check the class order", images=reordered, classes=["green", "red", "blue"])
    assert any("against your stated 99.0%" in p for p in problems)
    stops(made, "check the class order",
          preprocessing={**settings()["preprocessing"], "channel_order": "BGR"})


def test_expected_accuracy_reads_the_conformal_calibration_images_never_the_test_images(made, tmp_path):
    # Own split with a supplied build, so every calibration/ image is a conformal-calibration image.
    good_calibration = tmp_path / "a"
    write_images(good_calibration / "calibration", {c: 70 for c in CLASSES}, seed=21)
    write_images(good_calibration / "test", {c: 70 for c in CLASSES}, seed=22, mislabelled=True)
    result = check(made, images=good_calibration, split="own", **shrunk())["checks"]["expected_accuracy"]
    assert result["part"] == "conformal_calibration" and result["n_items"] == 210
    assert result["measured"] == 1.0  # the test/ pictures, all mislabelled, were not read
    bad_calibration = tmp_path / "b"  # the same, the other way round: now the check must stop the run
    write_images(bad_calibration / "calibration", {c: 70 for c in CLASSES}, seed=23, mislabelled=True)
    write_images(bad_calibration / "test", {c: 70 for c in CLASSES}, seed=24)
    stops(made, "clean FP32 top-1 0.0% on 210 conformal-calibration images", images=bad_calibration,
          split="own", **shrunk())


def correct_answers(n, n_correct, seed=0):
    """A made-up 0/1 array: n_correct right answers among n images, in a random order."""
    answers = np.array([1] * n_correct + [0] * (n - n_correct))
    return np.random.default_rng(seed).permutation(answers)


STATED = {"top1": 0.90, "n_images": 500, "measured_on": "own validation"}


def test_a_correct_model_at_200_images_with_a_4_point_chance_gap_passes():
    check = expected_accuracy(correct_answers(200, 172), STATED, 3)  # 86% against a stated 90%
    assert check["pass"] and check["n_items"] == 200
    assert check["allowed_gap"] == max(0.05, check["half_width_99"]) and check["half_width_99"] > 0.05


def test_the_gap_must_exceed_both_5_points_and_the_99_percent_half_width():
    # Stated 80% on 200 images: the 99% half-width is 2.576 x sqrt(0.8 x 0.2 / 200), about 7.3 points.
    stated = {**STATED, "top1": 0.80}
    assert expected_accuracy(correct_answers(200, 146), stated, 3)["within_tolerance"]  # 7 points: chance
    assert not expected_accuracy(correct_answers(200, 144), stated, 3)["within_tolerance"]  # 8 points
    # Stated 95% on 5,000 images: the half-width is under 1 point, so the 5-point line decides.
    stated = {**STATED, "top1": 0.95}
    assert expected_accuracy(correct_answers(5000, 4500), stated, 3)["within_tolerance"]  # exactly 5 points
    assert not expected_accuracy(correct_answers(5000, 4499), stated, 3)["within_tolerance"]


def test_a_low_accuracy_must_still_be_above_chance():
    low = expected_accuracy(correct_answers(100, 34), {**STATED, "top1": 0.34}, 3)
    assert low["within_tolerance"] and not low["above_chance"] and not low["pass"]


def test_a_shuffled_class_order_stops_the_run_at_about_200_images(made, tmp_path):
    images = tmp_path / "images"
    write_images(images / "calibration", {c: 70 for c in CLASSES}, seed=31)
    write_images(images / "test", {c: 70 for c in CLASSES}, seed=32)
    right_order = check(made, images=images, split="own", **shrunk())["checks"]["expected_accuracy"]
    assert right_order["pass"] and right_order["n_items"] == 210
    stops(made, "clean FP32 top-1 0.0% on 210 conformal-calibration images", "check the class order",
          images=images, split="own", classes=["green", "blue", "red"], **shrunk())


@pytest.fixture(scope="module")
def own(made):
    """An own-split folder: calibration/ (all conformal calibration, with a supplied build) and test/."""
    images = made / "own"
    write_images(images / "calibration", {c: 70 for c in CLASSES}, seed=41)
    write_images(images / "test", {c: 70 for c in CLASSES}, seed=42)
    return images


def refs_in(part, n, wrong=()):
    """n reference predictions from one part (spread over the classes); the numbers in `wrong` say blue."""
    files = [f"{part}/{c}/{i:05d}.png" for i in range(11) for c in CLASSES][:n]
    return [{"file": f, "top1": "blue" if k in wrong else f.split("/")[1]} for k, f in enumerate(files)]


def test_reference_predictions_naming_test_images_are_ignored_and_never_stop_the_run(made, own):
    every_one_wrong = [{"file": r["file"], "top1": "green" if r["top1"] != "green" else "red"}
                       for r in refs_in("test", 32)]
    plan = check(made, images=own, split="own", reference_predictions=every_one_wrong, **shrunk())
    assert plan["checks"]["reference_predictions"] == {
        "given": 32, "ignored_test_images": 32, "checked": 0, "matched": 0, "mismatched": 0,
        "can_stop_the_run": False}
    assert {"kind": "few usable reference predictions", "usable": 0, "ignored_test_images": 32,
            "mismatched": 0, "below": 20} in plan["warnings"]
    mixed = refs_in("calibration", 20) + every_one_wrong[:12]  # 20 usable and right, 12 test images wrong
    result = check(made, images=own, split="own", reference_predictions=mixed, **shrunk())
    assert result["checks"]["reference_predictions"]["ignored_test_images"] == 12
    assert result["checks"]["reference_predictions"]["can_stop_the_run"]


def test_twenty_usable_reference_predictions_can_stop_the_run_fewer_only_warn(made, own):
    stops(made, "Brokkr's FP32 says 'red', your code said 'blue'", images=own, split="own",
          reference_predictions=refs_in("calibration", 20, wrong=[0]), **shrunk())
    plan = check(made, images=own, split="own", reference_predictions=refs_in("calibration", 19, wrong=[0]),
                 **shrunk())
    assert plan["checks"]["reference_predictions"]["mismatched"] == 1
    assert any(w["kind"] == "few usable reference predictions" and w["mismatched"] == 1
               for w in plan["warnings"])
    assert not any("red/00000.png" in json.dumps(w) for w in plan["warnings"])  # counts only, no file names
    missing = refs_in("calibration", 8) + [{"file": "calibration/red/no_such_file.png", "top1": "red"}]
    stops(made, "calibration/red/no_such_file.png is not an image of the labelled folder", images=own,
          split="own", reference_predictions=missing, **shrunk())


def test_own_split_with_few_images_warns_and_names_small_classes(made, tmp_path):
    images = tmp_path / "images"
    write_images(images / "calibration", {"red": 80, "green": 80, "blue": 80}, seed=11)
    write_images(images / "test", {"red": 100, "green": 100, "blue": 10}, seed=12)
    plan = check(made, images=images, split="own", **shrunk())
    assert plan["split"]["mode"] == "own"
    assert plan["split"]["parts"]["test"]["per_class"] == [100, 100, 10]
    assert {"kind": "classes with few test images", "classes": ["blue"], "below": 20} in plan["warnings"]


def test_too_few_images_stop_the_run_naming_the_floor(made, tmp_path):
    images = write_images(tmp_path / "images", {c: 120 for c in CLASSES}, seed=13)
    stops(made, "conformal_calibration: 120 images; the floor is 200", images=images, **shrunk())


def test_the_plan_names_no_image_file_and_no_absolute_path(made):
    plan = check(made, calib=True)
    text = json.dumps(plan)
    names = [p.relative_to(made / "images").as_posix() for p in (made / "images").rglob("*.png")]
    names += [p.name for p in (made / "calib").rglob("*.png")]
    assert not any(n in text for n in names)
    for spelling in (str(made), str(made).replace("\\", "/"), str(made).replace("\\", "\\\\")):
        assert spelling not in text


def test_no_network_blocks_and_counts_every_attempt():
    with no_network() as attempts:
        with pytest.raises(NetworkBlocked):
            socket.create_connection(("example.com", 80), timeout=1)
        with pytest.raises(NetworkBlocked):
            socket.getaddrinfo("example.com", 80)
        s = socket.socket()
        try:
            with pytest.raises(NetworkBlocked):
                s.connect(("127.0.0.1", 9))
        finally:
            s.close()
    assert attempts == ["socket.create_connection", "socket.getaddrinfo", "socket.socket.connect"]
    assert socket.create_connection.__name__ == "create_connection"  # put back afterwards
    assert all(name not in vars(socket.socket) for name in ("connect", "connect_ex", "sendto"))  # inherited


def test_a_full_check_makes_no_connection_attempt(made):
    plan = check(made, **shrunk())
    assert plan["network"]["what"] == ("Python-level network connections blocked and counted; ONNX Runtime "
                                       "telemetry switched off") == NETWORK_WORDING
    assert plan["network"]["connection_attempts"] == 0
    assert plan["network"]["onnxruntime_telemetry_events"] == "switched off"


def test_the_last_line_says_what_a_failed_build_means_downstream(made):
    assert final_line(check(made, **shrunk())) == "Checks PASS."
    line = final_line(check(made, **shrunk("int8_rolled.onnx")))
    assert line.startswith("Checks PASS for the FP32 model and the images, but the supplied INT8 build "
                           "FAILED")
    assert "agrees with FP32 on 0.0% of 256 images (failed below 20%)" in line
    assert ("Downstream: FP32 numbers only; the INT8 rows will say \"INT8 build failed\" with this value, "
            "and no INT8 numbers (as for MobileNetV3-Small).") in line


def test_out_folder_inside_released_or_study_folders_is_refused(tmp_path):
    for inside in ("published/labels/x", "labels/x", "results/x"):
        assert check_out_folder(REPOSITORY / inside)
    assert check_out_folder(tmp_path / "out") == []


def command(*args):
    return subprocess.run([sys.executable, "-m", "brokkr_edge.cli", "test", *map(str, args)],
                          capture_output=True, text=True, cwd=REPOSITORY)


def test_the_command_checks_and_writes_the_run_plan(made, tmp_path):
    out = tmp_path / "out"
    ok = command("--config", write_settings(made, **shrunk()), "--images", made / "images", "--out", out)
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.splitlines()[-1] == "Checks PASS." and "not built yet" in ok.stdout
    assert f"network: {NETWORK_WORDING} (0 connection attempts)" in ok.stdout
    failed = command("--config", write_settings(made, **shrunk("int8_rolled.onnx")),
                     "--images", made / "images", "--out", tmp_path / "failed")
    assert failed.returncode == 0 and "INT8 build failed" in failed.stdout.splitlines()[-1]
    plan = json.loads((out / "run_plan.json").read_text(encoding="utf-8"))
    assert plan["kind"] == "brokkr-edge user run plan"
    bad = command("--config", write_settings(made, licences={}), "--images", made / "images",
                  "--out", tmp_path / "out2")
    assert bad.returncode == 2 and "Stopped" in bad.stderr and "licences.images is required" in bad.stderr
    assert not (tmp_path / "out2").exists()
    refused = REPOSITORY / "published" / "labels" / "made-up-user-label"
    inside = command("--config", write_settings(made), "--images", made / "images", "--out", refused)
    assert inside.returncode == 2 and "must not be inside" in inside.stderr and not refused.exists()
