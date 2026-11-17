"""Brokkr's INT8 build of a user's model (brokkr_edge.user_build; docs/user_models.md, plan of 10 October
2026, tests 7 and 8), on the made-up model and pictures of tests/user_made_up.py. Nothing here is a result."""

import json

import numpy as np
import pytest
from user_made_up import CLASSES, SIZE, tiny_model, write_images, write_settings, write_unlabelled

from brokkr_edge import user_build
from brokkr_edge.quantize import to_int8
from brokkr_edge.results import sha256_of
from brokkr_edge.user_plan import check_user_inputs, rebuild_parts
from brokkr_edge.user_settings import UserInputError

PER_CLASS = 390  # enough for 512 INT8 images, then at least 200 conformal-calibration and 200 test images


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = tmp_path_factory.mktemp("user_build")
    write_images(d / "images", {c: PER_CLASS for c in CLASSES})
    write_unlabelled(d / "calib", 520)
    tiny_model(d / "fp32.onnx")
    tiny_model(d / "fp32_batch_one.onnx", batch=1)
    return d


def plan_of(made, calib=False, **changes) -> dict:
    calib_images = made / "calib" if calib else None
    return check_user_inputs(write_settings(made, **changes), made / "images", calib_images, log=quiet)


@pytest.fixture(scope="module")
def plan(made):
    return plan_of(made)


@pytest.fixture(scope="module")
def parts(made, plan):
    return rebuild_parts(plan, made / "images")


def build(made, plan, parts, out):
    return user_build.build_int8(plan, made / "fp32.onnx", parts, out, log=quiet)


# ---- test 8: to_int8's input name ----

def test_to_int8_gives_the_same_file_with_and_without_input_name_images(tmp_path):
    model = tiny_model(tmp_path / "m.onnx")
    rng = np.random.default_rng(0)
    batches = [rng.uniform(-1, 1, (32, 3, SIZE, SIZE)).astype(np.float32) for _ in range(8)]
    a = to_int8(model, tmp_path / "a.onnx", batches, method="percentile99.99", group_batches=4)
    b = to_int8(model, tmp_path / "b.onnx", batches, method="percentile99.99", group_batches=4,
                input_name="images")
    assert sha256_of(a) == sha256_of(b)


def test_to_int8_builds_a_model_with_another_input_name(tmp_path):
    model = tiny_model(tmp_path / "m.onnx", input_name="pixels")
    batches = [np.zeros((32, 3, SIZE, SIZE), np.float32)] * 4
    with pytest.raises(Exception):  # noqa: B017 (onnxruntime's own error type)
        to_int8(model, tmp_path / "a.onnx", batches, method="percentile99.99", group_batches=4)
    assert to_int8(model, tmp_path / "b.onnx", batches, method="percentile99.99", group_batches=4,
                   input_name="pixels").is_file()


# ---- the parts, remade from the folders ----

def test_the_parts_are_remade_as_the_plan_describes_them(plan, parts):
    for name in ("int8_calibration", "conformal_calibration", "test"):
        assert len(parts[name]) == plan["split"]["parts"][name]["n_items"]
    paths = [e["path"] for name in parts for e in parts[name]]
    assert len(set(paths)) == len(paths)  # no image in two parts


def test_a_changed_folder_stops_the_run(made, plan, tmp_path):
    changed = tmp_path / "images"
    write_images(changed, {c: PER_CLASS for c in CLASSES}, seed=7)  # other pictures under the same names
    with pytest.raises(UserInputError, match="images folder changed"):
        rebuild_parts(plan, changed)
    with pytest.raises(UserInputError, match="leave it out"):
        rebuild_parts(plan, made / "images", made / "calib")


# ---- test 7: Brokkr's own INT8 build of the made-up model ----

@pytest.fixture(scope="module")
def built(made, plan, parts, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return out, build(made, plan, parts, out)


def test_the_build_follows_the_study_recipe(built):
    _, record = built
    r = record["recipe"]
    assert (r["calibration"], r["calibration_images"], r["batch"], r["group_images"]) == (
        "percentile99.99", 512, 32, 128)
    assert (r["weights"], r["activations"], r["skip_symbolic_shape"]) == (
        "int8, per-channel", "uint8, per-tensor", False)
    assert record["attempts"] == [{"skip_symbolic_shape": False, "error": None}]
    assert record["calibration"]["n_items"] == 512 and record["calibration"]["source"] == "labelled"


def test_the_build_is_checked_and_recorded(built, plan):
    out, record = built
    assert record["status"] == "usable" and record["failure"] is None
    assert record["checks"]["weights"]["pass"] and record["checks"]["outputs"]["pass"]
    # the first 256 conformal-calibration images, or all of them when there are fewer (219 here)
    conformal = plan["split"]["parts"]["conformal_calibration"]["n_items"]
    assert record["checks"]["agreement_with_fp32"]["n_items"] == min(256, conformal)
    assert record["file"]["sha256"] == sha256_of(out / "int8.onnx")
    assert json.loads((out / "int8_build.json").read_text(encoding="utf-8")) == record


def test_the_record_names_no_image_and_no_path(built, made):
    out, _ = built
    text = (out / "int8_build.json").read_text(encoding="utf-8")
    assert ".png" not in text and str(made) not in text and str(out) not in text
    assert "\\" not in text.replace("\\\\", "")  # no Windows path


def test_a_resumed_run_reuses_the_build(made, plan, parts, built, monkeypatch):
    out, record = built
    monkeypatch.setattr(user_build, "to_int8", lambda *a, **k: pytest.fail("the build ran again"))
    assert build(made, plan, parts, out) == record


def test_the_build_makes_its_out_folder(made, plan, parts, tmp_path):
    """Found 17 November 2026: a fresh --out made both attempts crash, as the folder did not exist yet."""
    record = build(made, plan, parts, tmp_path / "new" / "out")
    assert record["status"] == "usable" and record["attempts"][0]["error"] is None


def test_a_crash_is_retried_with_skip_symbolic_shape(made, plan, parts, tmp_path, monkeypatch):
    real = user_build.to_int8

    def crashes_first(*args, skip_symbolic_shape=False, **kwargs):
        if not skip_symbolic_shape:
            raise RuntimeError("C:\\Users\\someone\\secret path")  # a message that must not be kept
        return real(*args, skip_symbolic_shape=skip_symbolic_shape, **kwargs)

    monkeypatch.setattr(user_build, "to_int8", crashes_first)
    record = build(made, plan, parts, tmp_path)
    assert record["status"] == "usable" and record["recipe"]["skip_symbolic_shape"] is True
    assert record["attempts"] == [{"skip_symbolic_shape": False, "error": "RuntimeError"},
                                  {"skip_symbolic_shape": True, "error": None}]
    assert "secret path" not in (tmp_path / "int8_build.json").read_text(encoding="utf-8")


def test_a_build_that_crashes_twice_fails_with_no_number(made, plan, parts, tmp_path, monkeypatch):
    def always(*_, **__):
        raise ValueError("no")

    monkeypatch.setattr(user_build, "to_int8", always)
    record = build(made, plan, parts, tmp_path)
    assert record["status"] == "failed" and record["file"] is None
    assert record["failure"] == {"check": user_build.CRASHED, "value": None, "limit": None, "n_items": None,
                                 "split": None}
    assert not (tmp_path / "int8.onnx").exists() and not list(tmp_path.glob("*_prep.onnx"))


def test_calibration_images_from_the_calib_folder(made, tmp_path):
    plan = plan_of(made, calib=True)
    parts = rebuild_parts(plan, made / "images", made / "calib")
    record = build(made, plan, parts, tmp_path)
    assert record["calibration"]["source"] == "calib-images" and record["status"] == "usable"


def test_a_model_with_a_fixed_batch_of_one_is_fed_one_image_at_a_time(made, tmp_path):
    plan = plan_of(made, fp32="fp32_batch_one.onnx")
    parts = rebuild_parts(plan, made / "images")
    record = user_build.build_int8(plan, made / "fp32_batch_one.onnx", parts, tmp_path, log=quiet)
    recipe = record["recipe"]
    assert (recipe["batch"], recipe["group_images"], record["status"]) == (1, 128, "usable")
