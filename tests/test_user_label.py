"""The label of a user's own model (brokkr_edge.user_label; docs/user_models.md, plan of 10 October 2026,
section c, tests 1-4, 6, 10 and 11), made end to end on the made-up model and pictures of
tests/user_made_up.py: the checks, the INT8 build, the conditions, then the label. Nothing here is a
result."""

import json
from pathlib import Path

import numpy as np
import pytest
from user_made_up import (
    CLASSES,
    PREP,
    ROLLED,
    made_up_seed_sensitivity,
    quantized,
    tiny_model,
    write_images,
    write_settings,
)

from brokkr_edge import user_build
from brokkr_edge.label_render import unexplained_numbers
from brokkr_edge.label_schema import check_label
from brokkr_edge.user_build import build_int8
from brokkr_edge.user_label import (
    SEED_SENSITIVITY,
    LabelError,
    check_label_sources,
    make_label,
    random_condition_sentence,
    user_limits,
    warning_limit,
    write_label,
)
from brokkr_edge.user_plan import check_user_inputs, rebuild_parts, write_plan
from brokkr_edge.user_run import FP32, INT8_BROKKR, INT8_SUPPLIED, run_conditions

PER_CLASS = 390
BADGE = "User-submitted label — UNVERIFIED"


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = tmp_path_factory.mktemp("user_label")
    write_images(d / "images", {c: PER_CLASS for c in CLASSES})
    tiny_model(d / "fp32.onnx")
    quantized(d / "fp32.onnx", d / "int8.onnx")
    quantized(tiny_model(d / "rolled.onnx", weights=ROLLED), d / "int8_rolled.onnx")  # agrees on no image
    tiny_model(d / "small.onnx", size=16)
    return d


def labelled_run(made, out, fp32="fp32.onnx", **changes) -> tuple:
    """The whole user path, as the command will run it: returns (plan, label)."""
    plan = check_user_inputs(write_settings(made, fp32=fp32, **changes), made / "images", log=quiet)
    write_plan(plan, out)
    parts = rebuild_parts(plan, made / "images")
    builds = {FP32: made / fp32}
    if "shrunk" in plan["models"]:
        if plan["models"]["shrunk"]["status"] == "usable":
            builds[INT8_SUPPLIED] = made / changes["shrunk_model"]["file"]
    elif build_int8(plan, made / fp32, parts, out, log=quiet)["status"] == "usable":
        builds[INT8_BROKKR] = out / "int8.onnx"
    run_conditions(plan, builds, parts, out, log=quiet)
    label = make_label(plan, out)
    write_label(label, out)
    return plan, label


def supplied(file: str) -> dict:
    return {"shrunk_model": {"file": file, "precision": "int8", "made_by": "made-up test build"}}


@pytest.fixture(scope="module")
def brokkr_built(made, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return out, *labelled_run(made, out)


@pytest.fixture(scope="module")
def supplied_ok(made, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return out, *labelled_run(made, out, **supplied("int8.onnx"))


@pytest.fixture(scope="module")
def supplied_failed(made, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return out, *labelled_run(made, out, **supplied("int8_rolled.onnx"))


RUNS = ["brokkr_built", "supplied_ok", "supplied_failed"]


def pages(out: Path) -> dict:
    names = ("label.json", "label.md", "label.html")
    return {name: (out / name).read_text(encoding="utf-8") for name in names}


# ---- tests 1-4, on Brokkr's own build and on a supplied pair (usable and failed) ----

@pytest.mark.parametrize("run", RUNS)
def test_1_the_written_label_passes_check_label(run, request):
    out, _, label = request.getfixturevalue(run)
    assert check_label(json.loads((out / "label.json").read_text(encoding="utf-8"))) == []
    assert label["source"]["kind"] == "user-submitted" and label["generated"]["seed_scheme"] == "content-v1"


@pytest.mark.parametrize("run", RUNS)
def test_2_both_pages_show_the_unverified_badge(run, request):
    out, _, _ = request.getfixturevalue(run)
    text = pages(out)
    assert BADGE in text["label.md"] and BADGE in text["label.html"]


@pytest.mark.parametrize("run", RUNS)
def test_3_every_number_on_the_pages_is_a_label_number(run, request):
    out, _, label = request.getfixturevalue(run)
    text = pages(out)
    assert unexplained_numbers(label, text["label.md"]) == []
    assert unexplained_numbers(label, text["label.html"]) == []


@pytest.mark.parametrize("run", RUNS)
def test_4_no_image_name_or_path_anywhere_in_out(run, request, made):
    out, _, _ = request.getfixturevalue(run)
    files = [out / "label.json", out / "label.md", out / "label.html", out / "run_plan.json",
             *(out / "records").glob("*.json")] + ([out / "int8_build.json"] if run == "brokkr_built" else [])
    for path in files:
        text = path.read_text(encoding="utf-8")
        for bad in (".png", "red/0", "red\\\\0", str(made), str(out), str(Path.home())):
            assert bad not in text, (path.name, bad)


# ---- what the label holds ----

def test_the_label_uses_the_study_s_rules_and_its_own_numbers(brokkr_built):
    _, _, label = brokkr_built
    assert len(label["conditions"]) == 13 and label["summary"]["tested_conditions"] == 12
    metrics = {m["metric"] for m in label["measurements"]}
    assert {"top1", "coverage", "mean_set_size", "ece", "e_aurc", "damage_drop", "shrinking_cost"} <= metrics
    coverage = [m for m in label["measurements"] if m["metric"] == "coverage"]
    assert all(m["settings"]["method"] == "LAC" and m["settings"]["target_coverage"] == 0.9 for m in coverage)
    assert label["envelope"]["rule"]["name"] == "reliability envelope, version 1"


def test_a_brokkr_build_records_its_recipe_and_a_supplied_one_its_maker(brokkr_built, supplied_ok):
    built = next(b for b in brokkr_built[2]["builds"] if b["role"] == "labelled")
    assert built["recipe"]["method"] == "percentile99.99" and built["recipe"]["calibration"]["n_items"] == 512
    given = next(b for b in supplied_ok[2]["builds"] if b["role"] == "labelled")
    assert given["recipe"]["method"] == "supplied by the submitter" and given["recipe"]["made_by"]


def test_6_a_failed_supplied_build_has_fp32_rows_only(supplied_failed):
    out, _, label = supplied_failed
    lab = next(b for b in label["builds"] if b["role"] == "labelled")
    assert lab["status"] == "failed" and lab["failure"]["value"] == 0.0
    assert not [m for m in label["measurements"] if m["build_id"] == lab["build_id"]]
    rows = [r for r in label["envelope"]["rows"] if r["build_id"] == lab["build_id"]]
    assert rows and all(r["state"] == "INT8 build failed" for r in rows)
    md = pages(out)["label.md"]
    assert "It may be broken, or not made from this FP32 model; Brokkr cannot tell which." in md
    assert "not measured: INT8 build failed" in md


def test_a_build_that_crashed_twice_gives_a_label_with_no_file(made, tmp_path, monkeypatch):
    """Decision 10: FP32 rows and "INT8 build failed"; no number, no file, no exception message."""
    def always(*_, **__):
        raise RuntimeError("C:\\Users\\someone\\private")

    monkeypatch.setattr(user_build, "to_int8", always)
    _, label = labelled_run(made, tmp_path)
    lab = next(b for b in label["builds"] if b["role"] == "labelled")
    assert lab["file"] is None and lab["failure"]["value"] is None
    md = (tmp_path / "label.md").read_text(encoding="utf-8")
    assert "no file: the build crashed" in md and "private" not in md


def test_a_model_below_32_pixels_has_five_conditions_and_says_why(made, tmp_path):
    small = {**PREP, "resize": 16, "crop": 16}
    _, label = labelled_run(made, tmp_path, fp32="small.onnx", preprocessing=small)
    assert [c["condition_id"] for c in label["conditions"]] == [
        "clean", "brokkr/fog/3", "brokkr/darkness/5", "brokkr/defocus-blur/3", "brokkr/noise/3"]
    limits = " ".join(label["limits"])
    assert "The 8 ImageNet-C conditions were not tested" in limits and "16 x 16" in limits
    assert "imagecorruptions" not in limits and "Brokkr's own corruptions)" in limits


def test_a_changed_score_file_stops_the_label(made, tmp_path):
    plan, _ = labelled_run(made, tmp_path)
    npz = next((tmp_path / "records").glob("fp32_test_none_clean*.npz"))
    with np.load(npz) as data:
        saved = dict(data)
    saved["logits"] = saved["logits"] + 1.0
    np.savez_compressed(npz, **saved)
    with pytest.raises(LabelError, match="scores changed"):
        make_label(plan, tmp_path)


# ---- test 10: the random-condition sentence ----

def test_10_the_random_condition_figure_is_read_and_rounded_up(tmp_path):
    record = made_up_seed_sensitivity(tmp_path / "s.json", [0.004, -0.0123, 0.01])
    sentence = random_condition_sentence(record)
    assert "(MobileNetV3-Large, 1,234 images)" in sentence and "by up to 1.3 points;" in sentence
    exact = random_condition_sentence(made_up_seed_sensitivity(tmp_path / "t.json", [-0.01]))
    assert "by up to 1.0 point;" in exact  # 0.01 is 1.0 point, not rounded up to 1.1 by float dust


def test_10_the_sentence_appears_only_when_a_random_condition_ran(brokkr_built, tmp_path):
    _, plan, label = brokkr_built
    hw = label["hardware"][0]
    sensitivity = made_up_seed_sensitivity(tmp_path / "s.json", [0.02])
    random_ran = user_limits(plan, hw, [], plan["conditions"], sensitivity)
    assert any(s.startswith("Fog and noise damage use a random pattern.") for s in random_ran)
    calm = [c for c in plan["conditions"] if c["corruption"] in ("clean", "darkness", "contrast")]
    assert not any("random pattern" in s for s in user_limits(plan, hw, [], calm, tmp_path / "missing.json"))


# ---- test 11: each warning in summary.warnings and as a limits sentence ----

def test_11_each_warning_of_the_run_is_also_a_limit(brokkr_built):
    _, _, label = brokkr_built
    warnings = label["summary"]["warnings"]
    assert warnings  # the made-up folder is small: "few images" for both parts
    for w in warnings:
        assert warning_limit(w) in label["limits"]


@pytest.mark.parametrize("warning, words", [
    ({"kind": "few images", "part": "test", "n_items": 300, "recommended": 2000}, "300 images (2,000"),
    ({"kind": "classes with few test images", "classes": ["blue"], "below": 20}, "fewer than 20 test images"),
    ({"kind": "low agreement with FP32", "agreement": 0.85, "n_items": 256, "warn_below": 0.9},
     "85.00% of 256"),
    ({"kind": "few usable reference predictions", "usable": 12, "ignored_test_images": 3, "mismatched": 0,
      "below": 20}, "Only 12 reference predictions"),
])
def test_11_every_warning_kind_has_its_sentence_with_its_counts(warning, words):
    assert words in warning_limit(warning)


def test_10_a_missing_seed_record_is_found_before_the_run(brokkr_built, tmp_path):
    """check_label_sources stops a plan with random damage when the record is missing; a plan without random
    damage does not need it."""
    _, plan, _ = brokkr_built
    with pytest.raises(LabelError, match="this copy of Brokkr does not have"):
        check_label_sources(plan, tmp_path / "missing.json")
    no_random = {**plan, "conditions": [c for c in plan["conditions"]
                                        if c["corruption"] not in ("fog", "noise", "gaussian_noise")]}
    assert len(no_random["conditions"]) == 7  # clean and the 6 conditions without a random pattern
    check_label_sources(no_random, tmp_path / "missing.json")


@pytest.mark.skipif(not SEED_SENSITIVITY.is_file(), reason="Brokkr's seed record is not on this machine")
def test_10_the_real_seed_record_gives_the_sentence():
    """The other tests use a made-up stand-in (tests/conftest.py); this one reads Brokkr's own record."""
    sentence = random_condition_sentence(SEED_SENSITIVITY)
    assert sentence.startswith("Fog and noise damage use a random pattern. In one exploratory check (")
