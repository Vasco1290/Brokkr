"""What a user label's pages say (docs/label_schema.md, note of 6 October 2026; H's decisions of 10 October
2026), on the stand-in user label of tests/user_label_example.py. Its numbers mean nothing. Released study
labels must still render byte for byte (tests/test_published_labels.py)."""

import json

import pytest
from user_label_example import STUDY, changed, user_label

from brokkr_edge import label_render as LR
from brokkr_edge.label_render import to_html, to_markdown, unexplained_numbers

WARNINGS = [
    {"kind": "few images", "part": "test", "n_items": 300, "recommended": 2000},
    {"kind": "classes with few test images", "classes": ["blue"], "below": 20},
    {"kind": "low agreement with FP32", "agreement": 0.85, "n_items": 256, "warn_below": 0.9},
    {"kind": "few usable reference predictions", "usable": 12, "ignored_test_images": 3, "mismatched": 0,
     "below": 20},
]


@pytest.fixture(scope="module")
def user():
    return user_label()


@pytest.fixture(scope="module")
def warned(user):
    return changed(user, lambda lab: lab["summary"].update(warnings=WARNINGS))


def pages(label) -> list:
    return [to_markdown(label), to_html(label)]


def test_every_number_on_a_user_label_is_a_label_number(warned):
    for text in pages(warned):
        assert unexplained_numbers(warned, text) == []


def test_the_unverified_badge_is_on_both_pages(user):
    for text in pages(user):
        assert "User-submitted label — UNVERIFIED" in text


def test_each_warning_is_shown_under_the_badge_with_its_counts(warned):
    md = to_markdown(warned)
    block = md[md.index("## Warnings"):md.index("## Summary")]
    assert md.index("UNVERIFIED") < md.index("## Warnings")
    assert "the test part has 300 images; 2000 or more are recommended" in block
    assert "(fewer than 20 each): blue" in block
    assert "on 85.00% of 256 images (warning below 90%)" in block
    assert "12 could be checked (fewer than 20" in block and "3 named test images" in block
    assert "<h2>Warnings</h2>" in to_html(warned)


def test_no_warnings_no_warnings_section(user):
    for text in pages(user):
        assert "Warnings" not in text


def test_declared_licences_and_the_label_licence(user):
    md = to_markdown(user)
    licences = md[md.index("## Licences"):md.index("## Sources")]
    assert licences.count("(declared by the submitter, not checked by Brokkr)") == 4  # code, weights, images,
    assert "The images: CC0-1.0" in licences  # and the declaration that they were not used for training
    assert "The images were not used to train the model" in licences
    assert LR.USER_LABEL_LICENCE in licences
    assert "are Brokkr output" not in licences  # the study's licence line is not on a user label


def test_the_sanity_check_line_names_the_stated_figure(user):
    md = to_markdown(user)
    assert ("FP32 sanity check: clean top-1 75.00% on 5000 conformal-calibration images vs 76.00% stated by "
            "the submitter (300 images: my own validation images); allowed difference 5.00%: PASS.") in md
    assert LR.USER_TERMS["FP32 sanity check"] in md
    assert "figure torchvision publishes" not in md


def test_the_machine_kind_is_marked_as_stated(user):
    assert "(laptop, as stated by the submitter)" in to_markdown(user)


def test_a_label_made_outside_git_says_so(user):
    outside = changed(user, lambda lab: lab["generated"].update(commit=None, dirty=None))
    assert "at commit none (made outside a git checkout)" in to_markdown(outside)


@pytest.fixture(scope="module")
def failed():
    """MobileNetV3-Small's label (its INT8 build failed), rewritten as a user label."""
    return user_label("published/labels/mobilenet_v3_small/label.json")


def labelled(label) -> dict:
    return next(b for b in label["builds"] if b["role"] == "labelled")


def test_a_failed_supplied_build_may_be_broken_or_not_from_this_model(failed):
    def supplied(lab):
        labelled(lab)["recipe"] = {"method": "supplied by the submitter", "made_by": "a tool",
                                   "found": {"quantized_ops": 3}}
    md = to_markdown(changed(failed, supplied))
    assert "Do not use this INT8 build" in md
    assert "It may be broken, or not made from this FP32 model; Brokkr cannot tell which." in md
    assert "This recipe broke the model" not in md


def test_a_crashed_build_has_no_number(failed):
    check = "the INT8 build crashed, also with skip_symbolic_shape"

    def crashed(lab):
        labelled(lab)["failure"] = {"check": check, "value": None, "limit": None, "n_items": None,
                                    "split": None}
    md = to_markdown(changed(failed, crashed))
    assert f"FAILED: {check} |" in md
    assert f"it failed its build check ({check})" in md


def test_study_labels_keep_their_own_wording():
    md = to_markdown(json.loads(STUDY.read_text(encoding="utf-8")))
    assert "figure torchvision publishes" in md and "are Brokkr output" in md
    assert "as stated by the submitter" not in md and "Warnings" not in md
