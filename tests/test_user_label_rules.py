"""The validator's rules for labels of a user's own model, and the guard that keeps them off the released
labels and the site (docs/label_schema.md, note of 6 October 2026; H's decisions of 10 October 2026). The
label checked is a stand-in (tests/user_label_example.py); its numbers mean nothing."""

import json
import re
import sys
from pathlib import Path

import pytest
from user_label_example import changed, fp32_sanity, user_label

from brokkr_edge.label_schema import WARNING_FIELDS, check_label, study_label_problems

RELEASED = sorted(Path("published/labels").glob("*/label.json"))


@pytest.fixture(scope="module")
def user():
    return user_label()


def problems(label) -> str:
    return " | ".join(check_label(label))


def test_the_stand_in_user_label_passes(user):
    assert check_label(user) == []


def test_every_released_label_still_passes():
    assert len(RELEASED) == 10
    for path in RELEASED:
        assert check_label(json.loads(path.read_text(encoding="utf-8"))) == [], path


def set_in(path: str, value):
    """A change that sets label[a][b]... = value, for a dotted path."""
    def change(label):
        *parents, last = path.split(".")
        target = label
        for key in parents:
            target = target[int(key)] if isinstance(target, list) else target[key]
        target[last] = value
    return change


BROKEN = [
    (set_in("model.publisher", "torchvision"), "model.publisher is 'user'"),
    (set_in("model.model_id", "user/made-up-colours@000000000000"), "first 12 hex characters"),
    (set_in("source.how_made", "by hand"), "brokkr-edge test"),
    (set_in("source.verified", True), "always unverified"),
    (set_in("generated.seed_scheme", "position-study"), "seed_scheme"),
    (set_in("generated.commit", None), "both null"),  # dirty stays false: not allowed outside git
    (set_in("datasets.0.dataset_id", "imagenet/imagenet-1k-val:test"), "user/made-up-colours:<part>"),
    (set_in("datasets.0.split_mode", "random"), "split_mode"),
    (set_in("datasets.0.folder_fingerprint", "abc"), "folder_fingerprint"),
    (set_in("licences.images", ""), "images licence"),
    (set_in("licences.declared_by_submitter", False), "declared_by_submitter"),
    (set_in("declarations", {}), "images_not_used_to_train_the_model"),
    (set_in("checks.fp32_sanity.pass", False), "pass differs from the rule"),
    (set_in("checks.fp32_sanity.half_width_99", 0.2), "half_width_99 differs"),
    (set_in("checks.fp32_sanity.published.source", "my notes"), "stated by the submitter"),
    (set_in("summary.warnings", [{"kind": "something else"}]), "unknown kind"),
    (set_in("summary.warnings", [{"kind": "few images", "part": "test", "n_items": 3000,
                                  "recommended": 2000}]), "below its recommended size"),
    (set_in("summary.warnings", [{"kind": "low agreement with FP32", "agreement": 0.95, "n_items": 256,
                                  "warn_below": 0.9}]), "below its line"),
    (set_in("summary.warnings", [{"kind": "few usable reference predictions", "usable": 5}]), "missing"),
    (set_in("speed.0.status", "measured"), "one 'not measured' speed row per build"),
    (set_in("speed.0.reason", "not measured in step 3"), "one 'not measured' speed row per build"),
    (lambda lab: lab["speed"].pop(), "one 'not measured' speed row per build"),
    (lambda lab: lab["hardware"].append(dict(lab["hardware"][0], hardware_id="desktop:x:y")),
     "exactly one machine"),
    (set_in("hardware.0.kind", "phone"), "a kind in"),
    (set_in("model.description", "trained from C:\\Users\\me\\data"), "no paths from the user's machine"),
    (set_in("model.description", "see /home/me/images"), "no paths from the user's machine"),
]


@pytest.mark.parametrize("change, words", BROKEN, ids=[w for _, w in BROKEN])
def test_each_user_rule_is_enforced(user, change, words):
    assert words in problems(changed(user, change))


ALLOWED = [
    set_in("hardware.0.kind", "desktop"),
    set_in("hardware.0.kind", "other"),
    lambda lab: lab["generated"].update(commit=None, dirty=None),  # outside a git checkout (decision 12)
    set_in("generated.dirty", True),  # a dirty checkout is recorded, not refused (decision 12)
    set_in("source.submitted_by", "a lab in Kyoto"),
    set_in("model.description", "weights from https://example.org/model, licence CC0"),
]


@pytest.mark.parametrize("change", ALLOWED)
def test_what_a_user_label_may_do(user, change):
    assert check_label(changed(user, change)) == []


def test_each_warning_kind_passes_with_its_fields(user):
    warnings = [
        {"kind": "few images", "part": "test", "n_items": 300, "recommended": 2000},
        {"kind": "classes with few test images", "classes": [2], "below": 20},
        {"kind": "low agreement with FP32", "agreement": 0.85, "n_items": 256, "warn_below": 0.9},
        {"kind": "few usable reference predictions", "usable": 12, "ignored_test_images": 3, "mismatched": 0,
         "below": 20},
    ]
    assert check_label(changed(user, set_in("summary.warnings", warnings))) == []


def test_every_warning_kind_the_checks_emit_is_allowed():
    """Decision 9: a new warning kind in slice 1's checks fails here until the label format allows it."""
    source = "".join(Path(f).read_text(encoding="utf-8")
                     for f in ("brokkr_edge/user_images.py", "brokkr_edge/user_plan.py"))
    emitted = set(re.findall(r'warnings\.append\(\{"kind": "([^"]+)"', source))
    assert emitted == set(WARNING_FIELDS)


def test_a_supplied_build_names_its_maker(user):
    def supplied(lab, made_by):
        labelled = next(b for b in lab["builds"] if b["role"] == "labelled")
        labelled["recipe"] = {"method": "supplied by the submitter", "made_by": made_by,
                              "found": {"quantized_ops": 3, "int8_weight_tensors": 2}}
    assert check_label(changed(user, lambda lab: supplied(lab, "a tool"))) == []
    assert "made_by" in problems(changed(user, lambda lab: supplied(lab, "")))


def test_a_check_that_fails_by_the_rule_must_say_fail(user):
    calibration = user["checks"]["fp32_sanity"]["dataset_id"]
    far = fp32_sanity(0.40, 0.76, 5000, 1000, calibration)  # 36 points away: outside the allowed gap
    assert far["pass"] is False
    assert check_label(changed(user, set_in("checks.fp32_sanity", far))) == []


def test_desktop_is_a_user_kind_only():
    study = json.loads(RELEASED[0].read_text(encoding="utf-8"))
    study["hardware"][0]["kind"] = "desktop"
    assert "a kind in" in problems(study)


def test_the_guard_refuses_a_user_record_behind_a_released_label():
    """Decision 1, read literally: records. scripts/40 runs it on every released label's source files."""
    assert study_label_problems([{"source": "brokkr"}, {"no": "source field"}]) == []
    assert "made from a user's run" in " ".join(study_label_problems([{"source": "user-submitted"}]))


def test_the_site_still_accepts_a_user_label(user):
    """The website plan (step 2) shows a user-submitted label marked unverified; tests/test_site.py checks
    the words on the pages. The guard of decision 1 is about records, which the site never reads."""
    sys.path.insert(0, str(Path("web").resolve()))
    import site_labels  # the site's own modules live in web/, as in tests/test_site.py

    assert site_labels.label_problems({"mine": user}) == []
