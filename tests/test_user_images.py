"""Listing and splitting a user's images (brokkr_edge.user_images; docs/user_models.md, section 2).

The split rules are tested on made-up lists of labels (small floors where a test needs them), the listing on
small folders of made-up pictures. Nothing here is a result.
"""

import shutil

import numpy as np
import pytest
from PIL import Image
from user_made_up import write_images, write_unlabelled

from brokkr_edge.user_images import (
    CAPS,
    FLOORS,
    RECOMMENDED,
    fingerprint,
    overlap,
    plan_split,
    scan_labelled,
    scan_unlabelled,
    stratified_pick,
)
from brokkr_edge.user_settings import UserInputError

SMALL = {"floors": {"conformal_calibration": 5, "test": 5},
         "recommended": {"conformal_calibration": 50, "test": 100}, "int8_n": 30}


def labels_for(per_class: list) -> list:
    """Labels in a made-up canonical order: classes interleaved, so the order is not by class."""
    labels = [c for c, n in enumerate(per_class) for _ in range(n)]
    return list(np.random.default_rng(5).permutation(labels))


def split(labels, parts=None, n_classes=None, mode="brokkr", calib=None, builds=True, **kw):
    kw = {**SMALL, **kw}
    return plan_split(labels, parts or [None] * len(labels), n_classes or max(labels) + 1, mode, calib,
                      builds, **kw)


def test_stratified_pick_takes_exactly_n_and_keeps_each_class_share():
    labels = labels_for([100, 50, 10])
    picked = stratified_pick(list(range(160)), labels, 32, np.random.default_rng(1))
    counts = np.bincount([labels[i] for i in picked], minlength=3)
    assert len(picked) == len(set(picked)) == 32 and list(counts) == [20, 10, 2]


def test_brokkr_split_is_deterministic_disjoint_and_a_third_per_class():
    labels = labels_for([90, 60, 30])
    a, b = split(labels), split(labels)
    assert a == b
    int8 = set(a["int8_calibration"]["indices"])
    conformal, test = set(a["conformal_calibration"]), set(a["test"])
    assert a["int8_calibration"]["source"] == "labelled" and len(int8) == 30
    assert not (int8 & conformal or int8 & test or conformal & test)
    assert int8 | conformal | test == set(range(len(labels)))
    rest = np.bincount([labels[i] for i in range(len(labels)) if i not in int8], minlength=3)
    assert list(np.bincount([labels[i] for i in conformal], minlength=3)) == [n // 3 for n in rest]


def test_own_split_keeps_the_users_test_images_and_calibrates_int8_from_calibration_only():
    labels = labels_for([40, 40, 40])
    parts = ["test" if i % 4 == 0 else "calibration" for i in range(len(labels))]
    plan = split(labels, parts, mode="own")
    assert plan["test"] == [i for i in range(len(labels)) if parts[i] == "test"]
    used = set(plan["int8_calibration"]["indices"]) | set(plan["conformal_calibration"])
    assert all(parts[i] == "calibration" for i in used)
    assert set(plan["int8_calibration"]["indices"]).isdisjoint(plan["conformal_calibration"])


def test_unlabelled_calibration_folder_gives_the_int8_images():
    labels = labels_for([40, 40, 40])
    exact = split(labels, calib=30)
    assert exact["int8_calibration"] == {"source": "calib-images", "indices": list(range(30))}
    more = split(labels, calib=100)
    assert more["int8_calibration"]["source"] == "calib-images"
    assert len(more["int8_calibration"]["indices"]) == 30
    assert more == split(labels, calib=100)  # seeded
    assert len(more["conformal_calibration"]) + len(more["test"]) == len(labels)  # no labelled image taken
    with pytest.raises(UserInputError, match="needs at least 30"):
        split(labels, calib=29)


def test_a_supplied_shrunk_build_needs_no_int8_images_and_refuses_an_unused_folder():
    labels = labels_for([40, 40, 40])
    assert split(labels, builds=False)["int8_calibration"] is None
    with pytest.raises(UserInputError, match="would not be used"):
        split(labels, builds=False, calib=30)


def test_too_few_labelled_images_for_int8_calibration_is_refused():
    with pytest.raises(UserInputError, match="INT8 calibration needs 30 images"):
        split(labels_for([5, 5, 5]))


def test_caps_leave_images_unused_and_count_them():
    labels = labels_for([60, 60, 60])
    plan = split(labels, builds=False, caps={"conformal_calibration": 20, "test": 40})
    assert len(plan["conformal_calibration"]) == 20 and len(plan["test"]) == 40
    assert plan["unused"] == {"conformal_calibration": 40, "test": 80}


def test_below_a_floor_the_run_stops_naming_the_count_and_the_floor():
    with pytest.raises(UserInputError) as raised:
        split(labels_for([3, 3, 3]), builds=False)
    assert any("conformal_calibration: 3 images; the floor is 5" in p for p in raised.value.problems)


def test_the_real_floors_are_200_and_200():
    assert FLOORS == {"conformal_calibration": 200, "test": 200}
    assert RECOMMENDED == {"conformal_calibration": 1000, "test": 2000}
    assert CAPS == {"conformal_calibration": 5000, "test": 10000}
    labels = labels_for([180, 180, 180])  # 180 conformal calibration (a third), 360 test
    with pytest.raises(UserInputError, match="conformal_calibration: 180 images; the floor is 200"):
        plan_split(labels, [None] * len(labels), 3, "brokkr", None, False)
    plan = plan_split(labels_for([201, 201, 201]), [None] * 603, 3, "brokkr", None, False)
    assert len(plan["conformal_calibration"]) == 201 and len(plan["test"]) == 402


def test_every_class_needs_an_image_in_each_part():
    labels = labels_for([40, 40, 2])  # class 2: no conformal-calibration image (a third of 2 is 0)
    with pytest.raises(UserInputError, match=r"no image of the classes numbered \[2\]"):
        split(labels, builds=False)
    with pytest.raises(UserInputError, match=r"no image of the classes numbered \[3\]"):
        split(labels_for([40, 40, 40]), n_classes=4, builds=False)  # a class folder with no image at all


def test_between_floor_and_recommended_the_label_warns():
    plan = split(labels_for([25, 25, 25]), builds=False)  # per class: 8 conformal calibration, 17 test
    kinds = {(w["kind"], w.get("part")) for w in plan["warnings"]}
    assert ("few images", "conformal_calibration") in kinds and ("few images", "test") in kinds
    few = next(w for w in plan["warnings"] if w["kind"] == "classes with few test images")
    assert few["below"] == 20 and few["classes"] == [0, 1, 2]
    plan = split(labels_for([30, 30, 30]), builds=False)  # 20 test images per class: no class named
    assert not any(w["kind"] == "classes with few test images" for w in plan["warnings"])
    plan = split(labels_for([300, 300, 300]), builds=False)
    assert not any(w["kind"] == "classes with few test images" for w in plan["warnings"])


def test_listing_finds_classes_counts_other_files_and_keeps_canonical_order(tmp_path):
    root = write_images(tmp_path / "images", {"red": 3, "green": 2, "blue": 2})
    (root / "red" / "notes.txt").write_text("not an image")
    (root / "green" / "deeper").mkdir()
    shutil.copy(root / "green" / "00000.png", root / "green" / "deeper" / "x.png")
    (root / "green" / "00000.png").unlink()
    listed = scan_labelled(root, ["red", "green", "blue"], "brokkr")
    rels = [e["rel"] for e in listed["images"]]
    assert rels == sorted(rels) and "green/deeper/x.png" in rels and len(rels) == 7
    assert listed["other_files"] == 1
    assert {e["label"] for e in listed["images"] if e["rel"].startswith("blue/")} == {2}


def test_duplicates_and_unreadable_images_stop_the_run(tmp_path):
    root = write_images(tmp_path / "images", {"red": 2, "green": 2, "blue": 2})
    shutil.copy(root / "red" / "00000.png", root / "blue" / "copy.png")
    (root / "green" / "broken.png").write_bytes(b"not a picture")
    with pytest.raises(UserInputError) as raised:
        scan_labelled(root, ["red", "green", "blue"], "brokkr")
    found = raised.value.problems
    assert any("unreadable image: green/broken.png" in p for p in found)
    assert any("identical files" in p and "blue/copy.png" in p and "red/00000.png" in p for p in found)


def test_class_folders_must_match_the_settings_file(tmp_path):
    root = write_images(tmp_path / "images", {"red": 2, "green": 2})
    (root / "purple").mkdir()
    Image.new("RGB", (4, 4)).save(root / "loose.png")
    with pytest.raises(UserInputError) as raised:
        scan_labelled(root, ["red", "green", "blue"], "brokkr")
    found = raised.value.problems
    assert any("no folder for the classes ['blue']" in p for p in found)
    assert any("not in the settings file's classes: ['purple']" in p for p in found)
    assert any("images outside a class folder: loose.png" in p for p in found)


def test_own_split_needs_exactly_calibration_and_test_folders(tmp_path):
    write_images(tmp_path / "images" / "calibration", {"red": 2, "green": 2, "blue": 2})
    with pytest.raises(UserInputError, match="exactly the folders calibration/ and test/"):
        scan_labelled(tmp_path / "images", ["red", "green", "blue"], "own")
    write_images(tmp_path / "images" / "test", {"red": 2, "green": 2, "blue": 2}, seed=1)
    listed = scan_labelled(tmp_path / "images", ["red", "green", "blue"], "own")
    assert {e["part"] for e in listed["images"]} == {"calibration", "test"}
    assert all(e["rel"].startswith(e["part"] + "/") for e in listed["images"])


def test_a_calibration_image_that_is_also_labelled_is_found(tmp_path):
    root = write_images(tmp_path / "images", {"red": 2, "green": 2, "blue": 2})
    calib = write_unlabelled(tmp_path / "calib", 4)
    shutil.copy(root / "green" / "00001.png", calib / "same.png")
    found = overlap(scan_labelled(root, ["red", "green", "blue"], "brokkr")["images"],
                    scan_unlabelled(calib)["images"])
    assert found == ["calibration image same.png is also in the labelled folder as green/00001.png"]


def test_the_fingerprint_names_the_folder_not_the_order_and_changes_with_any_file(tmp_path):
    root = write_images(tmp_path / "images", {"red": 2, "green": 2, "blue": 2})
    entries = scan_labelled(root, ["red", "green", "blue"], "brokkr")["images"]
    assert fingerprint(entries) == fingerprint(list(reversed(entries)))
    write_images(tmp_path / "images", {"red": 1}, seed=7)  # rewrites red/00000.png with other pixels
    changed = scan_labelled(root, ["red", "green", "blue"], "brokkr")["images"]
    assert fingerprint(changed) != fingerprint(entries)
