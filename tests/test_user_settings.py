"""The settings file of a user's own model (brokkr_edge.user_settings; docs/user_models.md).

Everything here is made up for the tests; nothing is a result.
"""

import pytest
from user_made_up import settings, tiny_model

from brokkr_edge import user_settings
from brokkr_edge.user_settings import UserInputError, check_settings, load_settings


@pytest.fixture
def base(tmp_path):
    tiny_model(tmp_path / "fp32.onnx")
    tiny_model(tmp_path / "other.onnx")
    return tmp_path


def problems(base, **changes) -> list:
    return check_settings(settings(**changes), base)


def test_a_complete_settings_file_has_no_problems(base):
    assert problems(base) == []
    supplied = {"file": "other.onnx", "precision": "int8", "made_by": "a tool 1.0"}
    assert problems(base, shrunk_model=supplied) == []


@pytest.mark.parametrize("field", user_settings.LICENCE_FIELDS)
@pytest.mark.parametrize("value", [None, "", "  ", "unknown", "None", "N/A", "?"])
def test_every_licence_is_required_no_licence_no_run(base, field, value):
    licences = settings()["licences"]
    if value is None:
        del licences[field]
    else:
        licences[field] = value
    found = problems(base, licences=licences)
    assert any(f"licences.{field} is required" in p for p in found)


@pytest.mark.parametrize("declarations", [{}, {"images_not_used_to_train_the_model": False},
                                          {"images_not_used_to_train_the_model": "yes"}])
def test_the_training_declaration_is_required(base, declarations):
    assert any("declarations must be" in p for p in problems(base, declarations=declarations))


def test_a_missing_or_misspelt_field_is_named(base):
    data = settings()
    del data["outputs"]
    data["output"] = "logits"
    found = check_settings(data, base)
    assert "missing field 'outputs'" in found and any("unknown field 'output'" in p for p in found)


def test_only_int8_shrunk_builds_are_accepted_in_step_3(base):
    found = problems(base, shrunk_model={"file": "other.onnx", "precision": "fp16", "made_by": "a tool"})
    assert any("only INT8 builds" in p for p in found)
    found = problems(base, shrunk_model={"file": "other.onnx", "precision": "int8", "made_by": " "})
    assert any("made_by" in p for p in found)


def test_model_files_must_exist(base):
    assert any("fp32_model: file not found" in p for p in problems(base, fp32="missing.onnx"))
    found = problems(base, shrunk_model={"file": "missing.onnx", "precision": "int8", "made_by": "a tool"})
    assert any("shrunk_model.file: file not found" in p for p in found)


@pytest.mark.parametrize("change, words", [
    ({"crop": 9}, "crop cannot be larger"),
    ({"interpolation": "lanczos"}, "interpolation must be"),
    ({"std": [0.5, 0.0, 0.5]}, "every std must be above zero"),
    ({"mean": [0.5, 0.5]}, "mean must be three numbers"),
    ({"channel_order": "rgb"}, "channel_order"),
    ({"layout": "CHW"}, "layout"),
])
def test_preprocessing_mistakes_are_named(base, change, words):
    prep = {**settings()["preprocessing"], **change}
    assert any(words in p for p in problems(base, preprocessing=prep))


def test_class_names_must_be_different_and_usable_as_folder_names(base):
    assert any("must be different" in p for p in problems(base, classes=["red", "red", "blue"]))
    assert any("cannot be folder names" in p for p in problems(base, classes=["red", "a/b", "blue"]))
    assert any("at least two" in p for p in problems(base, classes=["red"]))


def test_expected_accuracy_is_a_fraction_with_its_image_count(base):
    found = problems(base, expected_accuracy={"top1": 91, "n_images": 0, "measured_on": ""})
    assert sum("expected_accuracy" in p for p in found) == 3
    assert any("expected_accuracy must be" in p for p in problems(base, expected_accuracy={"top1": 0.9}))


def test_raspberry_pi_5_only_on_a_machine_that_reports_one(base, tmp_path, monkeypatch):
    board = tmp_path / "model"
    monkeypatch.setattr(user_settings, "PI5_MODEL_FILE", board)
    assert any("does not report itself" in p for p in problems(base, device_kind="raspberry-pi-5"))
    board.write_text("Raspberry Pi 4 Model B Rev 1.4")
    assert any("does not report itself" in p for p in problems(base, device_kind="raspberry-pi-5"))
    board.write_text("Raspberry Pi 5 Model B Rev 1.0")
    assert problems(base, device_kind="raspberry-pi-5") == []
    assert any("device_kind must be" in p for p in problems(base, device_kind="phone"))


def test_reference_predictions_are_8_to_32_files_with_known_classes(base):
    refs = [{"file": f"red/{i:05d}.png", "top1": "red"} for i in range(8)]
    assert problems(base, reference_predictions=refs) == []
    assert any("8 to 32" in p for p in problems(base, reference_predictions=refs[:3]))
    wrong = refs[:7] + [{"file": "red/99999.png", "top1": "purple"}]
    assert any("top1 must be one of the classes" in p for p in problems(base, reference_predictions=wrong))
    twice = refs[:7] + refs[:1]
    assert any("each file may appear once" in p for p in problems(base, reference_predictions=twice))


def test_load_settings_raises_with_every_problem(base):
    bad = base / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(UserInputError, match="not valid JSON"):
        load_settings(bad)
    with pytest.raises(UserInputError) as raised:
        load_settings(base / "missing.json")
    assert "cannot read" in raised.value.problems[0]
