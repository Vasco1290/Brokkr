"""The settings file of a user's own model, and its checks (docs/user_models.md, sections 1, 3, 5 and 6).

A user describes their model in a small JSON file: the model files, the class names in output order, the
preprocessing, whether the model gives logits or probabilities, the accuracy they measured themselves, the
licences of the model and the images, and a declaration that the images were not used to train the model.

check_settings(data, base) returns every problem as a plain sentence (an empty list = the file is fine);
load_settings(path) reads the file and raises UserInputError if anything is wrong, so a run never starts on a
bad settings file. Paths in the file are relative to the settings file's folder.
"""

import json
import re
from pathlib import Path

SETTINGS_VERSION = 1
NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
INTERPOLATIONS = ("bilinear", "bicubic", "nearest")
CHANNEL_ORDERS = ("RGB", "BGR")
LAYOUTS = ("NCHW", "NHWC")
OUTPUTS = ("logits", "probabilities")
SPLITS = ("brokkr", "own")
DEVICE_KINDS = ("laptop", "desktop", "server", "raspberry-pi-5", "cloud-arm", "other")  # H, D11
LICENCE_FIELDS = ("model_code", "model_weights", "images", "images_source")
NOT_A_LICENCE = ("", "unknown", "none", "n/a", "?")
DECLARATION = "images_not_used_to_train_the_model"
REFERENCE_PREDICTIONS = (8, 32)  # fewest and most reference files (H, D7: optional)
REQUIRED = ("settings_version", "name", "fp32_model", "shrunk_model", "classes", "outputs", "preprocessing",
            "split", "expected_accuracy", "licences", "declarations", "device_kind")
OPTIONAL = ("reference_predictions", "submitted_by")
PREPROCESSING = ("resize", "crop", "interpolation", "mean", "std", "channel_order", "layout")
PI5_MODEL_FILE = Path("/proc/device-tree/model")


class UserInputError(Exception):
    """The user's inputs cannot be used. `problems` lists every reason, as plain sentences."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _text(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def reports_raspberry_pi_5() -> bool:
    """True only on a machine whose device tree names it a Raspberry Pi 5 (hard rule 2)."""
    try:
        return "Raspberry Pi 5" in PI5_MODEL_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _check_preprocessing(prep) -> list:
    if not isinstance(prep, dict):
        return ["preprocessing must be an object"]
    problems = [f"preprocessing: missing {k!r}" for k in PREPROCESSING if k not in prep]
    problems += [f"preprocessing: unknown field {k!r}" for k in prep if k not in PREPROCESSING]
    if problems:
        return problems
    for k in ("resize", "crop"):
        if not (_int(prep[k]) and prep[k] >= 1):
            problems.append(f"preprocessing: {k} must be a whole number of pixels")
    if not problems and prep["crop"] > prep["resize"]:
        problems.append("preprocessing: the crop cannot be larger than the resize")
    if prep["interpolation"] not in INTERPOLATIONS:
        problems.append(f"preprocessing: interpolation must be one of {', '.join(INTERPOLATIONS)}")
    for k in ("mean", "std"):
        v = prep[k]
        if not (isinstance(v, list) and len(v) == 3 and all(_number(x) for x in v)):
            problems.append(f"preprocessing: {k} must be three numbers (RGB order, pixel values 0 to 1)")
        elif k == "std" and not all(x > 0 for x in v):
            problems.append("preprocessing: every std must be above zero")
    if prep["channel_order"] not in CHANNEL_ORDERS:
        problems.append("preprocessing: channel_order must be RGB or BGR")
    if prep["layout"] not in LAYOUTS:
        problems.append("preprocessing: layout must be NCHW or NHWC")
    return problems


def _check_file(value, what: str, base: Path) -> list:
    if not _text(value):
        return [f"{what} must name a file"]
    if not (base / value).is_file():
        return [f"{what}: file not found ({value}, relative to the settings file)"]
    return []


def check_settings(data, base: Path) -> list:
    """Every way `data` (the parsed settings file) cannot be used, as sentences. Empty = fine."""
    if not isinstance(data, dict):
        return ["the settings file must hold one JSON object"]
    problems = [f"missing field {k!r}" for k in REQUIRED if k not in data]
    problems += [f"unknown field {k!r} (check its spelling)" for k in data if k not in REQUIRED + OPTIONAL]
    if problems:
        return problems
    if data["settings_version"] != SETTINGS_VERSION:
        problems.append(f"settings_version must be {SETTINGS_VERSION}")
    if not (isinstance(data["name"], str) and NAME.match(data["name"])):
        problems.append("name: lowercase letters, digits, '-', '_' and '.' only, starting with a letter "
                        "or digit")
    problems += _check_file(data["fp32_model"], "fp32_model", base)

    shrunk = data["shrunk_model"]
    if shrunk is not None:
        if not isinstance(shrunk, dict) or set(shrunk) != {"file", "precision", "made_by"}:
            problems.append('shrunk_model must be null or {"file", "precision", "made_by"}')
        else:
            problems += _check_file(shrunk["file"], "shrunk_model.file", base)
            if shrunk["precision"] != "int8":
                problems.append("shrunk_model.precision: only INT8 builds are accepted in step 3 "
                                "(FP16 pairs are parked)")
            if not _text(shrunk["made_by"]):
                problems.append("shrunk_model.made_by: name the tool (and its version) that made the build")

    classes = data["classes"]
    if not (isinstance(classes, list) and len(classes) >= 2 and all(_text(c) for c in classes)):
        problems.append("classes must list at least two class names, in the model's output order")
    else:
        if len(set(classes)) != len(classes):
            problems.append("classes: every class name must be different")
        bad = [c for c in classes if "/" in c or "\\" in c or c.strip() in (".", "..") or c != c.strip()]
        if bad:
            problems.append(f"classes: these names cannot be folder names: {bad}")

    if data["outputs"] not in OUTPUTS:
        problems.append('outputs must be "logits" or "probabilities"')
    problems += _check_preprocessing(data["preprocessing"])
    if data["split"] not in SPLITS:
        problems.append('split must be "brokkr" (Brokkr splits the folder) or "own" (calibration/ and test/)')

    expected = data["expected_accuracy"]
    if not (isinstance(expected, dict) and set(expected) == {"top1", "n_images", "measured_on"}):
        problems.append('expected_accuracy must be {"top1", "n_images", "measured_on"}')
    else:
        if not (_number(expected["top1"]) and 0 < expected["top1"] <= 1):
            problems.append("expected_accuracy.top1 must be a fraction above 0 and at most 1 (e.g. 0.91)")
        if not (_int(expected["n_images"]) and expected["n_images"] >= 1):
            problems.append("expected_accuracy.n_images must be the number of images it was measured on")
        if not _text(expected["measured_on"]):
            problems.append("expected_accuracy.measured_on must say which images it was measured on")

    licences = data["licences"]
    if not isinstance(licences, dict):
        problems.append("licences must be an object")
    else:
        for k in LICENCE_FIELDS:
            value = licences.get(k)
            if not isinstance(value, str) or value.strip().lower() in NOT_A_LICENCE:
                problems.append(f"licences.{k} is required: no licence recorded, no run (hard rule 8)")
        problems += [f"licences: unknown field {k!r}" for k in licences if k not in LICENCE_FIELDS]

    declarations = data["declarations"]
    if not (isinstance(declarations, dict) and set(declarations) == {DECLARATION}
            and declarations[DECLARATION] is True):
        problems.append(f'declarations must be {{"{DECLARATION}": true}}: images the model was trained on '
                        "would make every number look better than it is")

    if data["device_kind"] not in DEVICE_KINDS:
        problems.append(f"device_kind must be one of {', '.join(DEVICE_KINDS)}")
    elif data["device_kind"] == "raspberry-pi-5" and not reports_raspberry_pi_5():
        problems.append("device_kind raspberry-pi-5: this machine does not report itself as a Raspberry Pi 5")

    refs = data.get("reference_predictions")
    if refs is not None:
        low, high = REFERENCE_PREDICTIONS
        if not (isinstance(refs, list) and low <= len(refs) <= high
                and all(isinstance(r, dict) and set(r) == {"file", "top1"} for r in refs)):
            problems.append(f'reference_predictions must be null or {low} to {high} '
                            '{"file", "top1"} entries')
        else:
            if len({r["file"] for r in refs}) != len(refs):
                problems.append("reference_predictions: each file may appear once")
            if isinstance(classes, list):
                unknown = sorted({str(r["top1"]) for r in refs if r["top1"] not in classes})
                if unknown:
                    problems.append(f"reference_predictions: top1 must be one of the classes, not {unknown}")

    submitted_by = data.get("submitted_by")
    if submitted_by is not None and not _text(submitted_by):
        problems.append("submitted_by must be null or a name")
    return problems


def load_settings(path) -> tuple:
    """(settings, the settings file's folder). Raises UserInputError listing every problem."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise UserInputError([f"cannot read the settings file: {e}"]) from e
    except ValueError as e:
        raise UserInputError([f"the settings file is not valid JSON: {e}"]) from e
    problems = check_settings(data, path.parent)
    if problems:
        raise UserInputError(problems)
    return data, path.parent
