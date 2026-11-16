"""A stand-in user label for the tests of the validator and the renderer (tests only; never a result).

It is a released study label rewritten into the shape of a user label (docs/label_schema.md, note of
6 October 2026; H's decisions of 10 October 2026). Its numbers are the study label's and mean nothing here;
the label of a real user run is made by the user path (step B, later piece) and tested on the made-up model.
"""

import copy
import json
import math
from pathlib import Path

STUDY = Path("published/labels/mobilenet_v3_large/label.json")
NAME = "made-up-colours"
SPEED_REASON = "this version of brokkr-edge test does not measure speed"


def fp32_sanity(measured: float, stated: float, n_items: int, n_classes: int, dataset_id: str) -> dict:
    """The expected-accuracy check as a user label stores it (decision 5), worked out by the stated rule."""
    half_width = 2.576 * math.sqrt(stated * (1 - stated) / n_items)
    allowed = max(0.05, half_width)
    ci95 = [measured - 0.01, measured + 0.01]
    return {"measured": measured, "ci95": ci95, "n_items": n_items, "dataset_id": dataset_id,
            "published": {"value": stated, "source": "stated by the submitter", "n_images": 300,
                          "measured_on": "my own validation images"},
            "tolerance": 0.05, "half_width_99": half_width, "allowed_gap": allowed, "chance": 1 / n_classes,
            "pass": abs(measured - stated) <= allowed and ci95[0] > 1 / n_classes}


def user_label(study_path=STUDY) -> dict:
    study = json.loads(Path(study_path).read_text(encoding="utf-8"))
    sha = next(b for b in study["builds"] if b["role"] == "reference")["file"]["sha256"]
    model_id = f"user/{NAME}@{sha[:12]}"
    text = json.dumps(study).replace(study["model"]["model_id"], model_id)
    for d in study["datasets"]:
        text = text.replace(d["dataset_id"], f"user/{NAME}:{d['dataset_id'].split(':')[1]}")
    label = json.loads(text)

    label["source"] = {"kind": "user-submitted", "verified": False, "submitted_by": None,
                       "how_made": "brokkr-edge test 0.0.1"}
    label["model"]["publisher"] = "user"
    label["generated"]["seed_scheme"] = "content-v1"
    for d in label["datasets"]:
        d.update(split_mode="brokkr", folder_fingerprint="ab" * 32)
    label["licences"].update(images="CC0-1.0", declared_by_submitter=True)
    label["declarations"] = {"images_not_used_to_train_the_model": True}
    calibration = next(d["dataset_id"] for d in label["datasets"] if "conformal" in d["dataset_id"])
    label["checks"]["fp32_sanity"] = fp32_sanity(0.75, 0.76, 5000, label["model"]["outputs"]["n_classes"],
                                                 calibration)
    hardware_id = label["hardware"][0]["hardware_id"]
    label["speed"] = [{"build_id": b["build_id"], "hardware_id": hardware_id, "status": "not measured",
                       "reason": SPEED_REASON, "runtime_id": None} for b in label["builds"]]
    label["summary"]["warnings"] = []
    return label


def changed(label: dict, change) -> dict:
    """A deep copy of `label` with `change(copy)` applied."""
    new = copy.deepcopy(label)
    change(new)
    return new
