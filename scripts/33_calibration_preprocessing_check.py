"""Diagnostic: did each INT8 build's calibration images get exactly the preprocessing that model's
evaluation images got in the 4.1 sweep? (No model is run.)

Usage:  python scripts/33_calibration_preprocessing_check.py
Needs:  models/<model>_int8_percentile99.99.json and _int8.json build records, results/breadth/*_test_*.json,
        data/imagenet-1k/, and the git history (older code is read with `git show`)
Writes: results/checks/<model>_calibration_preprocessing.json (schema 2, kind "diagnostic": never a result)

For every model and each INT8 build it has (Percentile 99.99 for 4.1; MinMax "int8" for M2):
1. Settings: the preprocessing the build record states, the one every evaluation record states, and
   torchvision's (brokkr.export.preprocessing) must be the same.
2. Code: the evaluation functions (accuracy.open_image, resize_and_crop, normalize; export.preprocessing;
   sweep.build_caches, normalised) must have the same source at both sweep commits as today.
3. Behaviour: the 512 int8_calibration images preprocessed by the calibration code AS IT WAS AT THE
   BUILD COMMIT (brokkr/accuracy.py and brokkr/export.py read from git, called the way that commit's
   build script called them) must be bit-identical to today's evaluation path (resize_and_crop with the
   model's settings, then brokkr.sweep.normalised).
Prints PASS only if every check holds for every model.
"""

import ast
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import PIL

from brokkr import accuracy, sweep
from brokkr.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr.export import MODELS, preprocessing
from brokkr.fingerprint import machine_fingerprint
from brokkr.schema import condition, make_measurement, metric, save_measurement

SWEEP_COMMITS = ("5be2cb5", "fc965cd")  # the two commits the 4.1 test-split records come from
EVALUATION_FUNCTIONS = {"brokkr/accuracy.py": ("open_image", "resize_and_crop", "normalize"),
                        "brokkr/export.py": ("preprocessing",),
                        "brokkr/sweep.py": ("build_caches", "normalised")}
RUNTIME = {"name": "numpy and pillow (no model is run)", "version": f"numpy {np.__version__}, pillow "
           f"{PIL.__version__}", "execution_provider": "none", "threads": 1}


def source_at(commit: str | None, path: str) -> str:
    if commit is None:
        return Path(path).read_text(encoding="utf-8")
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def function_source(source: str, name: str) -> str | None:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    return None


def module_at(commit: str, path: str):
    """One file of the package as it was at `commit` (its own imports use today's package)."""
    module = types.ModuleType(f"{Path(path).stem}_at_{commit}")
    exec(compile(source_at(commit, path), f"{commit}:{path}", "exec"), module.__dict__)
    return module


# 2. The evaluation code is the same at both sweep commits as today.
code_same = True
for path, names in EVALUATION_FUNCTIONS.items():
    for name in names:
        today = function_source(source_at(None, path), name)
        for commit in SWEEP_COMMITS:
            same = function_source(source_at(commit, path), name) == today
            code_same &= same
            print(f"{path} {name}: at {commit} {'same as today' if same else 'DIFFERENT from today'}")

files = parquet_files("imagenet-1k-val")
positions = make_splits(count_images(files))["int8_calibration"]
raw = [image for image, _ in read_parquet_images(files, positions)]  # file bytes
print(f"{len(raw)} int8_calibration images")

eval_preps = {}
for p in Path("results/breadth").glob("*_test_*.json"):
    r = json.loads(p.read_text(encoding="utf-8"))
    stated = json.dumps(r["settings"]["preprocessing"], sort_keys=True)
    eval_preps.setdefault(r["model"]["name"], set()).add(stated)

machine = machine_fingerprint()
all_ok = code_same
for model in MODELS:
    prep = preprocessing(model)
    crops = np.stack([accuracy.resize_and_crop(accuracy.open_image(b), prep["resize"], prep["crop"],
                                               prep["interpolation"]) for b in raw])
    evaluated = sweep.normalised(crops)  # the sweep's evaluation path
    stated_eval = sorted(eval_preps[model])
    ok = stated_eval == [json.dumps(prep, sort_keys=True)]
    print(f"\n{model}: torchvision {prep}; evaluation records state {stated_eval}")
    builds = []
    for precision in ("int8_percentile99.99", "int8"):
        path = Path("models") / f"{model}_{precision}.json"
        if not path.exists():
            continue
        build = json.loads(path.read_text(encoding="utf-8"))
        commit = build["machine"]["git"]["commit"][:7]
        stated = (build.get("settings") or {}).get("preprocessing")
        old = module_at(commit, "brokkr/accuracy.py")
        if stated is not None:  # 4.1 builds: scripts/24 calls preprocess(image, resize, crop, interpolation)
            old_prep = module_at(commit, "brokkr/export.py").preprocessing(model)
            calibrated = np.stack([old.preprocess(old.open_image(b), old_prep["resize"], old_prep["crop"],
                                                  old_prep["interpolation"]) for b in raw])
            how = f"scripts/24 at {commit}: preprocess(image, {old_prep})"
            settings_same = stated == prep == old_prep
        else:  # Stage 1 (scripts/04) and Stage 3 (scripts/10) builds: preprocess(open_image(image)), defaults
            calibrated = np.stack([old.preprocess(old.open_image(b)) for b in raw])
            how = f"preprocess(image) with that commit's defaults, at {commit}"
            settings_same = True  # not stated in the record; the behaviour check below decides
        same = calibrated.dtype == evaluated.dtype and np.array_equal(calibrated, evaluated)
        largest = float(np.abs(calibrated.astype(np.float64) - evaluated).max())
        ok &= same and settings_same
        builds.append({"precision": precision, "build_commit": commit, "record_states": stated, "how": how,
                       "identical": bool(same), "largest_difference": largest})
        print(f"  {precision:<21} build {commit}, record states {stated}; {how}")
        print(f"    calibration arrays vs evaluation arrays: {'IDENTICAL' if same else 'DIFFERENT'} "
              f"(shape {calibrated.shape}, largest difference {largest})")
    print(f"  {'PASS' if ok else 'FAIL'} ({model})")
    all_ok &= ok

    spec = MODELS[model]
    record = make_measurement(
        "diagnostic", {"name": model, "weights": str(spec["weights"]), "licence": spec["licence"]},
        " and ".join(b["precision"] for b in builds) or "none", RUNTIME, "laptop", machine,
        {"dataset": "imagenet-1k-val", "split": "int8_calibration", "n_images": len(raw),
         "licence": DATASETS["imagenet-1k-val"]["licence"]},
        condition(),
        {"builds_checked": metric(len(builds)),
         "builds_identical": metric(sum(b["identical"] for b in builds)),
         "largest_difference": metric(max((b["largest_difference"] for b in builds), default=0.0)),
         "pass": metric(int(ok and code_same))},
        {"script": "scripts/33_calibration_preprocessing_check.py", "preprocessing": prep,
         "evaluation_records_state": stated_eval, "sweep_commits": list(SWEEP_COMMITS),
         "evaluation_code_same_at_sweep_commits": code_same,
         "note": "Diagnostic: calibration vs evaluation preprocessing. Not a result; no model is run."},
        derived_from=[{"file": f"models/{model}_{b['precision']}.json"} for b in builds])
    record["raw"] = {"builds": builds}
    save_measurement(record, Path("results/checks") / f"{model}_calibration_preprocessing.json")

print(f"\n{'PASS' if all_ok else 'FAIL'}: calibration preprocessing equals evaluation preprocessing for "
      f"{'every' if all_ok else 'NOT every'} model; records in "
      "results/checks/*_calibration_preprocessing.json")
sys.exit(0 if all_ok else 1)
