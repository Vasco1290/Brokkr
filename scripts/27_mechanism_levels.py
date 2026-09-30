"""Task 4.0, the mechanism test (M1): do low-contrast images use fewer INT8 levels?

Usage:  python scripts/27_mechanism_levels.py
Needs:  models/mobilenet_v3_large_int8.onnx (default, MinMax) and ..._int8_percentile99.99.onnx
Writes: results/levels/mobilenet_v3_large_<precision>_imagenet-1k-val_tuning_<condition>_levels.json
        (six schema-2 "levels" records) and results/final/mobilenet_v3_large_m1_verdict.json

Design and verdict rules: docs/hypotheses_stage4.md, "Task 4.0" and the dated note "how M1 is
computed", both committed before this script ran. First 500 tuning images (never test images);
conditions clean, darkness (Brokkr) s5, fog (Brokkr) s3. For every image and each of the 142
activation QuantizeLinear outputs, count the distinct 8-bit levels used; per image, take the median
over the 142. Then, paired over the same images (new minus old):
  (a) damaged minus clean, default INT8   (b) Percentile minus default, on the damaged images.
SUPPORTS if (a) < 0 with its interval below 0 and (a) <= -20% of default's clean value, and (b) > 0
with its interval above 0. REJECTS if (a)'s interval includes 0 or lies above 0. Otherwise INCONCLUSIVE.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from brokkr_edge.accuracy import bootstrap_ci, open_image, paired_bootstrap_diff, resize_and_crop
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.export import MODELS, file_info, preprocessing
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.levels import activation_quantize_outputs, distinct_levels, expose
from brokkr_edge.results import make_record, save_record
from brokkr_edge.schema import condition, condition_label, make_measurement, metric, save_measurement
from brokkr_edge.sweep import damaged_batch, normalised

MODEL, DATASET, N_IMAGES, BATCH, THREADS = "mobilenet_v3_large", "imagenet-1k-val", 500, 32, 4
PRECISIONS = {"default": "int8", "percentile": "int8_percentile99.99"}
CONDITIONS = [condition(), condition("darkness", "brokkr", 5), condition("fog", "brokkr", 3)]
REDUCTION = 0.20  # (a) must be at least 20% of the clean value (pre-registered)

prep = preprocessing(MODEL)
files = parquet_files(DATASET)
positions = make_splits(count_images(files))["tuning"][:N_IMAGES]
crops = np.stack([resize_and_crop(open_image(b), prep["resize"], prep["crop"], prep["interpolation"])
                  for b, _ in read_parquet_images(files, positions)])

paths = {k: Path("models") / f"{MODEL}_{p}.onnx" for k, p in PRECISIONS.items()}
names = {k: activation_quantize_outputs(onnx.load(str(p))) for k, p in paths.items()}
if not (len(names["default"]) == 142 and names["default"] == names["percentile"]):
    sys.exit(f"STOP: expected the same 142 activation tensors in both models, got "
             f"{len(names['default'])} and {len(names['percentile'])}")
tensors = names["default"]

per_image, per_tensor, check = {}, {}, {}
with tempfile.TemporaryDirectory() as tmp:
    for key, path in paths.items():
        plain = make_session(path, THREADS)
        exposed = make_session(expose(path, Path(tmp) / f"{key}.onnx", tensors), THREADS)
        for cond in CONDITIONS:
            levels = np.zeros((N_IMAGES, len(tensors)), dtype=np.int16)
            same_top, max_diff = True, 0.0
            for start in range(0, N_IMAGES, BATCH):
                idx = list(range(start, min(start + BATCH, N_IMAGES)))
                batch = normalised(damaged_batch(crops, positions, idx, cond["suite"], cond["corruption"],
                                                 cond["severity"]))
                outputs = exposed.run(None, {"images": batch})
                for t in range(len(tensors)):
                    levels[idx, t] = distinct_levels(outputs[1 + t])
                if cond["corruption"] == "clean":  # the check that exposing outputs changed nothing
                    reference = plain.run(None, {"images": batch})[0]
                    same_top &= bool(np.array_equal(reference.argmax(1), outputs[0].argmax(1)))
                    max_diff = max(max_diff, float(np.abs(reference - outputs[0]).max()))
            if cond["corruption"] == "clean":
                check[key] = {"same_top_answer_on_all_500": same_top, "largest_score_difference": max_diff}
                print(f"check, {key}: same top answer on all {N_IMAGES} clean images: {same_top}; "
                      f"largest score difference {max_diff:.3g}")
                if not same_top:
                    sys.exit("STOP: exposing the tensors changed the model's answers; nothing is judged")
            label = condition_label(cond)
            per_image[(key, label)] = np.median(levels, axis=1).astype(np.float64)
            per_tensor[(key, label)] = levels.mean(axis=0)

# Results: one schema-2 "levels" record per (model, condition).
machine = machine_fingerprint()
spec = MODELS[MODEL]
for key, precision in PRECISIONS.items():
    for cond in CONDITIONS:
        label = condition_label(cond)
        values = per_image[(key, label)]
        record = make_measurement(
            "levels", {"name": MODEL, "weights": str(spec["weights"]), "licence": spec["licence"]}, precision,
            {"name": "onnxruntime", "version": ort.__version__, "execution_provider": "CPUExecutionProvider",
             "threads": THREADS},
            "laptop", machine,
            {"dataset": DATASET, "split": "tuning", "n_images": N_IMAGES,
             "licence": DATASETS[DATASET]["licence"]},
            cond,
            {"mean_levels_per_image": metric(float(values.mean()), bootstrap_ci(values)),
             "median_levels_per_image": metric(float(np.median(values)))},
            {"script": "scripts/27_mechanism_levels.py", "tensors": len(tensors),
             "per_image_value": "median over the 142 activation tensors of the distinct 8-bit levels used",
             "preprocessing": prep, "damage_seed": "dataset position of each image", "batch_size": BATCH},
            derived_from=[{"file": str(paths[key]), "sha256": file_info(paths[key])["sha256"]}],
        )
        means = per_tensor[(key, label)].round(2).tolist()
        record["raw"] = {"mean_levels_per_tensor": dict(zip(tensors, means, strict=True))}
        name = f"{MODEL}_{precision}_{DATASET}_tuning_{cond['corruption']}_s{cond['severity']}_levels.json"
        save_measurement(record, Path("results/levels") / name)

# Verdicts, by the pre-registered rule.
print(f"\nMean over {N_IMAGES} tuning images of each image's median levels used (95% interval):")
for key in PRECISIONS:
    for cond in CONDITIONS:
        values = per_image[(key, condition_label(cond))]
        low, high = bootstrap_ci(values)
        print(f"  {key:<11} {condition_label(cond):<22} {values.mean():7.2f} ({low:.2f} to {high:.2f})")

verdicts, all_ok = {}, True
clean = condition_label(CONDITIONS[0])
for cond in CONDITIONS[1:]:
    label = condition_label(cond)
    clean_default = per_image[("default", clean)]
    a = paired_bootstrap_diff(clean_default, per_image[("default", label)])
    b = paired_bootstrap_diff(per_image[("default", label)], per_image[("percentile", label)])
    a_needed = -REDUCTION * clean_default.mean()
    supports = a[0] < 0 and a[2] < 0 and a[0] <= a_needed and b[0] > 0 and b[1] > 0
    rejects = a[2] >= 0
    verdict = "SUPPORTS" if supports else ("REJECTS" if rejects else "INCONCLUSIVE")
    extra = paired_bootstrap_diff(per_image[("percentile", clean)], per_image[("percentile", label)])
    verdicts[label] = {
        "verdict": verdict,
        "a_damaged_minus_clean_default": {"diff": a[0], "ci95": [a[1], a[2]]},
        "a_needed_for_supports": a_needed,
        "b_percentile_minus_default_damaged": {"diff": b[0], "ci95": [b[1], b[2]]},
        "reported_not_judged_percentile_damaged_minus_clean": {"diff": extra[0],
                                                               "ci95": [extra[1], extra[2]]},
    }
    print(f"\nM1, {label}: {verdict}")
    print(f"  (a) damaged minus clean, default INT8: {a[0]:+.2f} levels (95% interval {a[1]:+.2f} to "
          f"{a[2]:+.2f}); supports needs <= {a_needed:+.2f} with the interval below 0")
    print(f"  (b) Percentile minus default, damaged: {b[0]:+.2f} levels (95% interval {b[1]:+.2f} to "
          f"{b[2]:+.2f}); supports needs > 0 with the interval above 0")

verdict_record = make_record("verdicts", MODEL, "int8 and int8_percentile99.99", {
    "settings": {"prediction": "M1 (docs/hypotheses_stage4.md)", "split": "tuning", "n_images": N_IMAGES,
                 "bootstrap_resamples": 1000, "seed": 0, "reduction_needed": REDUCTION,
                 "script": "scripts/27_mechanism_levels.py"},
    "metrics": {"verdicts": verdicts, "exposure_check": check},
}, machine)
print(f"\nSaved {save_record(verdict_record, Path('results/final') / f'{MODEL}_m1_verdict.json')}")
