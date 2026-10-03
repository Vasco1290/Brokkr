"""How much does an INT8 model depend on WHICH calibration images it got? (Stage 3 noise floor)

Usage:  python scripts/14_int8_calibration_luck.py [--model NAME] [--seeds 6 7 8]
Needs:  models/<model>_fp32.onnx, results/choices/<model>_int8_method.json (scripts/11), the chosen
        model's clean tuning result (scripts/03), the tuning image cache (scripts/08 --split tuning)
        and data/imagenet-1k/
Writes: models/<model>_<best>_calibseed<seed>.onnx (+ .json record) per seed, and
        results/checks/<model>_<best>_calibration_luck.json (+ .npz with the new models' tuning scores)

The chosen INT8 method (Percentile 99.99) is rebuilt with other random 512-image calibration sets,
drawn from the 29,488 images that belong to no split (brokkr_edge.datasets.unassigned), one set per seed.
Everything else is identical: same method, groups of 128, per-channel weights. Each new model is
checked before it is kept, and the run stops at the first failure.

Reported on the clean tuning split, for the original model plus the new ones: top-1 and E-AURC, and
their spread (largest minus smallest, and the standard deviation). In the final run (task 3.7), a
difference between INT8 variants smaller than this spread is also labelled "within noise"
(docs/hypotheses_stage3.md). This adds an analysis; it changes no prediction.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from brokkr_edge.accuracy import normalize, open_image, preprocess
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import (
    DATASETS,
    count_images,
    parquet_files,
    read_parquet_images,
    unassigned,
)
from brokkr_edge.export import MODELS, file_info
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.quantize import CALIBRATION_BATCH as BATCH
from brokkr_edge.quantize import CALIBRATION_GROUP_BATCHES as GROUP_BATCHES
from brokkr_edge.quantize import INT8_METHODS, INT8_SETTINGS, check_int8_build, to_int8, weight_quantization
from brokkr_edge.results import load_arrays, make_record, save_arrays, save_record
from brokkr_edge.shift.reliability import confidence_and_correct
from brokkr_edge.shift.selective import aurc, optimal_aurc

DATASET = "imagenet-1k-val"
N_CALIBRATION = 512

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8])
args = parser.parse_args()

fp32_path = Path("models") / f"{args.model}_fp32.onnx"
choice_path = Path("results/choices") / f"{args.model}_int8_method.json"
crops_path = Path("data/cache") / f"{DATASET}_tuning_crops.npy"
for path in (fp32_path, choice_path, crops_path):
    if not path.exists():
        sys.exit(f"FAIL: {path} not found (see Needs in this script's docstring)")
choice = json.loads(choice_path.read_text())["metrics"]
best, method = choice["chosen"], choice["chosen_method"]
best_path = Path("models") / f"{args.model}_{best}.onnx"
original = load_arrays(Path("results/accuracy") / f"{args.model}_{best}_{DATASET}_tuning.json")

files = parquet_files(DATASET)
total = count_images(files)
free = unassigned(total)
crops = np.load(crops_path, mmap_mode="r")
check = np.stack([normalize(c) for c in crops[:256]])
fp32_top1 = make_session(fp32_path, 4).run(None, {"images": check})[0].argmax(1)
expected_weights = weight_quantization(best_path)
machine = machine_fingerprint()


def tuning_logits(path):
    session = make_session(path, num_threads=4)
    batches = (np.stack([normalize(c) for c in crops[i:i + BATCH]]) for i in range(0, len(crops), BATCH))
    return np.concatenate([session.run(None, {"images": b})[0] for b in batches]).astype(np.float32)


def top1_and_eaurc(logits, labels):
    confidence, correct = confidence_and_correct(logits, labels)
    return float(correct.mean()), aurc(confidence, correct) - optimal_aurc(correct)


builds = {"int8_calibration split (seed 1)": top1_and_eaurc(original["logits"], original["labels"])}
new_logits = {}
for seed in args.seeds:
    positions = np.sort(np.random.default_rng(seed).choice(free, N_CALIBRATION, replace=False))
    precision = f"{best}_calibseed{seed}"
    final = Path("models") / f"{args.model}_{precision}.onnx"
    if not final.exists():
        print(f"\n{precision}: building from {N_CALIBRATION} unassigned images (seed {seed})...", flush=True)
        samples = read_parquet_images(files, positions)
        images = np.stack([preprocess(open_image(image)) for image, _ in samples])
        temporary = final.with_name(final.stem + "_building.onnx")
        try:
            to_int8(fp32_path, temporary, [images[i:i + BATCH] for i in range(0, len(images), BATCH)],
                    method=method, group_batches=GROUP_BATCHES)
            checks = check_int8_build(temporary, check, fp32_top1, expected_weights)
            for name, ok in checks.items():
                print(f"    {'ok  ' if ok else 'FAIL'} {name}")
            if not all(checks.values()):
                raise RuntimeError(f"{precision} failed its checks")
            temporary.rename(final)
        except BaseException:
            temporary.unlink(missing_ok=True)
            temporary.with_name(temporary.stem + "_prep.onnx").unlink(missing_ok=True)
            print(f"FAIL: stopped at {precision}; no partial model was kept")
            raise
        final.with_suffix(".json").write_text(json.dumps({
            "model": args.model, "precision": precision, "derived_from": str(fp32_path),
            "licence": MODELS[args.model]["licence"],
            "settings": {**INT8_SETTINGS, "calibration_method": method,
                         "calibration_method_detail": INT8_METHODS[method]["description"],
                         "calibration": {"dataset": DATASET, "dataset_licence": DATASETS[DATASET]["licence"],
                                         "split": "unassigned (in no split)", "seed": seed,
                                         "n_images": N_CALIBRATION, "group_images": GROUP_BATCHES * BATCH,
                                         "positions": positions.tolist(),
                                         "purpose": "calibration-luck check only; not a Stage 3 candidate"}},
            "file": {"path": str(final), **file_info(final)}, "machine": machine}, indent=2))
    new_logits[seed] = tuning_logits(final)
    builds[f"unassigned images, seed {seed}"] = top1_and_eaurc(new_logits[seed], original["labels"])

print(f"\n{best} on the clean tuning split ({len(original['labels']):,} images)")
print(f"{'calibration images':<34}{'top-1':>8}{'E-AURC':>9}")
for name, (top1, eaurc) in builds.items():
    print(f"{name:<34}{top1:>8.2%}{eaurc:>9.4f}")
top1s, eaurcs = np.array([b[0] for b in builds.values()]), np.array([b[1] for b in builds.values()])
spread = {"top1_range": float(np.ptp(top1s)), "top1_sd": float(np.std(top1s, ddof=1)),
          "e_aurc_range": float(np.ptp(eaurcs)), "e_aurc_sd": float(np.std(eaurcs, ddof=1))}
print(f"{'range (largest - smallest)':<34}{spread['top1_range'] * 100:>6.2f}pt{spread['e_aurc_range']:>9.4f}")
print(f"{'standard deviation':<34}{spread['top1_sd'] * 100:>6.2f}pt{spread['e_aurc_sd']:>9.4f}")

record = make_record("check", args.model, best, {
    "settings": {"dataset": DATASET, "split": "tuning", "n_images": len(original["labels"]),
                 "calibration_seeds": args.seeds, "calibration_pool": "unassigned images",
                 "confidence": "probability of the top answer", "rule_in_3.7":
                 "differences between INT8 variants smaller than the range are labelled 'within noise'"},
    "metrics": {**spread, "builds": {name: {"top1": t, "e_aurc": e} for name, (t, e) in builds.items()}},
}, machine)
out = Path("results/checks") / f"{args.model}_{best}_calibration_luck.json"
save_arrays(record, out, labels=original["labels"], positions=original["positions"],
            **{f"logits_seed{seed}": z for seed, z in new_logits.items()})
save_record(record, out)
print(f"Saved to {out}\nPASS")
