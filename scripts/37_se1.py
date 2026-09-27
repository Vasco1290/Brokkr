"""SE1: does keeping the squeeze-and-excitation blocks in float remove INT8's extra loss under darkness?

Usage:  python scripts/37_se1.py
        python scripts/37_se1.py --dry-run --out <folder>   (a tool check, never a result: the first 64
        tuning images; everything, including the verdict file, goes to <folder>)
Needs:  models/<model>_fp32.onnx, _int8_percentile99.99.onnx, _int8_percentile99.99_seall.onnx and
        _seoutput.onnx with build records (scripts/24_build_models.py --keep-float), data/imagenet-1k/
Writes: results/se1/<model>_<precision>_tuning_<suite>_<corruption>_s<severity>.json (schema 2, kind
        "diagnostic") + .npz (logits, labels, positions); verdict in results/final/se1_verdict.json

Design, pass rule and H's confirmations: docs/hypotheses_stage4.md, "Squeeze-and-excitation diagnostic
(SE1)" and the notes after it. Rule code: brokkr/se1.py.
- Models: EfficientNet-B0 and MobileNetV3-Large (judged), RegNetY-400MF (control, judged),
  MobileNetV3-Small (reported). Builds: FP32, the baseline Percentile INT8, SE-all, SE-output.
- Images: the tuning split without its first 64 images (4,936), in split order; the first 64 were seen in
  the dry run, so they are left out (dated note of 27 September 2026). Each model's own preprocessing;
  damage seed = the image's dataset position. Conditions: clean; darkness (Brokkr) s5 (judged); contrast (ImageNet-C)
  s3 and fog (Brokkr) s3 (reported the same way).
- Every build is measured whatever its build check said; the check's result is printed and recorded.
- Resumable: a (model, build, condition) whose record and score file exist with a matching checksum is
  skipped. 8 threads, thread spinning off (as in 4.1).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr import se1
from brokkr.accuracy import accuracy_from_logits, open_image, resize_and_crop
from brokkr.benchmark import make_session
from brokkr.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr.export import MODELS, file_info, preprocessing
from brokkr.fingerprint import machine_fingerprint
from brokkr.judge import plain, top1_correct
from brokkr.judge_breadth import near_floor
from brokkr.results import make_record, save_record, sha256_of, written_atomically
from brokkr.schema import (
    check_build_record,
    condition,
    condition_label,
    load_measurement,
    make_measurement,
    metric,
    save_measurement,
)
from brokkr.sweep import damaged_batch, normalised

JUDGED, CONTROL, REPORTED = ["efficientnet_b0", "mobilenet_v3_large"], "regnet_y_400mf", "mobilenet_v3_small"
SE1_MODELS = [*JUDGED, CONTROL, REPORTED]
BASE = "int8_percentile99.99"
BUILDS = ["fp32", BASE, f"{BASE}_seall", f"{BASE}_seoutput"]
CLEAN, DARK = condition(), condition("darkness", "brokkr", 5)
CONDITIONS = [CLEAN, DARK, condition("contrast", "imagenet-c", 3), condition("fog", "brokkr", 3)]
DATASET, THREADS, BATCH = "imagenet-1k-val", 8, 32

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--out", default="results/se1")
args = parser.parse_args()
if args.dry_run and args.out == "results/se1":
    sys.exit("a dry run needs its own --out folder")
OUT = Path(args.out)
VERDICT_PATH = OUT / "se1_verdict_DRY_RUN.json" if args.dry_run else Path("results/final/se1_verdict.json")

files = parquet_files(DATASET)
tuning = make_splits(count_images(files))["tuning"]
positions = tuning[:64] if args.dry_run else tuning[64:]  # the dry run's 64 images are never in the run
machine = machine_fingerprint()
runtime = {"name": "onnxruntime", "version": ort.__version__, "execution_provider": "CPUExecutionProvider",
           "threads": THREADS, "spinning": "off"}


def build_status(model: str, build: str) -> dict:
    """The build record's own check, and what the record says about it (FP32: its export check)."""
    path = Path("models") / f"{model}_{build}.json"
    if not path.exists():
        sys.exit(f"FAIL: no build record {path}; build it first (scripts/24_build_models.py)")
    record = json.loads(path.read_text(encoding="utf-8"))
    status, problems = check_build_record(record, set())
    agreement = ((record.get("build") or {}).get("top1_agreement_with_fp32")
                 or (record.get("sanity_check") or {}).get("top1_agreement_with_fp32"))
    return {"usable": status == "usable" and not problems, "status": status, "problems": problems,
            "agreement_256_tuning": agreement, "kept_float": (record.get("settings") or {}).get("kept_float"),
            "size_bytes": file_info(Path("models") / f"{model}_{build}.onnx")["size_bytes"]}


def out_path(model: str, build: str, cond: dict) -> Path:
    suite = cond["suite"] or "none"
    return OUT / f"{model}_{build}_tuning_{suite}_{cond['corruption']}_s{cond['severity']}.json"


def complete(path: Path) -> bool:
    if not (path.exists() and path.with_suffix(".npz").exists()):
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))["arrays"]["sha256"]
    return saved == sha256_of(path.with_suffix(".npz"))


statuses = {(m, b): build_status(m, b) for m in SE1_MODELS for b in BUILDS}
print(f"SE1{' DRY RUN' if args.dry_run else ''}: {len(positions)} tuning images, {len(SE1_MODELS)} models, "
      f"{len(BUILDS)} builds, {len(CONDITIONS)} conditions -> {OUT}")
for (m, b), s in statuses.items():
    if b != "fp32":
        print(f"  build {m:<19} {b:<30} {s['status']:<8} "
              f"agreement(256 tuning) {s['agreement_256_tuning']:.3f}  "
              f"{s['size_bytes'] / 1e6:.2f} MB  kept float: {s['kept_float']}")

# 1. Measure: models with the same preprocessing share each damaged batch.
groups = {}
for m in SE1_MODELS:
    p = preprocessing(m)
    groups.setdefault((p["resize"], p["crop"], p["interpolation"]), []).append(m)
for (resize, crop, interpolation), models in groups.items():
    todo_any = [(m, b, c) for m in models for b in BUILDS for c in CONDITIONS
                if not complete(out_path(m, b, c))]
    if not todo_any:
        print(f"skip (complete): group {resize}/{crop}/{interpolation}")
        continue
    t0 = time.time()
    loaded = list(read_parquet_images(files, positions))
    crops = np.stack([resize_and_crop(open_image(b), resize, crop, interpolation) for b, _ in loaded])
    labels = np.array([label for _, label in loaded])
    sessions = {(m, b): make_session(Path("models") / f"{m}_{b}.onnx", THREADS, spinning=False)
                for m in models for b in BUILDS}
    print(f"group {resize}/{crop}/{interpolation} {models}: pictures ready ({time.time() - t0:.0f} s)")
    for cond in CONDITIONS:
        todo = [(m, b) for m in models for b in BUILDS if not complete(out_path(m, b, cond))]
        if not todo:
            continue
        t0 = time.time()
        logits = {job: [] for job in todo}
        for start in range(0, len(positions), BATCH):
            idx = list(range(start, min(start + BATCH, len(positions))))
            batch = normalised(damaged_batch(crops, positions, idx, cond["suite"], cond["corruption"],
                                             cond["severity"]))
            for job in todo:
                logits[job].append(sessions[job].run(None, {"images": batch})[0])
        for (m, b), parts in logits.items():
            scores = np.concatenate(parts).astype(np.float32)
            path = out_path(m, b, cond)
            OUT.mkdir(parents=True, exist_ok=True)
            with written_atomically(path.with_suffix(".npz")) as tmp, tmp.open("wb") as f:
                np.savez_compressed(f, logits=scores, labels=labels, positions=np.asarray(positions))
            acc = accuracy_from_logits(scores, labels)["metrics"]
            spec = MODELS[m]
            record = make_measurement(
                "diagnostic", {"name": m, "weights": str(spec["weights"]), "licence": spec["licence"]}, b,
                runtime, "laptop", machine,
                {"dataset": DATASET, "split": "tuning", "n_images": len(positions),
                 "licence": DATASETS[DATASET]["licence"]},
                cond,
                {"top1": metric(acc["top1"], acc["top1_ci95"]),
                 "top1_tied_images": metric(acc["top1_tied_images"])},
                {"script": "scripts/37_se1.py", "diagnostic": "SE1 (docs/hypotheses_stage4.md); not a result",
                 "dry_run": args.dry_run, "batch_size": BATCH,
                 "damage_seed": "dataset position of each image",
                 "preprocessing": {"resize": resize, "crop": crop, "interpolation": interpolation},
                 "build_status": statuses[(m, b)]["status"], "bootstrap_resamples": 1000, "seed": 0},
                {"file": path.with_suffix(".npz").name, "sha256": sha256_of(path.with_suffix(".npz"))},
                [{"file": f"models/{m}_{b}.onnx",
                  "sha256": file_info(Path("models") / f"{m}_{b}.onnx")["sha256"]}])
            save_measurement(record, path)
        print(f"  done: {condition_label(cond)}, {len(todo)} model-builds ({time.time() - t0:.0f} s)")

# 2. Per-image correctness from the saved scores (checksums checked), and the same images everywhere.
correct, reference = {}, None
for m in SE1_MODELS:
    for b in BUILDS:
        for cond in CONDITIONS:
            record, arrays = load_measurement(out_path(m, b, cond))
            images = (arrays["positions"].tolist(), arrays["labels"].tolist())
            reference = reference or images
            if images != reference:
                sys.exit(f"FAIL: {out_path(m, b, cond).name} does not hold the same images in the same order")
            correct[(m, b, condition_label(cond))] = top1_correct(arrays["logits"], arrays["labels"])
n = len(reference[0])


def numbers(m: str, label: str) -> dict:
    """Baseline extra gap and each variant's change under one condition, for one model."""
    c = lambda b, lab: correct[(m, b, lab)]  # noqa: E731
    base = se1.baseline_extra_gap(c("fp32", "clean"), c("fp32", label), c(BASE, "clean"), c(BASE, label))
    out = {"fp32_top1": float(c("fp32", label).mean()), "near_floor": near_floor(c("fp32", label).mean()),
           "baseline_extra_gap": base}
    for variant in BUILDS[2:]:
        ch = se1.change(c(BASE, "clean"), c(BASE, label), c(variant, "clean"), c(variant, label))
        out[variant] = {"change": ch, "share": se1.share(base, ch)}
    return out


table = {m: {condition_label(cd): numbers(m, condition_label(cd)) for cd in CONDITIONS[1:]}
         for m in SE1_MODELS}
dark = condition_label(DARK)
builds_ok = {m: statuses[(m, BASE)]["usable"] and statuses[(m, f"{BASE}_seall")]["usable"]
             for m in SE1_MODELS}
judged = {m: {"baseline": table[m][dark]["baseline_extra_gap"],
              "change": table[m][dark][f"{BASE}_seall"]["change"],
              "builds_ok": builds_ok[m]} for m in JUDGED}
control = {"change": table[CONTROL][dark][f"{BASE}_seall"]["change"], "builds_ok": builds_ok[CONTROL]}
v = se1.verdict(judged, control)


def pts(x: float) -> str:
    return f"{100 * x:+.2f}"


def ci(c: list) -> str:
    return f"({100 * c[0]:+.2f} to {100 * c[1]:+.2f})"


print(f"\nClean top-1 on these {n:,} tuning images (reported):")
for m in SE1_MODELS:
    print(f"  {m:<19} " + "  ".join(f"{b.replace(BASE, 'P99.99')} {100 * correct[(m, b, 'clean')].mean():.2f}"
                                     for b in BUILDS))
what = "DRY RUN, not a verdict" if args.dry_run else "darkness (Brokkr) s5, SE-all"
print(f"\nSE1 ({what}): {v['verdict']}")
for why in v["why"]:
    print(f"  not judged because: {why}")
for part, ok in v["parts"].items():
    print(f"  {'holds' if ok else 'does not hold'}: {part}")
for label in table[SE1_MODELS[0]]:
    judged_note = "judged (SE-all)" if label == dark else "reported, not judged"
    print(f"\n{label} ({judged_note}); points, paired 95% intervals over the same images")
    for m in SE1_MODELS:
        t = table[m][label]
        b = t["baseline_extra_gap"]
        role = "judged" if m in JUDGED else ("control" if m == CONTROL else "reported")
        floor = " (near floor)" if t["near_floor"] else ""
        line = (f"  {m:<19} [{role}] FP32 {100 * t['fp32_top1']:.2f}{floor}; "
                f"baseline extra gap {pts(b['value'])} {ci(b['ci95'])}")
        for variant in BUILDS[2:]:
            ch, sh = t[variant]["change"], t[variant]["share"]
            share_text = "share n/a" if sh is None else f"share {100 * sh:.0f}%"
            name = variant.replace(BASE + "_", "")
            line += f"; {name} change {pts(ch['value'])} {ci(ch['ci95'])} {share_text}"
        print(line)

verdict_record = make_record("verdicts", "SE1 (4 models)", f"fp32, {BASE}, _seall, _seoutput", {
    "settings": {"script": "scripts/37_se1.py", "split": "tuning", "n_images": n, "dry_run": args.dry_run,
                 "rules": "docs/hypotheses_stage4.md: SE1 and H's confirmations of 27 September 2026",
                 "builds": {f"{m} {b}": s for (m, b), s in statuses.items()}},
    "metrics": {"verdict": v},
    "raw": {"table": table},
}, machine)
print(f"\nSaved {save_record(plain(verdict_record), VERDICT_PATH)}")
