"""Task 4.1: judge H18-H22 (docs/hypotheses_stage4.md) from the breadth sweep's saved scores.

Usage:  python scripts/32_judge_breadth.py                    (the 4.1 test-split results)
        python scripts/32_judge_breadth.py --results <dry-run folder> --split tuning --out <file>
Needs:  schema-2 accuracy records with saved scores from scripts/28_breadth_sweep.py, and the model
        build records in models/
Writes: results/final/breadth_4.1_verdicts.json (or --out): every verdict and the numbers behind it

Rules: the H18-H22 rules committed before the sweep and the dated note of 27 September 2026, both in
docs/hypotheses_stage4.md; the judging code is brokkr/judge_breadth.py.
- Judged models: FP32 sanity check passed (clean test-split top-1 within 1.0 point of torchvision's
  published top-1; on other splits it cannot be judged and is reported as such) and a usable INT8
  build (brokkr.schema.check_build_record). Every other model is listed with the reason.
- Top-1 counts ties towards the lower class number, as everywhere (brokkr.judge.top1_correct), and
  must equal each record's saved top-1 exactly.
- Every record of every model must hold the same images in the same order (checked), so the
  intervals are paired.
After the verdicts it prints, for every model and condition, the absolute weakness next to the
compression-caused gap (reported, not judged, as the hypotheses file asks).
"""

import argparse
import json
import sys
from pathlib import Path

from brokkr import judge_breadth as jb
from brokkr.export import MODELS
from brokkr.fingerprint import machine_fingerprint
from brokkr.judge import plain, top1_correct
from brokkr.results import make_record, save_record
from brokkr.schema import check_build_record, condition, condition_label, load_measurement

INT8, TOLERANCE = "int8_percentile99.99", 0.010  # sanity-check tolerance, as in scripts/28_breadth_sweep.py
CLEAN = condition()
# The 12 damaged conditions of docs/hypotheses_stage4.md, "Conditions".
DAMAGED = [condition("fog", "brokkr", 3), condition("darkness", "brokkr", 5),
           condition("defocus_blur", "brokkr", 3), condition("noise", "brokkr", 3)] + [
    condition(c, "imagenet-c", s)
    for c in ("fog", "contrast", "defocus_blur", "gaussian_noise") for s in (3, 5)]
CONTRAST_3, NOISE_3 = condition("contrast", "imagenet-c", 3), condition("gaussian_noise", "imagenet-c", 3)
DARKNESS_5, CONTRAST_5 = condition("darkness", "brokkr", 5), condition("contrast", "imagenet-c", 5)

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/breadth")
parser.add_argument("--split", default="test")
parser.add_argument("--out", default="results/final/breadth_4.1_verdicts.json")
args = parser.parse_args()


def key(cond: dict) -> str:
    return condition_label(cond)


def pct(x: float) -> str:
    return f"{100 * x:+.2f}"


def ci_pts(ci: list) -> str:
    return f"({100 * ci[0]:+.2f} to {100 * ci[1]:+.2f})"


# 1. Records on this split, by (model, precision, condition).
paths = {}
for path in sorted(Path(args.results).glob("*.json")):
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("schema_version") == 2 and r["kind"] == "accuracy" and r["data"]["split"] == args.split:
        paths[(r["model"]["name"], r["precision"], key(r["condition"]))] = path
if not paths:
    sys.exit(f"FAIL: no {args.split}-split accuracy records in {args.results}")
all_models = [m for m in MODELS if (m, "fp32", "clean") in paths]

# 2. Per-image correctness (checked against each record's saved top-1), and the same images everywhere.
correct, top1, reference = {}, {}, None
problems = []
for (model, precision, label), path in sorted(paths.items()):
    record, arrays = load_measurement(path)
    c = top1_correct(arrays["logits"], arrays["labels"])
    if c.sum() / len(c) != record["metrics"]["top1"]["value"]:
        problems.append(f"{path.name}: top-1 {c.mean()} differs from the record's "
                        f"{record['metrics']['top1']['value']}")
    images = (arrays["positions"].tolist(), arrays["labels"].tolist())
    reference = reference or images
    if images != reference:
        problems.append(f"{path.name}: not the same images, in the same order, as the other records")
    p = "int8" if precision == INT8 else precision
    correct[(model, p, label)], top1[(model, p, label)] = c, c.sum() / len(c)
n_images = len(reference[0])
if problems:
    sys.exit("FAIL:\n  " + "\n  ".join(problems))

# 3. Which models are judged.
judged, left_out, sanity = [], {}, {}
for model in all_models:
    published = MODELS[model]["weights"].meta["_metrics"]["ImageNet-1K"]["acc@1"] / 100
    measured = top1[(model, "fp32", "clean")]
    if args.split == "test":
        ok = abs(measured - published) <= TOLERANCE
        sanity[model] = {"fp32_clean_top1": measured, "published": published, "pass": ok}
    else:
        ok = True
        sanity[model] = {"fp32_clean_top1": measured, "published": published, "pass": None,
                         "note": f"not judged: {args.split} split, not test"}
    status, build_problems = check_build_record(json.loads(
        (Path("models") / f"{model}_{INT8}.json").read_text(encoding="utf-8")), set())
    missing = [key(c) for c in [CLEAN, *DAMAGED] for p in ("fp32", "int8") if (model, p, key(c)) not in top1]
    if not ok:
        left_out[model] = f"FP32 sanity check failed ({measured:.4f} vs published {published:.4f})"
    elif status != "usable" or build_problems:
        left_out[model] = f"INT8 build {status}" + (f" ({'; '.join(build_problems)})"
                                                     if build_problems else "")
    elif missing:
        left_out[model] = f"missing records: {missing}"
    else:
        judged.append(model)

print(f"{args.split} split, {n_images:,} images per record, {len(paths)} records from {args.results}")
for model in all_models:
    s = sanity[model]
    check = "not judged (not the test split)" if s["pass"] is None else ("PASS" if s["pass"] else "FAIL")
    print(f"  {model:<19} FP32 clean {s['fp32_clean_top1']:.4f} (torchvision {s['published']:.4f}): "
          f"sanity {check}; "
          + ("judged" if model in judged else f"LEFT OUT: {left_out[model]}"))
print(f"Judged models: {len(judged)}")

# 4. The verdicts.
labels = [key(c) for c in DAMAGED]
correlations = jb.h18_h19(judged, labels, top1)
extras = {label: {m: jb.extra_gap(correct[(m, "fp32", "clean")], correct[(m, "int8", "clean")],
                                  correct[(m, "fp32", label)], correct[(m, "int8", label)]) for m in judged}
          for label in labels}
fp32_at = {label: {m: top1[(m, "fp32", label)] for m in judged} for label in labels}
verdicts = {
    **correlations,
    "H20": jb.large_extra_gap(judged, extras[key(CONTRAST_3)], fp32_at[key(CONTRAST_3)]),
    "H21": jb.large_extra_gap(judged, extras[key(DARKNESS_5)], fp32_at[key(DARKNESS_5)]),
    "H22": jb.contrast_vs_noise(judged, extras[key(CONTRAST_3)], extras[key(NOISE_3)],
                                fp32_at[key(CONTRAST_3)], fp32_at[key(NOISE_3)]),
}
RULES = {
    "H18a": "Spearman(clean FP32 top-1, INT8 top-1 under the condition): interval above zero in >= 8 of 12",
    "H18b": "Spearman(clean FP32 top-1, compression-caused gap): interval includes zero in >= 8 of 12",
    "H19": "Spearman(clean compression-caused gap, gap under the condition): "
           "interval above zero in >= 6 of 12",
    "H20": "contrast (ImageNet-C) s3: extra gap <= -5.0 points, paired interval below zero, >= 5 models",
    "H21": "darkness (Brokkr) s5: extra gap <= -5.0 points, paired interval below zero, >= 5 models",
    "H22": "contrast s3 extra gap minus Gaussian noise s3 extra gap < 0, interval below zero, >= 5 models",
}
for name, v in verdicts.items():
    print(f"\n{name}: {v['verdict']}  ({v['holding']} hold, {v['needed']} needed; "
          f"{v['judged']} of {v['total']} judged)\n  rule: {RULES[name]}")
    for item_name, item in v["items"].items():
        mark = "holds" if item["holds"] else ("not judged: " + item["why_not_judged"] if not item["judged"]
                                              else "does not hold")
        if "rho" in item:
            floor = item["left_out_near_floor"]
            left = f"; left out (near floor): {floor}" if floor else ""
            numbers = (f"rho {item['rho']:+.3f}, 95% interval {item['ci95'][0]:+.3f} to "
                       f"{item['ci95'][1]:+.3f} "
                       f"({item['side']}), {len(item['models'])} models, {item['redrawn']} resamples redrawn"
                       if item["judged"] else f"{len(item['models'])} models")
            print(f"  {item_name:<32} {numbers}{left}: {mark}")
        else:
            print(f"  {item_name:<19} {pct(item['value'])} points {ci_pts(item['ci95'])} "
                  f"({item['side']}): {mark}")

# 5. Reported, not judged: absolute weakness next to every compression-caused number.
print("\nReported, not judged. Points; paired 95% intervals over the same images; "
      "'floor' = FP32 below 10% (left out of every judged count).")
print(f"  {'condition':<30} {'model':<19} {'FP32':>6} {'FP32-clean':>10} {'INT8':>6} "
      f"{'INT8-FP32 (interval)':>24} {'INT8/FP32':>9} {'extra gap (interval)':>24}")
table = []
for label in ["clean", *labels]:
    for model in all_models:
        fp32 = top1[(model, "fp32", label)]
        row = {"condition": label, "model": model, "fp32_top1": fp32,
               "fp32_minus_clean": fp32 - top1[(model, "fp32", "clean")], "near_floor": jb.near_floor(fp32),
               "int8_top1": None, "gap": None, "relative": None, "extra_gap": None}
        if (model, "int8", label) in top1:
            g = jb.gap(correct[(model, "fp32", label)], correct[(model, "int8", label)])
            int8_top1 = top1[(model, "int8", label)]
            row.update(int8_top1=int8_top1, gap=g, relative=int8_top1 / fp32)
            if label != "clean" and model in judged:
                row["extra_gap"] = {k: v for k, v in extras[label][model].items() if k != "per_image"}
        table.append(row)
        int8 = "INT8 failed or not run" if row["gap"] is None else (
            f"{100 * row['int8_top1']:6.2f} {pct(g['value']):>7} {ci_pts(g['ci95']):>16} "
            f"{row['relative']:9.3f}")
        e = row["extra_gap"]
        extra = "" if e is None else f" {pct(e['value']):>7} {ci_pts(e['ci95'])}"
        print(f"  {label:<30} {model:<19} {100 * fp32:6.2f} {pct(row['fp32_minus_clean']):>10} {int8}{extra}"
              + ("  floor" if row["near_floor"] else ""))
print("  Contrast (ImageNet-C) s5 extra gaps are in this table: reported, not judged (hypotheses file, H20).")

# 6. Save.
record = make_record("verdicts", f"breadth ({len(judged)} models judged)", f"fp32 and {INT8}", {
    "settings": {"script": "scripts/32_judge_breadth.py", "split": args.split, "n_images": n_images,
                 "results": args.results, "rules": "docs/hypotheses_stage4.md (H18-H22 and the note of "
                 "27 September 2026)", "bootstrap_resamples": 1000, "seed": 0,
                 "judged_models": judged, "left_out": left_out, "sanity_check": sanity},
    "metrics": {"verdicts": verdicts},
    "raw": {"table": table, "rules": RULES},
}, machine_fingerprint())
print(f"\nSaved {save_record(plain(record), args.out)}")
