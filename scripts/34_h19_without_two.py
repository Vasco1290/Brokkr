"""EXPLORATORY, after the verdicts: H19's correlations without MobileNetV3-Large and EfficientNet-B0.

Usage:  python scripts/34_h19_without_two.py
Needs:  results/final/breadth_4.1_verdicts.json (scripts/32_judge_breadth.py)
Writes: results/checks/breadth_4.1_h19_without_two_<condition>.json (schema 2, kind "diagnostic", one per
        damaged condition: never a result, not judged, changes no verdict)

Question (asked after the verdicts, 27 September 2026): does H19's relation (a model's clean INT8 loss
ranks its INT8 loss under damage) depend on the two models with the largest INT8 losses? Same code as the
verdict (brokkr.judge_breadth.correlation: near-floor models left out, 1,000 resamples of the model list,
seed 0), with the rule of the later 27 September note: at least 6 models, else "not computed". The top-1
values are read from the saved verdict file, so no score file is reopened.
"""

import json
import sys
from pathlib import Path

import numpy as np

from brokkr import judge_breadth as jb
from brokkr.datasets import DATASETS
from brokkr.export import MODELS
from brokkr.fingerprint import machine_fingerprint
from brokkr.results import sha256_of
from brokkr.schema import condition, condition_label, make_measurement, metric, save_measurement

VERDICTS = Path("results/final/breadth_4.1_verdicts.json")
WITHOUT = ("mobilenet_v3_large", "efficientnet_b0")
RUNTIME = {"name": "numpy (no model is run)", "version": f"numpy {np.__version__}",
           "execution_provider": "none", "threads": 1}

v = json.loads(VERDICTS.read_text(encoding="utf-8"))
top1 = {}
for row in v["raw"]["table"]:
    top1[(row["model"], "fp32", row["condition"])] = row["fp32_top1"]
    if row["int8_top1"] is not None:
        top1[(row["model"], "int8", row["condition"])] = row["int8_top1"]
models = [m for m in v["settings"]["judged_models"] if m not in WITHOUT]
official = v["metrics"]["verdicts"]["H19"]["items"]
DAMAGED = [condition("fog", "brokkr", 3), condition("darkness", "brokkr", 5),
           condition("defocus_blur", "brokkr", 3), condition("noise", "brokkr", 3)] + [
    condition(c, "imagenet-c", s)
    for c in ("fog", "contrast", "defocus_blur", "gaussian_noise") for s in (3, 5)]
conditions = {condition_label(c): c for c in DAMAGED}
if set(conditions) != set(official):
    sys.exit("FAIL: the conditions in the verdict file are not the 12 damaged 4.1 conditions")

print(f"EXPLORATORY (after the verdicts, not judged). H19 without {', '.join(WITHOUT)}: "
      f"{len(models)} models: {models}; at least {jb.MIN_MODELS} models per correlation")
machine = machine_fingerprint()
clean_gap = {m: top1[(m, "int8", "clean")] - top1[(m, "fp32", "clean")] for m in models}
n_above = 0
for label, cond in conditions.items():
    fp32 = {m: top1[(m, "fp32", label)] for m in models}
    gap = {m: top1[(m, "int8", label)] - fp32[m] for m in models}
    r = jb.correlation(models, clean_gap, gap, fp32)  # MIN_MODELS: at least 6
    o = official[label]
    was = (f"all 9: rho {o['rho']:+.3f} ({o['ci95'][0]:+.3f} to {o['ci95'][1]:+.3f})" if o["judged"]
           else "all 9: not judged")
    metrics = {"models": metric(len(r["models"]))}
    if r["judged"]:
        n_above += r["side"] == "above zero"
        metrics["spearman_rho"] = metric(r["rho"], r["ci95"])
        now = (f"rho {r['rho']:+.3f}, 95% interval {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f} ({r['side']}), "
               f"{len(r['models'])} models, {r['redrawn']} redrawn")
    else:
        now = f"not computed: {r['why_not_judged']} ({len(r['models'])} models above the floor)"
    floor = f"; near floor: {r['left_out_near_floor']}" if r["left_out_near_floor"] else ""
    print(f"  {label:<32} {now}{floor}   [{was}]")

    record = make_measurement(
        "diagnostic",
        {"name": f"{len(models)} of the 9 judged 4.1 models (without {' and '.join(WITHOUT)})",
         "weights": "torchvision defaults (brokkr.export.MODELS)",
         "licence": {m: MODELS[m]["licence"] for m in models}},
        "fp32 and int8_percentile99.99", RUNTIME, "laptop", machine,
        {"dataset": "imagenet-1k-val", "split": v["settings"]["split"], "n_images": v["settings"]["n_images"],
         "licence": DATASETS["imagenet-1k-val"]["licence"]},
        cond, metrics,
        {"script": "scripts/34_h19_without_two.py", "exploratory": "after the verdicts; not judged",
         "left_out_models": list(WITHOUT), "min_models": jb.MIN_MODELS, "bootstrap_resamples": 1000,
         "seed": 0,
         "computed": r["judged"], "why_not_computed": r.get("why_not_judged"), "side": r["side"],
         "left_out_near_floor": r["left_out_near_floor"], "redrawn": r.get("redrawn"),
         "judged_h19_all_nine": {k: o[k] for k in ("judged", "rho", "ci95", "side")}},
        derived_from=[{"file": VERDICTS.as_posix(), "sha256": sha256_of(VERDICTS)}])
    stem = label.replace(" (", "_").replace(") ", "_").replace(" ", "_").lower()
    save_measurement(record, Path("results/checks") / f"breadth_4.1_h19_without_two_{stem}.json")

print(f"Conditions with the interval above zero without the two models: {n_above} "
      f"(with all 9, as judged: {v['metrics']['verdicts']['H19']['holding']})")
