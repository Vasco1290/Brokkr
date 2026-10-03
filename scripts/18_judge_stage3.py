"""Judge Stage 3's predictions H10-H17 from the final run's saved scores (task 3.7).

Usage:  python scripts/18_judge_stage3.py [--model NAME]
Needs:  the final run's test-split results (scripts/run_stage3_final.py), the Stage 2 test sweep
        (FP32 baselines), and the Stage 3 choices: results/choices/<model>_{int8_method,temperatures,
        shift_aware}.json and results/checks/<model>_<best>_calibration_luck.json
Writes: results/final/<model>_stage3_verdicts.json

Written, and tested on fake data (tests/test_judge.py), BEFORE the final run. The rules are in
brokkr_edge/judge.py and use only the thresholds written in docs/hypotheses_stage3.md. Each prediction
gets PASS, FAIL or NOT RUN, plus "within noise" where an INT8-vs-INT8 difference is smaller than the
calibration-luck range. Every coverage number is printed next to its average set size.
"""

import argparse
import json
from pathlib import Path

from brokkr_edge import judge
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.results import load_arrays, make_record, save_record
from brokkr_edge.shift import CORRUPTIONS

DATASET = "imagenet-1k-val"
HARMFUL = ([("defocus_blur", s) for s in (2, 3, 4, 5)] + [("motion_blur", s) for s in (2, 3, 4, 5)]
           + [("noise", s) for s in (3, 4, 5)] + [("fog", 5)])  # fixed in the design rules
V2_RESULT = Path("results/accuracy/mobilenet_v3_large_fp32_imagenetv2-matched-frequency_all.json")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--out", default=None, help="default: results/final/<model>_stage3_verdicts.json")
args = parser.parse_args()
m = args.model


def read(path):
    return json.loads(Path(path).read_text())


def test_scores(precision, corruption="clean", severity=0):
    """Logits and labels of one test condition (checksum verified on loading)."""
    name = f"{m}_{precision}_{DATASET}_test_{corruption}_s{severity}.json"
    arrays = load_arrays(Path("results/sweep") / name)
    return arrays["logits"], arrays["labels"]


best = read(f"results/choices/{m}_int8_method.json")["metrics"]["chosen"]
luck = read(f"results/checks/{m}_{best}_calibration_luck.json")["metrics"]
top1_noise, e_aurc_noise = luck["top1_range"], luck["e_aurc_range"]
temperature = read(f"results/choices/{m}_temperatures.json")["metrics"]["temperatures"]["fp32"]
shift = read(f"results/choices/{m}_shift_aware.json")["metrics"]


def h10():
    logits, labels = test_scores(best)
    return judge.h10(logits, labels)


def h11():
    logits_b, labels = test_scores(best)
    logits_u, _ = test_scores(f"{best}_unrounded")
    size = {p: read(Path("models") / f"{m}_{p}.json")["file"]["size_bytes"]
            for p in (best, f"{best}_unrounded")}
    return judge.h11(logits_b, logits_u, labels, size[best], size[f"{best}_unrounded"], e_aurc_noise)


def h12():
    logits_b, labels = test_scores(best, "darkness", 5)
    logits_d, _ = test_scores(f"{best}_mixed_without_darkness", "darkness", 5)
    return judge.h12(logits_b, logits_d, labels, top1_noise)


def h13():
    logits_b, labels = test_scores(best)
    others = [test_scores(f"{best}_mixed_without_{c}")[0] for c in CORRUPTIONS]
    return judge.h13(logits_b, others, labels, top1_noise)


def h14():
    clean, labels = test_scores("fp32")
    return judge.h14(temperature, clean, {c: test_scores("fp32", c, 5)[0] for c in CORRUPTIONS}, labels)


def h15():
    clean, labels = test_scores("fp32")
    robust = shift["robust_conformal"]["fp32"]
    thresholds = {c: robust["held_out"][c]["threshold"] for c in CORRUPTIONS}
    return judge.h15(clean, {c: test_scores("fp32", c, 3)[0] for c in CORRUPTIONS}, labels,
                     robust["clean_only_threshold"], thresholds)


def h16():
    clean, labels = test_scores("fp32")
    harmful = {f"{c} s{s}": test_scores("fp32", c, s)[0] for c, s in HARMFUL}
    return judge.h16(shift["alarm"]["fp32"]["threshold"], clean, harmful, labels)


def h17():
    if not V2_RESULT.exists():
        return judge.h17(None, None, None)
    arrays = load_arrays(V2_RESULT)
    clean_threshold = shift["robust_conformal"]["fp32"]["clean_only_threshold"]
    return judge.h17(arrays["logits"], arrays["labels"], clean_threshold)


def extra_best_int8_robust_conformal():
    """Extra analysis (not a prediction): best INT8's robust conformal on held-out corruptions, s3."""
    clean, labels = test_scores(best)
    robust = shift["robust_conformal"][best]
    thresholds = {c: robust["held_out"][c]["threshold"] for c in CORRUPTIONS}
    return judge.h15(clean, {c: test_scores(best, c, 3)[0] for c in CORRUPTIONS}, labels,
                     robust["clean_only_threshold"], thresholds)


verdicts = {}
checks = [("H10", h10), ("H11", h11), ("H12", h12), ("H13", h13), ("H14", h14), ("H15", h15), ("H16", h16),
          ("H17", h17), ("EXTRA best INT8 robust conformal", extra_best_int8_robust_conformal)]
for name, check in checks:
    try:
        verdicts[name] = check()
    except FileNotFoundError as error:
        verdicts[name] = judge.not_run(f"missing result file: {error.filename or error}")
    r = verdicts[name]
    label = "EXTRA (no prediction)" if name.startswith("EXTRA") else r["verdict"]
    noise = "  (within noise)" if r["within_noise"] else ""
    print(f"\n{name}: {label}{noise}")
    for part, ok in r["parts"].items():
        print(f"    {'yes' if ok else 'NO '}  {part}")
    for key, value in r["numbers"].items():
        print(f"    {key}: {value}")
    for note in r["notes"]:
        print(f"    note: {note}")

record = make_record("verdicts", m, "all", {
    "settings": {"hypotheses": "docs/hypotheses_stage3.md", "split": "test",
                 "judging": "brokkr_edge/judge.py",
                 "noise_floor": {"top1_range": top1_noise, "e_aurc_range": e_aurc_noise}},
    "metrics": {name: {"verdict": r["verdict"], "within_noise": r["within_noise"]}
                for name, r in verdicts.items()},
    "raw": verdicts,
}, machine_fingerprint())
out = save_record(record, Path(args.out or Path("results/final") / f"{m}_stage3_verdicts.json"))
print(f"\nSaved to {out}")
