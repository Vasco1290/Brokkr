"""The Stage 3 final run (task 3.7): every chosen fix measured once on the 10,000 test images.

Usage:  python scripts/run_stage3_final.py [--model NAME] [--check-only]
Needs:  the commit tagged stage3-final-run checked out with no uncommitted changes, all Stage 3
        models, and the Stage 2 test sweep (FP32/FP16/default INT8 baselines, not rerun)
Writes: results/accuracy/*_test.json and results/sweep/*_test_*.json for the new INT8 models, and a
        log of every step in results/final/run_log.txt
Then:   python scripts/18_judge_stage3.py

Steps (docs/hypotheses_stage3.md, dated 3.7 note):
1. Clean test accuracy (scripts/03) for best INT8, best INT8 unrounded, and the five leave-one-out
   models. The sweep's clean check compares against these.
2. Best INT8 and unrounded: all 26 conditions (scripts/08, both on identical damaged images).
3. Each leave-one-out model: clean and its held-out corruption at severities 1-5 only.

Safety:
- Resumable: a step whose output files all exist and are complete is skipped, so after a crash or
  power loss the same command continues. A file that exists but is incomplete stops the run: that is
  a technical failure; delete the file, log the reason in docs/stage3_final_run_log.md, and rerun.
  Reruns are never allowed because of a result.
- Keeps the laptop awake with Windows' keep-awake request (SetThreadExecutionState), which ends when
  this script ends. No power setting is changed. Closing the lid may still sleep the laptop.
- At the end, every file the judging script needs is checked: it loads, its checksum matches, and it
  has finite scores for all 10,000 test images.
"""

import argparse
import ctypes
import datetime
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from brokkr_edge.results import load_arrays
from brokkr_edge.shift import CORRUPTIONS

DATASET = "imagenet-1k-val"
N_TEST, N_CLASSES = 10_000, 1000
TAG = "stage3-final-run"
HARMFUL = ([("defocus_blur", s) for s in (2, 3, 4, 5)] + [("motion_blur", s) for s in (2, 3, 4, 5)]
           + [("noise", s) for s in (3, 4, 5)] + [("fog", 5)])

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--check-only", action="store_true", help="only check that every file is complete")
args = parser.parse_args()
m, py = args.model, sys.executable
log_path = Path("results/final/run_log.txt")


def log(message: str):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def git(*command) -> str:
    return subprocess.run(["git", *command], capture_output=True, text=True).stdout.strip()


def keep_awake(on: bool):
    """Ask Windows not to sleep while this script runs (released when it ends or on=False)."""
    if platform.system() == "Windows":
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def sweep(precision, corruption, severity):
    return Path("results/sweep") / f"{m}_{precision}_{DATASET}_test_{corruption}_s{severity}.json"


def problem_with(path: Path):
    """None if the result file is complete, else what is wrong with it."""
    if not path.exists():
        return "missing"
    try:
        json.loads(path.read_text())
        arrays = load_arrays(path)  # also checks the .npz checksum
    except (ValueError, OSError, KeyError) as error:
        return f"unreadable ({error})"
    logits = arrays["logits"]
    if logits.shape != (N_TEST, N_CLASSES) or len(arrays["labels"]) != N_TEST:
        return f"wrong shape {logits.shape}"
    if not np.isfinite(logits).all():
        return "scores are not all finite"
    return None


best = json.loads((Path("results/choices") / f"{m}_int8_method.json").read_text())["metrics"]["chosen"]
new_models = [best, f"{best}_unrounded"] + [f"{best}_mixed_without_{c}" for c in CORRUPTIONS]
all_conditions = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in (1, 2, 3, 4, 5)]

steps = [(f"clean test accuracy, {p}",
          [py, "scripts/03_evaluate_accuracy.py", "--precision", p, "--split", "test"],
          [Path("results/accuracy") / f"{m}_{p}_{DATASET}_test.json"]) for p in new_models]
steps.append((f"sweep {best} and unrounded, all 26 conditions",
              [py, "scripts/08_corruption_sweep.py", "--precision", best, f"{best}_unrounded"],
              [sweep(p, c, s) for p in (best, f"{best}_unrounded") for c, s in all_conditions]))
for c in CORRUPTIONS:
    p = f"{best}_mixed_without_{c}"
    steps.append((f"sweep {p}: clean and {c} s1-s5",
                  [py, "scripts/08_corruption_sweep.py", "--precision", p, "--corruptions", c,
                   "--severities", "0", "1", "2", "3", "4", "5"],
                  [sweep(p, "clean", 0)] + [sweep(p, c, s) for s in (1, 2, 3, 4, 5)]))

# Stage 2 baselines the judging script reads (not rerun, only checked).
baselines = [sweep("fp32", "clean", 0)] + [sweep("fp32", c, s) for c in CORRUPTIONS for s in (3, 5)]
baselines += [sweep("fp32", c, s) for c, s in HARMFUL]

if not args.check_only:
    tagged = git("rev-parse", "HEAD") == git("rev-parse", f"{TAG}^{{commit}}")
    if not tagged or git("status", "--porcelain"):
        sys.exit(f"FAIL: check out the commit tagged {TAG}, with no uncommitted changes, before the run")
    log(f"final run at {git('rev-parse', '--short', 'HEAD')} (tag {TAG})")
    keep_awake(True)
    try:
        for name, command, outputs in steps:
            problems = {str(o): problem_with(o) for o in outputs}
            if all(p is None for p in problems.values()):
                log(f"skip (already complete): {name}")
                continue
            broken = {o: p for o, p in problems.items() if p not in (None, "missing")}
            if broken:
                log(f"STOP: incomplete files {broken}. Technical failure: delete them, log the reason in "
                    "docs/stage3_final_run_log.md, and run again.")
                sys.exit(1)
            log(f"start: {name}")
            start = time.time()
            exit_code = subprocess.run(command).returncode
            log(f"end:   {name} (exit {exit_code}, {(time.time() - start) / 60:.1f} min)")
            if exit_code != 0:
                log("STOP: a step failed. Fix the technical cause, log it, and run again.")
                sys.exit(1)
    finally:
        keep_awake(False)

# Completeness check of every file the judging script needs.
expected = list(dict.fromkeys([o for _, _, outputs in steps for o in outputs] + baselines))  # no repeats
problems = {str(path): problem_with(path) for path in expected}
bad = {path: p for path, p in problems.items() if p is not None}
for path, p in bad.items():
    print(f"    {p}: {path}")
log(f"completeness check: {len(expected) - len(bad)} of {len(expected)} expected files complete")
print("PASS: every expected result file exists and is complete" if not bad else "FAIL")
sys.exit(0 if not bad else 1)
