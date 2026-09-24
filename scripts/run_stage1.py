"""Reproduce every Stage 1 result with one command.

Runs the numbered scripts in order and stops at the first failure:
  01 export FP32 -> 04 make FP16/INT8 -> 02 speed -> 03 accuracy -> 05 results page

Usage:  python scripts/run_stage1.py [--full]
        --full also measures FP32 accuracy on all 50,000 images (adds about 15 minutes)
Needs:  data/imagenet-1k/ (see README). For stable speed numbers: plugged in,
        Windows power mode "Best performance", other programs closed.

Every result records the git commit it came from; commit your code first so results are
traceable (the script warns if there are uncommitted changes).
"""

import argparse
import subprocess
import sys
import time

from brokkr.fingerprint import git_info

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="also run FP32 accuracy on all 50,000 images")
args = parser.parse_args()

steps = [
    ["01_export_model.py"],
    ["04_quantize.py"],
    ["02_benchmark_speed.py", "--precision", "fp32", "fp16", "int8", "--threads", "1", "2", "4", "8"],
    ["03_evaluate_accuracy.py", "--precision", "fp32"],
    ["03_evaluate_accuracy.py", "--precision", "fp16"],
    ["03_evaluate_accuracy.py", "--precision", "int8"],
]
if args.full:
    steps.append(["03_evaluate_accuracy.py", "--precision", "fp32", "--n", "0"])
steps.append(["05_build_site.py"])

git = git_info()
print(f"Code version: {git['commit']}")
if git["dirty"]:
    print("WARNING: uncommitted changes - results won't match any commit exactly.")

start = time.time()
for i, step in enumerate(steps, 1):
    print(f"\n=== [{i}/{len(steps)}] {' '.join(step)} ===", flush=True)
    if subprocess.run([sys.executable, f"scripts/{step[0]}", *step[1:]]).returncode != 0:
        print(f"\nFAIL at step {i}: {step[0]}")
        sys.exit(1)

print(f"\nPASS: all {len(steps)} steps succeeded in {(time.time() - start) / 60:.0f} minutes")
