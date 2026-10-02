"""Does `brokkr-edge test` reproduce MobileNetV3-Large's 28 records of task 4.1? Prints PASS or FAIL.

Usage:  python scripts/43_check_reproduction.py [--out results/reproduction]
Needs:  results/breadth (the 4.1 records and scores), models/mobilenet_v3_large_{fp32,int8_percentile99.99},
        data/imagenet-1k/, a clean commit, and the laptop on mains power
Writes: <out>/repeatability.json (before step 2), <out>/mobilenet_v3_large/ (the 28 new records),
        <out>/reproduction_check.json

The rule was fixed before the command existed (docs/hypotheses_stage4.md, notes of 30 September and
3 October 2026):
1. Is INT8 repeatable here? The command runs twice, each in its own process, on tuning images 0-63.
   Repeatable = the INT8 scores on clean images are identical bit for bit. (The other conditions and
   FP32 are compared too; that is printed, not judged.)
2. The command runs once on the test and conformal_calibration images (28 records).
3. Each record is compared with the 4.1 record of the same name:
   - if repeatable: PASS = the same images in the same order and identical top-1 predictions;
   - if not: PASS = correct top-1 answers differ by at most 10 (test) or 5 (calibration) images, and
     the same model file, preprocessing, damage seed rule, batch size and thread count.
The run takes a while; finished records are kept, so the same command can be run again after a stop.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from brokkr_edge.fingerprint import git_info, machine_fingerprint, power_state
from brokkr_edge.results import make_record, save_record
from brokkr_edge.schema import check_record
from brokkr_edge.test_run import compare_scores

MODEL, INT8 = "mobilenet_v3_large", "int8_percentile99.99"
ORIGINAL = Path("results/breadth")
DRY_RUN_IMAGES = 64

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="results/reproduction")
args = parser.parse_args()
out = Path(args.out)

if git_info()["dirty"]:
    sys.exit("STOP: uncommitted changes. The reproduction must come from a clean commit.")
if power_state()["on_ac_power"] is False:
    sys.exit("STOP: the laptop is on battery. Plug it in and run the same command again.")


def brokkr_edge_test(*options: str) -> None:
    """Run the command itself, in its own process."""
    subprocess.run([sys.executable, "-m", "brokkr_edge.cli", "test", "--model", MODEL, *options], check=True)


def scores(path: Path) -> dict:
    with np.load(path.with_suffix(".npz")) as data:
        return {name: data[name] for name in ("logits", "labels", "positions")}


# 1. Repeatability, decided and recorded before anything is compared with 4.1.
repeat_path = out / "repeatability.json"
if not repeat_path.exists():
    runs = [out / "repeatability" / f"run{i}" for i in (1, 2)]
    for folder in runs:
        shutil.rmtree(folder, ignore_errors=True)  # always two fresh runs
        brokkr_edge_test("--split", "tuning", "--limit", str(DRY_RUN_IMAGES), "--out", str(folder))
    names = sorted(p.name for p in runs[0].glob("*.json"))
    same = {n: bool(np.array_equal(scores(runs[0] / n)["logits"], scores(runs[1] / n)["logits"]))
            for n in names}
    judged = [n for n in names if f"_{INT8}_" in n and "_none_clean_" in n]
    if len(judged) != 1:
        sys.exit(f"STOP: expected one INT8 clean record in the dry run, found {judged}")
    save_record(make_record("check", MODEL, INT8, {
        "settings": {"rule": "docs/hypotheses_stage4.md, notes of 30 September and 3 October 2026",
                     "split": "tuning", "images": DRY_RUN_IMAGES, "runs": 2, "judged_record": judged[0]},
        "metrics": {"repeatable": same[judged[0]], "records_compared": len(names),
                    "records_identical": sum(same.values()),
                    "records_that_differ": [n for n in names if not same[n]]},
    }, machine_fingerprint()), repeat_path)
repeat = json.loads(repeat_path.read_text(encoding="utf-8"))["metrics"]
repeatable = repeat["repeatable"]
print(f"Step 1: INT8 on clean tuning images, two separate runs: "
      f"{'identical bit for bit: REPEATABLE' if repeatable else 'NOT identical: NOT REPEATABLE'} "
      f"(reported, not judged: {repeat['records_identical']} of {repeat['records_compared']} score files of "
      f"all conditions and both builds are identical)")
print("        rule for step 3: " + ("identical top-1 predictions on every image" if repeatable else
                                     "top-1 within 0.1 points and identical build settings"))

# 2. The full run (finished records are kept).
new_dir = out / MODEL
brokkr_edge_test("--out", str(new_dir))

# 3. Each of the 28 records against its 4.1 record.
names = sorted(p.name for p in ORIGINAL.glob(f"{MODEL}_*.json"))
results, all_pass = [], len(names) == 28
if len(names) != 28:
    print(f"FAIL: expected 28 records of {MODEL} in {ORIGINAL}, found {len(names)}")
for name in names:
    old_record = json.loads((ORIGINAL / name).read_text(encoding="utf-8"))
    if not (new_dir / name).exists():
        results.append({"record": name, "pass": False, "why": "not produced by brokkr-edge test"})
        continue
    new_record = json.loads((new_dir / name).read_text(encoding="utf-8"))
    row = {"record": name, **compare_scores(scores(ORIGINAL / name), scores(new_dir / name), repeatable),
           "record_problems": check_record(new_record),
           "made_from_clean_commit": not new_record["code"]["dirty"]}
    same_settings = all([
        old_record["derived_from"] == new_record["derived_from"],  # the model file's SHA-256
        old_record["settings"]["preprocessing"] == new_record["settings"]["preprocessing"],
        old_record["settings"]["damage_seed"] == new_record["settings"]["damage_seed"],
        old_record["settings"]["batch_size"] == new_record["settings"]["batch_size"],
        old_record["runtime"]["threads"] == new_record["runtime"]["threads"],
    ])
    row["same_build_settings"] = same_settings
    passed = row["reproduced"] and (repeatable or same_settings)  # the second rule also needs the settings
    row["pass"] = bool(passed and not row["record_problems"] and row["made_from_clean_commit"])
    results.append(row)

for row in results:
    all_pass &= row["pass"]
    if "top1_differs_on" not in row:
        print(f"FAIL {row['record']}: {row.get('why', 'not the same images')}")
        continue
    identical = "identical" if row["scores_identical"] else "NOT identical"
    print(f"{'PASS' if row['pass'] else 'FAIL'} {row['record']}: top-1 differs on "
          f"{row['top1_differs_on']} of {row['images']} images; scores {identical} bit for bit (largest "
          f"difference {row['largest_score_difference']:.3g}); same build settings: "
          f"{'yes' if row['same_build_settings'] else 'NO'}")

save_record(make_record("check", MODEL, f"fp32 and {INT8}", {
    "settings": {"rule": "docs/hypotheses_stage4.md, notes of 30 September and 3 October 2026",
                 "original_records": ORIGINAL.as_posix(), "new_records": new_dir.as_posix(),
                 "repeatable": repeatable,
                 "pass_rule": "identical top-1 predictions" if repeatable else
                 "top-1 within 0.1 points and identical build settings"},
    "metrics": {"verdict": "PASS" if all_pass else "FAIL", "records": len(results),
                "records_passing": sum(r["pass"] for r in results),
                "records_with_identical_scores": sum(r.get("scores_identical", False) for r in results),
                "per_record": results},
}, machine_fingerprint()), out / "reproduction_check.json")
print(f"\n{'PASS' if all_pass else 'FAIL'}: {sum(r['pass'] for r in results)} of {len(results)} records "
      f"reproduced ({'repeatable' if repeatable else 'not repeatable'} rule); "
      f"{out / 'reproduction_check.json'}")
sys.exit(0 if all_pass else 1)
