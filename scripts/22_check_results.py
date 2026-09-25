"""Check every saved result against the unified result format (schema version 2, task 4.0).

Usage:  python scripts/22_check_results.py [--folder results]
Reads:  every .json under the folder. Stage 1-3 files (schema version 1) are converted in memory by
        brokkr.schema.from_v1; nothing on disk is changed.
Prints: measurements per kind, the decision records that are not measurements (choices, checks,
        verdicts: they have no single model/condition, so they are listed, not converted), every
        problem found, and PASS or FAIL.
"""

import argparse
import sys
from collections import Counter

from brokkr.datasets import DATASETS
from brokkr.export import MODELS
from brokkr.schema import check_record, load_measurements

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="results")
args = parser.parse_args()

models = {name: {"weights": str(spec["weights"]), "licence": spec["licence"]}
          for name, spec in MODELS.items()}
measurements, skipped = load_measurements(args.folder, models, DATASETS)

failures = [(path, problems) for path, record in measurements if (problems := check_record(record))]
per_kind = Counter(record["kind"] for _, record in measurements)
versions = Counter("converted from schema 1" if record["derived_from"] and "converted" in
                   record["derived_from"][0].get("note", "") else "schema 2" for _, record in measurements)

print(f"Measurements: {len(measurements)} ({dict(versions)})")
for kind, n in sorted(per_kind.items()):
    print(f"  {kind:<12} {n}")
print(f"Not measurements (listed, not converted): {len(skipped)}")
for kind, n in sorted(Counter(kind for _, kind in skipped).items()):
    print(f"  {kind:<18} {n}")

for path, problems in failures:
    print(f"FAIL {path}")
    for problem in problems:
        print(f"     - {problem}")
ok = bool(measurements) and not failures
print(f"\n{'PASS' if ok else 'FAIL'}: {len(measurements) - len(failures)} of {len(measurements)} "
      "measurement records pass the schema check")
sys.exit(0 if ok else 1)
