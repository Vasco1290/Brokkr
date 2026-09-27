"""Check every saved result, and every model build record (task 4.0).

Usage:  python scripts/22_check_results.py [--folder results] [--models models]
Reads:  every .json under the results folder. Stage 1-3 files (schema version 1) are converted in
        memory by brokkr.schema.from_v1; nothing on disk is changed. And every models/*.json build record.
Prints:
  1. Result records (schema 2): count per kind, with diagnostics counted apart (they explain results
     but are never results), and the files that are not measurements (choices, checks, verdicts,
     profiles), listed but not converted.
  2. Build records: each model file's status ("usable": its own sanity check passed; "failed"; "no
     sanity check"), any non-default setting, and problems: not from a clean commit, no licence, or
     built by code that had an option (brokkr.schema.BUILD_OPTIONS) without stating it.
  3. PASS only if every result record passes the schema check, every build record is acceptable, and
     no result record uses a model whose build failed or has no sanity check.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from brokkr.datasets import DATASETS
from brokkr.export import MODELS
from brokkr.schema import check_build_record, check_record, load_measurements, non_default_settings

# The commit that added each build option: code at or after it must state the option in its records.
OPTION_INTRODUCED = {"skip_symbolic_shape": "e33195e", "kept_float": "7c933be"}

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="results")
parser.add_argument("--models", default="models")
args = parser.parse_args()


def had_option(commit: str, introduced: str) -> bool:
    """True if `commit` already contained the commit that introduced the option."""
    return subprocess.run(["git", "merge-base", "--is-ancestor", introduced, commit],
                          capture_output=True).returncode == 0


# 1. Result records.
models = {name: {"weights": str(spec["weights"]), "licence": spec["licence"]}
          for name, spec in MODELS.items()}
measurements, skipped = load_measurements(args.folder, models, DATASETS)
failures = [(path, problems) for path, record in measurements if (problems := check_record(record))]
results = [r for _, r in measurements if r["kind"] != "diagnostic"]
diagnostics = [r for _, r in measurements if r["kind"] == "diagnostic"]
converted = sum("converted" in (r["derived_from"][0].get("note", "") if r["derived_from"] else "")
                for _, r in measurements)

print(f"1. Result records: {len(results)} results and {len(diagnostics)} diagnostics "
      f"({converted} converted from schema 1, {len(measurements) - converted} written as schema 2)")
for kind, n in sorted(Counter(r["kind"] for _, r in measurements).items()):
    print(f"     {kind:<12} {n}")
print(f"   Not measurements (listed, not converted): {len(skipped)}")
for kind, n in sorted(Counter(str(kind) for _, kind in skipped).items()):
    print(f"     {kind:<18} {n}")
for path, problems in failures:
    print(f"   FAIL {path}")
    for problem in problems:
        print(f"        - {problem}")

# 2. Build records.
build_problems, not_usable = [], {}
status_counts = Counter()
print(f"\n2. Build records in {args.models}/")
for path in sorted(Path(args.models).glob("*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    commit = ((record.get("machine") or {}).get("git") or {}).get("commit") or ""
    existed = {option for option, introduced in OPTION_INTRODUCED.items()
               if commit and had_option(commit, introduced)}
    status, problems = check_build_record(record, existed)
    status_counts[status] += 1
    if status != "usable":
        not_usable[(record["model"], record["precision"])] = status
    extra = non_default_settings(record)
    if status != "usable" or extra or problems:
        note = f"; non-default: {extra}" if extra else ""
        print(f"   {status.upper():<16} {path.name}{note}")
    for problem in problems:
        print(f"   FAIL {path.name}: {problem}")
        build_problems.append((path, problem))
print(f"   {dict(status_counts)} ({sum(status_counts.values())} records; usable ones without "
      "non-default settings are not listed)")

# 3. No result may use a model whose build failed or was never checked.
used_badly = sorted({(r["model"]["name"], r["precision"]) for r in results
                     if (r["model"]["name"], r["precision"]) in not_usable})
for model, precision in used_badly:
    print(f"   FAIL a result uses {model} {precision}, whose build is {not_usable[(model, precision)]!r}")

n_builds = sum(status_counts.values())
ok = bool(measurements) and not failures and not build_problems and not used_badly
print(f"\n{'PASS' if ok else 'FAIL'}: {len(measurements) - len(failures)} of {len(measurements)} result "
      f"records pass the schema check; {n_builds - len({p for p, _ in build_problems})} of {n_builds} "
      f"build records acceptable; {len(used_badly)} results use an unusable build")
sys.exit(0 if ok else 1)
