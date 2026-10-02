"""Print tables that read across the labels (read-only: it changes nothing and judges nothing).

Usage:  python scripts/42_label_tables.py [--labels labels] [--model convnext_tiny]
Needs:  <labels>/<model>/label.json (scripts/39_make_labels.py)

Every number is read from a label.json; the script only counts rows and measures how far an interval
is from its line. Sections:
1. Harmful rows by the line they failed (accuracy only, coverage only, both), for FP32 and INT8 rows.
2. What the ten labels say (the summary counts of each label).
3. For each model and build, the row closest to each line, with its interval and its distance to the
   line; and every borderline row.
4. Rows whose shrinking cost is marked "not informative" (FP32 near floor), with their envelope states.
With --model: that model's "hurt by shrinking" and "large shrinking cost" conditions, with intervals.
"""

import argparse
import json
from pathlib import Path

from brokkr_edge.model_list import load_model_list

parser = argparse.ArgumentParser()
parser.add_argument("--labels", default="labels")
parser.add_argument("--model")
args = parser.parse_args()

labels = {}
for model in load_model_list():
    path = Path(args.labels) / model / "label.json"
    if path.exists():
        labels[model] = json.loads(path.read_text(encoding="utf-8"))
print(f"{len(labels)} labels read from {args.labels}; commits: "
      f"{sorted({x['generated']['commit'][:7] for x in labels.values()})}")


def pts(x: float) -> str:
    return f"{100 * x:+.2f}"


def interval(m: dict) -> str:
    return f"{pts(m['value'])} ({pts(m['ci95'][0])} to {pts(m['ci95'][1])})"


def share(m: dict) -> str:
    return f"{100 * m['value']:.2f} ({100 * m['ci95'][0]:.2f} to {100 * m['ci95'][1]:.2f})"


def parts(label: dict) -> dict:
    """Lookups for one label: its two builds, condition names, envelope rows and measurements."""
    builds = {b["role"]: b for b in label["builds"]}
    return {
        "name": label["model"]["display_name"],
        "builds": {"FP32": builds["reference"], "INT8": builds["labelled"]},
        "cond": {c["condition_id"]: c["label"] for c in label["conditions"]},
        "rows": {(r["build_id"], r["condition_id"]): r for r in label["envelope"]["rows"]},
        "m": {(m["metric"], m["build_id"], m["condition_id"]): m for m in label["measurements"]},
        "rule": label["envelope"]["rule"],
    }


def distance(ci: list, line: float) -> float:
    """How far the interval is from its line: 0 if it straddles the line (borderline), positive if the
    whole interval clears it (the lower end's margin), negative if the whole interval is below it."""
    if ci[0] >= line:
        return ci[0] - line
    if ci[1] < line:
        return ci[1] - line
    return 0.0


print("\n1. Harmful rows by the line they failed (rows = damaged conditions; 'accuracy' = the whole "
      "damage-drop interval is below its line, 'coverage' = the whole coverage interval is below its line)")
print(f"{'model':20s} {'build':5s} {'rows':>4s} {'not harmful':>11s} {'borderline':>10s} {'harmful':>7s} "
      f"{'accuracy only':>13s} {'coverage only':>13s} {'both':>4s}")
totals = {}
for label in labels.values():
    p = parts(label)
    for build_name, build in p["builds"].items():
        rows = [r for (bid, _), r in p["rows"].items() if bid == build["build_id"]]
        if build["status"] != "usable":
            print(f"{p['name']:20s} {build_name:5s} {len(rows):4d}   INT8 build failed: no rows judged")
            continue
        counts = {
            "not harmful": sum(r["state"] == "not harmful" for r in rows),
            "borderline": sum(r["state"] == "borderline" for r in rows),
            "harmful": sum(r["state"] == "harmful" for r in rows),
            "accuracy only": sum(r["failed"] == ["damage drop"] for r in rows),
            "coverage only": sum(r["failed"] == ["coverage"] for r in rows),
            "both": sum(r["failed"] == ["damage drop", "coverage"] for r in rows),
        }
        print(f"{p['name']:20s} {build_name:5s} {len(rows):4d} {counts['not harmful']:11d} "
              f"{counts['borderline']:10d} {counts['harmful']:7d} {counts['accuracy only']:13d} "
              f"{counts['coverage only']:13d} {counts['both']:4d}")
        total = totals.setdefault(build_name, dict.fromkeys(counts, 0) | {"rows": 0})
        total["rows"] += len(rows)
        for key, value in counts.items():
            total[key] += value
for build_name, t in totals.items():
    print(f"{'all models':20s} {build_name:5s} {t['rows']:4d} {t['not harmful']:11d} {t['borderline']:10d} "
          f"{t['harmful']:7d} {t['accuracy only']:13d} {t['coverage only']:13d} {t['both']:4d}")

print("\n   INT8 'too hard for this model' rows (FP32 harmful too), by the line FP32 itself failed there")
print(f"{'model':20s} {'too hard':>8s} {'FP32 accuracy only':>18s} {'FP32 coverage only':>18s} "
      f"{'FP32 both':>9s}")
for label in labels.values():
    p = parts(label)
    too_hard = [c for line in label["summary"]["lines"] if line["cause"] == "too hard for this model"
                for c in line["conditions"]]
    fp32 = [p["rows"][(p["builds"]["FP32"]["build_id"], c)]["failed"] for c in too_hard]
    print(f"{p['name']:20s} {len(too_hard):8d} {fp32.count(['damage drop']):18d} "
          f"{fp32.count(['coverage']):18d} {fp32.count(['damage drop', 'coverage']):9d}")

print("\n2. What the ten labels say (counts of tested conditions, from each label's summary)")
print(f"{'model':20s} {'clean shrinking cost':>24s} {'tested':>6s} {'not harmful':>11s} {'borderline':>10s} "
      f"{'too hard':>8s} {'hurt':>4s} {'unclear':>7s} {'large cost':>10s} {'FP32 harmful':>12s} "
      f"{'coverage failed':>15s}")
for label in labels.values():
    p, s = parts(label), label["summary"]
    count = {(line["state"], line["cause"]): line["count"] for line in s["lines"]}
    int8 = p["builds"]["INT8"]
    if int8["status"] != "usable":
        print(f"{p['name']:20s} {'INT8 build failed':>24s} {s['tested_conditions']:6d} {'-':>11s} {'-':>10s} "
              f"{'-':>8s} {'-':>4s} {'-':>7s} {'-':>10s} {s['reference_harmful']['count']:12d} {'-':>15s}")
        continue
    cost = interval(p["m"][("shrinking_cost", int8["build_id"], "clean")])
    print(f"{p['name']:20s} {cost:>24s} {s['tested_conditions']:6d} "
          f"{count.get(('not harmful', None), 0):11d} {count.get(('borderline', None), 0):10d} "
          f"{count.get(('harmful', 'too hard for this model'), 0):8d} "
          f"{count.get(('harmful', 'hurt by shrinking'), 0):4d} "
          f"{count.get(('harmful', 'cause unclear'), 0):7d} {s['large_shrinking_cost']['count']:10d} "
          f"{s['reference_harmful']['count']:12d} {s['coverage_failed']['count']:15d}")

print("\n3. The row closest to each line, per model and build (distance in points: 0 = the interval "
      "straddles the line, + = the whole interval clears it, - = the whole interval is below it)")
borderline = []
for label in labels.values():
    p = parts(label)
    lines = {"coverage": p["rule"]["coverage_min"], "damage_drop": p["rule"]["damage_drop_min"]}
    for build_name, build in p["builds"].items():
        if build["status"] != "usable":
            continue
        for metric, line in lines.items():
            found = [(abs(distance(m["ci95"], line)), cid, m) for (name, bid, cid), m in p["m"].items()
                     if name == metric and bid == build["build_id"] and cid != "clean"]
            _, cid, m = min(found, key=lambda x: x[0])
            print(f"{p['name']:20s} {build_name:5s} {metric:11s} line {pts(line):>7s}: {p['cond'][cid]:30s} "
                  f"{interval(m):>26s}  distance {pts(distance(m['ci95'], line))}  "
                  f"[{p['rows'][(build['build_id'], cid)]['state']}]")
        borderline += [
            (p["name"], build_name, p["cond"][cid], r["why"])
            for (bid, cid), r in p["rows"].items()
            if bid == build["build_id"] and r["state"] == "borderline"
        ]
print(f"\n   Borderline rows in all labels: {len(borderline)}")
for name, build_name, cond, why in borderline:
    print(f"   {name:20s} {build_name:5s} {cond:30s} {'; '.join(why)}")

print("\n4. Rows whose shrinking cost is marked 'not informative' (FP32 top-1 under the condition below "
      "the near-floor line); their envelope states are judged as for any other row")
for label in labels.values():
    p = parts(label)
    fp32, int8 = p["builds"]["FP32"]["build_id"], p["builds"]["INT8"]["build_id"]
    for (bid, cid), r in p["rows"].items():
        if bid == int8 and r["shrinking_cost_flag"] == "not informative":
            fp32_top1 = 100 * p["m"][("top1", fp32, cid)]["value"]
            print(f"{p['name']:20s} {p['cond'][cid]:30s} FP32 top-1 {fp32_top1:.2f}%  "
                  f"shrinking cost {interval(p['m'][('shrinking_cost', int8, cid)])}  "
                  f"FP32 [{p['rows'][(fp32, cid)]['state']}]  INT8 [{r['state']}]")

if args.model:
    label = labels[args.model]
    p = parts(label)
    fp32, int8 = p["builds"]["FP32"]["build_id"], p["builds"]["INT8"]["build_id"]
    hurt = [c for line in label["summary"]["lines"] if line["cause"] == "hurt by shrinking"
            for c in line["conditions"]]
    large = label["summary"]["large_shrinking_cost"]["conditions"]
    for title, conditions in (("hurt by shrinking", hurt), ("large shrinking cost", large)):
        print(f"\n{p['name']}: {title} ({len(conditions)} conditions; top-1 and coverage in %, "
              f"others in points)")
        for cid in conditions:
            print(f"  {p['cond'][cid]}")
            for build_name, bid in (("FP32", fp32), ("INT8", int8)):
                row = p["rows"][(bid, cid)]
                print(f"    {build_name}: top-1 {share(p['m'][('top1', bid, cid)])}; damage drop "
                      f"{interval(p['m'][('damage_drop', bid, cid)])}; coverage "
                      f"{share(p['m'][('coverage', bid, cid)])}, set size "
                      f"{p['m'][('mean_set_size', bid, cid)]['value']:.2f}; [{row['state']}"
                      f"{': failed ' + ' and '.join(row['failed']) if row['failed'] else ''}]")
            print(f"    shrinking cost {interval(p['m'][('shrinking_cost', int8, cid)])} "
                  f"[{p['rows'][(int8, cid)]['shrinking_cost_flag']}]")
