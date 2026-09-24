"""Measure the speed of exported ONNX models on this machine.

For stable numbers: plug the laptop in, set Windows power mode to "Best performance",
and close other programs. The script records the power state either way.

Usage:  python scripts/02_benchmark_speed.py [--model NAME] [--precision fp32 fp16 int8]
                    [--threads 1 4] [--cores performance|efficiency|all] [--sessions 10]
                    [--pause 5] [--cooldown 0]
Needs:  models/<model>_<precision>.onnx  (scripts/01_export_model.py, scripts/04_quantize.py)
Writes: results/speed/<model>_<precision>_<N>threads_<cores>_<power>.json, one per precision and
        thread count; <power> is "ac", "battery", or "unknownpower", plus the Windows power mode

--cores: hybrid CPUs mix fast "performance" and slow "efficiency" cores. Unpinned, the OS moves
the benchmark between them and timings jump between two speeds, so by default the benchmark is
pinned to the performance cores. Thread counts larger than the chosen CPUs are skipped.
"""

import argparse
import sys
import time
from pathlib import Path

from brokkr.benchmark import benchmark_latency, combine_sessions, pin_to_cpus
from brokkr.fingerprint import machine_fingerprint
from brokkr.results import make_record, save_record

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--precision", nargs="+", default=["fp32"])
parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
parser.add_argument("--cores", choices=["performance", "efficiency", "all"], default="performance")
parser.add_argument("--sessions", type=int, default=10, help="repeat everything this many times")
parser.add_argument("--cooldown", type=float, default=0,
                    help="seconds to rest before starting, e.g. after heavy work heated the machine")
parser.add_argument("--pause", type=float, default=5, help="seconds to rest between sessions")
args = parser.parse_args()

paths = {p: Path("models") / f"{args.model}_{p}.onnx" for p in args.precision}
for path in paths.values():
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run scripts/01_export_model.py / 04_quantize.py first.")

machine = machine_fingerprint()
power = machine["power"]
print(f"Machine: {machine['cpu_model']}, {machine['os']} {machine['os_release']}")
print(f"Power:   on AC = {power['on_ac_power']}, mode = {power['power_mode']}")
if power["on_ac_power"] is False:
    print("WARNING: running on battery. Results will be recorded as such and may be slower.")
print(f"Models:  {', '.join(str(p) for p in paths.values())}")
power_label = {True: "ac", False: "battery", None: "unknownpower"}[power["on_ac_power"]]
if power["power_mode"]:  # e.g. "ac-bestperformance", so different modes never overwrite each other
    power_label += "-" + power["power_mode"].replace(" ", "")

# Choose and pin the CPUs. Unknown core layout -> use all CPUs and say so.
types = machine["core_types"]
if args.cores == "all" or types is None:
    cpu_ids = list(range(machine["cpu_count_logical"]))
    if args.cores != "all":
        print("WARNING: core types unknown on this machine - using all CPUs, not pinned.")
    cores_label = "all"
else:
    cpu_ids = types[args.cores]
    if not cpu_ids:
        sys.exit(f"FAIL: this machine has no {args.cores} cores ({types})")
    cores_label = args.cores
    pin_to_cpus(cpu_ids)
print(f"Cores:   {cores_label} -> logical CPUs {cpu_ids}")
threads_list = [t for t in args.threads if t <= len(cpu_ids)]
for t in sorted(set(args.threads) - set(threads_list)):
    print(f"Skipping {t} threads: more than the {len(cpu_ids)} chosen CPUs.")
print()

if args.cooldown:
    print(f"Cooling down for {args.cooldown:.0f} s before measuring...")
    time.sleep(args.cooldown)

# Each session takes turns over every (precision, threads) pair, so if the machine speeds up
# or slows down over time, every combination is affected equally and comparisons stay fair.
combos = [(p, t) for p in args.precision for t in threads_list]
runs = {combo: [] for combo in combos}
for session in range(args.sessions):
    if session > 0:
        time.sleep(args.pause)
    for precision, threads in combos:
        runs[(precision, threads)].append(benchmark_latency(paths[precision], num_threads=threads))
    print(f"session {session + 1}/{args.sessions} done")

print(f"\n{'precision':>9} {'threads':>7} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}   "
      "p50 range across sessions")
# IQR = spread of the middle half of sessions; full spread = fastest vs slowest session.
passed = True
for (precision, threads), sessions in runs.items():
    result = combine_sessions(sessions)
    result["settings"].update({"cores": cores_label, "cpu_ids": cpu_ids})
    record = make_record("speed", args.model, precision, result, machine)
    filename = f"{args.model}_{precision}_{threads}threads_{cores_label}_{power_label}.json"
    save_record(record, Path("results/speed") / filename)

    m = result["metrics"]
    print(f"{precision:>9} {threads:>7} {m['p50_ms']:>8.2f} {m['p95_ms']:>8.2f} {m['p99_ms']:>8.2f}   "
          f"{m['p50_ms_min']:.2f}-{m['p50_ms_max']:.2f} "
          f"(IQR {m['p50_iqr_pct']:.0f}%, full spread {m['p50_spread_pct']:.0f}%)")
    # Sanity checks: percentiles in order, every session and run recorded.
    passed &= 0 < m["p50_ms"] <= m["p95_ms"] <= m["p99_ms"]
    passed &= len(result["raw"]["sessions"]) == args.sessions
    passed &= all(len(s["latencies_ms"]) == result["settings"]["timed_runs"]
                  for s in result["raw"]["sessions"])

print("\nValues are medians across sessions. Saved to results/speed/")
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
