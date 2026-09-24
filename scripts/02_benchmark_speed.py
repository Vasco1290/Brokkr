"""Measure the speed of an exported ONNX model on this machine.

For stable numbers: plug the laptop in, set Windows power mode to "Best performance",
and close other programs. The script records the power state either way.

Usage:  python scripts/02_benchmark_speed.py [--model NAME] [--precision fp32]
                                             [--threads 1 4] [--sessions 5] [--pause 5]
Needs:  models/<model>_<precision>.onnx  (run scripts/01_export_model.py first)
Writes: results/speed/<model>_<precision>_<N>threads_<power>.json, one per thread count,
        where <power> is "ac", "battery", or "unknownpower" (so runs in each state are all kept)
"""

import argparse
import sys
import time
from pathlib import Path

from brokkr.benchmark import benchmark_latency, combine_sessions
from brokkr.fingerprint import machine_fingerprint
from brokkr.results import make_record, save_record

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--precision", default="fp32")
parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
parser.add_argument("--sessions", type=int, default=5, help="repeat everything this many times")
parser.add_argument("--pause", type=float, default=5, help="seconds to rest between sessions")
args = parser.parse_args()

onnx_path = Path("models") / f"{args.model}_{args.precision}.onnx"
if not onnx_path.exists():
    sys.exit(f"FAIL: {onnx_path} not found. Run scripts/01_export_model.py first.")

machine = machine_fingerprint()
power = machine["power"]
print(f"Machine: {machine['cpu_model']}, {machine['os']} {machine['os_release']}")
print(f"Power:   on AC = {power['on_ac_power']}, mode = {power['power_mode']}")
if power["on_ac_power"] is False:
    print("WARNING: running on battery. Results will be recorded as such and may be slower.")
print(f"Model:   {onnx_path}\n")
power_label = {True: "ac", False: "battery", None: "unknownpower"}[power["on_ac_power"]]

# Sessions go round-robin over thread counts (1, 4, 1, 4, ...) so that if the machine
# speeds up or slows down over time, every thread count is affected equally.
runs = {threads: [] for threads in args.threads}
for session in range(args.sessions):
    if session > 0:
        time.sleep(args.pause)
    for threads in args.threads:
        runs[threads].append(benchmark_latency(onnx_path, num_threads=threads))
    print(f"session {session + 1}/{args.sessions} done")

print(f"\n{'threads':>7} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}   p50 range across sessions")
passed = True
for threads, sessions in runs.items():
    result = combine_sessions(sessions)
    record = make_record("speed", args.model, args.precision, result, machine)
    filename = f"{args.model}_{args.precision}_{threads}threads_{power_label}.json"
    save_record(record, Path("results/speed") / filename)

    m = result["metrics"]
    print(f"{threads:>7} {m['p50_ms']:>8.2f} {m['p95_ms']:>8.2f} {m['p99_ms']:>8.2f}   "
          f"{m['p50_ms_min']:.2f}-{m['p50_ms_max']:.2f} (spread {m['p50_spread_pct']:.0f}%)")
    # Sanity checks: percentiles in order, every session and run recorded.
    passed &= 0 < m["p50_ms"] <= m["p95_ms"] <= m["p99_ms"]
    passed &= len(result["raw"]["sessions"]) == args.sessions
    passed &= all(len(s["latencies_ms"]) == result["settings"]["timed_runs"]
                  for s in result["raw"]["sessions"])

print("\nValues are medians across sessions. Saved to results/speed/")
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
