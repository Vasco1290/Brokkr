"""Laptop latency for the 19 builds of the labels, by the method fixed before any timing.

Usage:  python scripts/44_laptop_latency.py          (the real run: the laptop is left alone meanwhile)
        python scripts/44_laptop_latency.py --smoke --models mobilenet_v3_large --out <folder>
Needs:  models/<model>_fp32.onnx and _int8_percentile99.99.onnx with their build records (scripts/24)
Writes: <out>/<model>_<precision>_laptop_<threads>threads.json (schema-2 speed record) + .npz (every
        timed run of every kept session), and <out>/run_log.txt

The method (docs/hypotheses_stage4.md, notes of 29-30 September and 1-3 October 2026; rules in
brokkr_edge.latency):
- 19 builds: FP32 of all 10 models, Percentile INT8 of the 9 whose build is usable. Batch 1, one random
  picture (seed 0), the model only. Pinned to the performance cores; 1 and 4 threads.
- Before each model: the machine must be on mains power, in "best performance" mode, battery saver
  off, under 10% CPU use over 10 seconds, with no other Brokkr job running; the power state is checked
  again after the model. If a check fails the run stops (exit code 2) and writes nothing for that model.
- Each model: 60 s cool-down, then 10 sessions with 30 s between them. A session times FP32 and INT8
  back to back at 1 thread, then at 4 threads; odd sessions start with FP32, even ones with INT8.
  Each timing is 20 warm-up runs and 300 timed runs.
- A session in which the laptop slept or paused (a run or a gap longer than the larger of 1.0 s and 20
  times the timing's median run) is discarded whole, logged, and repeated. More than 3 discarded
  sessions for a model = FAIL for that model: no record is written for it.
- Resumable: a model whose records all exist is skipped, so the same command can be run again.

--smoke is a check that the script works, not a measurement: one model, 2 sessions, 2 s cool-downs,
machine checks printed but not enforced, records marked "smoke_test" and refused inside results/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr_edge import latency
from brokkr_edge.benchmark import pin_to_cpus
from brokkr_edge.fingerprint import core_types, git_info, machine_fingerprint, physical_cores, power_state
from brokkr_edge.model_list import load_model_list
from brokkr_edge.results import sha256_of, written_atomically
from brokkr_edge.schema import make_measurement, metric, save_measurement
from brokkr_edge.test_run import keep_awake, usable_precisions

REPO = Path(__file__).resolve().parents[1]
METHOD = "docs/hypotheses_stage4.md, notes of 29-30 September and 1-3 October 2026"

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=list(load_model_list()))
parser.add_argument("--out", default="results/latency")
parser.add_argument("--smoke", action="store_true", help="check that the script works; not a measurement")
args = parser.parse_args()
out_dir = Path(args.out)
inside_results = (REPO / "results") in [out_dir.resolve(), *out_dir.resolve().parents]
if args.smoke and (len(args.models) != 1 or inside_results):
    sys.exit("--smoke needs exactly one model and an --out folder outside results/")
sessions_wanted = 2 if args.smoke else latency.SESSIONS
cooldown_model = 2 if args.smoke else latency.COOLDOWN_BEFORE_MODEL_S
cooldown_session = 2 if args.smoke else latency.COOLDOWN_BETWEEN_SESSIONS_S
out_dir.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {message}"
    print(line, flush=True)
    with (out_dir / "run_log.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def record_path(model: str, precision: str, threads: int) -> Path:
    return out_dir / f"{model}_{precision}_laptop_{threads}threads.json"


def check_machine(when: str, with_idle: bool) -> dict:
    """The machine checks; stops the run if one fails (a smoke test only prints them)."""
    power = power_state()
    idle = latency.idle_cpu_percent() if with_idle else None
    jobs = latency.other_brokkr_jobs(REPO)
    problems = latency.machine_problems(power, idle, jobs)
    if problems:
        log(f"machine check {when}: " + "; ".join(problems))
        if not args.smoke:
            log("STOP: the machine is not ready. Fix the above and run the same command again.")
            keep_awake(False)
            sys.exit(2)
    return {"power": power, "idle_cpu_percent": idle, "other_brokkr_jobs": jobs}


if git_info()["dirty"] and not args.smoke:
    sys.exit("STOP: uncommitted changes. Latency records must come from a clean commit.")
cores = core_types()
pinned = cores["performance"]
pin_to_cpus(pinned)
core_map = physical_cores()
vnni = latency.vnni()
models = load_model_list()
keep_awake(True)
smoke = " (SMOKE TEST, not a measurement)" if args.smoke else ""
log(f"laptop latency at {git_info()['commit'][:7]}{smoke}; pinned to logical CPUs {pinned}; "
    f"{sessions_wanted} sessions; threads {list(latency.THREAD_COUNTS)}; {vnni['text']}")

failed_models = []
for model in args.models:
    builds = usable_precisions(model)
    wanted = [record_path(model, p, t) for p in builds for t in latency.THREAD_COUNTS]
    if all(p.exists() and p.with_suffix(".npz").exists() for p in wanted):
        log(f"skip (complete): {model}")
        continue
    before = check_machine(f"before {model}", with_idle=True)
    time.sleep(cooldown_model)
    kept, discarded = [], []  # kept: one {(precision, threads): timing} per session
    while len(kept) < sessions_wanted and len(discarded) <= latency.MAX_DISCARDED_SESSIONS:
        number = len(kept) + 1
        order = latency.session_order(builds, number)
        timings, pause = {}, None
        for threads in latency.THREAD_COUNTS:
            for precision in order:
                timing = latency.time_build(Path("models") / f"{model}_{precision}.onnx", threads)
                pause = latency.interruption(timing["starts"], timing["ends"])
                if pause:
                    pause |= {"session": number, "precision": precision, "threads": threads,
                              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    break
                timings[(precision, threads)] = timing
            if pause:
                break
        if pause:
            discarded.append(pause)
            log(f"DISCARDED {model} session {number} ({pause['precision']}, {pause['threads']} threads): "
                f"longest run {pause['longest_run_s']:.2f} s, longest gap {pause['longest_gap_s']:.2f} s, "
                f"limit {pause['limit_s']:.2f} s; discarded so far: {len(discarded)}")
        else:
            kept.append(timings)
            log(f"{model} session {number} of {sessions_wanted} done (order: {', '.join(order)})")
        if len(kept) < sessions_wanted:
            time.sleep(cooldown_session)
    after = check_machine(f"after {model}", with_idle=False)
    if len(kept) < sessions_wanted:
        failed_models.append(model)
        log(f"FAIL {model}: {len(discarded)} sessions discarded (more than "
            f"{latency.MAX_DISCARDED_SESSIONS}); no latency record written for it")
        continue

    machine = machine_fingerprint()
    entry = models[model]
    for precision in builds:
        build = json.loads((Path("models") / f"{model}_{precision}.json").read_text(encoding="utf-8"))
        for threads in latency.THREAD_COUNTS:
            runs = [session[(precision, threads)] for session in kept]
            summary = latency.summarise([r["latencies_ms"] for r in runs])
            path = record_path(model, precision, threads)
            with written_atomically(path.with_suffix(".npz")) as tmp, tmp.open("wb") as f:
                np.savez_compressed(f, latencies_ms=np.array([r["latencies_ms"] for r in runs]),
                                    run_starts=np.array([r["starts"] for r in runs]),
                                    run_ends=np.array([r["ends"] for r in runs]))
            record = make_measurement(
                "speed", {"name": model, "weights": entry["weights"], "licence": entry["licence"]},
                precision,
                {"name": "onnxruntime", "version": ort.__version__,
                 "execution_provider": "CPUExecutionProvider", "threads": threads,
                 "intra_op_threads": threads, "inter_op_threads": 1,
                 "graph_optimisation": latency.GRAPH_OPTIMISATION, "spinning": "on (ONNX Runtime's default)"},
                "laptop", machine, None, None,
                {name: metric(summary[name]) for name in
                 ("p50_ms", "p95_ms", "p99_ms", "p50_ms_fastest_session", "p50_ms_slowest_session",
                  "spread_pct")},
                {"script": "scripts/44_laptop_latency.py", "method": METHOD, "smoke_test": args.smoke,
                 "batch": 1, "input_shape": [1, 3, 224, 224], "input": "random, seed 0",
                 "what_is_timed": "model only; excludes loading and pre-processing",
                 "warmup_runs": latency.WARMUP_RUNS, "timed_runs": latency.TIMED_RUNS,
                 "sessions": len(kept),
                 "session_order": "FP32 first in odd sessions, INT8 first in even ones; 1 thread, then 4",
                 "pinned_cpus": pinned, "physical_core_of_each_cpu": core_map,
                 "cooldown_s": {"before_model": cooldown_model, "between_sessions": cooldown_session},
                 "unstable": summary["unstable"], "unstable_above_pct": latency.UNSTABLE_ABOVE_PCT,
                 "pause_rule": {"floor_s": latency.PAUSE_FLOOR_S, "times_median": latency.PAUSE_TIMES_MEDIAN,
                                "max_discarded_sessions": latency.MAX_DISCARDED_SESSIONS},
                 "discarded_sessions": discarded, "vnni": vnni,
                 "clock_disagreement_s": max(r["clock_disagreement_s"] for r in runs),
                 "machine_before": before, "machine_after": after},
                {"file": path.with_suffix(".npz").name, "sha256": sha256_of(path.with_suffix(".npz"))},
                [{"file": f"models/{model}_{precision}.onnx", "sha256": build["file"]["sha256"]}],
            )
            save_measurement(record, path)
            log(f"wrote {path.name}: p50 {summary['p50_ms']:.2f} ms, p95 {summary['p95_ms']:.2f} ms, p99 "
                f"{summary['p99_ms']:.2f} ms, spread {summary['spread_pct']:.1f}%"
                f"{' UNSTABLE' if summary['unstable'] else ''}; discarded sessions: {len(discarded)}")

keep_awake(False)
done = sum(record_path(m, p, t).exists() for m in args.models for p in usable_precisions(m)
           for t in latency.THREAD_COUNTS)
expected = sum(len(usable_precisions(m)) * len(latency.THREAD_COUNTS) for m in args.models)
verdict = "PASS" if done == expected and not failed_models else "FAIL"
log(f"{verdict}: {done} of {expected} latency records exist"
    + (f"; failed models: {failed_models}" if failed_models else ""))
sys.exit(0 if verdict == "PASS" else 1)
