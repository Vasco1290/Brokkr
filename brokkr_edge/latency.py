"""Latency measured in sessions, by the method fixed before any timing (docs/hypotheses_stage4.md, notes
of 29-30 September and 1-3 October 2026). This module holds the rules; scripts/44_laptop_latency.py
runs them. Nothing here is specific to one device.

- One timing = one build at one thread count: 20 warm-up runs (thrown away), then 300 timed runs of
  one random picture (batch 1, seed 0). Only the model is timed, not loading or pre-processing.
- One session = a model's builds timed back to back at each thread count. A model gets 10 sessions;
  p50, p95 and p99 are the medians across sessions; the spread is the interquartile range of the
  sessions' p50s as a share of their median, and above 10% the result is flagged "unstable".
- A session interrupted by sleep or a pause is discarded, never recorded: a timing is interrupted if
  any timed run, or any gap between two timed runs, lasts longer than the larger of 1.0 second and 20
  times that timing's median run time (confirmed by H, 2 October 2026).
- The machine must be ready: on mains power, "best performance" mode, battery saver off, less than
  10% CPU use over 10 seconds, and no other Brokkr job running.
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

WARMUP_RUNS, TIMED_RUNS, SESSIONS = 20, 300, 10
THREAD_COUNTS = (1, 4)
COOLDOWN_BEFORE_MODEL_S, COOLDOWN_BETWEEN_SESSIONS_S = 60, 30
IDLE_CHECK_S, IDLE_LIMIT_PCT = 10, 10.0
UNSTABLE_ABOVE_PCT = 10.0
PAUSE_FLOOR_S, PAUSE_TIMES_MEDIAN = 1.0, 20
MAX_DISCARDED_SESSIONS = 3  # per model; one more is a FAIL for that model
GRAPH_OPTIMISATION = "ORT_ENABLE_ALL"  # ONNX Runtime's default, set explicitly so the record can say so


def make_session(onnx_path, threads: int) -> ort.InferenceSession:
    """An ONNX Runtime CPU session with exactly `threads` threads and the settings the records state."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, GRAPH_OPTIMISATION)
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def time_build(onnx_path, threads: int, input_shape=(1, 3, 224, 224), seed: int = 0) -> dict:
    """One timing. Each timed run's latency comes from the precise timer; its start and end are also
    read from the wall clock, which keeps counting while the machine sleeps (to spot a pause)."""
    session = make_session(onnx_path, threads)
    name = session.get_inputs()[0].name
    picture = np.random.default_rng(seed).standard_normal(input_shape).astype(np.float32)
    for _ in range(WARMUP_RUNS):
        session.run(None, {name: picture})
    latencies_ms, starts, ends = [], [], []
    first = time.perf_counter()
    for _ in range(TIMED_RUNS):
        starts.append(time.time())
        t0 = time.perf_counter_ns()
        session.run(None, {name: picture})
        latencies_ms.append((time.perf_counter_ns() - t0) / 1e6)
        ends.append(time.time())
    by_timer, by_wall_clock = time.perf_counter() - first, ends[-1] - starts[0]
    return {"latencies_ms": latencies_ms, "starts": starts, "ends": ends,
            "clock_disagreement_s": abs(by_wall_clock - by_timer)}


def interruption(starts: list, ends: list) -> dict | None:
    """None if the timing ran without a pause; otherwise what was seen (the session is then discarded).

    starts, ends: wall-clock seconds at the start and end of each timed run.
    """
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    runs = ends - starts
    gaps = starts[1:] - ends[:-1]
    limit = max(PAUSE_FLOOR_S, PAUSE_TIMES_MEDIAN * float(np.median(runs)))
    longest_run, longest_gap = float(runs.max()), float(gaps.max()) if len(gaps) else 0.0
    if longest_run <= limit and longest_gap <= limit:
        return None
    return {"longest_run_s": longest_run, "longest_gap_s": longest_gap, "limit_s": limit,
            "median_run_s": float(np.median(runs))}


def session_order(builds: list, session_number: int) -> list:
    """Builds in the order they are timed in session 1, 2, 3...: first to last in odd sessions, reversed
    in even ones, so slow drift (the laptop warming up) affects FP32 and INT8 alike."""
    return list(builds) if session_number % 2 == 1 else list(reversed(builds))


def summarise(sessions_ms: list) -> dict:
    """One build at one thread count, over its sessions (each a list of latencies in milliseconds)."""
    per_session = np.array([[np.percentile(s, q) for q in (50, 95, 99)] for s in sessions_ms])
    p50s = per_session[:, 0]
    median_p50 = float(np.median(p50s))
    q1, q3 = np.percentile(p50s, [25, 75])
    spread = float((q3 - q1) / median_p50 * 100)
    return {
        "p50_ms": median_p50,
        "p95_ms": float(np.median(per_session[:, 1])),
        "p99_ms": float(np.median(per_session[:, 2])),
        "p50_ms_fastest_session": float(p50s.min()),
        "p50_ms_slowest_session": float(p50s.max()),
        "spread_pct": spread,
        "unstable": spread > UNSTABLE_ABOVE_PCT,
    }


def is_brokkr_job(cmdline: list, cwd: str | None, repo: Path) -> bool:
    """Is this command line a Python script from the repository's scripts/ folder, or a brokkr command?"""
    scripts = (Path(repo) / "scripts").resolve()
    for arg in cmdline:
        text = arg.replace("\\", "/").lower()
        if Path(text).name.startswith("brokkr-edge") or text == "brokkr_edge.cli":
            return True
        if text.endswith(".py"):
            path = Path(arg) if Path(arg).is_absolute() else Path(cwd or ".") / arg
            if scripts in path.resolve().parents:
                return True
    return False


def other_brokkr_jobs(repo: Path) -> list:
    """Command lines of other running Brokkr jobs (this process, its parents and its children are not
    counted). Uses psutil, so the same check works on Windows and Linux."""
    import psutil

    me = psutil.Process()
    family = {me.pid} | {p.pid for p in me.parents()} | {p.pid for p in me.children(recursive=True)}
    jobs = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
        pid, cmdline, cwd = process.info["pid"], process.info["cmdline"], process.info["cwd"]
        if pid not in family and cmdline and is_brokkr_job(cmdline, cwd, repo):
            jobs.append(" ".join(cmdline))
    return jobs


def idle_cpu_percent(seconds: float = IDLE_CHECK_S) -> float:
    """Total CPU use over `seconds`, in percent (the machine is idle if this is low)."""
    import psutil

    return float(psutil.cpu_percent(interval=seconds))


def machine_problems(power: dict, idle_percent: float | None, other_jobs: list) -> list:
    """Every reason the machine is not ready for a timing, as sentences. Empty = ready.

    idle_percent None = not measured this time (the end-of-model check looks at the power state only).
    """
    problems = []
    if power.get("on_ac_power") is not True:
        problems.append("not on mains power")
    if power.get("power_mode") != "best performance":
        problems.append(f"power mode is {power.get('power_mode')!r}, not 'best performance'")
    if power.get("battery_saver"):
        problems.append("battery saver is on")
    if idle_percent is not None and idle_percent >= IDLE_LIMIT_PCT:
        problems.append(f"CPU use {idle_percent:.1f}% over {IDLE_CHECK_S} s (limit {IDLE_LIMIT_PCT:.0f}%)")
    if other_jobs:
        problems.append(f"another Brokkr job is running: {other_jobs[0]}")
    return problems


def vnni() -> dict:
    """Whether the CPU has the VNNI instructions that speed up INT8, read with py-cpuinfo. py-cpuinfo
    reports AVX512-VNNI but has no check for AVX-VNNI, so that one stays unknown (H, 30 September 2026)."""
    import cpuinfo

    flags = cpuinfo.get_cpu_info().get("flags", [])
    has = "avx512_vnni" in flags or "avx512vnni" in flags
    return {"avx512_vnni": has, "avx_vnni": None, "source": f"py-cpuinfo {cpuinfo.CPUINFO_VERSION_STRING}",
            "text": f"AVX512-VNNI: {'yes' if has else 'no'}; AVX-VNNI: unknown (not reported by py-cpuinfo)"}
