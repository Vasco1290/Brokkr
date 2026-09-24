"""Measure how fast an ONNX model runs on the current machine.

The rules (from the project brief) are enforced in code: a fixed thread count,
at least 20 warm-up runs that are thrown away, then at least 100 timed runs.
This module has no Brokkr-specific assumptions: it only needs an .onnx file.
"""

import os
import platform
import time

import numpy as np
import onnxruntime as ort

MIN_WARMUP = 20
MIN_RUNS = 100


def pin_to_cpus(cpu_ids: list) -> None:
    """Make this process (and every thread it starts) run only on the given logical CPUs.

    On hybrid CPUs, pinning to one kind of core stops the OS from moving the benchmark
    between fast and slow cores, which otherwise makes timings jump between two speeds.
    """
    if platform.system() == "Linux":
        os.sched_setaffinity(0, set(cpu_ids))
    elif platform.system() == "Windows":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        mask = sum(1 << cpu for cpu in cpu_ids)
        if not kernel32.SetProcessAffinityMask(ctypes.c_void_p(kernel32.GetCurrentProcess()),
                                               ctypes.c_size_t(mask)):
            raise OSError(f"could not pin to CPUs {cpu_ids}")
    else:
        raise NotImplementedError(f"CPU pinning not supported on {platform.system()}")


def make_session(onnx_path, num_threads: int) -> ort.InferenceSession:
    """Open an ONNX Runtime session that uses exactly `num_threads` CPU threads."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads  # threads used inside one operation (e.g. a convolution)
    options.inter_op_num_threads = 1  # run operations one after another, not in parallel
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def summarise(latencies_ms) -> dict:
    """Summary statistics of a list of timings, in milliseconds."""
    t = np.asarray(latencies_ms)
    return {
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std()),
        "min_ms": float(t.min()),
        "p50_ms": float(np.percentile(t, 50)),  # median: half the runs were faster than this
        "p95_ms": float(np.percentile(t, 95)),  # 95% of runs were faster than this
        "p99_ms": float(np.percentile(t, 99)),
        "max_ms": float(t.max()),
    }


def benchmark_latency(onnx_path, input_shape=(1, 3, 224, 224), num_threads: int = 4,
                      warmup: int = MIN_WARMUP, runs: int = MIN_RUNS, seed: int = 0) -> dict:
    """Time single inferences of an ONNX model. Returns settings, summary metrics, and raw timings."""
    if warmup < MIN_WARMUP or runs < MIN_RUNS:
        raise ValueError(f"need warmup >= {MIN_WARMUP} and runs >= {MIN_RUNS}")

    session = make_session(onnx_path, num_threads)
    input_name = session.get_inputs()[0].name
    inputs = np.random.default_rng(seed).standard_normal(input_shape).astype(np.float32)

    # Warm-up: the first runs are slower (memory allocation, caches), so they are not counted.
    for _ in range(warmup):
        session.run(None, {input_name: inputs})

    latencies_ms = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        session.run(None, {input_name: inputs})
        latencies_ms.append((time.perf_counter_ns() - start) / 1e6)

    return {
        "settings": {
            "input_shape": list(input_shape),
            "num_threads": num_threads,
            "warmup_runs": warmup,
            "timed_runs": runs,
            "seed": seed,
            "execution_provider": "CPUExecutionProvider",
            "onnxruntime_version": ort.__version__,
        },
        "metrics": summarise(latencies_ms),
        "raw": {"latencies_ms": latencies_ms},
    }


def combine_sessions(sessions: list) -> dict:
    """Combine several benchmark_latency() results for the same model and settings.

    One session can be unlucky (a background program, the CPU changing speed), so we run
    several and report the MEDIAN of each statistic across sessions, plus how much the
    sessions disagreed ("spread"). A large spread means the machine was not stable.
    """
    if len(sessions) < 2:
        raise ValueError("need at least 2 sessions to measure how much they vary")
    per_session = [s["metrics"] for s in sessions]
    p50s = [m["p50_ms"] for m in per_session]

    metrics = {key: float(np.median([m[key] for m in per_session]))
               for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")}
    metrics["p50_ms_min"] = float(min(p50s))
    metrics["p50_ms_max"] = float(max(p50s))
    # Spread: gap between the fastest and slowest session, as a % of the median.
    # One unlucky session (e.g. the laptop heating up) is enough to make this large.
    metrics["p50_spread_pct"] = float((max(p50s) - min(p50s)) / metrics["p50_ms"] * 100)
    # IQR spread: gap between the 25th and 75th percentile sessions, as a % of the median.
    # It ignores the most extreme sessions, so it measures how stable the typical session was.
    q1, q3 = np.percentile(p50s, [25, 75])
    metrics["p50_iqr_pct"] = float((q3 - q1) / metrics["p50_ms"] * 100)

    return {
        "settings": {**sessions[0]["settings"], "sessions": len(sessions)},
        "metrics": metrics,
        "raw": {"sessions": [{"metrics": s["metrics"], "latencies_ms": s["raw"]["latencies_ms"]}
                             for s in sessions]},
    }
