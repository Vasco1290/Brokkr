"""The latency rules (brokkr_edge.latency) on made-up timings whose right answer is known.

The numbers here are invented for the tests; they are never results.
"""

from pathlib import Path

import numpy as np

from brokkr_edge import latency


def timing(run_s: float, runs: int = 300, gap_s: float = 0.0001) -> tuple:
    """Wall-clock starts and ends of `runs` back-to-back runs of `run_s` seconds each."""
    starts = np.arange(runs) * (run_s + gap_s)
    return starts.tolist(), (starts + run_s).tolist()


def test_a_normal_timing_is_not_an_interruption():
    assert latency.interruption(*timing(0.008)) is None
    starts, ends = timing(0.008)
    ends[5] += 0.05  # one slow run (the system briefly busy) is real latency, and is kept
    assert latency.interruption(starts, ends) is None


def test_a_long_gap_between_runs_is_an_interruption():
    starts, ends = timing(0.008)
    for i in range(100, 300):  # the laptop slept for 30 s between run 99 and run 100
        starts[i] += 30
        ends[i] += 30
    found = latency.interruption(starts, ends)
    assert found and found["longest_gap_s"] > 29 and found["limit_s"] == 1.0


def test_a_long_run_is_an_interruption():
    starts, ends = timing(0.008)
    ends[100] += 5  # the laptop slept during run 100 (later runs overlap in this toy example)
    found = latency.interruption(starts, ends)
    assert found and found["longest_run_s"] > 5


def test_the_limit_grows_with_a_slow_model():
    starts, ends = timing(0.2)  # a slow model: 20 times its median run is 4 s, above the 1 s floor
    for i in range(100, 300):
        starts[i] += 2
        ends[i] += 2
    assert latency.interruption(starts, ends) is None  # a 2 s gap is under the 4 s limit
    for i in range(200, 300):
        starts[i] += 5
        ends[i] += 5
    assert abs(latency.interruption(starts, ends)["limit_s"] - 4.0) < 1e-9


def test_sessions_alternate_which_build_goes_first():
    builds = ["fp32", "int8_percentile99.99"]
    assert latency.session_order(builds, 1) == builds
    assert latency.session_order(builds, 2) == builds[::-1]
    assert latency.session_order(["fp32"], 2) == ["fp32"]  # a model with no usable INT8 build


def test_summary_takes_medians_across_sessions_and_flags_a_large_spread():
    steady = [[10.0 + 0.01 * s] * 300 for s in range(10)]  # ten sessions, p50 from 10.00 to 10.09 ms
    summary = latency.summarise(steady)
    assert abs(summary["p50_ms"] - 10.045) < 1e-9 and not summary["unstable"]
    assert summary["p50_ms_fastest_session"] == 10.0 and summary["p50_ms_slowest_session"] == 10.09
    jumpy = [[10.0] * 300] * 5 + [[13.0] * 300] * 5  # half the sessions 30% slower
    assert latency.summarise(jumpy)["unstable"] and latency.summarise(jumpy)["spread_pct"] > 10


def test_other_brokkr_jobs_are_recognised_by_their_command_line(tmp_path):
    repo = tmp_path / "Brokkr"
    (repo / "scripts").mkdir(parents=True)
    assert latency.is_brokkr_job(["python", "scripts/28_breadth_sweep.py"], str(repo), repo)
    assert latency.is_brokkr_job(["python.exe", str(repo / "scripts" / "37_se1.py")], "C:/", repo)
    assert latency.is_brokkr_job([str(Path("venv") / "Scripts" / "brokkr-edge.exe"), "test"], None, repo)
    assert latency.is_brokkr_job(["python", "-m", "brokkr_edge.cli", "test"], None, repo)
    assert not latency.is_brokkr_job(["python", "scripts/train.py"], str(tmp_path / "Other"), repo)
    assert not latency.is_brokkr_job(["chrome.exe", "--type=renderer"], None, repo)


def test_the_machine_must_be_ready():
    ready = {"on_ac_power": True, "power_mode": "best performance", "battery_saver": False}
    assert latency.machine_problems(ready, 3.0, []) == []
    assert latency.machine_problems(ready, None, []) == []  # the end-of-model check: power only
    assert latency.machine_problems(ready | {"on_ac_power": False}, 3.0, [])
    assert latency.machine_problems(ready | {"on_ac_power": None}, 3.0, [])  # "could not tell" is not ready
    assert latency.machine_problems(ready | {"power_mode": "balanced"}, 3.0, [])
    assert latency.machine_problems(ready | {"battery_saver": True}, 3.0, [])
    assert latency.machine_problems(ready, 10.0, [])  # the limit itself is not idle
    assert latency.machine_problems(ready, 3.0, ["python scripts/28_breadth_sweep.py"])


def test_the_confirmed_numbers_are_the_ones_in_the_code():
    # H, 2 October 2026: larger of 1.0 s and 20 times the median; at most 3 discarded sessions.
    assert (latency.PAUSE_FLOOR_S, latency.PAUSE_TIMES_MEDIAN, latency.MAX_DISCARDED_SESSIONS) == (1.0, 20, 3)
    assert (latency.WARMUP_RUNS, latency.TIMED_RUNS, latency.SESSIONS) == (20, 300, 10)
    assert latency.THREAD_COUNTS == (1, 4)
