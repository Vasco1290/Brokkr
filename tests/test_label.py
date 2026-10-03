"""The label rules (brokkr_edge.label), the label validator (brokkr_edge.label_schema) and the renderer's
number check (brokkr_edge.label_render), on made-up inputs whose right answer is known.

The numbers here are invented for the tests; they are never results.
"""

import copy
import html

import numpy as np

from brokkr_edge import label as lb
from brokkr_edge.label_render import (
    SLOWER,
    TERMS,
    _do_not_use,
    _envelope_cell,
    _next_step,
    _title,
    readme_broken_example,
    readme_example,
    to_html,
    to_markdown,
    unexplained_numbers,
)
from brokkr_edge.label_schema import (
    build_id,
    check_label,
    condition_id,
    hardware_id,
    model_id,
    runtime_id,
    slug,
)


def test_envelope_states_and_their_ends():
    ok_cov, ok_drop = [0.80, 0.90], [-0.10, -0.02]  # both lower ends exactly on the lines: still clear
    assert lb.envelope_state(ok_cov, ok_drop)[0] == "not harmful"
    assert lb.envelope_state([0.70, 0.7999], ok_drop)[0] == "harmful"  # whole coverage interval below 80%
    assert lb.envelope_state(ok_cov, [-0.20, -0.1001])[0] == "harmful"  # whole drop interval below -10
    assert lb.envelope_state([0.79, 0.85], ok_drop)[0] == "borderline"  # straddles the coverage line
    state, why = lb.envelope_state(ok_cov, [-0.12, -0.05])
    assert state == "borderline" and why == ["damage drop interval straddles its line"]


def test_shrinking_cost_flags():
    assert lb.shrinking_cost_flags([-0.08, -0.06], 0.60) == ["large shrinking cost"]
    assert lb.shrinking_cost_flags([-0.08, -0.04], 0.60) == []  # interval reaches above -5 points
    assert lb.shrinking_cost_flags([-0.03, -0.01], 0.05) == ["not informative"]  # FP32 near floor
    # both apply: "not informative" no longer hides the large shrinking cost (note of 3 October 2026)
    assert lb.shrinking_cost_flags([-0.08, -0.06], 0.09) == ["large shrinking cost", "not informative"]


def test_harm_cause_reads_fp32_state_only():
    assert lb.harm_cause("harmful") == "too hard for this model"  # FP32 also fails
    assert lb.harm_cause("not harmful") == "hurt by shrinking"  # FP32 copes, INT8 doesn't
    assert lb.harm_cause("borderline") == "cause unclear"  # FP32 neither clearly copes nor fails


def test_failed_lines_name_each_whole_interval_below_its_line():
    ok_cov, ok_drop, low_cov, low_drop = [0.85, 0.9], [-0.05, 0.0], [0.70, 0.75], [-0.20, -0.15]
    assert lb.failed_lines(ok_cov, ok_drop) == []
    assert lb.failed_lines(ok_cov, low_drop) == ["damage drop"]
    assert lb.failed_lines(low_cov, ok_drop) == ["coverage"]
    assert lb.failed_lines(low_cov, low_drop) == ["damage drop", "coverage"]
    assert lb.failed_lines([0.79, 0.85], [-0.12, -0.05]) == []  # straddling is borderline, not failed


def rows_for(cases: dict) -> list:
    """Envelope rows from {condition: (INT8 state, FP32 state, INT8 failed lines, INT8 flag)}."""
    rows = []
    for cond, (int8, fp32, failed, flag) in cases.items():
        rows.append({"build_id": "int8", "condition_id": cond, "state": int8, "failed": failed,
                     "shrinking_cost_flags": flag})
        rows.append({"build_id": "fp32", "condition_id": cond, "state": fp32, "failed": [],
                     "shrinking_cost_flags": []})
    return rows


CASES = {
    "a": ("not harmful", "harmful", [], ["large shrinking cost"]),
    "b": ("harmful", "harmful", ["damage drop"], ["large shrinking cost"]),
    "c": ("harmful", "not harmful", ["coverage"], []),
    "d": ("harmful", "borderline", ["damage drop", "coverage"], []),
    "e": ("borderline", "not harmful", [], []),
}


def test_summary_groups_by_the_shrunk_build_first_then_by_cause():
    lines = lb.summary(rows_for(CASES), "int8")["lines"]
    assert [(x["state"], x["cause"], x["count"], x["conditions"]) for x in lines] == [
        ("not harmful", None, 1, ["a"]),
        ("borderline", None, 1, ["e"]),
        ("harmful", "too hard for this model", 1, ["b"]),
        ("harmful", "hurt by shrinking", 1, ["c"]),
        ("harmful", "cause unclear", 1, ["d"]),  # INT8 harmful, FP32 borderline: under harmful
        ("not tested", None, None, []),
    ]
    assert lines[4]["by_failed"] == [{"failed": ["damage drop", "coverage"], "count": 1, "conditions": ["d"]}]


def test_every_large_shrinking_cost_is_named_in_the_summary_whatever_its_group():
    summary = lb.summary(rows_for(CASES), "int8")
    flagged = [c for c, case in CASES.items() if "large shrinking cost" in case[3]]
    # "not harmful" (a) and "too hard" (b) alike
    assert summary["large_shrinking_cost"] == {"count": 2, "conditions": flagged} and flagged == ["a", "b"]
    label = tiny_label()  # its one condition is "not harmful": the flag must still reach the summary
    label["envelope"]["rows"][1]["shrinking_cost_flags"] = ["large shrinking cost"]
    label["summary"] = lb.summary(label["envelope"]["rows"], label["label_id"])
    assert check_label(label) == []
    for text in (to_markdown(label), to_html(label)):
        assert "(large shrinking cost) in 1 of 1 tested conditions: fog (suite) s3 (-2.00 points)" in text


def test_the_three_opening_counts_come_from_the_envelope_rows():
    summary = lb.summary(rows_for(CASES), "int8")
    assert summary["reference_harmful"] == {"count": 2, "conditions": ["a", "b"]}  # FP32 itself harmful
    assert summary["coverage_failed"] == {"count": 2, "conditions": ["c", "d"]}  # INT8 fails coverage
    label = tiny_label()  # nothing harmful, nothing flagged: the three lines still appear, with 0
    for text in (to_markdown(label), to_html(label)):
        assert "on clean images the shrinking cost is -1.00 points (-2.00 to +0.00)" in text
        assert "even at full size, in 0 of 1 tested conditions." in text
        assert "unreliable (coverage below 80%) in 0 of 1 tested conditions." in text
        assert text.index("Shrinking") < text.index("Harsh conditions") < text.index("Uncertainty signal")
        assert text.index("Uncertainty signal") < text.index("Not harmful in our tests")  # before the groups


NEXT_STEPS = {  # (failed lines, cause) -> suggested next step (note of 3 October 2026)
    # both lines failed: the cause-based advice first, re-calibrating second
    ("both", "too hard for this model"): "consider a stronger model; re-calibrate on your own images",
    ("both", "hurt by shrinking"): "try another recipe or model; re-calibrate on your own images",
    ("both", "cause unclear"): "try another recipe or a stronger model; re-calibrate on your own images",
    # accuracy only: the cause-based advice only
    ("accuracy", "too hard for this model"): "consider a stronger model",
    ("accuracy", "hurt by shrinking"): "try another recipe or model",
    ("accuracy", "cause unclear"): "try another recipe or a stronger model",
    # coverage only: re-calibrate first; "try another recipe" only where shrinking is involved
    ("coverage", "too hard for this model"): "re-calibrate on your own images",
    ("coverage", "hurt by shrinking"): "re-calibrate on your own images; try another recipe",
    ("coverage", "cause unclear"): "re-calibrate on your own images; try another recipe",
}
FAILED = {"both": ["damage drop", "coverage"], "accuracy": ["damage drop"], "coverage": ["coverage"]}


def test_the_suggested_next_step_follows_the_line_that_failed():
    for (failed, cause), expected in NEXT_STEPS.items():
        assert _next_step({"state": "harmful", "failed": FAILED[failed]}, cause) == expected, (failed, cause)


def test_a_coverage_only_row_never_suggests_a_stronger_model():
    for cause in ("too hard for this model", "hurt by shrinking", "cause unclear"):
        assert "stronger model" not in _next_step({"state": "harmful", "failed": ["coverage"]}, cause)


def test_only_a_harmful_row_gets_a_suggested_next_step():
    assert _next_step({"state": "not harmful", "failed": []}, None) == "—"
    assert _next_step({"state": "borderline", "failed": []}, None) == "—"
    assert _next_step({"state": "INT8 build failed", "failed": []}, None) == "—"
    assert _next_step(None, None) == "—"


def test_a_harmful_cell_says_which_line_it_failed():
    rule = lb.ENVELOPE_RULE
    row = {"state": "harmful", "failed": ["damage drop", "coverage"]}
    assert _envelope_cell(row, rule) == (
        "harmful: accuracy dropped and uncertainty signal unreliable (coverage below 80%)"
    )
    one = {"state": "harmful", "failed": ["damage drop"]}
    assert _envelope_cell(one, rule) == "harmful: accuracy dropped"
    assert _envelope_cell({"state": "not harmful", "failed": []}, rule) == "not harmful in our tests"


def test_paired_counts_whole_items():
    new, old = np.zeros(1000), np.zeros(1000)
    new[:300], old[:250] = 1, 1
    p = lb.paired(new, old)
    assert p["count"] == 50 and p["value"] == 0.05 and p["ci95"][0] > 0 and p["n_items"] == 1000


def test_ids_are_stable_and_lowercase():
    mid = model_id("torchvision", "mobilenet_v3_large", "MobileNet_V3_Large_Weights.IMAGENET1K_V2")
    assert mid == "torchvision/mobilenet_v3_large@imagenet1k_v2"
    assert build_id(mid, "int8", "ab" * 32) == f"{mid}#int8:{'ab' * 6}"
    assert (
        hardware_id("laptop", "12th Gen Intel(R) Core(TM) i5-1235U", "Windows 11")
        == "laptop:12th-gen-intel-r-core-tm-i5-1235u:windows-11"
    )
    assert (
        condition_id(None, "clean", 0) == "clean"
        and condition_id("imagenet-c", "contrast", 3) == "imagenet-c/contrast/3"
    )
    assert slug("  Fog (Brokkr)  ") == "fog-brokkr"


RT = runtime_id("rt", "v1", "cpu", 8, "off")


def tiny_label() -> dict:
    """A small valid label: one reference and one labelled build, one damaged condition."""
    sha_a, sha_b, sha_f = "a" * 64, "b" * 64, "f" * 64
    mid = "maker/tiny@v1"
    ref, lab = build_id(mid, "fp32", sha_a), build_id(mid, "int8", sha_b)
    hw = "laptop:test-cpu:test-os"
    src = [{"file": "results/x.json", "sha256": sha_f, "field": "metrics.top1"}]

    def m(metric, build, cond, value, ci):
        return {
            "metric": metric,
            "build_id": build,
            "dataset_id": "maker/data:test",
            "condition_id": cond,
            "hardware_id": hw,
            "runtime_id": RT,
            "value": value,
            "ci95": ci,
            "n_items": 100,
            "unit": "fraction",
            "settings": {},
            "sources": src,
        }

    measurements = []
    for b in (ref, lab):
        for cond in ("clean", "suite/fog/3"):
            measurements += [
                m("top1", b, cond, 0.7, [0.6, 0.8]),
                m("coverage", b, cond, 0.9, [0.85, 0.95]),
                m("mean_set_size", b, cond, 2.5, [2.0, 3.0]),
            ]
        measurements.append(m("damage_drop", b, "suite/fog/3", -0.01, [-0.03, 0.01]))
    measurements.append(m("shrinking_cost", lab, "suite/fog/3", -0.02, [-0.04, 0.0]))
    measurements.append(m("shrinking_cost", lab, "clean", -0.01, [-0.02, 0.0]))
    rows = [
        {
            "build_id": b,
            "condition_id": "suite/fog/3",
            "hardware_id": hw,
            "state": "not harmful",
            "why": [],
            "failed": [],
            "shrinking_cost_flags": [],
        }
        for b in (ref, lab)
    ]
    return {
        "schema_version": 1,
        "label_id": lab,
        "source": {"kind": "official", "verified": True, "submitted_by": None, "how_made": "test"},
        "generated": {
            "by": "test",
            "commit": "c" * 40,
            "dirty": False,
            "date_utc": "test",
            "ci_level": 0.95,
            "label_licence": "CC BY 4.0",
            "envelope_rule": "test",
        },
        "model": {
            "model_id": mid,
            "name": "tiny",
            "publisher": "maker",
            "weights": "v1",
            "task": "classification",
            "modality": "image",
            "input": {},
            "outputs": {},
            "licence": {"code": "MIT", "weights": "MIT"},
        },
        "builds": [
            {
                "build_id": ref,
                "role": "reference",
                "precision": "fp32",
                "recipe": {"method": "export"},
                "file": {"sha256": sha_a, "size_bytes": 1000},
                "status": "usable",
                "failure": None,
            },
            {
                "build_id": lab,
                "role": "labelled",
                "precision": "int8",
                "recipe": {"method": "p99"},
                "file": {"sha256": sha_b, "size_bytes": 500},
                "status": "usable",
                "failure": None,
            },
        ],
        "hardware": [
            {
                "hardware_id": hw,
                "kind": "laptop",
                "cpu_model": "test cpu",
                "os": "test os",
            }
        ],
        "runtimes": [
            {"runtime_id": RT, "name": "rt", "version": "v1", "execution_provider": "cpu", "threads": 8,
             "spinning": "off", "sources": src}
        ],
        "datasets": [
            {
                "dataset_id": "maker/data:test",
                "name": "data",
                "split": "test",
                "n_items": 100,
                "item_kind": "image",
                "licence": "CC0",
            }
        ],
        "conditions": [
            {"condition_id": "clean", "modality": "image", "label": "clean"},
            {"condition_id": "suite/fog/3", "modality": "image", "label": "fog (suite) s3"},
        ],
        "checks": {
            "fp32_sanity": {
                "measured": 0.7,
                "published": 0.7,
                "tolerance": 0.01,
                "pass": True,
                "published_source": "test",
                "sources": src,
            }
        },
        "measurements": measurements,
        "envelope": {"rule": dict(lb.ENVELOPE_RULE), "rows": rows},
        "summary": lb.summary(rows, lab),
        "speed": [
            {
                "build_id": b,
                "hardware_id": hw,
                "hardware_kind": "laptop",
                "runtime_id": None,
                "status": "not measured",
                "reason": "test",
                "settings": {"threads": 1},
            }
            for b in (ref, lab)
        ],
        "details": {
            "conformal": {b: {"target": 0.9, "threshold": 0.5, "calibration_items": 100} for b in (ref, lab)}
        },
        "limits": ["test"],
        "licences": {
            "brokkr_code": "Apache-2.0",
            "model_code": "MIT",
            "model_weights": "MIT",
            "weights_trained_on": None,
            "label_data": "CC BY 4.0",
            "sources": src,
        },
        "sources": [{"file": "results/x.json", "sha256": sha_f, "check": "test"}],
    }


def test_a_valid_label_passes_and_renders_with_every_number_explained():
    label = tiny_label()
    assert check_label(label) == []
    for text in (to_markdown(label), to_html(label)):
        assert unexplained_numbers(label, text) == []


def test_the_validator_catches_each_rule():
    def broken(change):
        label = copy.deepcopy(tiny_label())
        change(label)
        return check_label(label)

    assert broken(lambda x: x["source"].update(kind="user-submitted"))  # user-submitted must be unverified
    assert broken(lambda x: x["generated"].update(dirty=True))  # official labels need a clean commit
    assert broken(lambda x: x["model"]["licence"].update(weights=""))  # hard rule 8
    assert broken(lambda x: x["measurements"].pop(2))  # coverage without its set size
    assert broken(lambda x: x["builds"][1].update(status="failed"))  # failed build without failure/with rows
    assert broken(lambda x: x.update(schema_version=2))
    assert broken(lambda x: x["summary"]["lines"][0].update(count=5))  # summary disagrees with envelope
    assert broken(lambda x: x["sources"][0].update(file="C:/Users/someone/Brokkr/x.json"))  # machine path
    assert broken(lambda x: x["measurements"][0]["sources"][0].update(file="/home/someone/x.json"))
    assert broken(lambda x: x["envelope"]["rows"][1].update(state="harmful"))  # harmful, no failed line
    assert broken(lambda x: x["summary"]["coverage_failed"].update(count=3))  # an opening count is wrong
    # a large shrinking cost left out of the summary
    assert broken(lambda x: x["envelope"]["rows"][1].update(shrinking_cost_flags=["large shrinking cost"]))
    # the flags are a list in a fixed order
    assert broken(lambda x: x["envelope"]["rows"][1].update(
        shrinking_cost_flags=["not informative", "large shrinking cost"]))
    assert broken(lambda x: x["hardware"][0].update(runtime={"name": "rt"}))  # runtime has its own field
    assert broken(lambda x: x["measurements"][0].update(runtime_id="nowhere@1:cpu:1t:spin-on"))
    assert broken(lambda x: x["licences"].update(model_weights=""))  # an empty licence line
    assert broken(lambda x: x["speed"][0].update(runtime_id=RT))  # a "not measured" row has no runtime


def test_titles_use_display_names_and_fall_back_to_internal_names():
    label = tiny_label()
    assert _title(label) == "tiny · INT8"
    label["model"]["display_name"] = "Tiny-Net"
    label["builds"][1]["display_name"] = "INT8 (percentile calibration, 99.99%)"
    assert _title(label) == "Tiny-Net · INT8 (percentile calibration, 99.99%)"
    for text in (to_markdown(label), to_html(label)):
        assert "Tiny-Net · INT8 (percentile calibration, 99.99%)" in text
        assert unexplained_numbers(label, text) == []  # "99.99%" is a label string, not a loose number


def test_a_failed_build_gets_one_plain_do_not_use_sentence():
    build = {"status": "failed", "failure": {"check": "top-1 agreement with FP32 >= 20%", "value": 0.0273,
                                             "limit": 0.2, "n_items": 256, "split": "tuning"}}
    sentence = _do_not_use(build)
    assert sentence.startswith("This recipe broke the model, so do not use this INT8 build")
    assert "2.73%" in sentence and "256 tuning images" in sentence and "20%" in sentence
    assert sentence.endswith(".") and ". " not in sentence  # one sentence: a single full stop, at the end
    assert _do_not_use({"status": "usable", "failure": None}) is None


def test_every_explanation_is_plain_text_without_numbers():
    label = tiny_label()
    assert unexplained_numbers(label, " ".join(TERMS.values())) == []
    for text in (to_markdown(label), to_html(label)):
        assert "What the words mean" in text
    assert "<abbr title=" in to_html(label)


def test_the_readme_example_shows_only_label_numbers():
    label = tiny_label()
    text = readme_example(label)
    assert "tiny · INT8" in text and "(1 of 1): fog (suite) s3" in text
    assert unexplained_numbers(label, text) == []
    label["builds"][1].update(status="failed", failure={"check": "top-1 agreement with FP32 >= 20%",
                                                        "value": 0.0273, "limit": 0.2, "n_items": 256,
                                                        "split": "tuning"})
    sentence = readme_broken_example(label)
    assert "2.73% of 256 tuning images" in sentence and unexplained_numbers(label, sentence) == []


def test_a_number_not_in_the_label_is_caught():
    label = tiny_label()
    text = to_markdown(label) + "\nTop-1 on another machine: 81.23%\n"
    assert unexplained_numbers(label, text) == ["81.23%"]
    assert unexplained_numbers(label, "<html><style>p{line-height:1.5}</style><p>INT8 top-1</p></html>") == []


def test_not_informative_never_hides_a_large_shrinking_cost():
    """A made-up row where FP32 is near floor AND the whole shrinking-cost interval is below -5 points:
    both flags are kept, the summary counts it, and both renders show both (note of 3 October 2026)."""
    label = tiny_label()
    rule = label["envelope"]["rule"]
    flags = lb.shrinking_cost_flags([-0.08, -0.06], 0.09, rule)
    assert flags == ["large shrinking cost", "not informative"]
    label["envelope"]["rows"][1]["shrinking_cost_flags"] = flags
    label["summary"] = lb.summary(label["envelope"]["rows"], label["label_id"])
    assert check_label(label) == []
    assert label["summary"]["large_shrinking_cost"]["conditions"] == ["suite/fog/3"]
    for text in (to_markdown(label), to_html(label)):
        assert "[large shrinking cost; not informative]" in text
        assert "(large shrinking cost) in 1 of 1 tested conditions" in text


def speed_row(build: str, threads: int, p50: float, spread: float, ref_p50: float | None = None) -> dict:
    """A made-up measured laptop speed row (tests only; the numbers are invented)."""
    row = {
        "build_id": build, "hardware_id": "laptop:test-cpu:test-os", "hardware_kind": "laptop",
        "runtime_id": RT, "status": "measured", "reason": None,
        "settings": {"threads": threads, "batch": 1, "input": "random, seed 0",
                     "what_is_timed": "model only"},
        "p50_ms": p50, "p95_ms": p50 + 0.5, "p99_ms": p50 + 1.0, "spread_pct": spread,
        "unstable": spread > 10.0, "unstable_above_pct": 10.0, "sessions": 10, "warmup_runs": 20,
        "timed_runs": 300, "discarded_sessions": 0,
        "timed_utc": {"first_start": "2026-01-01T00:00:00+00:00", "last_end": "2026-01-01T01:00:00+00:00"},
        "pinning": {"logical_cpus": [0, 1, 2, 3], "physical_cores": [0, 2], "n_logical": 4, "n_physical": 2,
                    "core_kind": "performance", "reported_by": "Windows", "read_back": False},
        "vnni": "AVX512-VNNI: no",
        "sources": [{"file": "results/x.json", "sha256": "f" * 64, "field": "metrics.p50_ms.value"}],
    }
    if ref_p50 is not None:
        ratio = p50 / ref_p50
        row["time_vs_reference"] = {"ratio_p50": ratio, "reference_p50_ms": ref_p50, "slower": ratio > 1,
                                    "sources": row["sources"]}
    return row


def timed_label(int8_p50: dict, unstable_fp32_at: int | None = None) -> dict:
    """tiny_label with FP32 at 4.0 ms and INT8 at the given p50 per thread count (made-up numbers)."""
    label = tiny_label()
    ref, lab = (b["build_id"] for b in label["builds"])
    label["speed"] = []
    for threads, p50 in int8_p50.items():
        label["speed"].append(speed_row(ref, threads, 4.0, 12.5 if threads == unstable_fp32_at else 2.5))
        label["speed"].append(speed_row(lab, threads, p50, 3.5, ref_p50=4.0))
    return label


def test_int8_slower_is_said_plainly_and_never_as_a_negative_speed_up():
    label = timed_label({1: 6.0, 4: 5.0})  # INT8 slower at both thread counts
    assert check_label(label) == []
    for text in (to_markdown(label), to_html(label)):
        assert "INT8 takes 1.50× the time of FP32 (slower)" in text
        assert "INT8 takes 1.25× the time of FP32 (slower)" in text
        assert f"{SLOWER}." in text
        assert "speed-up" not in text.lower() and "faster" not in text.lower()
        assert unexplained_numbers(label, text) == []


def test_int8_faster_has_no_slower_wording_and_a_partial_case_names_the_thread_count():
    faster = timed_label({1: 2.0, 4: 3.0})
    for text in (to_markdown(faster), to_html(faster)):
        assert "INT8 takes 0.50× the time of FP32 |" in text or "INT8 takes 0.50× the time of FP32<" in text
        assert "(slower)" not in text and SLOWER not in text
    partial = timed_label({1: 6.0, 4: 3.0})
    text = to_markdown(partial)
    assert "INT8 is slower than FP32 on this laptop CPU at 1 thread (relative comparison only)." in text
    assert unexplained_numbers(partial, text) == []


def test_an_unstable_row_says_so_in_plain_words_and_the_ratio_is_still_shown():
    label = timed_label({1: 2.0, 4: 3.0}, unstable_fp32_at=4)
    assert check_label(label) == []
    for text in (to_markdown(label), to_html(label)):
        assert "unstable: speed varied a lot between repeat runs (spread 12.5%); treat as rough" in text
        assert "INT8 takes 0.75× the time of FP32; one or both timings unstable, treat as rough" in text
        assert "A row is marked unstable when its spread is above 10.0%." in text
        assert "Spread" in text and "Unstable" in text  # glossary entries
        assert unexplained_numbers(label, text) == []
    assert label["speed"][2]["unstable"]  # FP32 at 4 threads, spread above the line
    label["speed"][2]["unstable"] = False  # the flag must agree with the spread and its line
    assert check_label(label)


def test_the_cores_sentence_and_the_pin_limitation_come_from_the_row():
    text = to_markdown(timed_label({1: 2.0, 4: 3.0}))
    assert ("Pinned to the laptop's 2 performance cores (4 hardware threads, as reported by Windows). The "
            "4-thread setting therefore runs on 2 physical cores. The pin was not read back after it was set."
            ) in text


def test_the_licence_section_and_model_card_link():
    label = tiny_label()
    md, page = to_markdown(label), html.unescape(to_html(label))
    assert 'license_link: "#licences"' in md and "\n## Licences\n" in md and '<h2 id="licences">' in page
    for text in (md, page):
        assert "Brokkr's code, which made this label: Apache-2.0." in text
        assert "The model's weights keep their original licence: MIT" in text
        assert "This label's own numbers and text are Brokkr output, licensed CC BY 4.0." in text
        assert "ImageNet" not in text.split("Licences")[-1]  # not trained on ImageNet: no ImageNet line
    label["licences"]["weights_trained_on"] = "ImageNet-1k"
    assert "trained on ImageNet-1k and carry ImageNet's non-commercial terms of access" in to_markdown(label)


def test_every_page_states_the_label_schema_version():
    label = tiny_label()
    for text in (to_markdown(label), to_html(label)):
        assert "label schema version 1" in text


def test_the_readme_example_uses_the_approved_threshold_wording():
    text = readme_example(tiny_label())
    assert ("a line whose value was written down before these results existed and adopted for the labels "
            "afterwards, unchanged") in text
