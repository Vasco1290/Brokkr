"""The label rules (brokkr_edge.label), the label validator (brokkr_edge.label_schema) and the renderer's
number check (brokkr_edge.label_render), on made-up inputs whose right answer is known.

The numbers here are invented for the tests; they are never results.
"""

import copy

import numpy as np

from brokkr_edge import label as lb
from brokkr_edge.label_render import TERMS, _do_not_use, _title, to_html, to_markdown, unexplained_numbers
from brokkr_edge.label_schema import build_id, check_label, condition_id, hardware_id, model_id, slug


def test_envelope_states_and_their_ends():
    ok_cov, ok_drop = [0.80, 0.90], [-0.10, -0.02]  # both lower ends exactly on the lines: still clear
    assert lb.envelope_state(ok_cov, ok_drop)[0] == "not harmful"
    assert lb.envelope_state([0.70, 0.7999], ok_drop)[0] == "harmful"  # whole coverage interval below 80%
    assert lb.envelope_state(ok_cov, [-0.20, -0.1001])[0] == "harmful"  # whole drop interval below -10
    assert lb.envelope_state([0.79, 0.85], ok_drop)[0] == "borderline"  # straddles the coverage line
    state, why = lb.envelope_state(ok_cov, [-0.12, -0.05])
    assert state == "borderline" and why == ["damage drop interval straddles its line"]


def test_shrinking_cost_flag():
    assert lb.shrinking_cost_flag([-0.08, -0.06], 0.60) == "large shrinking cost"
    assert lb.shrinking_cost_flag([-0.08, -0.04], 0.60) is None  # interval reaches above -5 points
    assert lb.shrinking_cost_flag([-0.08, -0.06], 0.05) == "not informative"  # FP32 near floor wins


def test_summary_groups_use_only_the_envelope_states():
    g = lb.summary_group
    assert g("not harmful", "harmful") == "fine"  # the summary is about the INT8 build
    assert g("harmful", "harmful") == "too hard for this model"
    assert g("harmful", "not harmful") == "hurt by shrinking"
    assert g("harmful", "borderline") == "borderline"  # FP32 neither copes nor fails
    assert g("borderline", "not harmful") == g("borderline", "harmful") == "borderline"
    assert g("INT8 build failed", "harmful") == "INT8 build failed"


def test_summary_lines_follow_the_groups_and_name_a_mixed_borderline():
    def row(build, cond, state):
        return {"build_id": build, "condition_id": cond, "state": state}

    states = {"a": ("not harmful", "harmful"), "b": ("harmful", "harmful"), "c": ("harmful", "not harmful"),
              "d": ("harmful", "borderline"), "e": ("borderline", "not harmful")}
    rows = [row("int8", c, s8) for c, (s8, _) in states.items()] + [row("fp32", c, s) for c, (_, s) in
                                                                      states.items()]
    lines = lb.summary(rows, "int8", {c: c.upper() for c in states})["lines"]
    assert [(x["group"], x["count"], x["conditions"]) for x in lines] == [
        ("fine", 1, ["A"]),
        ("too hard for this model", 1, ["B"]),
        ("hurt by shrinking", 1, ["C"]),
        ("borderline", 2, ["D (INT8 harmful, FP32 borderline)", "E"]),
        ("not tested", None, ["every damage type and severity not listed above"]),
    ]


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
    rows = [
        {
            "build_id": b,
            "condition_id": "suite/fog/3",
            "hardware_id": hw,
            "state": "not harmful",
            "why": [],
            "shrinking_cost_flag": None,
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
                "runtime": {"name": "rt", "version": "v", "execution_provider": "cpu"},
            }
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
        "summary": lb.summary(rows, lab, {"suite/fog/3": "fog (suite) s3"}),
        "speed": [
            {
                "build_id": b,
                "hardware_id": hw,
                "hardware_kind": "laptop",
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


def test_a_number_not_in_the_label_is_caught():
    label = tiny_label()
    text = to_markdown(label) + "\nTop-1 on another machine: 81.23%\n"
    assert unexplained_numbers(label, text) == ["81.23%"]
    assert unexplained_numbers(label, "<html><style>p{line-height:1.5}</style><p>INT8 top-1</p></html>") == []
