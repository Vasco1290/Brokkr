"""Render a label.json as a Markdown model card (Hugging Face metadata at the top) and as an HTML page.

Both read ONLY the label; they compute nothing. Every number goes through fmt(), so
unexplained_numbers() can check that each number in the output is a label.json number, shown in one
of the formats below. Values are rounded only here.

Every technical term has a one-line plain explanation (TERMS): listed under "What the words mean" in
both formats, and shown as a hover note on table headings, envelope states and summary groups in HTML.
The explanations hold no numbers, so the number check still covers everything a reader sees.
"""

import html
import json
import re

FORMATS = {
    "pct": lambda v: f"{100 * v:.2f}%",  # top-1, coverage
    "pct0": lambda v: f"{100 * v:.0f}%",  # interval level, coverage target and line
    "pts": lambda v: f"{100 * v:+.2f}",  # damage drop, shrinking cost (points)
    "pts0": lambda v: f"{100 * v:+.0f}",  # the damage-drop and shrinking-cost lines
    "size": lambda v: f"{v:.2f}",  # average prediction-set size (classes)
    "ece": lambda v: f"{v:.3f}",
    "eaurc": lambda v: f"{v:.4f}",
    "mb": lambda v: f"{v / 1e6:.2f}",  # file size in MB (from bytes)
    "int": lambda v: f"{int(v):d}",
    "thr": lambda v: f"{v:.6f}",  # conformal threshold
}
# Words that contain digits but are not numbers.
WORDS_WITH_DIGITS = (
    "top-1",
    "top-5",
    "Top-1",
    "Top-5",
    "INT8",
    "FP32",
    "FP16",
    "E-AURC",
    "ImageNet-1k",
    "ImageNet-C",
    "CC BY 4.0",
    "Apache-2.0",
    "Raspberry Pi 5",
    "p50",
    "p95",
    "p99",
    "imagenet-1k",
    "sha256",
    "SHA-256",
)
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
STATE_MARK = {
    "not harmful": "not harmful in our tests",
    "harmful": "harmful",
    "borderline": "borderline",
    "not tested": "not tested",
    "INT8 build failed": "INT8 build failed",
}
# The summary (docs/label_schema.md, second note of 30 September 2026) as a reader sees it: lines by the
# shrunk build's own state, harmful ones by cause.
STATE_TITLE = {
    "not harmful": "Not harmful in our tests",
    "borderline": "Borderline (too close to a line to call)",
    "harmful": "Harmful",
    "INT8 build failed": "INT8 build failed",
    "not tested": "Not tested",
}
CAUSE_TITLE = {
    "too hard for this model": "Too hard for this model (FP32 also fails)",
    "hurt by shrinking": "Hurt by shrinking (FP32 copes, INT8 doesn't)",
    "cause unclear": "Cause unclear (FP32 is borderline)",
}
# A suggested next step for a harmful row, by its cause (note of 1 October 2026). General suggestions,
# not results: the label says so under the damage table and in its limits.
NEXT_STEP = {
    "hurt by shrinking": "try another recipe or model",
    "too hard for this model": "consider a stronger model",
    "cause unclear": "try another recipe or a stronger model",
}
RECALIBRATE = "re-calibrate on your own images"
DAMAGE_NOTE = (
    "Damage drop and shrinking cost in points (positive = better). The suggested next steps are general "
    "suggestions; they were not tested for this model."
)

# One plain line per technical term on the label. No numbers here (see the module docstring).
TERMS = {
    "Official label": "made by Brokkr from its own runs. A user-submitted label is always marked unverified.",
    "FP32": "the original model, not shrunk: every number in it is stored at full precision.",
    "INT8": "the shrunk model: most of its numbers are stored as small whole numbers, so the file is "
    "smaller.",
    "Shrinking": "turning the FP32 model into an INT8 one (also called quantization).",
    "Recipe": "how the INT8 build was made. Percentile calibration sets each layer's range of values from "
    "sample images and ignores the rarest extreme values; the percentage is the share of values kept inside.",
    "Reference and labelled build": "the label describes the labelled (shrunk) build; the reference (FP32) "
    "build is shown beside it for comparison.",
    "Top-1": "the share of images where the model's first answer is the right one.",
    "Top-5": "the share of images where the right answer is among the model's five best guesses.",
    "Damage": "simulated fog, darkness, blur, noise or low contrast, added to the test images.",
    "Brokkr and ImageNet-C": "the two sets of damage: Brokkr's own, and ImageNet-C, a widely used public "
    "benchmark (made here with its official code).",
    "Severity (s3, s5)": "how strong the damage is; a higher number is stronger.",
    "Points": "percentage points: the plain difference between two percentages. Negative means worse.",
    "Damage drop": "top-1 under the damage minus top-1 on clean images, for the same build and the same "
    "images. Negative means the damage hurt.",
    "Shrinking cost": "INT8 top-1 minus FP32 top-1 on the same images. Negative means shrinking lost "
    "accuracy.",
    "Large shrinking cost": "the whole interval of the shrinking cost is below the line given under "
    "\"What was tested\".",
    "Not informative": "FP32 itself gets almost every image wrong here, so the shrinking cost says little.",
    "Prediction set": "the classes the model says could be right. A small set means it is sure; a big set is "
    "its way of saying \"I'm not sure\".",
    "Coverage": "the share of images whose prediction set contains the right answer. The sets are tuned on "
    "clean images to reach a target; under damage there is no promise, so the label shows what was measured.",
    "Set size": "the average number of classes in a prediction set.",
    "Conformal threshold": "the cut-off that decides which classes go into a prediction set, set once on "
    "clean calibration images.",
    "Calibration images": "clean images used only to set things up (INT8's ranges of values, the conformal "
    "threshold), never for testing.",
    "Test images": "the images every accuracy and coverage number here is measured on; none of them is used "
    "for calibration.",
    "Tuning images": "images kept apart from the test images, used for checks while building.",
    "Envelope": "the verdict for each build in each condition, from its coverage and damage drop and the "
    "lines under \"What was tested\".",
    "Not harmful in our tests": "both whole intervals clear their lines in this condition. It is not a "
    "guarantee: other damage or other images may give other results.",
    "Harmful": "a whole interval is below its line: coverage is too low, or accuracy fell too far.",
    "Borderline": "an interval crosses a line, so the result could go either way.",
    "Not tested": "this damage type or severity was not run.",
    "INT8 build failed": "the INT8 build did not pass its build check, so it was not tested.",
    "Top-1 agreement with FP32": "the share of images where INT8's first answer is the same as FP32's.",
    "Too hard for this model": "FP32 fails here too: the model struggles even before shrinking. The "
    "shrinking-cost column shows whether shrinking made it worse.",
    "Hurt by shrinking": "FP32 copes here but INT8 does not: the shrinking caused the failure.",
    "Cause unclear": "INT8 is harmful here, but FP32 is borderline, so it is not clear whether the model "
    "or the shrinking is to blame.",
    "Accuracy dropped": "the whole damage-drop interval is below its line: top-1 fell too far under this "
    "damage.",
    "Uncertainty signal unreliable": "the whole coverage interval is below its line: the prediction sets "
    "miss the right answer too often, so the model's \"I'm not sure\" can no longer be trusted here.",
    "Harsh conditions": "conditions where the FP32 original is itself harmful: the damage is too much for "
    "this model even at full size.",
    "Uncertainty signal": "the model's way of saying \"I'm not sure\": its prediction sets. It can be "
    "trusted only where coverage stays above its line.",
    "Re-calibrate": "set the conformal threshold again, on images like the ones the model will really "
    "see instead of clean ones.",
    "Suggested next step": "a general suggestion for a harmful condition. It was not tested for this model.",
    "Interval (in brackets)": "the range the true value most likely lies in, found by resampling the test "
    "images many times (bootstrap); its level is given under \"What was tested\".",
    "Paired": "both sides of a difference are computed on the same resampled images, so how hard each image "
    "is cancels out.",
    "ECE": "expected calibration error: how far the model's confidence is from how often it is right. "
    "Lower is better.",
    "E-AURC": "how far the model is from ordering its answers perfectly by confidence, so that its wrong "
    "answers come last. Lower is better.",
    "Latency": "the time the model takes for one image.",
    "p50, p95, p99": "latency percentiles: p50 is the typical time; p95 and p99 are slow runs, with only a "
    "small share of runs slower still.",
    "Threads": "how many CPU workers the model may use at once.",
    "FP32 sanity check": "our FP32 top-1 on clean images compared with the figure torchvision publishes, to "
    "catch a broken setup.",
    "onnxruntime, CPUExecutionProvider": "the program that runs the model (ONNX Runtime), here on the CPU.",
    "SHA-256": "a fingerprint of a file; it changes if even one byte of the file changes.",
    "Commit": "the exact version of Brokkr's code that made this label.",
    "CC BY 4.0": "the licence of this label's data: anyone may reuse it, with credit.",
}
# Which term explains each table heading, envelope state and summary group (HTML hover notes).
HOVER = {
    "Condition": "Damage",
    "FP32 top-1": "Top-1",
    "INT8 top-1": "Top-1",
    "Top-1": "Top-1",
    "FP32 damage drop": "Damage drop",
    "INT8 damage drop": "Damage drop",
    "FP32 envelope": "Envelope",
    "INT8 envelope": "Envelope",
    "INT8 coverage": "Coverage",
    "Coverage": "Coverage",
    "Shrinking cost": "Shrinking cost",
    "Role": "Reference and labelled build",
    "Build": "Reference and labelled build",
    "Recipe": "Recipe",
    "Latency": "p50, p95, p99",
    "Threads": "Threads",
    "not harmful in our tests": "Not harmful in our tests",
    "not harmful": "Not harmful in our tests",
    "harmful": "Harmful",
    "borderline": "Borderline",
    "not tested": "Not tested",
    "INT8 build failed": "INT8 build failed",
    "too hard for this model": "Too hard for this model",
    "hurt by shrinking": "Hurt by shrinking",
    "cause unclear": "Cause unclear",
    "shrinking": "Shrinking cost",
    "harsh conditions": "Harsh conditions",
    "uncertainty signal": "Uncertainty signal",
    "Suggested next step": "Suggested next step",
}


def _first_upper(text: str) -> str:
    """Capitalise only the first letter (str.capitalize would turn "INT8" into "Int8")."""
    return text[:1].upper() + text[1:]


def fmt(value, kind: str) -> str:
    return FORMATS[kind](value)


def _leaves(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


def unexplained_numbers(label: dict, text: str) -> list:
    """Numbers a reader sees in `text` (Markdown, or an HTML page's visible text) that are not a label.json
    number in one of FORMATS. Empty = every number is explained. Checksums and commits shown shortened
    (their first 12 characters) count as identifiers, not numbers."""
    if "<html" in text:  # only what a reader sees: no stylesheet, no tags
        text = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", " ", text, flags=re.S)))
    strings = {s for s in _leaves(label) if isinstance(s, str) and re.search(r"\d", s)}
    strings |= {s[:12] for s in strings if re.fullmatch(r"[0-9a-f]{40,64}", s)}
    strings = sorted(strings, key=len, reverse=True)
    for s in strings + list(WORDS_WITH_DIGITS):
        text = text.replace(s, " ").replace(html.escape(s), " ")
    numbers = [v for v in _leaves(label) if isinstance(v, (int, float)) and not isinstance(v, bool)]
    allowed = {f(v) for v in numbers for kind, f in FORMATS.items() if kind != "int" or float(v).is_integer()}
    return [t for t in NUMBER.findall(text) if t not in allowed]


# ---- what both formats show ----


def _by(label: dict) -> dict:
    """Measurements indexed by (metric, build, condition), and other lookups."""
    index = {(m["metric"], m["build_id"], m["condition_id"]): m for m in label["measurements"]}
    builds = {b["role"]: b for b in label["builds"]}
    cond_label = {c["condition_id"]: c["label"] for c in label["conditions"]}
    rows = {(r["build_id"], r["condition_id"]): r for r in label["envelope"]["rows"]}
    return {"m": index, "builds": builds, "cond": cond_label, "rows": rows}


def _value(m: dict | None, kind: str, ci: bool = True) -> str:
    if m is None:
        return "—"
    text = fmt(m["value"], kind)
    if ci and m.get("ci95"):
        text += f" ({fmt(m['ci95'][0], kind)} to {fmt(m['ci95'][1], kind)})"
    return text


def _coverage(ix: dict, build: str, cond: str) -> str:
    cov, size = ix["m"].get(("coverage", build, cond)), ix["m"].get(("mean_set_size", build, cond))
    if cov is None:
        return "—"
    return f"{_value(cov, 'pct')}, set size {_value(size, 'size', ci=False)}"


def _build_name(build: dict) -> str:
    return build.get("display_name") or build["precision"].upper()


def _title(label: dict) -> str:
    labelled = next(b for b in label["builds"] if b["role"] == "labelled")
    model = label["model"].get("display_name") or label["model"]["name"]
    return f"{model} · {_build_name(labelled)}"


def _do_not_use(build: dict) -> str | None:
    """One plain sentence when the labelled build failed its build check (H's Checkpoint 1 decision 4)."""
    if build["status"] != "failed":
        return None
    f = build["failure"]
    if "agreement" in f["check"] and f.get("n_items"):
        return (
            f"This recipe broke the model, so do not use this INT8 build: its first answer matched FP32's "
            f"on only {fmt(f['value'], 'pct')} of {fmt(f['n_items'], 'int')} {f['split']} images, and a "
            f"usable build needs at least {fmt(f['limit'], 'pct0')}."
        )
    return (
        f"This recipe broke the model, so do not use this INT8 build: it failed its build check "
        f"({f['check']})."
    )


def _failed_text(failed: list, rule: dict) -> str:
    """Which line a harmful row failed, in plain words (H's review note 3)."""
    words = {
        "damage drop": "accuracy dropped",
        "coverage": f"uncertainty signal unreliable (coverage below {fmt(rule['coverage_min'], 'pct0')})",
    }
    return " and ".join(words[name] for name in failed)


def _envelope_cell(row: dict | None, rule: dict) -> str:
    if row is None:
        return "—"
    reason = _failed_text(row.get("failed", []), rule)
    return STATE_MARK[row["state"]] + (f": {reason}" if reason else "")


def _next_step(row: dict | None, cause: str | None) -> str:
    """The suggested next step for a harmful row of the labelled build; nothing for any other row."""
    if row is None or row["state"] != "harmful":
        return "—"
    steps = [NEXT_STEP[cause]] + ([RECALIBRATE] if "coverage" in row["failed"] else [])
    return "; ".join(steps)


def _top_lines(label: dict, ix: dict) -> list:
    """The three lines that open the summary: shrinking, harsh conditions, uncertainty signal."""
    rule, summary, labelled = label["envelope"]["rule"], label["summary"], label["label_id"]
    of = f" of {fmt(summary['tested_conditions'], 'int')} tested conditions"

    def counted(part, with_cost=False):
        names = []
        for c in part["conditions"]:
            cost = ix["m"].get(("shrinking_cost", labelled, c)) if with_cost else None
            names.append(ix["cond"][c] + (f" ({fmt(cost['value'], 'pts')} points)" if cost else ""))
        return f"{fmt(part['count'], 'int')}{of}" + (": " + ", ".join(names) if names else "")

    harsh = (
        f": the FP32 original is itself harmful, even at full size, in "
        f"{counted(summary['reference_harmful'])}."
    )
    if ix["builds"]["labelled"]["status"] != "usable":
        shrinking = signal = ": not measured: INT8 build failed."
    else:
        clean = ix["m"][("shrinking_cost", labelled, "clean")]
        shrinking = (
            f": on clean images the shrinking cost is {fmt(clean['value'], 'pts')} points "
            f"({fmt(clean['ci95'][0], 'pts')} to {fmt(clean['ci95'][1], 'pts')}). Shrinking made it much "
            f"worse (large shrinking cost) in {counted(summary['large_shrinking_cost'], with_cost=True)}."
        )
        signal = (
            f": unreliable (coverage below {fmt(rule['coverage_min'], 'pct0')}) in "
            f"{counted(summary['coverage_failed'])}. The prediction sets were calibrated on clean images; "
            f"they can be re-calibrated on your own images."
        )
    return [
        (0, "shrinking", "Shrinking", shrinking),
        (0, "harsh conditions", "Harsh conditions", harsh),
        (0, "uncertainty signal", "Uncertainty signal", signal),
    ]


def _summary(label: dict, ix: dict) -> list:
    """(indent, hover key, title, rest of the line) for each summary line: the three opening lines, then
    the conditions by the shrunk build's state and, for harmful ones, by cause."""
    rule, summary = label["envelope"]["rule"], label["summary"]
    of = f" of {fmt(summary['tested_conditions'], 'int')}"

    def names(ids):
        return ", ".join(ix["cond"][c] for c in ids)

    out = _top_lines(label, ix)
    for line in summary["lines"]:
        state, cause = line["state"], line["cause"]
        if state == "not tested":
            continue
        count = f" ({fmt(line['count'], 'int')}{of})"
        if cause is None:
            out.append((0, state, STATE_TITLE[state], f"{count}: {names(line['conditions'])}"))
            continue
        if not any(x[1] == "harmful" for x in out):
            out.append((0, "harmful", STATE_TITLE["harmful"], ", by cause:"))
        parts = [
            f"{_failed_text(part['failed'], rule)} ({fmt(part['count'], 'int')}): {names(part['conditions'])}"
            for part in line["by_failed"]
        ]
        out.append((1, cause, CAUSE_TITLE[cause], f"{count}: " + "; ".join(parts)))
    not_tested = ": every damage type and severity not listed above"
    out.append((0, "not tested", STATE_TITLE["not tested"], not_tested))
    return out


def _sections(label: dict) -> dict:
    """Plain rows of text for each part of the label, shared by Markdown and HTML."""
    ix = _by(label)
    ref, lab = ix["builds"]["reference"], ix["builds"]["labelled"]
    rule = label["envelope"]["rule"]
    level = label["generated"]["ci_level"]
    failed = lab["status"] == "failed"
    out = {
        "badge": (
            "Official label (made by Brokkr from its own runs)"
            if label["source"]["kind"] == "official"
            else "User-submitted label — UNVERIFIED"
        ),
        "do_not_use": _do_not_use(lab),
    }
    out["summary"] = _summary(label, ix)
    ds = {d["dataset_id"]: d for d in label["datasets"]}
    test = ds[label["measurements"][0]["dataset_id"]] if label["measurements"] else next(iter(ds.values()))
    hw = label["hardware"][0]
    out["tested"] = [
        f"Images: {test['name']}, split {test['split']}, {fmt(test['n_items'], 'int')} items; "
        f"licence: {test['licence']}",
        f"Machine: {hw['cpu_model']}, {hw['os']} ({hw['kind']}); {hw['runtime']['name']} "
        f"{hw['runtime']['version']}, "
        f"{hw['runtime']['execution_provider']}",
        f"Intervals: {fmt(level, 'pct0')} bootstrap intervals over the same items (paired for differences).",
        f"Envelope: {rule['name']} ({rule['fixed_in']}); {rule['harm_definition']}. A condition is "
        f"harmful if "
        f"the whole coverage interval is below {fmt(rule['coverage_min'], 'pct0')} or the whole damage-drop "
        f"interval is below {fmt(rule['damage_drop_min'], 'pts0')} points; not harmful if both whole "
        f"intervals "
        f"clear those lines; otherwise borderline.",
        f"Shrinking cost is marked \"large shrinking cost\" when its whole interval is below "
        f"{fmt(rule['large_shrinking_cost_below'], 'pts0')} points, and \"not informative\" when FP32 top-1 "
        f"under that condition is below {fmt(rule['near_floor_fp32_top1_below'], 'pct0')}.",
        f"Label made by {label['generated']['by']} at commit {label['generated']['commit'][:12]}; label data "
        f"licence {label['generated']['label_licence']}.",
    ]
    out["builds"] = [
        [
            b["role"],
            _build_name(b),
            fmt(b["file"]["size_bytes"], "mb") + " MB",
            "usable"
            if b["status"] == "usable"
            else f"FAILED: {b['failure']['check']} ({fmt(b['failure']['value'], 'pct')}; line "
            f"{fmt(b['failure']['limit'], 'pct0')})",
        ]
        for b in (ref, lab)
    ]
    clean_rows = []
    for b in (ref, lab):
        if b["status"] == "failed":
            clean_rows.append([b["precision"].upper(), "INT8 build failed", "—", "—"])
            continue
        cost = ix["m"].get(("shrinking_cost", b["build_id"], "clean")) if b is lab else None
        clean_rows.append(
            [
                b["precision"].upper(),
                _value(ix["m"].get(("top1", b["build_id"], "clean")), "pct"),
                _coverage(ix, b["build_id"], "clean"),
                _value(cost, "pts") if cost else "reference",
            ]
        )
    out["clean"] = clean_rows
    damage_rows = []
    cause = {c: line["cause"] for line in label["summary"]["lines"] for c in line["conditions"]}
    for c in label["conditions"]:
        cid = c["condition_id"]
        if cid == "clean":
            continue
        r_row, l_row = ix["rows"].get((ref["build_id"], cid)), ix["rows"].get((lab["build_id"], cid))
        row = [
            c["label"],
            _value(ix["m"].get(("top1", ref["build_id"], cid)), "pct", ci=False),
            _value(ix["m"].get(("damage_drop", ref["build_id"], cid)), "pts"),
            _envelope_cell(r_row, rule),
        ]
        if failed:
            row += ["—", "—", "—", STATE_MARK["INT8 build failed"], "—", "—"]
        else:
            cost = ix["m"].get(("shrinking_cost", lab["build_id"], cid))
            flag = l_row.get("shrinking_cost_flag") if l_row else None
            row += [
                _value(ix["m"].get(("top1", lab["build_id"], cid)), "pct", ci=False),
                _value(ix["m"].get(("damage_drop", lab["build_id"], cid)), "pts"),
                _coverage(ix, lab["build_id"], cid),
                _envelope_cell(l_row, rule),
                _value(cost, "pts") + (f" [{flag}]" if flag else ""),
                _next_step(l_row, cause.get(cid)),
            ]
        damage_rows.append(row)
    out["damage"] = damage_rows
    out["speed"] = [
        [
            b["precision"].upper(),
            s.get("hardware_kind") or s.get("hardware_id"),
            fmt(s["settings"]["threads"], "int") if s.get("settings", {}).get("threads") else "—",
            (
                f"p50 {fmt(s['p50_ms'], 'size')} ms, p95 {fmt(s['p95_ms'], 'size')} ms, p99 "
                f"{fmt(s['p99_ms'], 'size')} ms"
                if s["status"] == "measured"
                else f"not measured: {s['reason']}"
            ),
        ]
        for s in label["speed"]
        for b in (ix["builds"]["reference"], lab)
        if s["build_id"] == b["build_id"]
    ]
    details = []
    for b in (ref, lab):
        if b["status"] == "failed":
            continue
        e, a = ix["m"].get(("ece", b["build_id"], "clean")), ix["m"].get(("e_aurc", b["build_id"], "clean"))
        t5 = ix["m"].get(("top5", b["build_id"], "clean"))
        conf = label["details"]["conformal"].get(b["build_id"], {})
        details.append(
            f"{b['precision'].upper()}, clean: ECE {_value(e, 'ece')} (positive = worse); E-AURC "
            f"{_value(a, 'eaurc')} (positive = worse); top-5 {_value(t5, 'pct')}; prediction sets "
            f"for a {fmt(conf['target'], 'pct0')} target, threshold {fmt(conf['threshold'], 'thr')} from "
            f"{fmt(conf['calibration_items'], 'int')} clean calibration items."
        )
    s = label["checks"]["fp32_sanity"]
    details.append(
        f"FP32 sanity check: clean top-1 {fmt(s['measured'], 'pct')} vs published "
        f"{fmt(s['published'], 'pct')} "
        f"({s['published_source']}): {'PASS' if s['pass'] else 'FAIL'}."
    )
    out["details"] = details
    out["limits"] = label["limits"]
    out["sources"] = [f"{x['file']} (SHA-256 {x['sha256'][:12]}…)" for x in label["sources"]]
    return out


DAMAGE_HEAD = [
    "Condition",
    "FP32 top-1",
    "FP32 damage drop",
    "FP32 envelope",
    "INT8 top-1",
    "INT8 damage drop",
    "INT8 coverage",
    "INT8 envelope",
    "Shrinking cost",
    "Suggested next step",
]
BUILDS_HEAD = ["Role", "Recipe", "File size", "Status"]
CLEAN_HEAD = ["Build", "Top-1", "Coverage", "Shrinking cost"]
SPEED_HEAD = ["Build", "Hardware", "Threads", "Latency"]
TERMS_NOTE = "Every technical word is explained under \"What the words mean\" below."


def _md_table(head: list, rows: list) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def to_markdown(label: dict) -> str:
    s = _sections(label)
    meta = label["model"]
    front = [
        "---",
        "license: other",
        "license_name: see-label-licences",
        "library_name: onnx",
        "tags:",
        "- brokkr",
        "- brokkr-label",
        "- onnx",
        "- quantization",
    ]
    if meta["modality"] == "image" and meta["task"] == "classification":
        front.append("pipeline_tag: image-classification")
    front.append("---")
    parts = ["\n".join(front), f"# {_title(label)}"]
    if s["do_not_use"]:
        parts.append(f"**{s['do_not_use']}**")
    parts += [
        f"**{s['badge']}**",
        "## Summary",
        "\n".join(f"{'  ' * indent}- **{title}**{rest}" for indent, _, title, rest in s["summary"]),
        TERMS_NOTE,
        "## What was tested",
        "\n".join(f"- {line}" for line in s["tested"]),
        "## Builds",
        _md_table(BUILDS_HEAD, s["builds"]),
        "## Clean images",
        _md_table(CLEAN_HEAD, s["clean"]),
        "## Under damage",
        DAMAGE_NOTE,
        _md_table(DAMAGE_HEAD, s["damage"]),
        "## Speed",
        _md_table(SPEED_HEAD, s["speed"]),
        "## Details",
        "\n".join(f"- {line}" for line in s["details"]),
        "## What the words mean",
        "\n".join(f"- **{term}**: {text}" for term, text in TERMS.items()),
        "## Limits",
        "\n".join(f"- {line}" for line in s["limits"]),
        "## Licences",
        f"- Model code: {meta['licence']['code']}\n- Model weights: {meta['licence']['weights']}\n"
        f"- This label's data: {label['generated']['label_licence']}",
        "## Sources",
        "\n".join(f"- {line}" for line in s["sources"]),
    ]
    return "\n\n".join(parts) + "\n"


CSS = """
:root{--bg:#fbfbf8;--fg:#1d1d1b;--muted:#5f5f58;--line:#d9d8d0;--head:#efeee7;--good:#1b6e3a;--bad:#a3221b;--mid:#8a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe4;--muted:#a9a89f;--line:#3a3a35;
--head:#23231f;
--good:#6fcf8f;--bad:#ff8b80;--mid:#f0c060}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:auto}h1{font-size:1.5rem;margin:.2rem 0}h2{font-size:1.1rem;margin-top:1.8rem}
.badge{display:inline-block;border:1px solid var(--line);border-radius:4px;padding:2px 8px;color:var(--muted)}
.unverified{color:var(--bad);border-color:var(--bad)}
.warning{border-left:4px solid var(--bad);padding:6px 12px;color:var(--bad);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid var(--line);padding:4px 6px;
text-align:left;vertical-align:top}th{background:var(--head)}.wrap{overflow-x:auto}
abbr[title]{text-decoration:underline dotted;cursor:help}
.s-not-harmful-in-our-tests{color:var(--good)}.s-harmful,.s-INT8-build-failed{color:var(--bad);font-weight:600}
.s-borderline{color:var(--mid)}li{margin:2px 0;overflow-wrap:anywhere}small{color:var(--muted)}
dt{font-weight:600}dd{margin:0 0 6px 16px}li.sub{margin-left:24px}
"""


def _hover(text: str) -> str:
    """The escaped text, with its term's explanation as a hover note when it has one."""
    term = HOVER.get(text)
    if term is None:
        return html.escape(text)
    return f'<abbr title="{html.escape(TERMS[term])}">{html.escape(text)}</abbr>'


def _html_table(head: list, rows: list) -> str:
    def cell(c):
        state, _, reason = str(c).partition(": ")  # an envelope cell: "harmful: accuracy dropped"
        if state not in STATE_MARK.values():
            return f"<td>{html.escape(str(c))}</td>"
        reason = html.escape(f": {reason}") if reason else ""
        return f'<td class="s-{state.replace(" ", "-")}">{_hover(state)}{reason}</td>'

    return (
        '<div class="wrap"><table><tr>'
        + "".join(f"<th>{_hover(h)}</th>" for h in head)
        + "</tr>"
        + "".join("<tr>" + "".join(cell(c) for c in r) + "</tr>" for r in rows)
        + "</table></div>"
    )


def to_html(label: dict) -> str:
    s = _sections(label)
    meta = label["model"]
    unverified = label["source"]["kind"] != "official"

    def items(lines):
        return "<ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in lines) + "</ul>"

    summary = "".join(
        f'<li{" class=sub" if indent else ""}><strong><abbr title="{html.escape(TERMS[HOVER[key]])}">'
        f"{html.escape(title)}</abbr></strong>{html.escape(rest)}</li>"
        for indent, key, title, rest in s["summary"]
    )
    body = [f"<h1>{html.escape(_title(label))}</h1>"]
    if s["do_not_use"]:
        body.append(f'<p class="warning">{html.escape(s["do_not_use"])}</p>')
    body += [
        f'<p><span class="badge{" unverified" if unverified else ""}">{html.escape(s["badge"])}</span></p>',
        "<h2>Summary</h2>",
        f"<ul>{summary}</ul>",
        "<p><small>Hover over a dotted word for a plain explanation; all of them are listed under "
        "\"What the words mean\" below.</small></p>",
        "<h2>What was tested</h2>",
        items(s["tested"]),
        "<h2>Builds</h2>",
        _html_table(BUILDS_HEAD, s["builds"]),
        "<h2>Clean images</h2>",
        _html_table(CLEAN_HEAD, s["clean"]),
        "<h2>Under damage</h2>",
        f"<p><small>{html.escape(DAMAGE_NOTE)}</small></p>",
        _html_table(DAMAGE_HEAD, s["damage"]),
        "<h2>Speed</h2>",
        _html_table(SPEED_HEAD, s["speed"]),
        "<h2>Details</h2>",
        items(s["details"]),
        "<h2>What the words mean</h2>",
        "<dl>"
        + "".join(f"<dt>{html.escape(t)}</dt><dd>{html.escape(x)}</dd>" for t, x in TERMS.items())
        + "</dl>",
        "<h2>Limits</h2>",
        items(s["limits"]),
        "<h2>Licences</h2>",
        items(
            [
                f"Model code: {meta['licence']['code']}",
                f"Model weights: {meta['licence']['weights']}",
                f"This label's data: {label['generated']['label_licence']}",
            ]
        ),
        "<h2>Sources</h2>",
        items(s["sources"]),
    ]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(_title(label))} · Brokkr label</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )


def load(path) -> dict:
    return json.loads(open(path, encoding="utf-8").read())
