"""Render a label.json as a Markdown model card (Hugging Face metadata at the top) and as an HTML page.

Both read ONLY the label; they compute nothing. Every number goes through fmt(), so
unexplained_numbers() can check that each number in the output is a label.json number, shown in one
of the formats below. Values are rounded only here.
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


def _title(label: dict) -> str:
    b = {x["role"]: x for x in label["builds"]}
    labelled = b["labelled"]
    return f"{label['model']['name']} · {labelled['precision'].upper()} ({labelled['recipe']['method']})"


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
        )
    }
    out["summary"] = [
        f"{STATE_MARK[line['state']].capitalize()}"
        + (
            f" ({fmt(line['count'], 'int')} of {fmt(label['summary']['tested_conditions'], 'int')})"
            if line["count"] is not None
            else ""
        )
        + ": "
        + ", ".join(line["conditions"])
        for line in label["summary"]["lines"]
    ]
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
        f"Label made by {label['generated']['by']} at commit {label['generated']['commit'][:12]}; label data "
        f"licence {label['generated']['label_licence']}.",
    ]
    out["builds"] = [
        [
            b["role"],
            b["precision"].upper(),
            b["recipe"]["method"],
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
    for c in label["conditions"]:
        cid = c["condition_id"]
        if cid == "clean":
            continue
        r_row, l_row = ix["rows"].get((ref["build_id"], cid)), ix["rows"].get((lab["build_id"], cid))
        row = [
            c["label"],
            _value(ix["m"].get(("top1", ref["build_id"], cid)), "pct", ci=False),
            _value(ix["m"].get(("damage_drop", ref["build_id"], cid)), "pts"),
            STATE_MARK[r_row["state"]] if r_row else "—",
        ]
        if failed:
            row += ["—", "—", "—", STATE_MARK["INT8 build failed"], "—"]
        else:
            cost = ix["m"].get(("shrinking_cost", lab["build_id"], cid))
            flag = l_row.get("shrinking_cost_flag") if l_row else None
            row += [
                _value(ix["m"].get(("top1", lab["build_id"], cid)), "pct", ci=False),
                _value(ix["m"].get(("damage_drop", lab["build_id"], cid)), "pts"),
                _coverage(ix, lab["build_id"], cid),
                STATE_MARK[l_row["state"]] if l_row else "—",
                _value(cost, "pts") + (f" [{flag}]" if flag else ""),
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
]


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
    parts = [
        "\n".join(front),
        f"# {_title(label)}",
        f"**{s['badge']}**",
        "## Summary",
        "\n".join(f"- {line}" for line in s["summary"]),
        "## What was tested",
        "\n".join(f"- {line}" for line in s["tested"]),
        "## Builds",
        _md_table(["Role", "Precision", "Recipe", "File size", "Status"], s["builds"]),
        "## Clean images",
        _md_table(["Build", "Top-1", "Coverage", "Shrinking cost"], s["clean"]),
        "## Under damage",
        "Damage drop and shrinking cost in points (positive = better).",
        _md_table(DAMAGE_HEAD, s["damage"]),
        "## Speed",
        _md_table(["Build", "Hardware", "Threads", "Latency"], s["speed"]),
        "## Details",
        "\n".join(f"- {line}" for line in s["details"]),
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
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid var(--line);padding:4px 6px;
text-align:left;vertical-align:top}th{background:var(--head)}.wrap{overflow-x:auto}
.s-not-harmful-in-our-tests{color:var(--good)}.s-harmful,.s-INT8-build-failed{color:var(--bad);font-weight:600}
.s-borderline{color:var(--mid)}li{margin:2px 0}small{color:var(--muted)}
"""


def _html_table(head: list, rows: list) -> str:
    def cell(c):
        text = html.escape(str(c))
        cls = next((f' class="s-{v.replace(" ", "-")}"' for v in STATE_MARK.values() if str(c) == v), "")
        return f"<td{cls}>{text}</td>"

    return (
        '<div class="wrap"><table><tr>'
        + "".join(f"<th>{html.escape(h)}</th>" for h in head)
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

    body = [
        f"<h1>{html.escape(_title(label))}</h1>",
        f'<p><span class="badge{" unverified" if unverified else ""}">{html.escape(s["badge"])}</span></p>',
        "<h2>Summary</h2>",
        items(s["summary"]),
        "<h2>What was tested</h2>",
        items(s["tested"]),
        "<h2>Builds</h2>",
        _html_table(["Role", "Precision", "Recipe", "File size", "Status"], s["builds"]),
        "<h2>Clean images</h2>",
        _html_table(["Build", "Top-1", "Coverage", "Shrinking cost"], s["clean"]),
        "<h2>Under damage</h2>",
        "<p><small>Damage drop and shrinking cost in points (positive = better).</small></p>",
        _html_table(DAMAGE_HEAD, s["damage"]),
        "<h2>Speed</h2>",
        _html_table(["Build", "Hardware", "Threads", "Latency"], s["speed"]),
        "<h2>Details</h2>",
        items(s["details"]),
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
        f"<title>{html.escape(meta['name'])} label</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(body)}</main></body></html>\n"
    )


def load(path) -> dict:
    return json.loads(open(path, encoding="utf-8").read())
