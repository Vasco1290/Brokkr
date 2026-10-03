"""Draw Brokkr's figures from the released labels and the records, and check every plotted value.

Usage:  python scripts/47_figures.py [--check]
Needs:  published/labels/*/label.json (committed), results/final/breadth_4.1_verdicts.json (H21 and the
        4.1 table) and MobileNetV3-Large's three clean test records in results/accuracy/ (figure 4)
Writes: docs/figures/<figure>-light.svg and -dark.svg, docs/figures/figures.json (every plotted value
        with the file, SHA-256 and field it came from) and the hero block between the "figure-hero"
        markers in README.md

Figures (plan approved by H, 4 October 2026; docs/label_schema.md, note of 4 October 2026):
1. hero-shrinking-cost: shrinking cost, clean vs darkness (Brokkr) s5, with H21's verdict (README)
2. grid-shrinking-cost: 9 models x 12 damaged conditions (exploratory)
3. mnv3l-uncertainty: MobileNetV3-Large coverage with average set size, FP32 and INT8
4. mnv3l-clean-accuracy: MobileNetV3-Large clean top-1, FP32 vs default INT8 vs percentile INT8
5. speed-ratio: INT8 time as a multiple of FP32 time (p50), laptop CPU, relative comparison only

--check writes nothing and exits 1 unless all of these hold (scripts/40_check_labels.py runs it too):
1. up to date: every SVG, figures.json and the README block equal a fresh render;
2. values: every plotted value equals the field it names, re-read from disk with its fingerprint
   (brokkr_edge.results.sha256_of); the
   hero's shrinking costs also equal the 4.1 table's gaps; the grid's 5-point colour edge equals the
   labels' large-cost line;
3. positions: every mark, interval, bar, line and tick, converted back through its axis, is within half
   a unit of its value; every grid cell has the colour of its step, the grey of a near-floor cell only
   when flagged, and a dot exactly when flagged "large shrinking cost";
4. stray numbers: every number a reader sees (SVG text, hover notes, captions, alt text, README block)
   is a manifest value, an axis tick or a colour-step edge;
5. claims: every figure has its entry in docs/claims.json, naming a script that exists.
Font size and colour contrast are checked by tests/test_figures.py.
"""

import argparse
import html
import json
import re
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

from brokkr_edge import figures as F
from brokkr_edge.results import sha256_of

LABELS = Path("published/labels")
VERDICTS = Path("results/final/breadth_4.1_verdicts.json")
ACCURACY = Path("results/accuracy")
MODEL_LIST = Path("brokkr_edge/model_list.json")
OUT = Path("docs/figures")
MANIFEST = OUT / "figures.json"
README = Path("README.md")
CLAIMS = Path("docs/claims.json")
START = "<!-- figure-hero:start (written by scripts/47_figures.py; do not edit by hand) -->"
END = "<!-- figure-hero:end -->"
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?")
WORDS_WITH_DIGITS = ("FP32", "INT8", "H21", "ImageNet-C", "ImageNet-1k", "p50", "top-1", "Stage 1", "Stage 3")
SVG = "{http://www.w3.org/2000/svg}"


def resolve(data, field: list):
    """The value at `field`, a list of keys and list positions (e.g. ["measurements", 12, "value"])."""
    for part in field:
        data = data[part]
    return data


def path_of(field) -> list:
    """A field written as "a.b.3" becomes ["a", "b", 3]; a list is kept (for keys that contain dots)."""
    if isinstance(field, list):
        return field
    return [int(p) if p.isdigit() else p for p in field.split(".")]


class Sources:
    """The values of one figure, each read from a JSON file and recorded with where it came from."""

    files = {}  # path -> (data, sha256), shared by all figures

    def __init__(self):
        self.values = {}

    @classmethod
    def load(cls, path: Path):
        if path not in cls.files:
            cls.files[path] = (json.loads(path.read_text(encoding="utf-8")), sha256_of(path))
        return cls.files[path][0]

    def add(self, vid, path: Path, field, fmt, *, where=None, value_key=None, ci_key=None):
        data = self.load(path)
        field = path_of(field)
        obj = resolve(data, field)
        if where and any(obj[k] != x for k, x in where.items()):
            sys.exit(f"FAIL: {path} {field} is not {where}")
        source = {"file": path.as_posix(), "sha256": self.files[path][1], "field": field}
        entry = {"value": obj[value_key] if value_key else obj, "fmt": fmt}
        if where:
            source["where"] = where
        if value_key:
            source["value_key"] = value_key
        if ci_key:
            source["ci_key"] = ci_key
            entry["ci95"] = obj[ci_key]
        entry["source"] = source
        self.values[vid] = entry
        return vid

    def measurement(self, vid, model, label, metric, build_id, cid, fmt):
        i = next(
            i
            for i, m in enumerate(label["measurements"])
            if (m["metric"], m["build_id"], m["condition_id"]) == (metric, build_id, cid)
        )
        return self.add(
            vid,
            label_path(model),
            f"measurements.{i}",
            fmt,
            value_key="value",
            ci_key="ci95",
            where={"metric": metric, "build_id": build_id, "condition_id": cid},
        )

    def condition_label(self, vid, model, label, cid):
        i = next(i for i, c in enumerate(label["conditions"]) if c["condition_id"] == cid)
        return self.add(vid, label_path(model), f"conditions.{i}.label", "text")


def label_path(model: str) -> Path:
    return LABELS / model / "label.json"


def labels() -> dict:
    return {p.parent.name: Sources.load(p) for p in sorted(LABELS.glob("*/label.json"))}


def wrap(text: str) -> str:
    return textwrap.fill(text, width=104, break_on_hyphens=False)


# ---- the five figures: each returns its manifest entry and its two SVGs ----


def hero_figure(labs: dict) -> tuple:
    src, verdicts = Sources(), Sources.load(VERDICTS)
    judged = verdicts["settings"]["judged_models"]
    first = labs[judged[0]]
    dark = "brokkr/darkness/5"
    src.condition_label("dark_label", judged[0], first, dark)
    src.add("large_cost_line", label_path(judged[0]), "envelope.rule.large_shrinking_cost_below", "pts1")
    src.add("large_cost_abs", label_path(judged[0]), "envelope.rule.large_shrinking_cost_below", "abs0")
    src.add("ci_level", label_path(judged[0]), "generated.ci_level", "pct0")
    for key in ("needed", "holding", "judged", "verdict"):
        src.add(f"h21_{key}", VERDICTS, f"metrics.verdicts.H21.{key}", "text" if key == "verdict" else "int")
    src.add("n_images", VERDICTS, "settings.n_images", "int")
    failed = next(iter(verdicts["settings"]["left_out"]))
    src.add("failed_name", label_path(failed), "model.display_name", "text")
    rows = []
    for m in judged:
        lab = labs[m]
        rows.append(
            {
                "name": src.add(f"{m}.name", label_path(m), "model.display_name", "text"),
                "clean": src.measurement(
                    f"{m}.clean", m, lab, "shrinking_cost", lab["label_id"], "clean", "pts1"
                ),
                "dark": src.measurement(f"{m}.dark", m, lab, "shrinking_cost", lab["label_id"], dark, "pts1"),
                "gap": src.add(
                    f"{m}.h21_gap",
                    VERDICTS,
                    f"metrics.verdicts.H21.items.{m}",
                    "pts1",
                    value_key="value",
                    ci_key="ci95",
                ),
                "holds": src.add(f"{m}.h21_holds", VERDICTS, f"metrics.verdicts.H21.items.{m}.holds", "bool"),
            }
        )
    v = src.values
    rows.sort(key=lambda r: (v[r["dark"]]["value"], v[r["name"]]["value"]))
    svgs, axes = {}, None
    for theme in F.THEMES:
        svgs[theme], axes = F.hero(rows, v, theme)
    s = lambda vid: F.show(v, vid)  # noqa: E731
    caption = (
        f"Pre-registered condition: {s('dark_label')} (H21). {s('h21_judged')} models, {s('n_images')} test "
        f"images; bars are {s('ci_level')} paired intervals (one narrower than its marker is hidden behind "
        f"it). H21 predicted a large extra gap (the shrinking cost under the damage minus the shrinking cost "
        f"on clean images, as defined in docs/hypotheses_stage4.md) in at least {s('h21_needed')} of the "
        f"{s('h21_judged')} models; {s('h21_holding')} showed it; verdict "
        f"{s('h21_verdict')}. The dashed line is the label's large-cost line (whole interval more than "
        f"{s('large_cost_abs')} points below full size), adopted for the labels after these results existed; "
        f"it is not what H21 judged."
    )
    wide, wide_axes = {}, None
    for theme in F.THEMES:
        wide[theme], wide_axes = F.hero_wide(rows, v, theme)
    # Two layouts of one figure (H, 4 October 2026): the README shows the wide one on screens at least
    # 768 px wide and the narrow one on phones.
    narrow_uses = ["README.md (phones)", "website: Why labels? (phones)"]
    wide_uses = ["README.md (wide screens)", "website: Why labels?", "Report 2"]
    return [
        (figure_entry("hero-shrinking-cost", narrow_uses, "fig-hero", 343, caption, svgs, axes, v), svgs),
        (
            figure_entry("hero-shrinking-cost-wide", wide_uses, "fig-hero", 600, caption, wide, wide_axes, v),
            wide,
        ),
    ]


def grid_figure(labs: dict) -> tuple:
    src = Sources()
    judged = Sources.load(VERDICTS)["settings"]["judged_models"]
    first_model = judged[0]
    first = labs[first_model]
    src.add("large_cost_line", label_path(first_model), "envelope.rule.large_shrinking_cost_below", "abs0")
    src.add("near_floor_line", label_path(first_model), "envelope.rule.near_floor_fp32_top1_below", "pct0")
    src.add("ci_level", label_path(first_model), "generated.ci_level", "pct0")
    columns = []
    for c in first["conditions"]:
        if c["condition_id"] == "clean":
            continue
        columns.append(
            {
                "cid": c["condition_id"],
                "suite": c["suite"],
                "damage": c["damage"],
                "severity": c["severity"],
                "label": src.condition_label(
                    f"label.{c['condition_id']}", first_model, first, c["condition_id"]
                ),
            }
        )
    i = next(i for i, m in enumerate(first["measurements"]) if m["metric"] == "shrinking_cost")
    src.add("n_images", label_path(first_model), f"measurements.{i}.n_items", "int")
    rows = []
    for m in judged:
        lab = labs[m]
        cells = []
        for col in columns:
            r = next(
                k
                for k, row in enumerate(lab["envelope"]["rows"])
                if (row["build_id"], row["condition_id"]) == (lab["label_id"], col["cid"])
            )
            cells.append(
                {
                    "cost": src.measurement(
                        f"{m}.{col['cid']}.cost",
                        m,
                        lab,
                        "shrinking_cost",
                        lab["label_id"],
                        col["cid"],
                        "pts1",
                    ),
                    "flags": src.add(
                        f"{m}.{col['cid']}.flags",
                        label_path(m),
                        f"envelope.rows.{r}.shrinking_cost_flags",
                        "flags",
                    ),
                }
            )
        rows.append(
            {
                "name": src.add(f"{m}.name", label_path(m), "model.display_name", "text"),
                "count": src.add(f"{m}.count", label_path(m), "summary.large_shrinking_cost.count", "int"),
                "cells": cells,
            }
        )
    v = src.values
    rows.sort(key=lambda r: (-v[r["count"]]["value"], v[r["name"]]["value"]))
    svgs = {theme: F.grid(rows, columns, v, theme)[0] for theme in F.THEMES}
    s = lambda vid: F.show(v, vid)  # noqa: E731
    caption = (
        f"Exploratory, not pre-registered. Every usable INT8 build in the released labels, "
        f"{s('n_images')} test images per condition. Shrinking cost = INT8 minus FP32 top-1 on the same "
        f"images; each cell's hover note gives its {s('ci_level')} paired interval. Colour: the measured "
        f"cost. Dot: large cost (whole interval more than {s('large_cost_line')} points below full size). "
        f"Grey with a dash: near floor (FP32 below {s('near_floor_line')} top-1), where a small cost proves "
        f"nothing. Rows are sorted by their "
        f"number of large costs."
    )
    return figure_entry(
        "grid-shrinking-cost", ["website: Why labels?", "Report 2"], "fig-grid", 720, caption, svgs, {}, v
    ), svgs


def uncertainty_figure(labs: dict) -> tuple:
    src, m = Sources(), "mobilenet_v3_large"
    lab = labs[m]
    src.add("model_name", label_path(m), "model.display_name", "text")
    builds = []
    for k, b in enumerate(lab["builds"]):
        builds.append(
            {
                "key": b["role"],
                "id": b["build_id"],
                "name": src.add(f"build.{b['role']}", label_path(m), f"builds.{k}.display_name", "text"),
                "shape": "circle" if b["role"] == "reference" else "square",
                "colour": "fp32" if b["role"] == "reference" else "int8",
            }
        )
    i = next(i for i, x in enumerate(lab["measurements"]) if x["metric"] == "coverage")
    src.add("target", label_path(m), f"measurements.{i}.settings.target_coverage", "pct0")
    src.add("calibration_items", label_path(m), f"measurements.{i}.settings.calibration_items", "int")
    src.add("n_images", label_path(m), f"measurements.{i}.n_items", "int")
    src.add("coverage_line", label_path(m), "envelope.rule.coverage_min", "pct0")
    src.add("ci_level", label_path(m), "generated.ci_level", "pct0")
    rows = []
    for c in lab["conditions"]:
        cid = c["condition_id"]
        rows.append(
            {
                "label": src.condition_label(f"label.{cid}", m, lab, cid),
                "cells": {
                    b["key"]: {
                        "coverage": src.measurement(
                            f"{b['key']}.{cid}.coverage", m, lab, "coverage", b["id"], cid, "pct1"
                        ),
                        "size": src.measurement(
                            f"{b['key']}.{cid}.size", m, lab, "mean_set_size", b["id"], cid, "size"
                        ),
                    }
                    for b in builds
                },
            }
        )
    v = src.values
    svgs, axes = {}, None
    for theme in F.THEMES:
        svgs[theme], axes = F.uncertainty(rows, builds, v, theme)
    s = lambda vid: F.show(v, vid)  # noqa: E731
    caption = (
        f"{s('model_name')}, {s('n_images')} test images per condition; bars are {s('ci_level')} intervals "
        f"(one narrower than its marker is hidden behind it). Prediction sets were calibrated on clean "
        f"images ({s('calibration_items')} conformal-calibration images) for a {s('target')} target; under "
        f"damage there is no {s('target')} promise. The dashed "
        f"{s('coverage_line')} line is the label's coverage line. Coverage is always shown with its average "
        f"set size."
    )
    return figure_entry(
        "mnv3l-uncertainty",
        ["website: Why labels?", "Report 2"],
        "fig-uncertainty",
        343,
        caption,
        svgs,
        axes,
        v,
    ), svgs


def accuracy_figure(labs: dict) -> tuple:
    src, m = Sources(), "mobilenet_v3_large"
    src.add("model_name", label_path(m), "model.display_name", "text")
    # Brokkr's interval level, as stated on its label (the records name their intervals top1_ci95).
    src.add("ci_level", label_path(m), "generated.ci_level", "pct0")
    rows = []
    for key, file, colour in (
        ("fp32", "fp32", "fp32"),
        ("int8_minmax", "int8", "int8"),
        ("int8_percentile99.99", "int8_percentile99.99", "int8"),
    ):
        record = ACCURACY / f"{m}_{file}_imagenet-1k-val_test.json"
        rows.append(
            {
                "name": src.add(f"{key}.name", MODEL_LIST, ["precision_display_names", key], "text"),
                "top1": src.add(
                    f"{key}.top1", record, "metrics", "pct1", value_key="top1", ci_key="top1_ci95"
                ),
                "ties": src.add(f"{key}.ties", record, "metrics.top1_tied_images", "int"),
                "n": src.add(f"{key}.n_images", record, "settings.n_images", "int"),
                "split": src.add(f"{key}.split", record, "settings.split", "text"),
                "colour": colour,
            }
        )
    v = src.values
    if len({v[r["n"]]["value"] for r in rows}) != 1 or {v[r["split"]]["value"] for r in rows} != {"test"}:
        sys.exit("FAIL: the three clean records do not share one split and image count")
    svgs, axes = {}, None
    for theme in F.THEMES:
        svgs[theme], axes = F.clean_accuracy(rows, v, theme)
    s = lambda vid: F.show(v, vid)  # noqa: E731
    ties = "; ".join(f"{s(r['name'])}: {s(r['ties'])}" for r in rows)
    caption = (
        f"{s('model_name')}, clean {s(rows[0]['split'])} split, {s(rows[0]['n'])} images (Stage 1 and "
        f"Stage 3 records). Each interval is a {s('ci_level')} bootstrap interval for one build on its own, "
        f"not a paired comparison between builds. Exact score ties are broken toward the lower class "
        f"number; tied images: {ties}."
    )
    return figure_entry(
        "mnv3l-clean-accuracy",
        ["Report 2", "website: methods page"],
        "fig-clean-accuracy",
        343,
        caption,
        svgs,
        axes,
        v,
    ), svgs


def speed_figure(labs: dict) -> tuple:
    src = Sources()
    first_model = "mobilenet_v3_large"
    first = labs[first_model]
    rows, threads = [], set()
    for m, lab in labs.items():
        path = label_path(m)
        bi = next(k for k, b in enumerate(lab["builds"]) if b["role"] == "labelled")
        row = {"name": src.add(f"{m}.name", path, "model.display_name", "text"), "ratios": {}, "unstable": {}}
        row["failed"] = lab["builds"][bi]["status"] == "failed"
        if row["failed"]:
            src.add("failed_text", VERDICTS, f"settings.left_out.{m}", "text")
            rows.append(row)
            continue
        ref_id = next(b["build_id"] for b in lab["builds"] if b["role"] == "reference")
        for k, sp in enumerate(lab["speed"]):
            if sp["build_id"] != lab["label_id"] or sp["status"] != "measured":
                continue
            t = sp["settings"]["threads"]
            threads.add(t)
            row["ratios"][t] = src.add(
                f"{m}.{t}t.ratio", path, f"speed.{k}.time_vs_reference.ratio_p50", "ratio"
            )
            j = next(
                j
                for j, x in enumerate(lab["speed"])
                if x["build_id"] == ref_id and x["status"] == "measured" and x["settings"]["threads"] == t
            )
            row["unstable"][t] = [
                src.add(f"{m}.{t}t.int8_unstable", path, f"speed.{k}.unstable", "bool"),
                src.add(f"{m}.{t}t.fp32_unstable", path, f"speed.{j}.unstable", "bool"),
            ]
            src.add(f"threads.{t}", path, f"speed.{k}.settings.threads", "int")
        rows.append(row)
    k = next(
        k
        for k, sp in enumerate(first["speed"])
        if sp["build_id"] == first["label_id"] and sp["status"] == "measured"
    )
    sp = first["speed"][k]
    p = label_path(first_model)
    src.add("unstable_line", p, f"speed.{k}.unstable_above_pct", "pctv")
    src.add("sessions", p, f"speed.{k}.sessions", "int")
    src.add("batch", p, f"speed.{k}.settings.batch", "int")
    src.add("n_physical", p, f"speed.{k}.pinning.n_physical", "int")
    src.add("n_logical", p, f"speed.{k}.pinning.n_logical", "int")
    src.add("reported_by", p, f"speed.{k}.pinning.reported_by", "text")
    h = next(i for i, x in enumerate(first["hardware"]) if x["hardware_id"] == sp["hardware_id"])
    src.add("cpu", p, f"hardware.{h}.cpu_model", "text")
    src.add("os", p, f"hardware.{h}.os", "text")
    v = src.values
    keys = sorted(threads)
    thread_marks = [
        {
            "key": t,
            "label": f"{F.show(v, f'threads.{t}')} thread{'s' if t > 1 else ''}",
            "shape": shape,
            "colour": colour,
        }
        for t, shape, colour in zip(keys, ("circle", "triangle"), ("int8", "second"), strict=True)
    ]
    rows.sort(
        key=lambda r: (
            r["failed"],
            v[r["ratios"][keys[0]]]["value"] if not r["failed"] else 0,
            v[r["name"]]["value"],
        )
    )
    svgs, axes = {}, None
    for theme in F.THEMES:
        svgs[theme], axes = F.speed(rows, thread_marks, v, theme)
    s = lambda vid: F.show(v, vid)  # noqa: E731
    most = f"threads.{keys[-1]}"
    caption = (
        f"Laptop latency on {s('cpu')}, {s('os')}, batch {s('batch')}. Pinned to the laptop's "
        f"{s('n_physical')} performance cores ({s('n_logical')} hardware threads, as reported by "
        f"{s('reported_by')}); the {s(most)}-thread setting therefore runs on {s('n_physical')} physical "
        f"cores, and the pin was not read back. Each ratio is INT8's p50 divided by FP32's p50, each the "
        f"median across {s('sessions')} sessions. Hollow marker: speed varied a lot between repeat runs "
        f"(spread above {s('unstable_line')}) for INT8 or FP32; treat as rough. A relative comparison on one "
        f"laptop CPU only, not a measure of speed on edge devices."
    )
    return figure_entry(
        "speed-ratio", ["website: compare page"], "fig-speed", 343, caption, svgs, axes, v
    ), svgs


def figure_entry(name, appears_in, claim_id, min_px, caption, svgs, axes, values) -> dict:
    desc = re.search(r'<desc id="desc">(.*?)</desc>', svgs["light"], re.S).group(1)
    return {
        "name": name,
        "files": {theme: (OUT / f"{name}-{theme}.svg").as_posix() for theme in svgs},
        "appears_in": appears_in,
        "claim_id": claim_id,
        "min_display_px": min_px,
        "caption": caption,
        "alt": html.unescape(desc),
        "axes": axes,
        "values": values,
    }


def render_all() -> tuple:
    labs = labels()
    built = []
    for make in (hero_figure, grid_figure, uncertainty_figure, accuracy_figure, speed_figure):
        made = make(labs)
        built += made if isinstance(made, list) else [made]
    manifest = {
        "generated_by": "scripts/47_figures.py",
        "about": "Every value plotted in docs/figures/*.svg, with the file, SHA-256 and field it was read "
        "from. Generated; never edit by hand.",
        "step_edges": list(F.STEP_EDGES),
        "figures": {entry.pop("name"): entry for entry, _ in built},
    }
    svgs = {}
    for entry, theme_svgs in built:
        for theme, text in theme_svgs.items():
            svgs[entry["files"][theme]] = text
    # The wide hero on screens at least 768 px wide, the narrow one on phones; each in light and dark.
    hero, wide = manifest["figures"]["hero-shrinking-cost"], manifest["figures"]["hero-shrinking-cost-wide"]
    wide_screen, dark = "(min-width: 768px)", "(prefers-color-scheme: dark)"
    block = (
        "\n<picture>\n"
        f'  <source media="{wide_screen} and {dark}" srcset="{wide["files"]["dark"]}">\n'
        f'  <source media="{wide_screen}" srcset="{wide["files"]["light"]}">\n'
        f'  <source media="(prefers-color-scheme: dark)" srcset="{hero["files"]["dark"]}">\n'
        f'  <img src="{hero["files"]["light"]}" alt="{html.escape(hero["alt"], quote=True)}">\n'
        "</picture>\n\n"
        f"{wrap('*' + hero['caption'] + '*')}\n"
    )
    return manifest, svgs, block


def manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"


# ---- checks ----


def check_values(manifest: dict) -> list:
    """Check 2: every value re-read from its file, the hero against the 4.1 table, the grid's step edge."""
    problems = []
    for name, fig in manifest["figures"].items():
        for vid, entry in fig["values"].items():
            s = entry["source"]
            if sha256_of(s["file"]) != s["sha256"]:
                problems.append(f"{name} {vid}: {s['file']} changed since the figure was drawn")
                continue
            obj = resolve(json.loads(Path(s["file"]).read_text(encoding="utf-8")), s["field"])
            if any(obj[k] != x for k, x in s.get("where", {}).items()):
                problems.append(f"{name} {vid}: {s['field']} is not {s['where']}")
            value = obj[s["value_key"]] if "value_key" in s else obj
            if value != entry["value"] or ("ci_key" in s and obj[s["ci_key"]] != entry["ci95"]):
                problems.append(f"{name} {vid}: differs from {s['file']} {s['field']}")
    hero = manifest["figures"]["hero-shrinking-cost"]["values"]
    table = json.loads(VERDICTS.read_text(encoding="utf-8"))["raw"]["table"]
    for vid, entry in hero.items():
        if vid.endswith((".clean", ".dark")):
            model, which = vid.rsplit(".", 1)
            condition = "clean" if which == "clean" else hero["dark_label"]["value"]
            row = next(r for r in table if r["model"] == model and r["condition"] == condition)
            if (row["gap"]["value"], row["gap"]["ci95"]) != (entry["value"], entry["ci95"]):
                problems.append(f"hero {vid}: differs from the 4.1 table's gap")
    line = manifest["figures"]["grid-shrinking-cost"]["values"]["large_cost_line"]["value"]
    if manifest["step_edges"][1] != -line:
        problems.append("grid: the 5-point colour edge is not the labels' large-cost line")
    return problems


def position_checker(sc, where: str, problems: list):
    """near(x, value, what): record a problem if x is more than half a unit from where `value` belongs."""

    def near(x, value, what):
        if abs(float(x) - sc(value)) > 0.5:
            problems.append(f"{where}: {what} drawn at {x}, value says {sc(value):.2f}")

    return near


def check_positions(manifest: dict) -> list:
    """Check 3: every drawn element sits where its value says."""
    problems = []
    for name, fig in manifest["figures"].items():
        values, axes = fig["values"], fig["axes"]
        seen = {}
        for theme, file in fig["files"].items():
            colours = F.THEMES[theme]
            root = ET.fromstring(Path(file).read_text(encoding="utf-8"))
            ids = set()
            for el in root.iter():
                tag, part, vid = el.tag.replace(SVG, ""), el.get("data-part"), el.get("data-v")
                if vid is not None and vid not in values:
                    problems.append(f"{file}: data-v {vid} is not a manifest value")
                    continue
                if vid:
                    ids.add(vid)
                where = f"{file} {vid or el.get('data-tick')}"
                if part == "cell":
                    flags = values[el.get("data-flags")]["value"]
                    floor = "not informative" in flags
                    want = colours["floor"] if floor else colours["steps"][F.step_of(values[vid]["value"])]
                    if el.get("fill") != want or (el.get("data-floor") == "1") != floor:
                        problems.append(f"{where}: cell colour or near-floor mark does not match its value")
                    continue
                if part is None:
                    continue
                axis = axes[el.get("data-axis")]
                near = position_checker(F.scale(axis["domain"], axis["range"]), where, problems)
                if part == "tick":
                    t = float(el.get("data-tick"))
                    near(el.get("x"), t, "tick")
                    if axis["ticks"].get(el.text) != t:
                        problems.append(f"{where}: tick text {el.text!r} does not name its position")
                elif part == "line":
                    value = 1.0 if el.get("data-at") == "1" else values[vid]["value"]
                    near(el.get("x1"), value, "line")
                    near(el.get("x2"), value, "line")
                elif part == "ci":
                    near(el.get("x1"), values[vid]["ci95"][0], "interval start")
                    near(el.get("x2"), values[vid]["ci95"][1], "interval end")
                elif part == "bar":
                    near(el.get("x"), 0.0, "bar start")
                    near(float(el.get("x")) + float(el.get("width")), values[vid]["value"], "bar end")
                elif part == "mark":
                    if tag == "circle":
                        x = float(el.get("cx"))
                    elif tag == "rect":
                        x = float(el.get("x")) + float(el.get("width")) / 2
                    else:
                        x = float(el.get("points").split()[0].split(",")[0])
                    near(x, values[vid]["value"], "mark")
            dots = {el.get("data-dot") for el in root.iter() if el.get("data-dot")}
            flagged = {
                vid
                for vid, e in values.items()
                if e["fmt"] == "flags" and "large shrinking cost" in e["value"]
            }
            if dots != flagged:
                problems.append(f"{file}: dots differ from the 'large shrinking cost' flags")
            seen[theme] = ids
        if len({frozenset(s) for s in seen.values()}) != 1:
            problems.append(f"{name}: the light and dark files draw different values")
    return problems


def visible_text(svg_text: str) -> list:
    root = ET.fromstring(svg_text)
    return [el.text or "" for el in root.iter() if el.tag.replace(SVG, "") in ("text", "title", "desc")]


def stray_numbers(fig: dict, texts: list) -> list:
    """Numbers in `texts` that are not a value of `fig` (in its own format), an axis tick or a step edge."""
    values = fig["values"]
    names = sorted({e["value"] for e in values.values() if e["fmt"] == "text"}, key=len, reverse=True)
    allowed = set()
    for e in values.values():
        if e["fmt"] in F.FORMATS and e["fmt"] != "text" and isinstance(e["value"], (int, float)):
            for x in [e["value"], *e.get("ci95", [])]:
                allowed |= set(NUMBER.findall(F.FORMATS[e["fmt"]](x)))
    for axis in fig["axes"].values():
        for label in axis["ticks"]:
            allowed |= set(NUMBER.findall(label))
    for label in F.step_names():
        allowed |= set(NUMBER.findall(label))
    loose = []
    for text in texts:
        for s in [*names, *WORDS_WITH_DIGITS]:
            text = text.replace(s, " ")
        loose += [t for t in NUMBER.findall(text) if t not in allowed]
    return loose


def check_numbers(manifest: dict, readme_block: str) -> list:
    """Check 4: no number a reader sees is unexplained."""
    problems = []
    for name, fig in manifest["figures"].items():
        texts = [fig["caption"], fig["alt"]]
        for file in fig["files"].values():
            texts += visible_text(Path(file).read_text(encoding="utf-8"))
        if name == "hero-shrinking-cost":
            texts.append(re.sub(r"<[^>]+>", " ", html.unescape(readme_block.replace("docs/figures/", ""))))
        loose = stray_numbers(fig, texts)
        if loose:
            problems.append(
                f"{name}: numbers not explained by its values, ticks or step edges: {sorted(set(loose))}"
            )
    return problems


def check_claims(manifest: dict) -> list:
    """Check 5: every figure has a complete claims-register entry naming a script that exists."""
    claims = {c["id"]: c for c in json.loads(CLAIMS.read_text(encoding="utf-8"))["claims"]}
    problems = []
    for name, fig in manifest["figures"].items():
        c = claims.get(fig["claim_id"])
        if c is None:
            problems.append(f"{name}: no claim {fig['claim_id']} in {CLAIMS}")
            continue
        missing = [
            k
            for k in ("claim", "figure", "scope", "status", "prior_work", "command", "check")
            if not c.get(k)
        ]
        script = c.get("command", "").split()[1:2]
        figures = c["figure"] if isinstance(c.get("figure"), list) else [c.get("figure")]
        if missing or name not in figures or not script or not Path(script[0]).exists():
            problems.append(f"{name}: claim {c['id']} is incomplete or names a missing script ({missing})")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest, svgs, block = render_all()
    text = README.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        sys.exit("FAIL: README.md needs exactly one pair of figure-hero markers")
    before, rest = text.split(START)
    old_block, after = rest.split(END)

    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)
        for file, svg in svgs.items():
            Path(file).write_text(svg, encoding="utf-8", newline="\n")
        MANIFEST.write_text(manifest_text(manifest), encoding="utf-8", newline="\n")
        README.write_text(before + START + block + END + after, encoding="utf-8", newline="\n")
        print(
            f"wrote {len(svgs)} SVG files and {MANIFEST} ({len(manifest['figures'])} figures, "
            f"{sum(len(f['values']) for f in manifest['figures'].values())} values), and the README's "
            f"hero block"
        )
        return

    stale = [
        f for f, svg in svgs.items() if not Path(f).exists() or Path(f).read_text(encoding="utf-8") != svg
    ]
    if not MANIFEST.exists() or MANIFEST.read_text(encoding="utf-8") != manifest_text(manifest):
        stale.append(MANIFEST.as_posix())
    if old_block != block:
        stale.append("README.md (figure-hero block)")
    committed = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else manifest
    results = [
        ("up to date", [f"out of date: {f}; run scripts/47_figures.py" for f in stale]),
        ("values match their sources", check_values(committed)),
        ("positions match values", check_positions(committed)),
        ("no stray numbers", check_numbers(committed, old_block)),
        ("claims register", check_claims(committed)),
    ]
    ok = True
    n_values = sum(len(f["values"]) for f in committed["figures"].values())
    for what, problems in results:
        ok &= not problems
        print(
            f"{'PASS' if not problems else 'FAIL'}: figures, {what}"
            + (
                f" ({n_values} values, {len(committed['figures'])} figures, {len(svgs)} SVG files)"
                if what == "values match their sources" and not problems
                else ""
            )
        )
        for p in problems[:10]:
            print(f"     {p}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
