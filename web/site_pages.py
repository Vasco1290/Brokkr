"""The website's page types, one function each (docs/website_v0_plan.md, sections 5c and 12e).

Each function takes the Site (web/site_build.py) and returns a list of (path, Page). It reads only the
released labels and docs/figures/figures.json, and formats every number with brokkr_edge.label_render's
formats, so web/site_checks.py can trace each number on a page back to the labels and figures the page
names. Wording shared with the labels (glossary, suggested next steps, speed and licence lines) comes from
label_render too. PAGE_TYPES lists the page types in navigation order; adding one is one new function and
one new entry.
"""

import html
from dataclasses import dataclass, field

import site_meter as M
import themes
from site_labels import (
    SOURCE_WORDS,
    address,
    build_id,
    builds,
    condition_names,
    envelope_row,
    label_dir,
    measurement,
    the_test_split,
)

from brokkr_edge import label_render as LR

FEATURED = "mobilenet_v3_large"  # the bench meter's model (the prototypes' choice, approved by H)
# The meter's four conditions (H, 4 October 2026, section 12e).
METER_CONDITIONS = ["clean", "brokkr/darkness/5", "imagenet-c/fog/3", "imagenet-c/contrast/5"]
HERO = ("hero-shrinking-cost-wide", "hero-shrinking-cost")  # wide screens, phones
GITHUB = "https://github.com/Vasco1290/Brokkr"
ICON = {"not harmful": "✓", "borderline": "◐", "harmful": "⚠", "not tested": "○", "INT8 build failed": "✕"}
KIND = {
    "not harmful": "ok",
    "borderline": "borderline",
    "harmful": "harm",
    "not tested": "untested",
    "INT8 build failed": "failed",
}


@dataclass
class Page:
    title: str
    body: str
    labels: list  # the labels (folder names) whose numbers may appear on the page
    figures: list = field(default_factory=list)  # entries of docs/figures/figures.json used on the page
    allowed: dict = field(default_factory=dict)  # other numbers shown, each with its reason


@dataclass
class PageType:
    key: str
    nav: str | None  # title in the navigation; None = not in it
    path: str  # the page the navigation links to
    make: object  # function(site) -> [(path, Page)]


def esc(x) -> str:
    return html.escape(str(x))


def f(value, kind: str) -> str:
    return LR.fmt(value, kind)


def count(n: int) -> str:
    """A whole number with a thousands separator (10,000)."""
    return f"{n:,}"


def interval(m: dict, kind: str) -> str:
    return f"{f(m['ci95'][0], kind)} to {f(m['ci95'][1], kind)}"


def hover(term: str, text: str) -> str:
    """`text` with the label glossary's plain explanation of `term` as a hover note."""
    return f'<abbr title="{esc(LR.TERMS[term])}">{esc(text)}</abbr>'


def up(path: str) -> str:
    return "../" * path.count("/")


def short_condition(label_text: str) -> str:
    """'fog (Brokkr) s3' -> 'fog s3' for a key; the readout shows the full name with its suite."""
    return label_text.replace(" (Brokkr)", "").replace(" (ImageNet-C)", "")


def name_list(label: dict, ids: list) -> str:
    names = condition_names(label)
    return "".join(f'<li class="chip">{esc(names[c])}</li>' for c in ids)


def collapsed(summary: str, label: dict, ids: list) -> str:
    """A condition list, collapsed by default (section 12f); nothing when the list is empty."""
    if not ids:
        return ""
    chips = f'<ul class="chips">{name_list(label, ids)}</ul>'
    return f'<details class="names"><summary>{summary}</summary>{chips}</details>'


# ---- landing ----


def pixel_hero(cols: int = 28, rows: int = 9) -> str:
    """Squares cooling from ember (left) to stone (right) in discrete steps: quantization as heat."""
    steps = len(themes.THEMES["ember"]["heat"])
    cells = []
    for r in range(rows):
        for c in range(cols):
            jitter = ((r * 7 + c * 13) % 5 - 2) * 0.35  # a fixed pattern, so the picture never changes
            step = min(steps - 1, max(0, int((c / (cols - 1)) * steps + jitter)))
            size = 13 if step < 3 else 12
            cells.append(
                f'<rect x="{c * 16 + (14 - size) / 2:.1f}" y="{r * 16 + (14 - size) / 2:.1f}" '
                f'width="{size}" height="{size}" rx="1.5" fill="var(--heat-{step})"/>'
            )
    return (
        f'<svg class="hero-pixels" viewBox="0 0 {cols * 16} {rows * 16}" aria-hidden="true" '
        f'focusable="false">{"".join(cells)}</svg>'
    )


def meter_keys(site, label: dict, dom: tuple) -> str:
    """The tactile condition keys (radio buttons), in one row; the readout names each condition's suite.
    Each key carries the texts and geometry the script shows, all written here from the label."""
    lab = builds(label)[1]
    names = condition_names(label)
    keys = []
    for cid in METER_CONDITIONS:
        cost = measurement(label, "shrinking_cost", lab, cid)
        kind, icon, text = M.verdict(label, cost, cid)
        data = {
            "cost-text": f(cost["value"], "pts"),
            "interval-text": interval(cost, "pts"),
            "verdict": text,
            "verdict-kind": kind,
            "icon": icon,
            "label": names[cid],
            "angle": f"{M.angle(cost['value'], dom):.3f}",
            "band": M.band(cost, dom),
        }
        attrs = " ".join(f'data-{k}="{esc(v)}"' for k, v in data.items())
        keys.append(
            f'<label class="key"><input type="radio" name="cond" value="{esc(cid)}" {attrs}'
            f"{' checked' if cid == 'clean' else ''}><span>{esc(short_condition(names[cid]))}</span></label>"
        )
    return f'<div class="key-group">{"".join(keys)}</div>'


def meter_table(label: dict) -> str:
    """The same four readings as a plain table, for readers without JavaScript."""
    lab = builds(label)[1]
    names = condition_names(label)
    rows = []
    for cid in METER_CONDITIONS:
        cost = measurement(label, "shrinking_cost", lab, cid)
        kind, icon, text = M.verdict(label, cost, cid)
        rows.append(
            f'<tr><th scope="row">{esc(names[cid])}</th><td class="mono">{f(cost["value"], "pts")} '
            f'<span class="ci">({interval(cost, "pts")})</span></td>'
            f'<td class="v-{kind}"><span aria-hidden="true">{icon}</span> {text}</td></tr>'
        )
    return (
        '<div class="sheet no-js-only"><table class="data compact"><thead><tr><th scope="col">Condition</th>'
        '<th scope="col">Shrinking cost (points)</th><th scope="col">Verdict</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def finding(site) -> str:
    """The two-sentence finding under the hero figure (claims register: site-finding), generated from the
    hero figure's values (H21, from the 4.1 verdicts) and the featured label."""
    v = site.figure(HERO[0])["values"]
    label = site.labs[FEATURED]
    ref, lab = builds(label)
    dark = "brokkr/darkness/5"
    models = [k.split(".")[0] for k in v if k.endswith(".h21_holds") and v[k]["value"] is True]
    held = " and ".join(
        v[f"{m}.name"]["value"] for m in sorted(models, key=lambda m: v[f"{m}.name"]["value"])
    )
    clean = measurement(label, "shrinking_cost", lab, "clean")
    darkness = measurement(label, "shrinking_cost", lab, dark)
    top1 = measurement(label, "top1", lab, "clean")
    return (
        f"In our pre-registered test under {esc(v['dark_label']['value'])}, "
        f"{f(v['h21_holding']['value'], 'int')} of {f(v['h21_judged']['value'], 'int')} shrunk models lost "
        f"much more to shrinking than on clean images ({esc(held)}); we had predicted at least "
        f"{f(v['h21_needed']['value'], 'int')}, so the prediction (H21) is a "
        f"{esc(v['h21_verdict']['value'])}. For example, {esc(label['model']['display_name'])}'s shrunk "
        f"build scores {f(top1['value'], 'pct')} on clean images, a shrinking cost of "
        f"{f(clean['value'], 'pts')} points against its full-precision build, but "
        f"{f(darkness['value'], 'pts')} points under {esc(condition_names(label)[dark])}."
    )


def figure_html(site, path: str, names: tuple) -> str:
    """A committed figure in the visitor's theme mode: wide layout on screens at least 768 px wide, tall
    layout on phones; light and dark files swap with the theme (web/themes.py: --fig-light, --fig-dark)."""
    wide, tall = site.figure(names[0]), site.figure(names[1])
    out = []
    for mode in ("light", "dark"):
        out.append(
            f'<picture class="fig-{mode}"><source media="(min-width: 768px)" '
            f'srcset="{up(path)}{site.asset(wide["files"][mode])}" width="800" height="526">'
            f'<img src="{up(path)}{site.asset(tall["files"][mode])}" alt="{esc(tall["alt"])}" '
            f'width="600" height="1148"></picture>'
        )
    return "".join(out)


def landing(site) -> list:
    path = "index.html"
    label = site.labs[FEATURED]
    lab = builds(label)[1]
    dom = M.domain(label, METER_CONDITIONS)
    clean = measurement(label, "shrinking_cost", lab, "clean")
    kind, icon, text = M.verdict(label, clean, "clean")
    ci_level = f(label["generated"]["ci_level"], "pct0")
    n_test = the_test_split(label)["n_items"]
    names = condition_names(label)
    model_page = f"{label_dir(label)}/index.html"
    buttons = "".join(
        f'<a class="button" href="{up(path)}{site.page_types[k].path}">{esc(site.page_types[k].nav)}</a>'
        for k in ("catalog", "why-labels")
        if k in site.page_types
    )
    caption = site.figure(HERO[0])["caption"]
    body = f"""
<section class="hero">
  {pixel_hero()}
  <div class="hero-text">
    <p class="kicker mono">THE FORGE'S TEST BENCH</p>
    <h1>Shrink AI models for small hardware, and find out honestly what you lost.</h1>
    <p>Brokkr shrinks a model to INT8, then measures what it lost under fog, darkness, blur, noise and low
    contrast. Every number comes from a checked result file: one laptop CPU, simulated damage.</p>
  </div>
</section>
<section class="bench" id="bench" aria-labelledby="bench-h">
  <div class="bench-head"><h2 id="bench-h" class="kicker mono">ON THE BENCH · SHRINKING COST</h2>
    <p class="bench-what">{esc(label["model"]["display_name"])} · INT8 vs FP32</p></div>
  <div class="bench-grid">
    <div class="meter-face">{M.svg(label, clean, dom)}</div>
    <div class="readout" aria-live="polite">
      <p class="readout-label">Shrinking cost, points ·
        <span class="mono" id="r-cond">{esc(names["clean"])}</span></p>
      <p class="nixie mono" id="r-cost">{f(clean["value"], "pts")}</p>
      <p class="mono readout-sub">{ci_level} interval
        <span id="r-interval">{interval(clean, "pts")}</span></p>
      <p class="verdict v-{kind}" id="r-verdict"><span aria-hidden="true">{icon}</span>
        <span>{text}</span></p>
      <fieldset class="keys js-only"><legend>Test condition</legend>
        {meter_keys(site, label, dom)}</fieldset>
    </div>
  </div>
  {meter_table(label)}
  <p class="note">INT8 minus FP32 top-1 on the same {count(n_test)} test images; the band is its {ci_level}
  interval. The verdict uses the whole interval, not the needle.
  <a href="{up(path)}{model_page}#conditions">See all {f(label["summary"]["tested_conditions"], "int")}
  damaged conditions</a></p>
</section>
<section class="finding" aria-labelledby="finding-h">
  <h2 id="finding-h">What we found</h2>
  <p class="finding-text">{finding(site)}</p>
  <figure class="sheet figure">{figure_html(site, path, HERO)}
    <details class="caption"><summary>About this figure</summary><p>{esc(caption)}</p></details>
  </figure>
  <p class="cta">{buttons}</p>
</section>"""
    page = Page(
        "Brokkr · the forge's test bench",
        body,
        labels=[FEATURED],
        figures=list(HERO),
        allowed={t: "meter tick label (the dial's scale, every 10 points)" for t in M.tick_labels(dom)},
    )
    return [(path, page)]


# ---- catalog ----


def source_badge(label: dict) -> str:
    kind = label["source"]["kind"]
    if kind == "official":
        return f'<span class="source">{SOURCE_WORDS[kind]}</span>'
    return f'<span class="source unverified">{SOURCE_WORDS[kind]} · unverified</span>'


def catalog(site) -> list:
    path = "models/index.html"
    cards = []
    for label in site.labs.values():
        ref, lab = builds(label)
        tested = f(label["summary"]["tested_conditions"], "int")
        link = f"{label_dir(label).removeprefix('models/')}/index.html"
        title = f'<h2><a href="{link}">{esc(label["model"]["display_name"])}</a></h2>'
        build = f'<span class="build-id mono">{build_id(label, lab)}</span>'
        ids = f'<p class="ids">{build} {source_badge(label)}</p>'
        if lab["status"] == "failed":
            body = (
                f'<p class="card-alarm"><span aria-hidden="true">✕</span> INT8 build failed: do not use</p>'
                f'<p class="card-note">{esc(LR._do_not_use(lab))}</p>'
            )
            cls = " card-failed"
        else:
            cost = measurement(label, "shrinking_cost", lab, "clean")
            large = label["summary"]["large_shrinking_cost"]
            body = (
                f'<dl class="card-facts"><div><dt>Shrinking cost, clean</dt><dd class="mono">'
                f'{f(cost["value"], "pts")} <span class="unit">points</span>'
                f'<span class="ci">{interval(cost, "pts")}</span></dd></div>'
                f'<div><dt>Large shrinking cost</dt><dd class="mono{" v-harm" if large["count"] else ""}">'
                f'{f(large["count"], "int")} <span class="unit">of {tested} conditions</span></dd></div></dl>'
            )
            cls = ""
        open_link = (
            f'<p class="card-link"><a href="{link}">Open the label <span aria-hidden="true">→</span></a></p>'
        )
        cards.append(
            f'<article class="card{cls}">{title}<p class="card-build">{esc(lab["display_name"])}</p>'
            f"{ids}{body}{open_link}</article>"
        )
    ci_level = f(next(iter(site.labs.values()))["generated"]["ci_level"], "pct0")
    body = f"""
<header class="page-head">
  <h1>Catalog</h1>
  <p class="note">One label per model: the shrunk (INT8) build, measured against its full-precision (FP32)
  original. The shrinking cost is INT8 minus FP32 top-1 on the same clean test images, with its {ci_level}
  interval; a large shrinking cost means the whole interval is below the label's line.</p>
</header>
<div class="cards">{"".join(cards)}</div>"""
    return [(path, Page("Catalog · Brokkr", body, labels=list(site.labs)))]


# ---- model pages ----


def value_ci(m: dict, kind: str) -> str:
    """A measured value and, in brackets, its interval."""
    return f'<span class="mono">{f(m["value"], kind)}</span> <span class="ci">({interval(m, kind)})</span>'


def td(label: str, inner: str, cls: str = "") -> str:
    """A table cell; `label` names it on the phone card its row becomes."""
    return f'<td data-th="{esc(label)}"{f" class={cls}" if cls else ""}>{inner}</td>'


def table(head: list, rows: list) -> str:
    """A datasheet table: a table on wide screens, one card per row on phones (CSS only). `head` holds the
    column headings as HTML; each row starts with its own <th scope="row">."""
    ths = "".join(f'<th scope="col">{h}</th>' for h in head)
    return (
        '<div class="sheet scroll"><table class="data cards-on-phone">'
        f"<thead><tr>{ths}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def row(name: str, cells: list) -> str:
    return f'<tr><th scope="row">{esc(name)}</th>{"".join(cells)}</tr>'


def fact(name: str, value: str, note: str, more: str = "") -> str:
    return (
        f'<div class="fact"><p class="fact-name">{name}</p><p class="fact-value mono">{value}</p>'
        f'<p class="fact-note">{note}</p>{more}</div>'
    )


def key_facts(label: dict) -> str:
    """Counts only; the condition names sit in collapsed lists (section 12e)."""
    ref, lab = builds(label)
    summary, rule = label["summary"], label["envelope"]["rule"]
    failed = lab["status"] == "failed"
    of = f'<span class="unit">of {f(summary["tested_conditions"], "int")} conditions</span>'
    not_measured = '<span class="v-failed">not measured</span>'
    fp32_top1 = f"FP32: {f(measurement(label, 'top1', ref, 'clean')['value'], 'pct')}"
    large, harsh, cov = (summary[k] for k in ("large_shrinking_cost", "reference_harmful", "coverage_failed"))
    out = []
    if failed:
        out.append(fact("Clean top-1, INT8", '<span class="v-failed">build failed</span>', fp32_top1))
        out.append(fact("Shrinking cost, clean", not_measured, "INT8 build failed"))
    else:
        cost = measurement(label, "shrinking_cost", lab, "clean")
        level = f(label["generated"]["ci_level"], "pct0")
        out.append(
            fact("Clean top-1, INT8", f(measurement(label, "top1", lab, "clean")["value"], "pct"), fp32_top1)
        )
        out.append(
            fact(
                "Shrinking cost, clean",
                f'{f(cost["value"], "pts")} <span class="unit">points</span>',
                f"{level} interval {interval(cost, 'pts')}",
            )
        )
    out.append(
        fact(
            "File size, INT8",
            f'{f(lab["file"]["size_bytes"], "mb")} <span class="unit">MB</span>',
            f"FP32: {f(ref['file']['size_bytes'], 'mb')} MB",
        )
    )
    line = f(rule["large_shrinking_cost_below"], "pts0")
    which = "Which conditions"
    if failed:
        out.append(fact("Large shrinking cost", not_measured, "INT8 build failed"))
    else:
        out.append(
            fact(
                "Large shrinking cost",
                f"{f(large['count'], 'int')} {of}",
                f"whole interval below the {line} point line",
                collapsed(which, label, large["conditions"]),
            )
        )
    out.append(
        fact(
            "Harsh even at full size",
            f"{f(harsh['count'], 'int')} {of}",
            "FP32 is harmful there too",
            collapsed(which, label, harsh["conditions"]),
        )
    )
    if failed:
        out.append(fact("Uncertainty signal unreliable", not_measured, "INT8 build failed"))
    else:
        out.append(
            fact(
                "Uncertainty signal unreliable",
                f"{f(cov['count'], 'int')} {of}",
                f"coverage below the {f(rule['coverage_min'], 'pct0')} line",
                collapsed(which, label, cov["conditions"]),
            )
        )
    return f'<div class="facts">{"".join(out)}</div>'


def envelope_groups(label: dict) -> str:
    """The conditions grouped by the INT8 build's own state, harmful ones by cause (the label's summary)."""
    summary = label["summary"]
    terms = {
        "not harmful": "Not harmful in our tests",
        "borderline": "Borderline",
        "harmful": "Harmful",
        "INT8 build failed": "INT8 build failed",
    }
    groups = []
    for state, term in terms.items():
        lines = [x for x in summary["lines"] if x["state"] == state]
        if not lines and state == "INT8 build failed":
            continue
        count_html = ""
        if state == "harmful":  # one count per cause; the label stores no total
            inner = "".join(
                f'<p class="cause">{hover(x["cause"].capitalize(), LR.CAUSE_TITLE[x["cause"]])}: '
                f'<span class="mono">{f(x["count"], "int")}</span></p>'
                + collapsed("Which conditions", label, x["conditions"])
                for x in lines
            )
        else:
            inner = collapsed("Which conditions", label, [c for x in lines for c in x["conditions"]])
            if lines:
                count_html = f' <span class="mono count">{f(lines[0]["count"], "int")}</span>'
        groups.append(
            f'<div class="group g-{KIND[state]}"><h3><span aria-hidden="true">{ICON[state]}</span> '
            f"{hover(term, LR.STATE_TITLE[state])}{count_html}</h3>{inner or '<p class=muted>none</p>'}</div>"
        )
    tested = f(summary["tested_conditions"], "int")
    return (
        f'<p class="note">The {tested} tested conditions, grouped by the INT8 build\'s own state; harmful '
        "ones by cause. A condition is harmful when a whole interval is below its line (coverage or "
        "damage drop).</p>"
        f'<div class="groups">{"".join(groups)}</div>'
    )


def state_cell(label: dict, env: dict | None, cause: str | None) -> str:
    """The INT8 build's state in one condition, its cause and the lines it failed."""
    if env is None:
        return '<span class="muted">reference condition (not judged)</span>'
    rule = label["envelope"]["rule"]
    state = env["state"]
    out = f'<span class="v-{KIND[state]}"><span aria-hidden="true">{ICON[state]}</span> {esc(state)}</span>'
    if cause:
        out += f'<span class="sub">{hover(cause.capitalize(), cause)}</span>'
    if env["failed"]:
        words = {
            "damage drop": "accuracy dropped",
            "coverage": f"coverage below {f(rule['coverage_min'], 'pct0')}",
        }
        out += f'<span class="sub">{"; ".join(words[x] for x in env["failed"])}</span>'
    return out


def conditions_table(label: dict) -> str:
    """Every tested condition, clean first."""
    ref, lab = builds(label)
    failed = lab["status"] == "failed"
    level = f(label["generated"]["ci_level"], "pct0")
    cause_of = {c: x["cause"] for x in label["summary"]["lines"] if x["cause"] for c in x["conditions"]}
    head = [
        "Condition",
        "FP32 top-1",
        "INT8 top-1",
        f"Shrinking cost (points, {level} interval)",
        "INT8 damage drop (points)",
        "INT8 coverage (set size)",
        "INT8 state",
        "Suggested next step",
    ]
    rows = []
    for c in label["conditions"]:
        cid = c["condition_id"]
        cells = [
            td(
                "FP32 top-1",
                f'<span class="mono">{f(measurement(label, "top1", ref, cid)["value"], "pct")}</span>',
            )
        ]
        if failed:
            cells.append('<td colspan="4" class="muted">not measured: INT8 build failed</td>')
            cells.append(
                td(
                    "INT8 state",
                    '<span class="v-failed"><span aria-hidden="true">✕</span> INT8 build failed</span>',
                )
            )
            cells.append(td("Next step", '<span class="muted">—</span>'))
            rows.append(row(c["label"], cells))
            continue
        env = envelope_row(label, lab, cid)
        cost, drop = (
            measurement(label, "shrinking_cost", lab, cid),
            measurement(label, "damage_drop", lab, cid),
        )
        cov, size = measurement(label, "coverage", lab, cid), measurement(label, "mean_set_size", lab, cid)
        flags = "".join(
            f' <span class="flag-large">⚠ {esc(fl)}</span>'
            if fl == "large shrinking cost"
            else f' <span class="flag">{esc(fl)}</span>'
            for fl in (env["shrinking_cost_flags"] if env else [])
        )
        step = LR._next_step(env, envelope_row(label, ref, cid))
        cells += [
            td(
                "INT8 top-1",
                f'<span class="mono">{f(measurement(label, "top1", lab, cid)["value"], "pct")}</span>',
            ),
            td("Shrinking cost", value_ci(cost, "pts") + flags),
            td("INT8 damage drop", value_ci(drop, "pts") if drop else '<span class="muted">reference</span>'),
            td(
                "INT8 coverage",
                f'<span class="mono">{f(cov["value"], "pct")}</span>'
                f'<span class="sub">set size <span class="mono">{f(size["value"], "size")}</span></span>',
            ),
            td("INT8 state", state_cell(label, env, cause_of.get(cid))),
            td("Next step", esc(step) if step != "—" else '<span class="muted">—</span>'),
        ]
        rows.append(row(c["label"], cells))
    return table([esc(h) for h in head], rows) + f'<p class="note">{esc(LR.DAMAGE_NOTE)}</p>'


def speed_section(label: dict) -> str:
    """The label's speed rows and notes (label_render's wording): unstable rows and INT8 slower than FP32
    are said in words and marked; a slower INT8 also gets a callout above the table."""
    ref, lab = builds(label)
    rows, notes = LR._speed(label, ref, lab)
    head = LR.SPEED_HEAD
    trs = []
    for r in rows:
        cells = []
        for h, x in zip(head[1:], r[1:], strict=True):
            if x.startswith("unstable"):
                cells.append(td(h, esc(f"⚠ {x}"), "warn"))
            elif "(slower)" in x:
                cells.append(td(h, esc(x), "warn"))
            else:
                cells.append(td(h, esc(x), "muted" if x.startswith("not measured") else ""))
        trs.append(row(r[0], cells))
    slower = [n for n in notes if n.startswith(LR.SLOWER.split(" (")[0])]
    callout = f'<p class="callout"><span aria-hidden="true">⚠</span> {esc(slower[0])}</p>' if slower else ""
    rest = "".join(f"<li>{esc(n)}</li>" for n in notes if n not in slower)
    head_html = [hover(LR.HOVER[h], h) if h in LR.HOVER else esc(h) for h in head]
    return callout + table(head_html, trs) + f'<ul class="notes">{rest}</ul>'


def uncertainty_section(label: dict) -> str:
    """Coverage with its set size on clean images, how often it fails under damage, the rest in details."""
    ref, lab = builds(label)
    rule, summary = label["envelope"]["rule"], label["summary"]
    conformal = label["details"]["conformal"]
    rows, more = [], []
    for b in (ref, lab):
        name = b["precision"].upper()
        if b["status"] == "failed":
            rows.append(row(name, ['<td colspan="3" class="v-failed">INT8 build failed: not measured</td>']))
            continue
        cov, size = (
            measurement(label, "coverage", b, "clean"),
            measurement(label, "mean_set_size", b, "clean"),
        )
        rows.append(
            row(
                name,
                [
                    td("Coverage, clean", value_ci(cov, "pct")),
                    td("Set size", f'<span class="mono">{f(size["value"], "size")}</span>'),
                    td(
                        "Threshold",
                        f'<span class="mono">{f(conformal[b["build_id"]]["threshold"], "thr")}</span>',
                    ),
                ],
            )
        )
        top5 = measurement(label, "top5", b, "clean")
        more.append(
            row(
                name,
                [
                    td("ECE", value_ci(measurement(label, "ece", b, "clean"), "ece")),
                    td("E-AURC", value_ci(measurement(label, "e_aurc", b, "clean"), "eaurc")),
                    td("Top-5", f'<span class="mono">{f(top5["value"], "pct")}</span>'),
                ],
            )
        )
    conf = conformal[ref["build_id"]]
    intro = (
        f"<p>Each build's {hover('Prediction set', 'prediction sets')} are tuned on "
        f"{count(conf['calibration_items'])} clean calibration images for a {f(conf['target'], 'pct0')} "
        "coverage target. On clean test images:</p>"
    )
    head = [
        "Build",
        hover("Coverage", "Coverage, clean"),
        hover("Set size", "Set size"),
        hover("Conformal threshold", "Threshold"),
    ]
    if lab["status"] == "failed":
        damage = "<p>Under damage: not measured for INT8 (the build failed).</p>"
    else:
        cov = summary["coverage_failed"]
        damage = (
            f"<p>Under damage, the INT8 build's {hover('Uncertainty signal', 'uncertainty signal')} is "
            f"unreliable (the whole coverage interval below the {f(rule['coverage_min'], 'pct0')} line) in "
            f'<strong class="mono">{f(cov["count"], "int")}</strong> of '
            f"{f(summary['tested_conditions'], 'int')} conditions. The sets were calibrated on clean images; "
            "they can be re-calibrated on your own images.</p>"
            + collapsed("Which conditions", label, cov["conditions"])
        )
    details = ""
    if more:
        more_head = [
            "Build",
            f"{hover('ECE', 'ECE')} (lower is better)",
            f"{hover('E-AURC', 'E-AURC')} (lower is better)",
            hover("Top-5", "Top-5"),
        ]
        details = (
            '<details class="more"><summary>More measures on clean images (ECE, E-AURC, top-5)</summary>'
            f"{table(more_head, more)}</details>"
        )
    return intro + table(head, rows) + damage + details


def provenance_section(label: dict) -> str:
    """Licences, how the label was made, the two builds, the limits and every source file."""
    ref, lab = builds(label)
    gen, rule = label["generated"], label["envelope"]["rule"]
    test, hw = the_test_split(label), label["hardware"][0]
    sanity = label["checks"]["fp32_sanity"]
    facts = [
        (
            "Made by",
            f"{esc(label['source']['how_made'])}; {esc(gen['by'])} at commit "
            f"<span class=mono>{gen['commit'][:12]}</span>",
        ),
        ("Label schema version", f"<span class=mono>{f(label['schema_version'], 'int')}</span>"),
        (
            "Test images",
            f"{esc(test['name'])}, {esc(test['split'])} split, {count(test['n_items'])} images; "
            f"licence: {esc(test['licence'])}",
        ),
        ("Machine", f"{esc(hw['cpu_model'])}, {esc(hw['os'])} ({esc(hw['kind'])})"),
        ("Envelope rule", f"{esc(rule['name'])} ({esc(rule['fixed_in'])})"),
        (
            "FP32 sanity check",
            f"clean top-1 {f(sanity['measured'], 'pct')} vs published "
            f"{f(sanity['published'], 'pct')} ({esc(sanity['published_source'])}): "
            f"{'PASS' if sanity['pass'] else 'FAIL'}",
        ),
    ]
    if lab["recipe"].get("skip_symbolic_shape"):
        facts.append(
            (
                "Non-default build setting",
                "skip_symbolic_shape (ONNX Runtime's symbolic shape "
                "inference was skipped in the preparation step)",
            )
        )
    build_rows = [
        row(
            b["display_name"],
            [
                td("Role", esc(b["role"])),
                td("File size", f'<span class="mono">{f(b["file"]["size_bytes"], "mb")} MB</span>'),
                td("SHA-256", f'<span class="mono">{b["file"]["sha256"][:12]}…</span>'),
                td("Status", esc(b["status"]), "v-failed" if b["status"] == "failed" else ""),
            ],
        )
        for b in (ref, lab)
    ]
    sources = "".join(
        f'<li><span class="mono">{esc(s["file"])}</span> '
        f'<span class="ci">SHA-256 {s["sha256"][:12]}…</span></li>'
        for s in label["sources"]
    )

    def bullets(lines):
        return '<ul class="notes">' + "".join(f"<li>{esc(x)}</li>" for x in lines) + "</ul>"

    return (
        f"<h3>Licences</h3>{bullets(LR._licences(label))}"
        '<h3>How it was made</h3><dl class="kv">'
        + "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in facts)
        + "</dl>"
        + table(["Build", "Role", "File size", "SHA-256", "Status"], build_rows)
        + f"<h3>Limits</h3>{bullets(label['limits'])}"
        + '<details class="more"><summary>Source files and their fingerprints</summary>'
        f'<ul class="notes">{sources}</ul></details>'
    )


def raw_section(label: dict) -> str:
    return (
        '<p>The label as data: <a href="label.json">label.json</a> (the only source of this page), and as a '
        f'single page: <a href="label.html">label.html</a>. '
        f"Licensed {esc(label['licences']['label_data'])}.</p>"
    )


def alarm(lab: dict) -> str:
    """The broken-build warning: first thing under the title, before any number."""
    if lab["status"] != "failed":
        return ""
    return (
        '<section class="alarm" aria-labelledby="alarm-h"><p class="alarm-icon" aria-hidden="true">✕</p>'
        '<div><h2 id="alarm-h">Do not use this INT8 build</h2>'
        f"<p>{esc(LR._do_not_use(lab))}</p>"
        "<p>The FP32 original was measured as usual; its results are below.</p></div></section>"
    )


# The model page's sections, in order (section 12e); the in-page menu is made from the same list.
SECTIONS = [
    ("facts", "Key facts", key_facts),
    ("envelope", "Where it holds up", envelope_groups),
    ("conditions", "Conditions", conditions_table),
    ("speed", "Speed", speed_section),
    ("uncertainty", "Uncertainty", uncertainty_section),
    ("provenance", "Licences and provenance", provenance_section),
    ("raw", "Raw label", raw_section),
]


def model_page(site, key: str) -> tuple:
    label = site.labs[key]
    ref, lab = builds(label)
    path = f"{label_dir(label)}/index.html"
    name = esc(label["model"]["display_name"])
    menu = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t, _ in SECTIONS)
    sections = "".join(
        f'<section id="{a}" aria-labelledby="{a}-h"><h2 id="{a}-h">{t}</h2>{make(label)}</section>'
        for a, t, make in SECTIONS
    )
    build = (
        f'<span class="build-id mono" title="Brokkr build ID, generated from the page address">'
        f"{build_id(label, lab)}</span>"
    )
    body = f"""
<article class="sheet-page">
  <p class="crumbs"><a href="{up(path)}models/index.html">Catalog</a> / {name}</p>
  <header class="model-head">
    <h1>{name}</h1>
    <p class="model-sub">{esc(lab["display_name"])}, compared with {esc(ref["display_name"])}</p>
    <p class="ids">{build} <span class="mono addr">{esc(address(label))}</span> {source_badge(label)}</p>
  </header>
  {alarm(lab)}
  <nav class="page-menu" aria-label="On this page"><ul>{menu}</ul></nav>
  {sections}
</article>"""
    title = f"{label['model']['display_name']} · {lab['display_name']} · Brokkr"
    return path, Page(title, body, labels=[key])


def model_pages(site) -> list:
    return [model_page(site, key) for key in site.labs]


PAGE_TYPES = [
    PageType("landing", None, "index.html", landing),
    PageType("catalog", "Catalog", "models/index.html", catalog),
    PageType("model", None, "models/index.html", model_pages),  # reached from the catalog, not the nav
]
