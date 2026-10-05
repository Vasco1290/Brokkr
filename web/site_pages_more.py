"""The website's slice B pages: Compare, Why labels?, Methods and Roadmap (docs/website_v0_plan.md, sections
1, 1a, 9, 11, 12 and 14). Same rules as web/site_pages.py: committed sources only, every number formatted
from a label, a figure entry or the README's checked findings, and each page names its sources so
web/site_checks.py can check every number it shows.
"""

import site_sources as S
from site_labels import builds, label_dir, measurement, the_test_split
from site_pages import (
    FEATURED,
    HERO,
    Page,
    count,
    esc,
    f,
    figure_block,
    hover,
    quote,
    table,
    up,
    value_ci,
)

from brokkr_edge import label_render as LR

# The ImageNet-C sentence approved with the "Why labels?" text (docs/website_v0_plan.md, section 11).
IMAGENET_C = (
    "The ImageNet-C conditions were generated with the imagecorruptions package v1.1.2 (Michaelis et al., "
    "2019, arXiv:1907.07484; an extension of the ImageNet-C code of Hendrycks & Dietterich, 2019, "
    "arXiv:1903.12261), with a one-line fix so that fog runs on NumPy 2, tested pixel-identical to the "
    "unmodified package's fog. They are not directly comparable to the released ImageNet-C files."
)
IMAGENET_C_WORDS = {
    "v1.1.2": "the imagecorruptions package's version",
    "Michaelis et al., 2019, arXiv:1907.07484": "citation",
    "Hendrycks & Dietterich, 2019, arXiv:1903.12261": "citation",
    "NumPy 2": "a library version",
}


# ---- Compare ----


def sort_td(label: str, inner: str, key) -> str:
    """A table cell with the value it sorts by (the script reads data-sort; it computes nothing)."""
    return f'<td data-th="{esc(label)}" data-sort="{"" if key is None else key}">{inner}</td>'


def compare_row(path: str, label: dict) -> str:
    ref, lab = builds(label)
    summary = label["summary"]
    failed = lab["status"] == "failed"
    name = label["model"]["display_name"]
    cells = [
        f'<th scope="row" data-sort="{esc(name.lower())}"><a href="{up(path)}{label_dir(label)}/index.html">'
        f"{esc(name)}</a></th>"
    ]
    fp32 = measurement(label, "top1", ref, "clean")["value"]
    cells.append(sort_td("FP32 top-1", f'<span class="mono">{f(fp32, "pct")}</span>', fp32))
    nothing = '<span class="muted">—</span>'  # beside "build failed" in the same row
    if failed:
        failed_cell = '<span class="v-failed"><span aria-hidden="true">✕</span> build failed</span>'
        cells += [sort_td("INT8 top-1", failed_cell, None), sort_td("Shrinking cost", nothing, None)]
        cells.append(sort_td("Large shrinking cost", nothing, None))
    else:
        top1 = measurement(label, "top1", lab, "clean")["value"]
        cost = measurement(label, "shrinking_cost", lab, "clean")
        large = summary["large_shrinking_cost"]["count"]
        cells.append(sort_td("INT8 top-1", f'<span class="mono">{f(top1, "pct")}</span>', top1))
        cells.append(sort_td("Shrinking cost", value_ci(cost, "pts"), cost["value"]))
        cells.append(
            sort_td(
                "Large shrinking cost",
                f'<span class="mono{" v-harm" if large else ""}">{f(large, "int")}</span>',
                large,
            )
        )
    harsh = summary["reference_harmful"]["count"]
    cells.append(sort_td("FP32 harmful", f'<span class="mono">{f(harsh, "int")}</span>', harsh))
    if failed:
        cells.append(sort_td("Uncertainty unreliable", nothing, None))
    else:
        cov = summary["coverage_failed"]["count"]
        cells.append(sort_td("Uncertainty unreliable", f'<span class="mono">{f(cov, "int")}</span>', cov))
    size = lab["file"]["size_bytes"]
    cells.append(
        sort_td(
            "INT8 file size",
            f'<span class="mono">{f(size, "mb")} MB</span>'
            f'<span class="sub">FP32 {f(ref["file"]["size_bytes"], "mb")} MB</span>',
            size,
        )
    )
    rows = [s for s in label["speed"] if s["build_id"] == lab["build_id"] and s["status"] == "measured"]
    if not rows:
        cells += [sort_td("INT8 p50", nothing, None), sort_td("Time vs FP32", nothing, None)]
        return f"<tr>{''.join(cells)}</tr>"
    s = max(rows, key=lambda r: r["settings"]["threads"])
    unstable = '<span class="sub warn">⚠ unstable</span>' if s["unstable"] else ""
    cells.append(
        sort_td("INT8 p50", f'<span class="mono">{f(s["p50_ms"], "ms")} ms</span>{unstable}', s["p50_ms"])
    )
    ratio = s["time_vs_reference"]
    slower = '<span class="sub warn">(slower)</span>' if ratio["slower"] else ""
    cells.append(
        sort_td(
            "Time vs FP32",
            f'<span class="mono">{f(ratio["ratio_p50"], "ratio")}×</span>{slower}',
            ratio["ratio_p50"],
        )
    )
    return f"<tr>{''.join(cells)}</tr>"


def compare(site) -> list:
    path = "compare/index.html"
    first = site.labs[FEATURED]
    threads = max(
        s["settings"]["threads"]
        for lab in site.labs.values()
        for s in lab["speed"]
        if s["status"] == "measured"
    )
    t = f(threads, "int")
    n = f(first["summary"]["tested_conditions"], "int")
    head = [
        "Model",
        hover("Top-1", "FP32 top-1, clean"),
        hover("Top-1", "INT8 top-1, clean"),
        hover("Shrinking cost", "Shrinking cost, clean (points)"),
        hover("Large shrinking cost", f"Large shrinking cost (of {n})"),
        hover("Harsh conditions", f"FP32 harmful (of {n})"),
        hover("Uncertainty signal unreliable", f"Uncertainty unreliable (of {n})"),
        "INT8 file size",
        hover("p50, p95, p99", f"INT8 p50, {t} threads"),
        hover("Time vs FP32", f"INT8 time vs FP32, {t} threads"),
    ]
    rows = [compare_row(path, label) for label in site.labs.values()]
    level = f(first["generated"]["ci_level"], "pct0")
    hw = first["hardware"][0]
    body = f"""
<header class="page-head">
  <h1>Compare</h1>
  <p class="note">Every label side by side. Shrinking cost is INT8 minus FP32 top-1 on the same clean test
  images, with its {level} interval; the three counts are out of the tested damaged conditions. Speed is
  laptop latency ({esc(hw["cpu_model"])}, {esc(hw["os"])}), never the speed of an edge device. Select a
  column heading to sort; without JavaScript the table stays in name order.</p>
</header>
<p class="scroll-hint" aria-hidden="true">scroll →</p>
<div class="sheet scroll compare-wrap">
  <table class="data sortable compare-table"><thead><tr>
  {"".join(f'<th scope="col">{h}</th>' for h in head)}</tr>
  </thead><tbody>{"".join(rows)}</tbody></table>
</div>
<section aria-labelledby="grid-h"><h2 id="grid-h">Every model in every condition</h2>
  <p class="note">Exploratory, not pre-registered. On a phone, scroll the figure sideways.</p>
  {figure_block(site, path, "grid-shrinking-cost", scroll=True)}</section>
<section aria-labelledby="speed-h"><h2 id="speed-h">Time on this laptop CPU</h2>
  <p class="note">INT8's typical time as a multiple of FP32's, per thread count: a relative comparison on
  one machine only.</p>
  {figure_block(site, path, "speed-ratio")}</section>"""
    page = Page(
        "Compare · Brokkr", body, labels=list(site.labs), figures=["grid-shrinking-cost", "speed-ratio"]
    )
    return [(path, page)]


# ---- Why labels? ----


def citation(c: tuple) -> str:
    key, authors, year, title, where, link = c
    return (
        f'<li id="ref-{key}"><a href="{link}">{esc(title)}</a>. {esc(authors)}. {esc(year)} '
        f'<span class="muted">({esc(where)})</span>.</li>'
    )


def why_labels(site) -> list:
    path = "why-labels/index.html"
    claim = site.claims["fig-uncertainty"]["claim"]
    refs = S.related_work()  # docs/related_work.md
    body = f"""
<article class="prose-page">
<header class="page-head"><h1>Why labels?</h1></header>
<p class="lead" data-claim="site-finding"><strong>A normal accuracy check is not enough.</strong>
{quote(site, "headline")}</p>
<p data-claim="why-example">{quote(site, "example")}</p>
{figure_block(site, path, *HERO)}
<section aria-labelledby="prereg-h"><h2 id="prereg-h">What we predicted, and what happened</h2>
  <p data-claim="why-prereg">{quote(site, "prereg")}</p></section>
<section aria-labelledby="explore-h"><h2 id="explore-h">Wider checks</h2>
  <p data-claim="why-explore">{quote(site, "explore")}</p>
  {figure_block(site, path, "grid-shrinking-cost", scroll=True)}
  <p data-claim="why-noise-blur">{quote(site, "noise_blur")}</p></section>
<section aria-labelledby="unc-h"><h2 id="unc-h">The "I'm not sure" signal</h2>
  <p data-claim="fig-uncertainty">{esc(claim)}</p>
  {figure_block(site, path, "mnv3l-uncertainty")}</section>
<section aria-labelledby="label-h"><h2 id="label-h">What a label does about it</h2>
  <p data-claim="why-label-design">That is why every Brokkr label shows each tested condition separately,
  says whether the full-precision model also fails there, and marks what was not tested. See the
  <a href="{up(path)}models/index.html">catalog</a>, and how each number is measured on the
  <a href="{up(path)}methods/index.html">methods page</a>.</p>
  <p data-claim="imagenet-c-method">{esc(IMAGENET_C)}</p></section>
<section aria-labelledby="refs-h"><h2 id="refs-h">Related work</h2>
  <p class="note">Cited on this page; each checked on its own page (title, authors, year) on
  3 October 2026.</p>
  <ul class="refs">{"".join(citation(c) for c in refs["cited"])}</ul>
  <p class="note">Also read, not cited above:</p>
  <ul class="refs">{"".join(citation(c) for c in refs["also read"])}</ul></section>
</article>"""
    words = dict(IMAGENET_C_WORDS)
    for c in refs["cited"] + refs["also read"]:
        words.update({c[2]: "citation year", c[4]: "citation (arXiv number, venue)", c[5]: "citation link"})
    words["3 October 2026"] = "the date the citations were checked"
    page = Page(
        "Why labels? · Brokkr",
        body,
        labels=[],
        figures=[*HERO, "grid-shrinking-cost", "mnv3l-uncertainty"],
        findings=True,
        words=words,
    )
    return [(path, page)]


# ---- Methods ----

SPLIT_USE = {
    "test": "every accuracy and coverage number",
    "conformal_calibration": "setting the prediction-set threshold (clean images only)",
}
HONESTY = [
    "Every number on a label and on this site comes from a checked result file; the site is not built if a "
    "page shows a number that is not in a label, a checked figure or the checked findings.",
    'Nothing is estimated: what was not measured says "not measured", what was not run says "not tested".',
    'Laptop latency is always called laptop latency. "Raspberry Pi 5" appears only for results measured on '
    'one; cloud ARM results are labelled "cloud ARM".',
    "Predictions are written down and committed before measuring; outcomes are added, never edited; analyses "
    "made after the verdicts are labelled exploratory.",
    "No setting is tuned on the test images.",
    "Weakness the full-precision model already has is shown apart from what shrinking caused.",
    "Coverage is never shown without its average set size.",
    "User-submitted results are always marked as such and never mixed with Brokkr's own.",
    "Every model and dataset has its licence recorded; no licence, no label.",
]


def methods(site) -> list:
    path = "methods/index.html"
    label = site.labs[FEATURED]
    ref, lab = builds(label)
    rule = label["envelope"]["rule"]
    test = the_test_split(label)
    calib = next(d for d in label["datasets"] if d["split"] == "conformal_calibration")
    int8 = lab["recipe"]["calibration"]
    data_rows = [
        f'<tr><th scope="row">{esc(d["name"])}, {esc(d["split"].replace("_", " "))} split</th>'
        f'<td data-th="Images" class="mono">{count(d["n_items"])}</td>'
        f'<td data-th="Used for">{esc(SPLIT_USE[d["split"]])}</td></tr>'
        for d in (test, calib)
    ]
    data_rows.append(
        f'<tr><th scope="row">{esc(test["name"])}, {esc(int8["dataset_id"].split(":")[-1])} split</th>'
        f'<td data-th="Images" class="mono">'
        f'{count(int8["n_items"])}</td><td data-th="Used for">setting INT8\'s value ranges</td></tr>'
    )
    suite = {None: "none (reference)", "brokkr": "Brokkr's own", "imagenet-c": "ImageNet-C"}
    cond_rows = [
        f'<tr><th scope="row">{esc(c["label"])}</th><td data-th="Damage set">{esc(suite[c["suite"]])}</td>'
        f'<td data-th="Damage">{esc(c["damage"].replace("_", " "))}</td>'
        f'<td data-th="Severity" class="mono">{f(c["severity"], "int") if c["severity"] else "—"}</td></tr>'
        for c in label["conditions"]
    ]
    tested = LR._sections(label)["tested"]
    rules = [x for x in tested if x.startswith(("Intervals:", "Envelope:", "Shrinking cost is marked"))]
    chance = next(x for x in label["limits"] if x.startswith("With "))
    cov, drop, large = (
        f(rule[k], "pts0" if k != "coverage_min" else "pct0")
        for k in ("coverage_min", "damage_drop_min", "large_shrinking_cost_below")
    )
    target = f(label["details"]["conformal"][ref["build_id"]]["target"], "pct0")
    history = (
        f"<p>The coverage line ({cov}) first appears in commit a957652 (24 September 2026), in a Stage 2 "
        "prediction written before any Stage 2 result; it became the harm line in commit 237effa "
        "(25 September 2026), with the Stage 2 results known. "
        f"The damage-drop line ({drop} points) was first committed in 621cc43 (26 September 2026), after "
        "Stage 3's final results. Both were committed before any result of Stage 4's task 4.1 existed. The "
        "labels call each of them a line whose value was written down before these results existed and "
        "adopted for the labels afterwards, unchanged.</p>"
        f"<p>The large-shrinking-cost line ({large} points) is the size that the pre-registered predictions "
        "H20 and H21 called large; the labels apply it to the plain shrinking cost, not to those "
        "predictions' "
        "extra gap, and adopted it after the results of task 4.1 existed. Each line is recorded in "
        "docs/hypotheses_stage4.md.</p>"
        f"<p>Why these values, without looking at the data: the prediction sets are tuned for {target} "
        f"coverage, so below {cov} they miss at least twice as many images as they are built to; a fall of "
        "ten points means at least one image in ten that was right on clean images is now wrong; and a cost "
        "of five points is large enough to change which model a user should pick.</p>"
    )
    _rows, notes = LR._speed(label, ref, lab)
    # The label's speed notes, without its own timing window (dates) and its slower-INT8 sentence.
    speed_notes = [
        n.split("(UTC). ", 1)[1] if n.startswith("Timed between") else n
        for n in notes
        if not n.startswith(LR.SLOWER.split(" (")[0])
    ]
    terms = ["p50, p95, p99", "Spread", "Unstable", "Time vs FP32", "Pinned", "Hardware threads"]
    body = f"""
<article class="prose-page">
<header class="page-head"><h1>Methods</h1>
<p class="lead">How every number on a Brokkr label is measured and judged: on one laptop CPU, with
simulated damage, on images kept apart from everything used for set-up.</p></header>
<section aria-labelledby="data-h"><h2 id="data-h">Images</h2>
  {table(["Images", "Count", "Used for"], data_rows)}
  <p class="note">The test images and the prediction-set calibration images share no image (recorded
  in every label). Test images: {esc(test["licence"])}</p></section>
<section aria-labelledby="cond-h"><h2 id="cond-h">Test conditions</h2>
  <p>Clean images and {f(label["summary"]["tested_conditions"], "int")} damaged conditions. Brokkr's own
  damage
  and ImageNet-C's are named separately everywhere; darkness is Brokkr's own (ImageNet-C has none).</p>
  {table(["Condition", "Damage set", "Damage", "Severity"], cond_rows)}
  <p>{esc(IMAGENET_C)}</p></section>
<section aria-labelledby="rules-h"><h2 id="rules-h">Thresholds and the envelope</h2>
  <ul class="notes">{"".join(f"<li>{esc(x)}</li>" for x in rules)}</ul>
  <p class="callout">{esc(chance)}</p>
  <h3>Where the lines come from</h3>{history}</section>
<section aria-labelledby="speed-h"><h2 id="speed-h">How speed was measured</h2>
  <ul class="notes">{"".join(f"<li>{esc(n)}</li>" for n in speed_notes)}</ul>
  <dl class="kv">{"".join(f"<div><dt>{esc(k)}</dt><dd>{esc(LR.TERMS[k])}</dd></div>" for k in terms)}
  </dl></section>
<section aria-labelledby="honest-h"><h2 id="honest-h">Honesty rules</h2>
  <ul class="notes">{"".join(f"<li>{esc(x)}</li>" for x in HONESTY)}</ul></section>
<section aria-labelledby="lim-h"><h2 id="lim-h">Limitations</h2>
  <ul class="notes">{"".join(f"<li>{esc(x)}</li>" for x in label["limits"])}</ul></section>
</article>"""
    words = dict(IMAGENET_C_WORDS)
    for w in ("a957652", "237effa", "621cc43", "24 September 2026", "25 September 2026", "26 September 2026"):
        words[w] = "commit or date in the threshold history (docs/hypotheses_stage4.md)"
    for w in ("Stage 2", "Stage 3", "Stage 4", "task 4.1", "docs/hypotheses_stage4.md", "Raspberry Pi 5"):
        words[w] = "a name, not a number"
    page = Page("Methods · Brokkr", body, labels=[FEATURED], words=words)
    return [(path, page)]


# ---- Roadmap ----


def roadmap(site) -> list:
    path = "roadmap/index.html"
    plan = S.roadmap()

    def block(key, title, note):
        items = "".join(f"<li>{esc(name)}</li>" for name in plan[key])
        head = f'<section class="phase phase-{key}"><h2>{title}</h2><p class="note">{note}</p>'
        return f"{head}<ul>{items}</ul></section>"

    body = f"""
<article class="prose-page">
<header class="page-head"><h1>Roadmap</h1>
<p class="lead">What exists, what is being built, and what comes after, in order. Generated from the items
ROADMAP.md marks public, without dates; each step is finished before the next starts.</p></header>
<div class="phases">
{block("built", "Built", "Marked done in ROADMAP.md.")}
{block("now", "Now", "In progress.")}
{block("next", "Next", "The next step.")}
{block("later", "Later", "The remaining steps, in order.")}
</div>
<section aria-labelledby="lv-h"><h2 id="lv-h">Later versions</h2>
  <p class="note">Planned, not scheduled.</p>
  <ul class="notes">{"".join(f"<li>{esc(x)}</li>" for x in plan["later_versions"])}</ul></section>
</article>"""
    words = {name: "a name from ROADMAP.md" for k in ("built", "now", "next", "later") for name in plan[k]}
    words.update({x: "an item from ROADMAP.md" for x in plan["later_versions"]})
    return [(path, Page("Roadmap · Brokkr", body, labels=[], words=words))]
