"""Brokkr's figures as plain SVG, one light and one dark file each (no plotting library).

Every function here takes plain values and returns SVG text; nothing here reads a file.
scripts/47_figures.py reads the labels and records, calls these functions, and checks what they drew.

Conventions on every figure, so a reader learns them once:
- FP32 (full precision) is a grey circle, INT8 a blue square: shape and colour both say which build.
- Every data mark has data-v="<value id>" and a hover <title>. scripts/47 converts each mark's position
  back into a value and compares it with the value it claims to show.
- Colours are fixed hex values per theme, because GitHub's <picture> switches whole files. Text never
  uses a data colour. Colours were checked with the data-viz palette validator (4 October 2026).

`v` is always the figure's values: {value id: {"value", "fmt", optional "ci95"}}, the same dict that
scripts/47 writes to docs/figures/figures.json with each value's source.
"""

import html
import math

FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    # Light: white, like GitHub's light page.
    "light": {
        "surface": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "grid": "#d1d9e0",
        "fp32": "#6e7781",
        "int8": "#0072b2",
        "second": "#d55e00",
        "floor": "#e6eaef",
        "band": "#f6f8fa",
        "steps": ("#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"),
        "text_choices": ("#1f2328", "#ffffff"),
    },
    # Dark: a neutral grey with no hue (H, 4 October 2026), the same as the website's fixed datasheet panel,
    # so a figure looks the same under every site theme.
    "dark": {
        "surface": "#181818",
        "ink": "#f2f2f2",
        "muted": "#a3a3a3",
        "grid": "#404040",
        "fp32": "#8b949e",
        "int8": "#3987e5",
        "second": "#d95926",
        "floor": "#2e2e2e",
        "band": "#222222",
        "steps": ("#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"),
        "text_choices": ("#f2f2f2", "#181818"),
    },
}

# Points lost (as fractions of top-1) where the grid's colour steps change (H, 4 October 2026):
# INT8 equal or better · under 1 · 1 to 5 · 5 to 15 · 15 or more points lost.
STEP_EDGES = (0.01, 0.05, 0.15)

# Short column names for the grid (the key under the grid gives each condition's full name).
SHORT_DAMAGE = {
    "fog": "fog",
    "darkness": "dark",
    "defocus_blur": "blur",
    "noise": "noise",
    "contrast": "contr",
    "gaussian_noise": "noise",
}

# What each short column name stands for, for the grid's one-line key.
FULL_DAMAGE = {
    "fog": "fog",
    "darkness": "darkness",
    "defocus_blur": "defocus blur",
    "noise": "noise",
    "contrast": "contrast",
    "gaussian_noise": "gaussian noise",
}


def abbreviations(columns: list) -> str:
    """'dark = darkness · blur = defocus blur · ...' for every short name that is not the full name. A short
    name that stands for different damage in different suites names each, with its suite."""
    meanings = {}
    for col in columns:
        suite = "Brokkr" if col["suite"] == "brokkr" else "ImageNet-C"
        meanings.setdefault(SHORT_DAMAGE[col["damage"]], {}).setdefault(FULL_DAMAGE[col["damage"]], suite)
    parts = []
    for short, fulls in meanings.items():
        if list(fulls) == [short]:
            continue
        if len(fulls) == 1:
            parts.append(f"{short} = {next(iter(fulls))}")
        else:
            parts.append(f"{short} = " + ", ".join(f"{full} ({suite})" for full, suite in fulls.items()))
    return " · ".join(parts + ["s = severity"])


FORMATS = {
    "pts1": lambda x: f"{100 * x:+.1f}",  # shrinking cost, extra gap (points)
    "abs0": lambda x: f"{-100 * x:.0f}",  # a negative line written as "N points below"
    "pct1": lambda x: f"{100 * x:.1f}%",  # top-1, coverage
    "pct0": lambda x: f"{100 * x:.0f}%",  # coverage target and line, near-floor line
    "size": lambda x: f"{x:.2f}",  # average prediction-set size (classes)
    "ratio": lambda x: f"{x:.2f}",  # INT8 time as a multiple of FP32 time
    "pctv": lambda x: f"{x:.1f}%",  # a value already in percent (latency spread line)
    "int": lambda x: f"{x:,}",
    "text": str,
}


def show(v: dict, vid: str) -> str:
    return FORMATS[v[vid]["fmt"]](v[vid]["value"])


def show_ci(v: dict, vid: str) -> str:
    lo, hi = v[vid]["ci95"]
    f = FORMATS[v[vid]["fmt"]]
    return f"({f(lo)} to {f(hi)})"


def step_of(cost: float) -> int:
    """The grid's colour step (0-4) for a shrinking cost (INT8 minus FP32, a fraction; negative = lost)."""
    loss = -cost
    return 0 if loss <= 0 else 1 + sum(loss >= edge for edge in STEP_EDGES)


def step_names() -> list:
    e = [f"{100 * x:g}" for x in STEP_EDGES]
    return [
        "INT8 equal or better",
        f"under {e[0]}",
        f"{e[0]} to {e[1]}",
        f"{e[1]} to {e[2]}",
        f"{e[2]} or more points lost",
    ]


def luminance(colour: str) -> float:
    def channel(c):
        c = int(c, 16) / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(colour[i : i + 2]) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two #rrggbb colours."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def text_on(fill: str, theme: str) -> str:
    """Dark or light text, whichever reads better on `fill`."""
    return max(THEMES[theme]["text_choices"], key=lambda t: contrast(t, fill))


def scale(domain, rng):
    (d0, d1), (r0, r1) = domain, rng
    return lambda x: r0 + (x - d0) / (d1 - d0) * (r1 - r0)


def ticks(d0: float, d1: float, step: float) -> list:
    return [round(k * step, 6) for k in range(math.ceil(d0 / step - 1e-9), math.floor(d1 / step + 1e-9) + 1)]


def text_width(s: str, size: float) -> float:
    """A rough width for laying out text (system sans: about 0.55 em per character)."""
    return 0.55 * size * len(s)


class Canvas:
    """Collects SVG parts in one theme; svg() wraps them with a background, title and description."""

    def __init__(self, width: int, theme: str, font_size: int, min_display_px: int):
        self.w, self.theme, self.fs, self.min_display_px = width, theme, font_size, min_display_px
        self.c = THEMES[theme]
        self.parts = []

    def colour(self, name: str) -> str:
        return self.c.get(name, name)

    def text(
        self, x, y, s, *, size=None, anchor="start", bold=False, colour="ink", on=None, halo=False, attrs=""
    ):
        """halo: a surface-coloured outline behind the letters, so gridlines never cross the text."""
        extra = f' text-anchor="{anchor}"' if anchor != "start" else ""
        extra += ' font-weight="600"' if bold else ""
        extra += (
            f' stroke="{self.c["surface"]}" stroke-width="5" stroke-linejoin="round" paint-order="stroke"'
            if halo
            else ""
        )
        extra += f' data-on="{on}"' if on else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size or self.fs}" '
            f'fill="{self.colour(colour)}"{extra}{attrs}>{html.escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, colour, width=1.0, dash=None, attrs=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{self.colour(colour)}" stroke-width="{width}"{d} '
            f'stroke-linecap="round"{attrs}/>'
        )

    def interval(self, lo, hi, y, colour, vid, axis, tip):
        """A 95% interval: a line with short end caps (one narrower than its marker hides behind it)."""
        self.line(lo, y, hi, y, colour, 3, attrs=f' data-v="{vid}" data-part="ci" data-axis="{axis}"')
        self.parts[-1] = self.parts[-1][:-2] + f"><title>{html.escape(tip)}</title></line>"
        for x in (lo, hi):
            self.line(x, y - 7, x, y + 7, colour, 2)

    def mark(self, shape, x, y, colour, vid, axis, tip, *, r=8.0, hollow=False):
        """A data marker with a 2-unit surface ring; hollow = surface fill with a coloured outline."""
        fill = self.c["surface"] if hollow else self.colour(colour)
        stroke = self.colour(colour) if hollow else self.c["surface"]
        style = f'fill="{fill}" stroke="{stroke}" stroke-width="{3 if hollow else 2}"'
        data = f'data-v="{vid}" data-part="mark" data-axis="{axis}"' + (' data-hollow="1"' if hollow else "")
        tip = f"<title>{html.escape(tip)}</title>"
        if shape == "circle":
            self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" {style} {data}>{tip}</circle>')
        elif shape == "square":
            s = r * 1.1
            self.parts.append(
                f'<rect x="{x - s:.2f}" y="{y - s:.2f}" width="{2 * s:.2f}" height="{2 * s:.2f}" '
                f'rx="1.5" {style} {data}>{tip}</rect>'
            )
        else:  # triangle, apex up; its apex is at x
            s = r * 1.2
            pts = f"{x:.2f},{y - s:.2f} {x + s:.2f},{y + s * 0.8:.2f} {x - s:.2f},{y + s * 0.8:.2f}"
            self.parts.append(f'<polygon points="{pts}" {style} {data}>{tip}</polygon>')

    def axis_ticks(self, axis, sc, values, labels, y_text, y_top, y_bottom):
        for t, label in zip(values, labels, strict=True):
            x = sc(t)
            self.line(x, y_top, x, y_bottom, "grid", 1)
            self.text(
                x,
                y_text,
                label,
                anchor="middle",
                colour="muted",
                attrs=f' data-part="tick" data-axis="{axis}" data-tick="{t}"',
            )

    def svg(self, height, title, desc) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {height:.0f}" '
            f'width="{self.w}" height="{height:.0f}" role="img" aria-labelledby="title desc" '
            f'font-family="{FONT}" data-theme="{self.theme}" data-surface="{self.c["surface"]}" '
            f'data-min-display-px="{self.min_display_px}">'
        )
        return (
            head + f'<title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc>'
            f'<rect width="{self.w}" height="{height:.0f}" rx="8" fill="{self.c["surface"]}"/>'
            + "".join(self.parts)
            + "</svg>\n"
        )


def legend_item(cv: Canvas, x, y, shape, colour, text, *, hollow=False, dash=None) -> float:
    """One legend entry at (x, y = text baseline); returns the x where the next entry can start."""
    if shape == "line":
        cv.line(x, y - 7, x + 30, y - 7, colour, 2, dash=dash)
        x += 38
    else:
        fill = cv.c["surface"] if hollow else cv.colour(colour)
        stroke = cv.colour(colour)
        if shape == "circle":
            cv.parts.append(
                f'<circle cx="{x + 8:.2f}" cy="{y - 7:.2f}" r="8" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="2"/>'
            )
        elif shape == "square":
            cv.parts.append(
                f'<rect x="{x - 0.8:.2f}" y="{y - 15.8:.2f}" width="17.6" height="17.6" rx="1.5" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )
        else:
            cv.parts.append(
                f'<polygon points="{x + 8:.2f},{y - 16.6:.2f} {x + 17.6:.2f},{y - 0.3:.2f} '
                f'{x - 1.6:.2f},{y - 0.3:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )
        x += 26
    cv.text(x, y, text)
    return x + text_width(text, cv.fs) + 24


# ---- 1. Hero: shrinking cost on clean images and under darkness ----


def hero(rows: list, v: dict, theme: str) -> tuple:
    """rows: [{"name", "clean", "dark", "gap", "holds"} value ids], sorted as they are drawn.
    Constants in v: dark_label, large_cost_line, failed_name.
    Phones: GitHub shows the README's images 310 px wide on a 375 px phone (measured 4 October 2026), so the
    smallest text is 22 units of 600 (11 px there)."""
    W, FS = 600, 22
    cv = Canvas(W, theme, FS, 310)
    lows = [v[r[k]]["ci95"][0] for r in rows for k in ("clean", "dark")] + [v["large_cost_line"]["value"]]
    highs = [v[r[k]]["ci95"][1] for r in rows for k in ("clean", "dark")] + [0.0]
    d0, d1 = math.floor((min(lows) - 0.005) / 0.05) * 0.05, math.ceil((max(highs) + 0.005) / 0.05) * 0.05
    domain, rng = (round(d0, 6), round(d1, 6)), (40.0, 568.0)
    sc = scale(domain, rng)
    step = 0.1 if d1 - d0 > 0.25 else 0.05
    tick_values = ticks(d0, d1, step)
    tick_labels = [f"{100 * t:.0f}" for t in tick_values]
    dark = show(v, "dark_label")

    cv.text(16, 34, f"Shrinking cost: clean vs {dark}", size=22, bold=True)
    cv.text(16, 64, "INT8 minus FP32 top-1 (points); left = worse", colour="muted")
    x = legend_item(cv, 16, 102, "circle", "fp32", "clean images")
    legend_item(cv, x, 102, "square", "int8", dark)
    legend_item(
        cv,
        16,
        136,
        "line",
        "muted",
        f"label's large-cost line ({show(v, 'large_cost_line')} points)",
        dash="7 6",
    )
    top = 190
    bottom = top + 100 * len(rows)
    cv.axis_ticks("x", sc, tick_values, tick_labels, 176, top, bottom)
    line_x = sc(v["large_cost_line"]["value"])
    cv.line(
        line_x,
        top,
        line_x,
        bottom,
        "muted",
        2,
        dash="7 6",
        attrs=' data-v="large_cost_line" data-part="line" data-axis="x"',
    )

    for i, r in enumerate(rows):
        y0 = top + 100 * i
        name = show(v, r["name"])
        cv.text(16, y0 + 26, name, bold=True, halo=True)
        y = y0 + 54
        xc, xd = sc(v[r["clean"]]["value"]), sc(v[r["dark"]]["value"])
        cv.line(xc, y, xd, y, "muted", 2)
        for vid, colour, label in ((r["dark"], "int8", dark), (r["clean"], "fp32", "clean images")):
            lo, hi = (sc(c) for c in v[vid]["ci95"])
            cv.interval(
                lo, hi, y, colour, vid, "x", f"{name}, {label}: 95% interval {show_ci(v, vid)} points"
            )
        cv.mark("square", xd, y, "int8", r["dark"], "x", f"{name}, {dark}: {show(v, r['dark'])} points")
        cv.mark(
            "circle",
            xc,
            y,
            "fp32",
            r["clean"],
            "x",
            f"{name}, clean images: {show(v, r['clean'])} points",
            r=6.5,
        )
        yes = "yes" if v[r["holds"]]["value"] else "no"
        cv.text(
            16,
            y0 + 88,
            f"Much worse in the dark? {yes}, {show(v, r['gap'])} {show_ci(v, r['gap'])}",
            colour="muted",
            halo=True,
        )
    cv.text(16, bottom + 36, f"Not shown: {show(v, 'failed_name')} (INT8 build failed).", colour="muted")
    title = f"Shrinking cost of each model, clean images vs {dark}"
    desc = "; ".join(
        f"{show(v, r['name'])}: clean {show(v, r['clean'])}, {dark} {show(v, r['dark'])} points" for r in rows
    )
    return cv.svg(bottom + 58, title, desc), {
        "x": {
            "domain": list(domain),
            "range": list(rng),
            "ticks": dict(zip(tick_labels, tick_values, strict=True)),
        }
    }


def hero_wide(rows: list, v: dict, theme: str) -> tuple:
    """The hero for screens at least 600 px wide (H, 4 October 2026): one line per model, H21 as a column.
    Same rows and values as hero(); the README shows this one on wide screens and hero() on phones."""
    W, FS, ROW = 800, 16, 38
    cv = Canvas(W, theme, FS, 600)
    lows = [v[r[k]]["ci95"][0] for r in rows for k in ("clean", "dark")] + [v["large_cost_line"]["value"]]
    highs = [v[r[k]]["ci95"][1] for r in rows for k in ("clean", "dark")] + [0.0]
    d0, d1 = math.floor((min(lows) - 0.005) / 0.05) * 0.05, math.ceil((max(highs) + 0.005) / 0.05) * 0.05
    domain, rng = (round(d0, 6), round(d1, 6)), (196.0, 536.0)
    sc = scale(domain, rng)
    tick_values = ticks(d0, d1, 0.1 if d1 - d0 > 0.25 else 0.05)
    tick_labels = [f"{100 * t:.0f}" for t in tick_values]
    dark = show(v, "dark_label")
    band = cv.c["band"]

    cv.text(16, 30, f"Shrinking cost: clean vs {dark}", size=18, bold=True)
    cv.text(16, 54, "INT8 minus FP32 top-1, in points; left = INT8 worse", colour="muted")
    x = legend_item(cv, 16, 88, "circle", "fp32", "clean images")
    x = legend_item(cv, x, 88, "square", "int8", dark)
    legend_item(
        cv,
        x,
        88,
        "line",
        "muted",
        f"label's large-cost line ({show(v, 'large_cost_line')} points)",
        dash="7 6",
    )
    top = 140
    bottom = top + ROW * len(rows)
    for i in range(0, len(rows), 2):  # light bands on every other row, so a row can be followed across
        cv.parts.append(
            f'<rect x="8" y="{top + ROW * i}" width="{W - 16}" height="{ROW}" rx="4" fill="{band}"/>'
        )
    cv.axis_ticks("x", sc, tick_values, tick_labels, 128, top, bottom)
    cv.text(566, 128, "Much worse in the dark?", bold=True)  # H21's large extra gap (H, 4 October 2026)
    line_x = sc(v["large_cost_line"]["value"])
    cv.line(
        line_x,
        top,
        line_x,
        bottom,
        "muted",
        2,
        dash="7 6",
        attrs=' data-v="large_cost_line" data-part="line" data-axis="x"',
    )

    for i, r in enumerate(rows):
        y = top + ROW * i + ROW / 2
        on = band if i % 2 == 0 else None
        name = show(v, r["name"])
        cv.text(16, y + 6, name, bold=True, on=on)
        xc, xd = sc(v[r["clean"]]["value"]), sc(v[r["dark"]]["value"])
        cv.line(xc, y, xd, y, "muted", 2)
        for vid, colour, label in ((r["dark"], "int8", dark), (r["clean"], "fp32", "clean images")):
            lo, hi = (sc(c) for c in v[vid]["ci95"])
            cv.interval(
                lo, hi, y, colour, vid, "x", f"{name}, {label}: 95% interval {show_ci(v, vid)} points"
            )
        cv.mark(
            "square", xd, y, "int8", r["dark"], "x", f"{name}, {dark}: {show(v, r['dark'])} points", r=6.5
        )
        cv.mark(
            "circle",
            xc,
            y,
            "fp32",
            r["clean"],
            "x",
            f"{name}, clean images: {show(v, r['clean'])} points",
            r=5.5,
        )
        yes = bool(v[r["holds"]]["value"])
        cv.text(
            566,
            y + 6,
            f"{'yes' if yes else 'no'}, {show(v, r['gap'])} {show_ci(v, r['gap'])}",
            bold=yes,
            colour="ink" if yes else "muted",
            on=on,
        )
    cv.text(16, bottom + 28, f"{show(v, 'failed_name')} is not shown: its INT8 build failed.", colour="muted")
    title = f"Shrinking cost of each model, clean images vs {dark}"
    desc = "; ".join(
        f"{show(v, r['name'])}: clean {show(v, r['clean'])}, {dark} {show(v, r['dark'])} points" for r in rows
    )
    return cv.svg(bottom + 44, title, desc), {
        "x": {
            "domain": list(domain),
            "range": list(rng),
            "ticks": dict(zip(tick_labels, tick_values, strict=True)),
        }
    }


# ---- 2. Grid: every model in every damaged condition ----


def grid(rows: list, columns: list, v: dict, theme: str) -> tuple:
    """rows: [{"name", "count", "cells": [{"cost", "flags"} value ids, one per column]}], sorted as drawn.
    columns: [{"label", "suite", "damage", "severity"}] (label is a value id; the others are plain text).
    Constants in v: large_cost_line, near_floor_line."""
    W, FS, CELL_W, CELL_H, NAME_W = 1000, 16, 56, 46, 190
    cv = Canvas(W, theme, FS, 720)
    c = cv.c
    x0 = 16 + NAME_W
    xs, x = [], x0
    for j, col in enumerate(columns):
        if j and col["suite"] != columns[j - 1]["suite"]:
            x += 16
        xs.append(x)
        x += CELL_W
    count_x = x + 44

    cv.text(16, 30, "Shrinking cost in every damaged condition (exploratory)", size=18, bold=True)
    cv.text(
        16, 54, "INT8 minus FP32 top-1, in points; each cell is one model in one condition", colour="muted"
    )
    for suite in dict.fromkeys(col["suite"] for col in columns):
        js = [j for j, col in enumerate(columns) if col["suite"] == suite]
        mid = (xs[js[0]] + xs[js[-1]] + CELL_W) / 2
        cv.text(mid, 86, "Brokkr" if suite == "brokkr" else "ImageNet-C", anchor="middle", bold=True)
        cv.line(xs[js[0]] + 2, 94, xs[js[-1]] + CELL_W - 2, 94, "grid", 1)
    for j, col in enumerate(columns):
        cv.text(xs[j] + CELL_W / 2, 114, SHORT_DAMAGE[col["damage"]], anchor="middle")
        cv.text(xs[j] + CELL_W / 2, 132, f"s{col['severity']}", anchor="middle", colour="muted")
    cv.text(count_x, 114, "large", anchor="middle")
    cv.text(count_x, 132, "costs", anchor="middle", colour="muted")

    top = 144
    for i, r in enumerate(rows):
        y = top + CELL_H * i
        name = show(v, r["name"])
        cv.text(16, y + 29, name)
        for j, cell in enumerate(r["cells"]):
            cost, flags = v[cell["cost"]]["value"], v[cell["flags"]]["value"]
            floor = "not informative" in flags
            fill = c["floor"] if floor else c["steps"][step_of(cost)]
            ink = c["ink"] if floor else text_on(fill, theme)
            label = show(v, columns[j]["label"])
            tip = (
                f"{name}, {label}: {show(v, cell['cost'])} points {show_ci(v, cell['cost'])}"
                + ("; large shrinking cost" if "large shrinking cost" in flags else "")
                + ("; near floor, so the cost says little" if floor else "")
            )
            floor_attr = ' data-floor="1"' if floor else ""
            cv.parts.append(
                f'<rect x="{xs[j] + 1:.2f}" y="{y + 1:.2f}" width="{CELL_W - 2}" height="{CELL_H - 2}" '
                f'rx="2" fill="{fill}" data-v="{cell["cost"]}" data-part="cell" '
                f'data-flags="{cell["flags"]}"{floor_attr}><title>{html.escape(tip)}</title></rect>'
            )
            cv.text(
                xs[j] + CELL_W / 2,
                y + 33,
                "–" if floor else show(v, cell["cost"]),
                anchor="middle",
                colour=ink,
                on=fill,
            )
            if "large shrinking cost" in flags:
                cv.parts.append(
                    f'<circle cx="{xs[j] + CELL_W - 9:.2f}" cy="{y + 10:.2f}" r="5" fill="{ink}" '
                    f'data-dot="{cell["flags"]}" data-on="{fill}"/>'
                )
        cv.text(count_x, y + 29, show(v, r["count"]), anchor="middle")

    # Key (compact, H, 4 October 2026): colour steps; near floor; H's sentence; the abbreviations in one line.
    y = top + CELL_H * len(rows) + 40
    x = 16
    for k, text in enumerate(step_names()):
        fill = c["steps"][k]
        cv.parts.append(f'<rect x="{x:.2f}" y="{y - 15:.2f}" width="22" height="20" rx="2" fill="{fill}"/>')
        cv.text(x + 30, y, text)
        x += 30 + text_width(text, FS) + 18
    y += 32
    cv.parts.append(f'<rect x="16" y="{y - 15:.2f}" width="22" height="20" rx="2" fill="{c["floor"]}"/>')
    cv.text(27, y, "–", anchor="middle", on=c["floor"])
    cv.text(46, y, f"near floor: FP32 below {show(v, 'near_floor_line')} top-1, so the cost says little")
    y += 32
    cv.text(
        16,
        y,
        f"Colour: the measured cost. Dot: large cost (whole interval more than "
        f"{show(v, 'large_cost_line')} points below full size).",
    )
    y += 32
    cv.text(16, y, abbreviations(columns), colour="muted")
    height = y + 20
    title = "Shrinking cost of every usable INT8 build in every damaged condition (exploratory)"
    desc = "; ".join(
        f"{show(v, r['name'])}: "
        + ", ".join(
            f"{show(v, columns[j]['label'])} {show(v, cell['cost'])}" for j, cell in enumerate(r["cells"])
        )
        for r in rows
    )
    return cv.svg(height, title, desc), {}


# ---- 3. MobileNetV3-Large: coverage and set size ----


def uncertainty(rows: list, builds: list, v: dict, theme: str) -> tuple:
    """rows: [{"label", "cells": {build key: {"coverage", "size"}}}], builds: [{"key", "name", "shape",
    "colour"}] (name is a value id). Constants in v: target, coverage_line, model_name."""
    W, FS = 600, 20
    cv = Canvas(W, theme, FS, 343)
    cov_axis, size_axis = ((0.0, 1.0), (40.0, 320.0)), None
    cov = scale(*cov_axis)
    size_max = max(v[b]["ci95"][1] for r in rows for b in (r["cells"][k]["size"] for k in r["cells"]))
    size_step = 5 if size_max > 10 else 2 if size_max > 4 else 1
    size_top = math.ceil(size_max / size_step) * size_step
    size_axis = ((0.0, float(size_top)), (380.0, 568.0))
    sz = scale(*size_axis)

    cv.text(16, 34, f"{show(v, 'model_name')}: coverage and set size", size=22, bold=True)
    cv.text(16, 62, "per condition, both builds", colour="muted")
    for k, b in enumerate(builds):
        legend_item(cv, 16, 98 + 32 * k, b["shape"], b["colour"], show(v, b["name"]))
    y = 98 + 32 * len(builds)
    x = legend_item(cv, 16, y, "line", "muted", f"{show(v, 'target')} target")
    legend_item(cv, x, y, "line", "muted", f"{show(v, 'coverage_line')} line", dash="7 6")
    head = y + 40
    cv.text(cov_axis[1][0] - 8, head, "Coverage", bold=True)
    cv.text(size_axis[1][0] - 8, head, "Set size (classes)", bold=True)
    top = head + 44
    bottom = top + 84 * len(rows)
    cov_ticks = [0.0, 0.5, 1.0]
    cv.axis_ticks("coverage", cov, cov_ticks, [f"{100 * t:.0f}%" for t in cov_ticks], head + 30, top, bottom)
    size_ticks = [0.0, size_top / 2, float(size_top)]
    cv.axis_ticks("size", sz, size_ticks, [f"{t:g}" for t in size_ticks], head + 30, top, bottom)
    for vid, dash in (("target", None), ("coverage_line", "7 6")):
        x = cov(v[vid]["value"])
        cv.line(
            x,
            top,
            x,
            bottom,
            "muted",
            2,
            dash=dash,
            attrs=f' data-v="{vid}" data-part="line" data-axis="coverage"',
        )

    for i, r in enumerate(rows):
        y0 = top + 84 * i
        label = show(v, r["label"])
        cv.text(16, y0 + 22, label, bold=True, halo=True)
        for k, b in enumerate(builds):
            y = y0 + 42 + 22 * k
            cell = r["cells"][b["key"]]
            bname = show(v, b["name"])
            for vid, sc, axis, what in (
                (cell["coverage"], cov, "coverage", "coverage"),
                (cell["size"], sz, "size", "average set size"),
            ):
                lo, hi = (sc(c) for c in v[vid]["ci95"])
                cv.interval(
                    lo,
                    hi,
                    y,
                    b["colour"],
                    vid,
                    axis,
                    f"{label}, {bname}: {what} 95% interval {show_ci(v, vid)}",
                )
                cv.mark(
                    b["shape"],
                    sc(v[vid]["value"]),
                    y,
                    b["colour"],
                    vid,
                    axis,
                    f"{label}, {bname}: {what} {show(v, vid)}",
                    r=7.0,
                )
    title = f"{show(v, 'model_name')}: coverage and average set size per condition"
    desc = "; ".join(
        f"{show(v, r['label'])}: "
        + ", ".join(
            f"{show(v, b['name'])} coverage {show(v, r['cells'][b['key']]['coverage'])} with set size "
            f"{show(v, r['cells'][b['key']]['size'])}"
            for b in builds
        )
        for r in rows
    )
    axes = {
        "coverage": {
            "domain": list(cov_axis[0]),
            "range": list(cov_axis[1]),
            "ticks": {f"{100 * t:.0f}%": t for t in cov_ticks},
        },
        "size": {
            "domain": list(size_axis[0]),
            "range": list(size_axis[1]),
            "ticks": {f"{t:g}": t for t in size_ticks},
        },
    }
    return cv.svg(bottom + 20, title, desc), axes


# ---- 4. MobileNetV3-Large: clean accuracy of three builds ----


def clean_accuracy(rows: list, v: dict, theme: str) -> tuple:
    """rows: [{"name", "top1", "colour"}] value ids. Constant in v: model_name."""
    W, FS = 600, 20
    cv = Canvas(W, theme, FS, 343)
    domain, rng = (0.0, 1.0), (24.0, 568.0)
    sc = scale(domain, rng)
    cv.text(16, 34, f"{show(v, 'model_name')}: clean top-1", size=22, bold=True)
    cv.text(16, 62, "three builds, clean test images", colour="muted")
    top, bottom = 120, 120 + 100 * len(rows)
    tick_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = [f"{100 * t:g}%" for t in tick_values]
    cv.axis_ticks("x", sc, tick_values, tick_labels, 104, top, bottom)
    for i, r in enumerate(rows):
        y0 = top + 100 * i
        name = show(v, r["name"])
        cv.text(16, y0 + 24, name, bold=True)
        value = v[r["top1"]]["value"]
        cv.parts.append(
            f'<rect x="{rng[0]:.2f}" y="{y0 + 36:.2f}" width="{sc(value) - rng[0]:.2f}" height="24" '
            f'rx="4" fill="{cv.colour(r["colour"])}" data-v="{r["top1"]}" data-part="bar" '
            f'data-axis="x"><title>{html.escape(name)}: top-1 {show(v, r["top1"])}</title></rect>'
        )
        lo, hi = (sc(c) for c in v[r["top1"]]["ci95"])
        cv.interval(lo, hi, y0 + 48, "ink", r["top1"], "x", f"{name}: 95% interval {show_ci(v, r['top1'])}")
        cv.text(
            16, y0 + 86, f"top-1 {show(v, r['top1'])}, 95% interval {show_ci(v, r['top1'])}", colour="muted"
        )
    title = f"{show(v, 'model_name')}: clean top-1 of three builds"
    desc = "; ".join(f"{show(v, r['name'])}: {show(v, r['top1'])} {show_ci(v, r['top1'])}" for r in rows)
    return cv.svg(bottom + 16, title, desc), {
        "x": {
            "domain": list(domain),
            "range": list(rng),
            "ticks": dict(zip(tick_labels, tick_values, strict=True)),
        }
    }


# ---- 5. Speed: INT8 time as a multiple of FP32 time ----


def speed(rows: list, threads: list, v: dict, theme: str) -> tuple:
    """rows: [{"name", "failed": bool, "ratios": {thread key: value id},
    "unstable": {thread key: [value ids of the INT8 and FP32 rows' unstable flags]}}].
    threads: [{"key", "label", "shape", "colour"}]. Constants in v: unstable_line, failed_text.
    A result is marked unstable when either of its two speed rows is."""
    W, FS = 600, 20
    cv = Canvas(W, theme, FS, 343)
    ratios = [v[r["ratios"][t["key"]]]["value"] for r in rows if not r["failed"] for t in threads]
    d0 = min(0.5, math.floor((min(ratios) - 0.05) / 0.25) * 0.25)
    d1 = max(1.25, math.ceil((max(ratios) + 0.05) / 0.25) * 0.25)
    domain, rng = (float(d0), float(d1)), (40.0, 568.0)
    sc = scale(domain, rng)
    tick_values = ticks(d0, d1, 0.25 if d1 - d0 <= 1.5 else 0.5)
    tick_labels = [f"{t:g}×" for t in tick_values]

    cv.text(16, 34, "INT8 time ÷ FP32 time (p50)", size=22, bold=True)
    cv.text(16, 62, "laptop CPU, relative comparison only", bold=True)
    x = 16
    for t in threads:
        x = legend_item(cv, x, 98, t["shape"], t["colour"], t["label"])
    x = legend_item(
        cv, 16, 130, "circle", "muted", f"unstable: spread above {show(v, 'unstable_line')}", hollow=True
    )
    legend_item(cv, x, 130, "line", "muted", "same as FP32")
    top = 206
    bottom = top + 82 * len(rows)
    cv.axis_ticks("x", sc, tick_values, tick_labels, 164, top, bottom)
    cv.text(rng[0], 194, "← INT8 takes less time", colour="muted")
    cv.text(rng[1], 194, "more (slower) →", anchor="end", colour="muted")
    one = sc(1.0)
    cv.line(one, top, one, bottom, "muted", 2, attrs=' data-part="line" data-axis="x" data-at="1"')

    for i, r in enumerate(rows):
        y0 = top + 82 * i
        name = show(v, r["name"])
        cv.text(16, y0 + 22, name, bold=True, halo=True)
        if r["failed"]:
            cv.text(16, y0 + 52, show(v, "failed_text") + ": no INT8 timing", colour="muted", halo=True)
            continue
        shaky = {t["key"]: any(v[u]["value"] for u in r["unstable"][t["key"]]) for t in threads}
        unstable = [t["label"] for t in threads if shaky[t["key"]]]
        if unstable:
            cv.text(
                W - 16, y0 + 22, "unstable: " + ", ".join(unstable), anchor="end", colour="muted", halo=True
            )
        for k, t in enumerate(threads):
            vid = r["ratios"][t["key"]]
            cv.mark(
                t["shape"],
                sc(v[vid]["value"]),
                y0 + 42 + 22 * k,
                t["colour"],
                vid,
                "x",
                f"{name}, {t['label']}: INT8 takes {show(v, vid)}× the time of FP32"
                + (" (slower)" if v[vid]["value"] > 1 else "")
                + (" (unstable)" if shaky[t["key"]] else ""),
                r=7.0,
                hollow=shaky[t["key"]],
            )
    title = "INT8 time as a multiple of FP32 time (p50), laptop CPU, relative comparison only"
    desc = "; ".join(
        f"{show(v, r['name'])}: "
        + (
            show(v, "failed_text")
            if r["failed"]
            else ", ".join(f"{t['label']} {show(v, r['ratios'][t['key']])}×" for t in threads)
        )
        for r in rows
    )
    return cv.svg(bottom + 16, title, desc), {
        "x": {
            "domain": list(domain),
            "range": list(rng),
            "ticks": dict(zip(tick_labels, tick_values, strict=True)),
        }
    }
