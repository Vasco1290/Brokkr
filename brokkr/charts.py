"""Small line charts as plain SVG, written directly (no plotting library needed).

Used by the results page. Colours come from CSS variables (--series-1, --grid, ...) defined by
the page, so charts follow its light/dark theme. Each series also has its own marker shape, so
identity never depends on colour alone, and every point has a hover tooltip (<title>).
The same numbers are always shown in a table next to the chart.
"""

import html

# Series order is fixed, so a precision keeps its colour and marker on every chart.
SERIES_STYLE = {"fp32": ("--series-1", "circle"), "fp16": ("--series-2", "square"),
                "int8": ("--series-3", "triangle")}

W, H = 220, 170  # one panel, in SVG units
LEFT, RIGHT, TOP, BOTTOM = 34, 8, 24, 30


def marker(shape: str, x: float, y: float, colour: str) -> str:
    style = f'fill="var({colour})" stroke="var(--surface)" stroke-width="2"'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" {style}/>'
    if shape == "square":
        # A little larger than the circle, so a square hidden behind a circle still shows its corners.
        return f'<rect x="{x - 5.5:.1f}" y="{y - 5.5:.1f}" width="11" height="11" rx="1" {style}/>'
    corners = f"M{x:.1f},{y - 5.5:.1f} L{x + 5:.1f},{y + 3.5:.1f} L{x - 5:.1f},{y + 3.5:.1f} Z"
    return f'<path d="{corners}" {style}/>'


def panel(title: str, series: dict, x_values: list, y_max: float = 100, reference: float | None = None,
          reference_label: str = "", unit: str = "%") -> str:
    """One chart panel: x = severity, y = 0..y_max, one line per series ({name: [y per x]})."""
    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM

    def px(i):  # x position of the i-th x value
        return LEFT + plot_w * i / (len(x_values) - 1)

    def py(v):
        return TOP + plot_h * (1 - v / y_max)

    parts = [f'<text x="{LEFT}" y="14" class="panel-title">{html.escape(title)}</text>']
    for tick in (0, y_max / 2, y_max):  # recessive hairline grid
        parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{py(tick):.1f}" y2="{py(tick):.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{LEFT - 5}" y="{py(tick) + 3.5:.1f}" class="tick" text-anchor="end">'
                     f'{tick:g}{unit}</text>')
    for i, x in enumerate(x_values):
        parts.append(f'<text x="{px(i):.1f}" y="{H - 14}" class="tick" text-anchor="middle">{x}</text>')
    parts.append(f'<text x="{LEFT + plot_w / 2:.1f}" y="{H - 2}" class="tick" '
                 f'text-anchor="middle">severity</text>')
    if reference is not None:
        parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{py(reference):.1f}" y2="{py(reference):.1f}" '
                     f'stroke="var(--ink-muted)" stroke-width="1"/>')
        # Label in the title row, where data can never cover it.
        parts.append(f'<text x="{W - RIGHT}" y="14" class="tick" text-anchor="end">'
                     f'grey line: {html.escape(reference_label)}</text>')

    # Draw in reverse order so FP32 ends up on top: FP16 often overlaps it almost exactly.
    drawing_order = list(SERIES_STYLE)[::-1]
    for name in drawing_order:  # lines first, then markers on top
        if name in series:
            points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(series[name]))
            colour = SERIES_STYLE[name][0]
            parts.append(f'<polyline points="{points}" fill="none" stroke="var({colour})" stroke-width="2" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')
    for name in drawing_order:
        if name not in series:
            continue
        colour, shape = SERIES_STYLE[name]
        for i, v in enumerate(series[name]):
            parts.append(marker(shape, px(i), py(v), colour))
            # A larger invisible circle so the tooltip is easy to hit.
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="11" fill="transparent">'
                         f'<title>{html.escape(title)}, severity {x_values[i]}, {name.upper()}: '
                         f'{v:.1f}{unit}</title></circle>')

    label = f"{title}: " + "; ".join(f"{n.upper()} " + ", ".join(f"{v:.1f}{unit}" for v in vals)
                                     for n, vals in series.items())
    return (f'<svg viewBox="0 0 {W} {H}" class="panel" role="img" aria-label="{html.escape(label)}">'
            + "".join(parts) + "</svg>")


def legend(names: list) -> str:
    """HTML legend: marker shape + colour + name, in the fixed series order."""
    items = []
    for name in SERIES_STYLE:
        if name in names:
            colour, shape = SERIES_STYLE[name]
            items.append(f'<span class="legend-item"><svg viewBox="0 0 16 16" width="16" height="16">'
                         f'{marker(shape, 8, 8, colour)}</svg>{name.upper()}</span>')
    return f'<div class="legend">{"".join(items)}</div>'
