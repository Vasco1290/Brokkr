"""The analog bench meter (docs/website_v0_plan.md, section 12a): one shrinking cost on a dial.

Needle at the measured value, a shaded band for its interval, two zones split at the label's large-cost
line (the one threshold the rules define). The verdict comes from the whole interval and the label's own
flag, never from the needle. The geometry functions are shared by the page and by its check
(web/site_checks.py), which recomputes every angle and band from a fresh read of the label.
"""

import math

from site_labels import builds, envelope_row, measurement

CX, CY, R = 160.0, 158.0, 122.0
SWEEP_FROM, SWEEP = 210.0, 240.0  # degrees: the dial runs from lower left, over the top, to lower right


def large_line(label: dict) -> float:
    return label["envelope"]["rule"]["large_shrinking_cost_below"]


def domain(label: dict, condition_ids: list) -> tuple:
    """The dial's range: from just below the lowest interval end (or the line) up to zero or above."""
    lab = builds(label)[1]
    costs = [measurement(label, "shrinking_cost", lab, cid) for cid in condition_ids]
    lows = [m["ci95"][0] for m in costs] + [large_line(label)]
    highs = [m["ci95"][1] for m in costs] + [0.0]
    return (math.floor(min(lows) / 0.1) * 0.1, max(0.0, math.ceil(max(highs) / 0.05) * 0.05))


def angle(value: float, dom: tuple) -> float:
    lo, hi = dom
    return SWEEP_FROM - (hi - value) / (hi - lo) * SWEEP


def point(a: float, r: float) -> tuple:
    return CX + r * math.cos(math.radians(a)), CY - r * math.sin(math.radians(a))


def arc(v_from: float, v_to: float, dom: tuple, r: float) -> str:
    """An arc path from the higher value to the lower one (clockwise on screen)."""
    a1, a2 = angle(max(v_from, v_to), dom), angle(min(v_from, v_to), dom)
    (x1, y1), (x2, y2) = point(a1, r), point(a2, r)
    return f"M{x1:.2f} {y1:.2f} A{r:.0f} {r:.0f} 0 {1 if a1 - a2 > 180 else 0} 1 {x2:.2f} {y2:.2f}"


def band(cost: dict, dom: tuple) -> str:
    return arc(cost["ci95"][0], cost["ci95"][1], dom, R - 4)


def ticks(dom: tuple) -> list:
    """A tick every 5 points; every 10 points is labelled."""
    lo, hi = dom
    return [round(k * 0.05, 4) for k in range(math.ceil(lo / 0.05 - 1e-9), math.floor(hi / 0.05 + 1e-9) + 1)]


def tick_labels(dom: tuple) -> list:
    return [f"{100 * t:.0f}" for t in ticks(dom) if abs(round(t * 100) % 10) == 0]


def verdict(label: dict, cost: dict, cid: str) -> tuple:
    """(kind, icon, text): the label's own flag (for clean, which has no envelope row, the same rule: the
    whole interval below the line), else which side of the line the whole interval is on."""
    row = envelope_row(label, builds(label)[1], cid)
    line = large_line(label)
    large = "large shrinking cost" in row["shrinking_cost_flags"] if row else cost["ci95"][1] < line
    if large:
        return "harm", "⚠", "Large shrinking cost"
    if cost["ci95"][0] > line:
        return "ok", "✓", "Small shrinking cost"
    return "borderline", "◐", "Too close to the line to call"


def svg(label: dict, cost: dict, dom: tuple) -> str:
    line = large_line(label)
    lo, hi = dom
    parts = [
        f'<path d="{arc(hi, line, dom, R)}" class="zone zone-small"/>',
        f'<path d="{arc(line, lo, dom, R)}" class="zone zone-large"/>',
    ]
    for t in ticks(dom):
        major = abs(round(t * 100) % 10) == 0
        (x1, y1), (x2, y2) = point(angle(t, dom), R - 10), point(angle(t, dom), R - (24 if major else 18))
        parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'class="tick{" major" if major else ""}"/>'
        )
        if major:
            x, y = point(angle(t, dom), R - 40)
            parts.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" class="tick-label">{100 * t:.0f}</text>')
    for v, name, cls in (
        ((hi + line) / 2, "small", "zone-label"),
        ((line + lo) / 2, "large", "zone-label zone-label-large"),
    ):
        x, y = point(angle(v, dom), R + 14)  # outside the arc, anchored away from it
        anchor = "end" if x < CX - 20 else "start" if x > CX + 20 else "middle"
        parts.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" class="{cls}" text-anchor="{anchor}">{name}</text>')
    parts.append(f'<text x="{CX}" y="{CY + 96}" class="dial-unit">INT8 − FP32, points</text>')
    parts.append(f'<path d="{band(cost, dom)}" class="band" id="meter-band"/>')
    a = angle(cost["value"], dom)
    parts.append(
        f'<g class="needle" id="meter-needle" style="transform: rotate({-a:.3f}deg)" '
        f'data-angle="{a:.3f}" data-rest="{angle(0.0, dom):.3f}">'
        f'<line x1="{CX - 18}" y1="{CY}" x2="{CX + R - 14}" y2="{CY}"/></g>'
    )
    parts.append(
        f'<circle cx="{CX}" cy="{CY}" r="11" class="hub"/>'
        f'<circle cx="{CX}" cy="{CY}" r="3.5" class="hub-dot"/>'
    )
    return (
        '<svg class="meter-dial" viewBox="-36 0 392 316" role="img" aria-labelledby="meter-title">'
        '<title id="meter-title">Shrinking cost on a dial: needle at the measured value, shaded band '
        "for its interval</title>"
        f'<circle cx="{CX}" cy="{CY}" r="{R + 30}" class="bezel"/>' + "".join(parts) + "</svg>"
    )
