"""The website's colour tokens: one entry per theme, plus the colours no theme may change.

A theme changes only the shell (backgrounds, panels, text, accent, glow, control styling). The data colours
(INT8 blue, FP32 grey, matching docs/figures/), the verdict colours (always with an icon) and the datasheet
panels behind tables and figures are FIXED: they depend only on whether the theme is light or dark, so the
committed SVG figures (made for those two surfaces) stay valid in every theme.

Adding a theme = adding one entry to THEMES; pages, CSS and the theme picker are generated from this file.
Besides colours, each theme's CSS block says which version of a committed figure to show (--fig-light,
--fig-dark), from its light or dark mode.
check() runs the automated rules: contrast (text 4.5:1, marks and controls 3:1), a theme's accent clearly
distinct from the INT8 blue, and no theme overriding a fixed colour.
"""

FIXED = {
    "light": {
        "int8": "#0072b2",
        "fp32": "#6e7781",  # as in brokkr_edge/figures.py
        "ok": "#1a7f37",
        "harm": "#b42318",
        "borderline": "#8a5a00",
        "untested": "#59636e",
        "failed": "#b42318",
        "sheet": "#ffffff",
        "sheet-ink": "#1f2328",
        "sheet-muted": "#59636e",
        "sheet-rule": "#1f2328",
        "sheet-line": "#d1d9e0",
        "large-zone": "#b42318",
        "small-zone": "#7d8590",
    },
    "dark": {
        "int8": "#3987e5",
        "fp32": "#8b949e",
        "ok": "#3fb950",
        "harm": "#ff7b72",
        "borderline": "#d29922",
        "untested": "#9198a1",
        "failed": "#ff7b72",
        # A neutral grey with no hue (H, 4 October 2026): the same surface as the dark figures.
        "sheet": "#181818",
        "sheet-ink": "#f2f2f2",
        "sheet-muted": "#a3a3a3",
        "sheet-rule": "#f2f2f2",
        "sheet-line": "#404040",
        "large-zone": "#ff7b72",
        "small-zone": "#6e7681",
    },
}

# The shell. "heat" is the pixel-cooling ramp, hottest first (the hero's squares cool from ember to stone).
THEMES = {
    "ember": {
        "name": "Ember forge",
        "mode": "dark",
        "bg": "#141210",
        "bg-2": "#1b1815",
        "panel": "#221e1a",
        "ink": "#efe6d8",
        "muted": "#aba193",
        "rule": "#3d352e",
        "accent": "#e0683c",
        "glow": "#ff9f57",
        "focus": "#ffc38a",
        "ctl": "#26211d",
        "ctl-hi": "#342d27",
        "ctl-lo": "#0b0908",
        "ctl-border": "#8a7b6d",
        "ctl-ink": "#efe6d8",
        "heat": ["#ffd27a", "#ff9f57", "#e0683c", "#a8432a", "#6e3a2c", "#4a3a33", "#3a332e", "#2c2824"],
    },
    "light": {
        "name": "Light",
        "mode": "light",
        "bg": "#f7f5f1",
        "bg-2": "#efebe4",
        "panel": "#fdfcfa",
        "ink": "#1d1b18",
        "muted": "#5b554c",
        "rule": "#d5cec3",
        "accent": "#b4461c",
        "glow": "#b4461c",
        "focus": "#1f5fbf",
        "ctl": "#f2eee8",
        "ctl-hi": "#ffffff",
        "ctl-lo": "#cfc6b9",
        "ctl-border": "#857a6d",
        "ctl-ink": "#1d1b18",
        "heat": ["#c4501f", "#b0451f", "#9c4a33", "#8d5644", "#9a8b80", "#b8aea4", "#d3cbc1", "#e6e0d8"],
    },
}
DEFAULT_FOR_MODE = {"dark": "ember", "light": "light"}  # the theme a first-time visitor gets


def tokens(theme_key: str) -> dict:
    """Every CSS token of one theme: its shell plus the fixed colours of its mode."""
    t = THEMES[theme_key]
    shell = {k: v for k, v in t.items() if k not in ("name", "mode", "heat")}
    shell.update({f"heat-{i}": c for i, c in enumerate(t["heat"])})
    return {**shell, **FIXED[t["mode"]]}


def css() -> str:
    """themes.css: each theme under [data-theme], and the default for each system setting."""

    def block(selector, key):
        mode = THEMES[key]["mode"]
        body = "".join(f"  --{k}: {v};\n" for k, v in tokens(key).items())
        # Which version of a committed figure (light or dark surface) shows under this theme.
        body += f"  --fig-light: {'block' if mode == 'light' else 'none'};\n"
        body += f"  --fig-dark: {'block' if mode == 'dark' else 'none'};\n"
        return f"{selector} {{\n  color-scheme: {mode};\n{body}}}\n"

    out = ["/* Generated from web/themes.py; never edit by hand. */\n"]
    out.append(block(":root", DEFAULT_FOR_MODE["light"]))
    out.append(
        "@media (prefers-color-scheme: dark) {\n"
        + block(':root:not([data-theme="light"])', DEFAULT_FOR_MODE["dark"])
        + "}\n"
    )
    out += [block(f':root[data-theme="{k}"]', k) for k in THEMES]
    return "".join(out)


# ---- checks ----


def _luminance(colour: str) -> float:
    def channel(c):
        c = int(c, 16) / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(colour[i : i + 2]) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_distance(a: str, b: str) -> float:
    """Perceptual distance between two colours (OKLab, x100); 15 or more reads as clearly different."""

    def oklab(colour):
        def lin(c):
            c = int(c, 16) / 255
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = (lin(colour[i : i + 2]) for i in (1, 3, 5))
        lms = [
            0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
            0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
            0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
        ]
        l_, m_, s_ = (x ** (1 / 3) for x in lms)
        return (
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        )

    return 100 * sum((x - y) ** 2 for x, y in zip(oklab(a), oklab(b), strict=True)) ** 0.5


TEXT_PAIRS = [  # (text token, background token): at least 4.5:1
    *[(ink, bg) for ink in ("ink", "muted", "accent", "glow") for bg in ("bg", "bg-2", "panel")],
    ("ctl-ink", "ctl"),
    ("accent", "ctl"),
    *[
        (ink, "sheet")
        for ink in ("sheet-ink", "sheet-muted", "ok", "harm", "borderline", "untested", "failed")
    ],
    *[(ink, bg) for ink in ("ok", "harm", "borderline", "untested") for bg in ("bg", "panel")],
]
MARK_PAIRS = [  # (mark or control token, background token): at least 3:1
    ("ctl-border", "ctl"),
    ("ctl-border", "bg"),
    ("ctl-border", "panel"),
    ("accent", "ctl"),
    ("focus", "bg"),
    ("focus", "panel"),
    ("focus", "ctl"),
    *[(m, bg) for m in ("int8", "fp32") for bg in ("sheet", "panel", "bg")],
    ("large-zone", "panel"),
    ("small-zone", "panel"),
    ("sheet-rule", "sheet"),
]


def check() -> list:
    """Every rule, for every theme; returns the problems (empty = PASS)."""
    problems = []
    fixed_keys = set(FIXED["light"])
    for key, theme in THEMES.items():
        overridden = fixed_keys & set(theme)
        if overridden:
            problems.append(f"{key}: overrides fixed colours {sorted(overridden)}")
        t = tokens(key)
        for ink, bg in TEXT_PAIRS:
            if contrast(t[ink], t[bg]) < 4.5:
                problems.append(f"{key}: text {ink} on {bg} {contrast(t[ink], t[bg]):.2f}:1 (needs 4.5)")
        for mark, bg in MARK_PAIRS:
            if contrast(t[mark], t[bg]) < 3:
                problems.append(f"{key}: {mark} on {bg} {contrast(t[mark], t[bg]):.2f}:1 (needs 3)")
        for heat in t["heat-0"], t["heat-2"]:  # the hottest squares must stand out from the page
            if contrast(heat, t["bg"]) < 3:
                problems.append(f"{key}: hero heat colour {heat} on bg below 3:1")
        if oklab_distance(t["accent"], t["int8"]) < 15:  # e.g. a frosty blue accent next to the INT8 blue
            problems.append(
                f"{key}: accent too close to the INT8 blue ({oklab_distance(t['accent'], t['int8']):.1f})"
            )
    return problems
