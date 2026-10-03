"""The figures in docs/figures/ (scripts/47_figures.py): readable on a phone, enough contrast in both themes,
and no meaning carried by colour alone.

Tests on made-up values (tests only, never shown as results) check the drawing rules; tests on the committed
SVGs check font size and contrast and run on a fresh clone. That every plotted value equals its source is
scripts/47_figures.py --check's job (it needs the records).
"""

import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from brokkr_edge import figures as F

FIGURES = Path("docs/figures")
SVG = "{http://www.w3.org/2000/svg}"
MIN_TEXT_PX = 11  # smallest text, in screen pixels, at the figure's smallest intended display width


def committed() -> list:
    return sorted(FIGURES.glob("*.svg"))


def tag(el) -> str:
    return el.tag.replace(SVG, "")


def made_up(value, fmt="pts1", ci=None):
    entry = {"value": value, "fmt": fmt}
    if ci is not None:
        entry["ci95"] = ci
    return entry


def load_script():
    spec = importlib.util.spec_from_file_location("figures_script", "scripts/47_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- drawing rules, on made-up values ----


def test_colour_steps_follow_hs_edges():
    assert [F.step_of(x) for x in (0.002, 0.0, -0.005, -0.01, -0.0499, -0.05, -0.1499, -0.15, -0.4)] == [
        0,
        0,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]
    assert F.step_names() == [
        "INT8 equal or better",
        "under 1",
        "1 to 5",
        "5 to 15",
        "15 or more points lost",
    ]


def grid_svg(theme: str) -> ET.Element:
    v = {
        "name": made_up("Model A", "text"),
        "count": made_up(1, "int"),
        "label.a": made_up("fog (Brokkr) s3", "text"),
        "label.b": made_up("darkness (Brokkr) s5", "text"),
        "a.cost": made_up(0.004, ci=[-0.001, 0.009]),
        "a.flags": made_up([], "flags"),
        "b.cost": made_up(-0.08, ci=[-0.09, -0.07]),
        "b.flags": made_up(["large shrinking cost", "not informative"], "flags"),
        "large_cost_line": made_up(-0.05, "abs0"),
        "near_floor_line": made_up(0.1, "pct0"),
    }
    columns = [
        {"label": "label.a", "suite": "brokkr", "damage": "fog", "severity": 3},
        {"label": "label.b", "suite": "brokkr", "damage": "darkness", "severity": 5},
    ]
    rows = [
        {
            "name": "name",
            "count": "count",
            "cells": [{"cost": "a.cost", "flags": "a.flags"}, {"cost": "b.cost", "flags": "b.flags"}],
        }
    ]
    return ET.fromstring(F.grid(rows, columns, v, theme)[0])


def test_a_cell_where_int8_did_better_gets_the_first_step():
    for theme in F.THEMES:
        cell = next(el for el in grid_svg(theme).iter() if el.get("data-v") == "a.cost")
        assert cell.get("fill") == F.THEMES[theme]["steps"][0]


def test_a_near_floor_cell_with_a_large_cost_shows_grey_a_dash_and_the_dot():
    """'Not informative' never hides a large shrinking cost (H's fix 2, 3 October 2026)."""
    for theme in F.THEMES:
        root = grid_svg(theme)
        cell = next(el for el in root.iter() if el.get("data-v") == "b.cost")
        assert cell.get("fill") == F.THEMES[theme]["floor"] and cell.get("data-floor") == "1"
        assert [el.get("data-dot") for el in root.iter() if el.get("data-dot")] == ["b.flags"]
        texts = [el.text for el in root.iter() if tag(el) == "text"]
        assert "–" in texts and "+0.4" in texts


def test_the_grid_key_has_hs_wording():
    texts = [el.text for el in grid_svg("light").iter() if tag(el) == "text"]
    assert (
        "Colour: the measured cost. Dot: large cost (whole interval more than 5 points below full size)."
        in texts
    )


def speed_svg() -> ET.Element:
    v = {
        "a.name": made_up("Model A", "text"),
        "b.name": made_up("Model B", "text"),
        "a.1": made_up(0.6, "ratio"),
        "a.4": made_up(1.3, "ratio"),
        "a.1.int8": made_up(False, "bool"),
        "a.1.fp32": made_up(False, "bool"),
        "a.4.int8": made_up(False, "bool"),
        "a.4.fp32": made_up(True, "bool"),
        "unstable_line": made_up(10.0, "pctv"),
        "failed_text": made_up("INT8 build failed", "text"),
    }
    threads = [
        {"key": 1, "label": "1 thread", "shape": "circle", "colour": "int8"},
        {"key": 4, "label": "4 threads", "shape": "triangle", "colour": "second"},
    ]
    rows = [
        {
            "name": "a.name",
            "failed": False,
            "ratios": {1: "a.1", 4: "a.4"},
            "unstable": {1: ["a.1.int8", "a.1.fp32"], 4: ["a.4.int8", "a.4.fp32"]},
        },
        {"name": "b.name", "failed": True, "ratios": {}, "unstable": {}},
    ]
    return ET.fromstring(F.speed(rows, threads, v, "light")[0])


def test_an_unstable_speed_result_is_hollow_and_says_so_in_words():
    root = speed_svg()
    marks = {el.get("data-v"): el for el in root.iter() if el.get("data-part") == "mark"}
    assert marks["a.4"].get("data-hollow") == "1" and marks["a.1"].get("data-hollow") is None
    texts = [el.text for el in root.iter() if tag(el) == "text"]
    assert "unstable: 4 threads" in texts


def test_a_failed_build_says_so_and_draws_no_mark():
    root = speed_svg()
    texts = [el.text for el in root.iter() if tag(el) == "text"]
    assert "INT8 build failed: no INT8 timing" in texts
    assert not [el for el in root.iter() if (el.get("data-v") or "").startswith("b.")]


def test_speed_wording_never_says_faster_or_speed_up():
    for path in FIGURES.glob("speed-ratio-*.svg"):
        text = path.read_text(encoding="utf-8").lower()
        assert "faster" not in text and "speed-up" not in text and "speedup" not in text


def test_every_label_condition_has_a_short_name():
    for path in Path("published/labels").glob("*/label.json"):
        for c in json.loads(path.read_text(encoding="utf-8"))["conditions"]:
            assert c["damage"] == "clean" or c["damage"] in F.SHORT_DAMAGE, (path, c["damage"])


def test_the_number_check_catches_a_planted_number():
    script = load_script()
    fig = {
        "values": {"x": made_up(-0.123, ci=[-0.15, -0.1]), "name": made_up("ResNet-18", "text")},
        "axes": {"x": {"ticks": {"-10": -0.1}}},
    }
    assert script.stray_numbers(fig, ["ResNet-18: -12.3 points (-15.0 to -10.0), tick -10"]) == []
    assert script.stray_numbers(fig, ["ResNet-18: -12.4 points"]) == ["-12.4"]


# ---- the committed SVGs ----


def test_every_figure_has_a_light_and_a_dark_file():
    manifest = json.loads((FIGURES / "figures.json").read_text(encoding="utf-8"))
    for fig in manifest["figures"].values():
        assert set(fig["files"]) == {"light", "dark"}
        assert all(Path(f).exists() for f in fig["files"].values())
    assert len(committed()) == 2 * len(manifest["figures"])


def test_text_is_readable_at_the_smallest_display_width():
    """Font size in SVG units, scaled to the figure's smallest intended width, is at least 11 px."""
    for path in committed():
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        width = float(root.get("viewBox").split()[2])
        smallest = min(float(el.get("font-size")) for el in root.iter() if tag(el) == "text")
        px = smallest * int(root.get("data-min-display-px")) / width
        assert px >= MIN_TEXT_PX, f"{path.name}: smallest text {px:.1f} px"


def test_text_and_marks_have_enough_contrast_in_both_themes():
    """Text at least 4.5:1 against what it sits on; data marks and lines at least 3:1 against the
    surface; coloured grid cells at least 2:1 (their value is also printed in them)."""
    for path in committed():
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        surface = root.get("data-surface")
        for el in root.iter():
            name, part = tag(el), el.get("data-part")
            if name == "text":
                ratio = F.contrast(el.get("fill"), el.get("data-on") or surface)
                assert ratio >= 4.5, f"{path.name}: text {el.text!r} at {ratio:.2f}:1"
            elif part == "cell" and el.get("data-floor") != "1":
                assert F.contrast(el.get("fill"), surface) >= 2.0, f"{path.name}: cell {el.get('fill')}"
            elif part in ("mark", "ci", "bar", "line") or el.get("data-dot"):
                colour = (
                    el.get("stroke")
                    if part == "ci" or part == "line" or el.get("data-hollow")
                    else el.get("fill")
                )
                against = el.get("data-on") or surface
                assert F.contrast(colour, against) >= 3.0, f"{path.name}: {part} {colour} on {against}"


def test_marks_carry_identity_by_shape_and_a_hover_note():
    """FP32 and INT8 differ by shape, not only colour; every data mark has a hover note."""
    for path in committed():
        root = ET.fromstring(path.read_text(encoding="utf-8"))
        for el in root.iter():
            if el.get("data-part") in ("mark", "ci", "bar", "cell"):
                assert el.find(f"{SVG}title") is not None, (
                    f"{path.name}: {el.get('data-v')} has no hover note"
                )
    hero = ET.fromstring((FIGURES / "hero-shrinking-cost-light.svg").read_text(encoding="utf-8"))
    shapes = {
        (el.get("data-v") or "").rsplit(".", 1)[-1]: tag(el)
        for el in hero.iter()
        if el.get("data-part") == "mark"
    }
    assert shapes == {"clean": "circle", "dark": "rect"}


def test_hero_rows_are_sorted_by_the_darkness_result():
    manifest = json.loads((FIGURES / "figures.json").read_text(encoding="utf-8"))
    values = manifest["figures"]["hero-shrinking-cost"]["values"]
    hero = ET.fromstring((FIGURES / "hero-shrinking-cost-light.svg").read_text(encoding="utf-8"))
    order = [
        el.get("data-v")
        for el in hero.iter()
        if el.get("data-part") == "mark" and el.get("data-v").endswith(".dark")
    ]
    assert len(order) == len({v for v in values if v.endswith(".dark")})
    darkness = [values[v]["value"] for v in order]
    assert darkness == sorted(darkness)


def test_figures_are_self_contained_with_a_title_and_description():
    for path in committed():
        text = path.read_text(encoding="utf-8")
        assert re.findall(r"https?://", text) == ["http://"], f"{path.name} refers to another site"
        assert '<title id="title">' in text and '<desc id="desc">' in text


def test_the_readme_shows_the_wide_hero_on_wide_screens_and_the_tall_one_on_phones():
    """H's option b (docs/label_schema.md, third note of 4 October 2026), in light and dark."""
    readme = Path("README.md").read_text(encoding="utf-8")
    block = readme[readme.index("figure-hero:start"):readme.index("figure-hero:end")]
    wide, dark = "(min-width: 768px)", "(prefers-color-scheme: dark)"
    assert f'media="{wide} and {dark}" srcset="docs/figures/hero-shrinking-cost-wide-dark.svg"' in block
    assert f'media="{wide}" srcset="docs/figures/hero-shrinking-cost-wide-light.svg"' in block
    assert f'media="{dark}" srcset="docs/figures/hero-shrinking-cost-dark.svg"' in block
    assert '<img src="docs/figures/hero-shrinking-cost-light.svg"' in block
