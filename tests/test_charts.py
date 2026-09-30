"""Checks for brokkr_edge.charts and the robustness section of the results page."""

import xml.etree.ElementTree as ET

from brokkr_edge.charts import legend, panel
from brokkr_edge.report import robustness_html

SVG = "{http://www.w3.org/2000/svg}"


def test_panel_is_valid_svg_with_a_tooltip_per_point():
    svg = panel("fog", {"fp32": [75, 74, 70, 60, 50, 40], "int8": [60, 55, 50, 40, 25, 10]},
                [0, 1, 2, 3, 4, 5], reference=90, reference_label="90% target")
    root = ET.fromstring(svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1))
    titles = [t.text for t in root.iter(f"{SVG}title")]
    assert len(titles) == 12  # 2 series x 6 severities
    assert "fog, severity 3, INT8: 40.0%" in titles
    assert len(list(root.iter(f"{SVG}polyline"))) == 2
    # Screen readers get the numbers too, not just "a chart".
    assert "FP32 75.0%, 74.0%" in root.get("aria-label")


def test_fp32_is_drawn_last_so_it_stays_visible():
    svg = panel("x", {"fp32": [1] * 6, "fp16": [1] * 6}, [0, 1, 2, 3, 4, 5])
    assert svg.rfind("FP32") > svg.rfind("FP16")


def test_values_are_placed_on_the_right_scale():
    svg = panel("x", {"fp32": [0, 100, 50, 50, 50, 50]}, [0, 1, 2, 3, 4, 5], y_max=100)
    # Plot area runs from y=24 (100%) to y=140 (0%), see TOP/BOTTOM in charts.py.
    points = svg.split('<polyline points="')[1].split('"')[0].split()
    assert points[0].endswith(",140.0") and points[1].endswith(",24.0") and points[2].endswith(",82.0")


def test_legend_keeps_fixed_order():
    html = legend(["int8", "fp32"])
    assert html.index("FP32") < html.index("INT8")


def test_no_sweep_means_no_robustness_section():
    assert robustness_html([], []) == ""
