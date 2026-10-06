"""The website's colour tokens (web/themes.py): every theme passes the automated rules, and the fixed colours
(datasheet panels, INT8 and FP32) are exactly the figures' colours, so the committed SVG figures sit on the
website's panels unchanged in every theme."""

import sys
from pathlib import Path

from brokkr_edge import figures

sys.path.insert(0, str(Path("web").resolve()))
import themes  # noqa: E402  (web/themes.py)


def test_every_theme_passes_contrast_and_the_other_rules():
    assert themes.check() == []


def test_the_datasheet_panels_and_data_colours_match_the_figures():
    for mode in ("light", "dark"):
        fixed, fig = themes.FIXED[mode], figures.THEMES[mode]
        assert fixed["sheet"] == fig["surface"], mode
        assert (fixed["int8"], fixed["fp32"]) == (fig["int8"], fig["fp32"]), mode


def test_the_dark_panel_is_a_neutral_grey():
    """H, 4 October 2026: no hue, so it works under every theme."""
    sheet = themes.FIXED["dark"]["sheet"]
    assert sheet[1:3] == sheet[3:5] == sheet[5:7]


def test_a_theme_whose_accent_is_close_to_the_int8_blue_is_refused():
    """The Runic frost rule, checked on a made-up theme (tests only)."""
    themes.THEMES["made-up"] = {**themes.THEMES["ember"], "name": "Made up", "accent": "#4a90e8"}
    try:
        assert any("accent too close to the INT8 blue" in p for p in themes.check())
    finally:
        del themes.THEMES["made-up"]


def test_the_meter_large_zone_must_stay_quieter_than_the_needle():
    """H, 4 October 2026: the needle and readout carry the eye (made-up theme, tests only)."""
    themes.THEMES["made-up"] = {**themes.THEMES["ember"], "name": "Made up", "glow": "#5a3a2a"}
    try:
        assert any("as loud as the needle" in p for p in themes.check())
    finally:
        del themes.THEMES["made-up"]
