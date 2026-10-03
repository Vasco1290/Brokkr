"""Released labels committed in published/labels/ (docs/label_schema.md, note of 3 October 2026).

Each must follow the label schema, come from a clean commit, render to exactly its committed HTML, and
show no number that is not in its label.json. Recomputing the numbers from their source records is
scripts/40_check_labels.py's job (it needs the records); this test runs on a fresh clone.
"""

import json
from pathlib import Path

from brokkr_edge.label_render import to_html, unexplained_numbers
from brokkr_edge.label_schema import check_label

PUBLISHED = Path("published/labels")


def published() -> list:
    return sorted(PUBLISHED.glob("*/label.json"))


def test_there_is_at_least_one_published_label():
    assert published(), "published/labels/ holds no label"


def test_every_published_label_is_valid_clean_and_rendered_from_its_json():
    for path in published():
        label = json.loads(path.read_text(encoding="utf-8"))
        assert check_label(label) == [], path
        assert label["generated"]["dirty"] is False and label["source"]["kind"] == "official", path
        page = (path.parent / "label.html").read_text(encoding="utf-8")
        assert page == to_html(label), f"{path.parent}/label.html is not a fresh render of its label.json"
        assert unexplained_numbers(label, page) == [], path


def test_published_labels_hold_no_path_from_this_machine():
    for path in published():
        for name in ("label.json", "label.html"):
            text = (path.parent / name).read_text(encoding="utf-8")
            assert "C:\\" not in text and "C:/" not in text and "/home/" not in text and "Users/" not in text
