"""scripts/48_compare_labels.py: a remake passes only with the always-allowed differences and the dated
changes listed in docs/label_changes.json; any other change fails. The "old" labels here are made from the
released ones by undoing the listed changes (tests only), so the test runs on a fresh clone."""

import html
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("compare_labels", Path("scripts/48_compare_labels.py"))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

MODEL = "mobilenet_v3_large"
CHANGES = json.loads(Path("docs/label_changes.json").read_text(encoding="utf-8"))["changes"]


def undo(text: str) -> str:
    """The released label.html as it was before the listed renderer and limit changes."""
    for c in CHANGES:
        for a, b in {(c["new"], c["old"]), (html.escape(c["new"]), html.escape(c["old"]))}:
            text = text.replace(a, b)
    return text


@pytest.fixture
def sets(tmp_path):
    old, new = tmp_path / "old" / MODEL, tmp_path / "new" / MODEL
    shutil.copytree(Path("published/labels") / MODEL, new)
    old.mkdir(parents=True)
    label = json.loads((new / "label.json").read_text(encoding="utf-8"))
    for c in CHANGES:
        if c["in"] == "label.json":
            C.parent(label, tuple(c["path"]))[c["path"][-1]] = c["old"]
    (old / "label.json").write_text(json.dumps(label, indent=2) + "\n", encoding="utf-8")
    (old / "label.html").write_text(undo((new / "label.html").read_text(encoding="utf-8")), encoding="utf-8")
    return tmp_path / "old", tmp_path / "new"


def run(sets, changes=CHANGES):
    return C.compare_model(*sets, MODEL, changes)


def edit_json(folder: Path, change) -> None:
    label = json.loads((folder / MODEL / "label.json").read_text(encoding="utf-8"))
    change(label)
    (folder / MODEL / "label.json").write_text(json.dumps(label, indent=2) + "\n", encoding="utf-8")
    (folder / MODEL / "label.html").write_text(C.to_html(label), encoding="utf-8")  # a consistent remake


def test_the_listed_changes_pass_and_are_named(sets):
    ok, line, details = run(sets)
    assert ok, (line, details)
    assert "differing fields: none; accepted listed changes: 2" in line
    assert [d for d in details if d.startswith("     accepted: [2026-10-05]")] == [
        f"     accepted: {C.name_of(c)}" for c in CHANGES
    ]


def test_an_unlisted_text_change_fails(sets):
    edit_json(
        sets[1], lambda lab: lab["limits"].__setitem__(0, lab["limits"][0] + " Also tested on a phone.")
    )
    ok, line, details = run(sets)
    assert not ok and "OTHER differences: 1" in line and any("other: limits/0" in d for d in details)


def test_an_unlisted_number_change_fails(sets):
    edit_json(sets[1], lambda lab: lab["checks"]["fp32_sanity"].__setitem__("tolerance", 0.02))
    ok, _, details = run(sets)
    assert not ok and any("checks/fp32_sanity/tolerance" in d for d in details)


def test_a_listed_field_changed_to_other_text_fails(sets):
    edit_json(sets[1], lambda lab: lab["limits"].__setitem__(1, lab["limits"][1].replace("v1.1.2", "v1.1.3")))
    ok, _, details = run(sets)
    assert not ok and any("other: limits/1" in d for d in details)


def test_an_unlisted_renderer_change_fails(sets):
    page = sets[0] / MODEL / "label.html"
    page.write_text(
        page.read_text(encoding="utf-8").replace("What the words mean", "Glossary"), encoding="utf-8"
    )
    ok, _, details = run(sets)
    assert not ok and any(
        "renderer changed in a way docs/label_changes.json does not list" in d for d in details
    )


def test_without_the_list_the_same_remake_fails(sets):
    ok, line, details = run(sets, changes=[])
    assert not ok and "accepted listed changes: 0" in line


def test_every_listed_change_is_dated_and_points_to_a_note():
    for c in CHANGES:
        assert set(c) == {"date", "note", "what", "in", "path", "old", "new"}
        assert c["in"] in ("label.json", "label.html") and (c["path"] is None) == (c["in"] == "label.html")
        assert c["note"].startswith("docs/") and Path(c["note"].split(",")[0]).exists()
        assert c["old"] != c["new"]
