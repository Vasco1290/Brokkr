"""Labels never live in results/ (decided by H, 30 September 2026).

results/ holds only measurement records. A label.json there carries "schema_version": 1, so the results
loader took it for a Stage 1-3 record and stopped (the Checkpoint 1 failure). Generated labels go to
the top-level labels/ folder instead.
"""

from pathlib import Path

LABEL_FILE_NAMES = {"label.json", "label.md", "label.html"}


def label_files_under(folder: Path) -> list:
    """Files in `folder` that are labels: named like one, or a JSON file whose start has "label_id"."""
    found = []
    for path in folder.rglob("*"):
        if path.name in LABEL_FILE_NAMES:
            found.append(path)
        elif path.suffix == ".json" and path.is_file():
            with open(path, "rb") as f:
                if b'"label_id"' in f.read(512):  # the second key of every label.json
                    found.append(path)
    return found


def test_the_finder_spots_labels_by_name_and_by_content(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "label.md").write_text("# a label")
    (tmp_path / "renamed.json").write_text('{"schema_version": 1, "label_id": "x"}')
    (tmp_path / "record.json").write_text('{"schema_version": 2, "kind": "accuracy"}')
    assert sorted(p.name for p in label_files_under(tmp_path)) == ["label.md", "renamed.json"]


def test_no_label_file_under_results():
    results = Path("results")
    if not results.exists():  # a fresh clone has no results, so nothing can be misplaced
        return
    assert label_files_under(results) == []


def test_the_label_scripts_default_to_the_labels_folder():
    scripts = {"scripts/39_make_labels.py": "--out", "scripts/40_check_labels.py": "--labels"}
    for script, option in scripts.items():
        text = Path(script).read_text(encoding="utf-8")
        assert f'parser.add_argument("{option}", default="labels")' in text, script
        assert "results/labels" not in text, script
