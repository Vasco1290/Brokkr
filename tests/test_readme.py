"""The README stays usable: every local file or image it points to exists, and old names are gone.

It has no hand-typed result numbers. Its label example (scripts/41_readme_label_example.py) and its Stage 4
findings (scripts/45_readme_findings.py) are generated blocks, checked here whenever the labels and records
are on the machine (they are not in git). Its label screenshots (scripts/46_readme_screenshots.py) must show
the released label as it is now.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

README = Path("README.md").read_text(encoding="utf-8")
# Markdown links and images "](target)", and HTML images 'src="target"'.
TARGETS = re.findall(r"\]\(([^)\s]+)\)", README) + re.findall(r'src="([^"]+)"', README)


def test_every_local_link_and_image_exists():
    local = [t for t in TARGETS if not t.startswith(("http://", "https://", "#"))]
    assert local, "the README should link to the roadmap, the report and its banner"
    missing = [t for t in local if not Path(t.split("#")[0]).exists()]
    assert missing == []


def test_every_contents_link_points_to_a_heading():
    headings = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("# ").lower()).replace(" ", "-")
        for line in README.splitlines()
        if line.startswith("#")
    }
    anchors = [t[1:] for t in TARGETS if t.startswith("#")]
    assert [a for a in anchors if a not in headings] == []


@pytest.mark.skipif(not Path("labels/mobilenet_v3_small/label.json").exists(),
                    reason="the working labels are not on this machine (they are not in git)")
def test_the_label_example_equals_a_fresh_render_of_the_labels():
    result = subprocess.run([sys.executable, "scripts/41_readme_label_example.py", "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not (Path("results/final/breadth_4.1_verdicts.json").exists()
                         and Path("labels/mobilenet_v3_large/label.json").exists()),
                    reason="the 4.1 records and labels are not on this machine (they are not in git)")
def test_the_findings_equal_a_fresh_render_of_the_records():
    result = subprocess.run([sys.executable, "scripts/45_readme_findings.py", "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_generated_blocks_are_present_once():
    for marker in ("label-example:start", "label-example:end", "findings:start", "findings:end"):
        assert README.count(marker) == 1, marker


def test_the_label_screenshots_show_the_released_label_as_it_is_now():
    record = json.loads(Path("docs/assets/label-summary.json").read_text(encoding="utf-8"))
    label = Path(record["label"])
    text_hash = hashlib.sha256(label.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    assert record["sha256"] == text_hash, "rerun scripts/46_readme_screenshots.py: the label has changed"
    for picture in record["pictures"]:
        assert (Path("docs/assets") / picture).exists() and f"docs/assets/{picture}" in README


def test_the_old_package_folder_name_is_gone():
    assert "brokkr/" not in README  # the package folder is brokkr_edge/ since 30 September 2026
    assert "Stages 1–2 of 7" not in README
