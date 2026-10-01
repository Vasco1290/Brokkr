"""The README stays usable: every local file or image it points to exists, and old names are gone.

It cannot check that the README's numbers are current (the result files are not in git); those are
re-read from the result files whenever the README is updated.
"""

import re
from pathlib import Path

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


def test_the_old_package_folder_name_is_gone():
    assert "brokkr/" not in README  # the package folder is brokkr_edge/ since 30 September 2026
    assert "Stages 1–2 of 7" not in README
