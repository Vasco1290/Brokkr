"""The "official" wording check (brokkr_edge.wording; H's fix list of 4 October 2026, item 1): no sentence in
the labels, the site or the README says "official" together with "ImageNet-C" or "corruption code"."""

import json
from pathlib import Path

from brokkr_edge.label_render import TERMS
from brokkr_edge.wording import (
    CORRECTION_NOTE,
    QUOTING_DOCS,
    check_docs,
    doc_overclaims,
    json_overclaims,
    official_overclaims,
    sentences,
)

# The released labels' wording before 5 October 2026 (the limit, and the glossary line), as committed at
# 15437fd.
OLD_LIMIT = (
    "The ImageNet-C conditions were made with the official corruption code on these test images, with fixed "
    "seeds and each model's own preprocessing, so they are not directly comparable to published ImageNet-C "
    "results."
)
OLD_TERM = (
    "Brokkr and ImageNet-C: the two sets of damage: Brokkr's own, and ImageNet-C, a widely used public "
    "benchmark (made here with its official code)."
)


def test_the_old_wording_fails():
    assert official_overclaims(OLD_LIMIT) == [OLD_LIMIT]
    assert official_overclaims(f"<li>{OLD_TERM}</li>") == [OLD_TERM]
    assert json_overclaims({"limits": ["Measured on one machine.", OLD_LIMIT]}) == [OLD_LIMIT]


def test_other_ways_of_saying_it_fail_too():
    for text in (
        "Made with the Official ImageNet-C code.",
        "We used the official corruption code.",
        "ImageNet-C (official release) conditions",
    ):
        assert official_overclaims(text) == [text], text


def test_the_new_wording_passes():
    label = json.loads(Path("published/labels/mobilenet_v3_large/label.json").read_text(encoding="utf-8"))
    limit = next(x for x in label["limits"] if x.startswith("The ImageNet-C conditions"))
    assert "imagecorruptions package v1.1.2" in limit and "released ImageNet-C files" in limit
    assert official_overclaims(limit) == []
    assert official_overclaims(TERMS["Brokkr and ImageNet-C"]) == []


def test_official_elsewhere_is_not_flagged():
    # The source kind "official", and "Official label" in its own list item, are about who made the label.
    assert (
        json_overclaims({"source": {"kind": "official"}, "limits": ["ImageNet-C is simulated damage."]}) == []
    )
    assert official_overclaims("<dt>Official label</dt><dd>made by Brokkr.</dd><h2>ImageNet-C</h2>") == []
    assert sentences("<li>One.</li><li>Two. Three</li>") == ["One.", "Two.", "Three"]


def test_released_labels_and_readme_pass():
    found = official_overclaims(Path("README.md").read_text(encoding="utf-8"))
    for folder in sorted(Path("published/labels").iterdir()):
        found += json_overclaims(json.loads((folder / "label.json").read_text(encoding="utf-8")))
        found += official_overclaims((folder / "label.html").read_text(encoding="utf-8"))
    assert found == []


def test_a_claim_wrapped_across_two_lines_is_caught():
    assert official_overclaims("We made the ImageNet-C conditions with the official\ncorruption code.") == [
        "We made the ImageNet-C conditions with the official corruption code."
    ]
    assert official_overclaims("## Official label\nImageNet-C is simulated.") == []  # a heading ends there


def test_the_docs_pass_and_the_correction_note_is_reported():
    problems, report = check_docs()
    assert problems == []
    assert report[0] == (
        "docs/hypotheses_stage4.md: correction note of 6 October 2026 found; 1 sentence(s) above it "
        "(pre-registered text, not edited) are not read"
    )
    assert all(any(line.startswith(name) for line in report) for name in QUOTING_DOCS)


def test_only_text_above_the_correction_note_is_ignored():
    old = "ImageNet-C conditions are made with the official corruption code."
    note = (
        f"{CORRECTION_NOTE}\n\nThe description quoted above, the official corruption code, is inaccurate.\n"
    )
    assert doc_overclaims(f"{old}\n\n{note}") == ([], True, 1)
    below = f"{old}\n\n{note}\n## A later note\n\n{old}\n"
    assert doc_overclaims(below) == ([old], True, 1)
    assert doc_overclaims(old) == ([old], False, 0)  # without the note, nothing is ignored
