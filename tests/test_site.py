"""The website generator (scripts/49_build_site.py, web/site_*.py): a fresh build passes every site check,
and each check fails when the thing it guards is broken (made-up changes, tests only). Needs only committed
files (published/labels/, docs/figures/), so it runs on a fresh clone."""

import copy
import html
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("web").resolve()))
import site_build  # noqa: E402  (web/site_build.py)
import site_checks  # noqa: E402
import site_labels  # noqa: E402
import site_pages  # noqa: E402
import site_sources  # noqa: E402

import brokkr_edge  # noqa: E402

COMMIT = "abc1234"  # a made-up commit for test builds
VERSION = brokkr_edge.__version__
MNV3L = "models/mobilenet-v3-large/int8-percentile-99.99/index.html"
MNV3S = "models/mobilenet-v3-small/int8-percentile-99.99/index.html"


@pytest.fixture(scope="module")
def built():
    labs, figures = site_build.load_inputs()
    pages, files = site_build.build(labs, figures, COMMIT)
    return labs, figures, pages, files


def problems_of(built, files=None) -> dict:
    labs, figures, pages, base = built
    results = site_checks.run_all(pages, files or base, labs, figures, COMMIT, VERSION, site_build.CLAIMS)
    return {what: p for what, p in results}


def test_a_fresh_build_passes_every_site_check(built):
    assert {what: p for what, p in problems_of(built).items() if p} == {}


def test_every_page_exists(built):
    labs, _, pages, _ = built
    assert {"index.html", "models/index.html", MNV3L, MNV3S} <= set(pages)
    assert {"compare/index.html", "why-labels/index.html", "methods/index.html", "roadmap/index.html"} <= set(
        pages
    )
    assert len([p for p in pages if p.startswith("models/") and p.count("/") == 3]) == len(labs)


def test_a_new_page_type_changes_no_existing_page_except_the_navigation(built):
    labs, figures, pages, files = built

    def made_up(site):  # tests only
        return [("made-up/index.html", site_pages.Page("Made up", "<p>Made up.</p>", labels=[]))]

    types = [*site_build.PAGE_TYPES, site_pages.PageType("made-up", "Made up", "made-up/index.html", made_up)]
    pages2, files2 = site_build.build(labs, figures, COMMIT, types)
    assert set(pages2) == set(pages) | {"made-up/index.html"}

    def without_nav(text):
        return re.sub(r'<nav aria-label="Main">.*?</nav>', "", text, flags=re.S)

    for path in pages:
        assert without_nav(files2[path]) == without_nav(files[path]), path
        assert 'href="' in files2[path] and "Made up</a>" in files2[path], path  # the nav shows the new page


def test_the_navigation_shows_only_pages_that_exist(built):
    nav = re.search(r'<nav aria-label="Main">(.*?)</nav>', built[3]["index.html"], re.S).group(1)
    assert re.findall(r">([^<]+)</a>", nav) == [
        "Catalog",
        "Compare",
        "Why labels?",
        "Methods",
        "Roadmap",
        "GitHub",
    ]
    buttons = re.findall(r'class="button" href="[^"]*">([^<]+)</a>', built[3]["index.html"])
    assert buttons == ["Catalog", "Why labels?"]  # from the page list, not typed on the landing page


def test_a_number_not_in_the_labels_is_caught(built):
    files = dict(built[3])
    files["models/index.html"] = files["models/index.html"].replace(
        "<h1>Catalog</h1>", "<h1>Catalog 42.42%</h1>"
    )
    assert any("42.42%" in p for p in problems_of(built, files)["numbers come from the labels and figures"])


def test_a_number_from_another_label_is_caught_on_a_model_page(built):
    """Each page may show only the numbers of the labels it names (MobileNetV3-Small's FP32 size here)."""
    labs, files = built[0], dict(built[3])
    size = site_pages.f(site_labels.builds(labs["mobilenet_v3_small"])[0]["file"]["size_bytes"], "mb")
    assert size not in files[MNV3L]
    files[MNV3L] = files[MNV3L].replace(
        '<h2 id="raw-h">Raw label</h2>', f'<h2 id="raw-h">Raw label {size}</h2>'
    )
    assert problems_of(built, files)["numbers come from the labels and figures"]


def test_a_meter_needle_that_does_not_match_its_label_is_caught(built):
    files = dict(built[3])
    angle = re.search(
        r'value="brokkr/darkness/5" data-cost-text="[^"]*" data-interval-text="[^"]*" '
        r'data-verdict="[^"]*" data-verdict-kind="[^"]*" data-icon="[^"]*" data-label="[^"]*" '
        r'data-angle="([^"]+)"',
        files["index.html"],
    ).group(1)
    files["index.html"] = files["index.html"].replace(f'data-angle="{angle}"', 'data-angle="12.000"', 1)
    assert problems_of(built, files)["meter needle, band, readout and verdicts match the label"]


def test_broken_links_anchors_and_outside_loads_are_caught(built):
    files = dict(built[3])
    files["models/index.html"] += '<a href="nowhere/index.html">x</a><a href="../index.html#no-such-id">y</a>'
    files["models/index.html"] += '<img src="https://example.org/x.png" alt="">'
    found = problems_of(built, files)["internal links resolve; nothing loaded from other sites"]
    assert len(found) == 3


def test_content_hidden_until_a_script_runs_is_caught(built):
    files = dict(built[3])
    files["models/index.html"] = files["models/index.html"].replace(
        '<div class="cards">', '<div class="cards" hidden>'
    )
    files["index.html"] = files["index.html"].replace(
        '<fieldset class="keys js-only">', '<fieldset class="keys js-only"><span>-0.05</span>'
    )
    found = problems_of(built, files)["pages work with JavaScript off"]
    assert any("hidden" in p for p in found) and any("-0.05" in p for p in found)


def test_every_number_the_script_can_show_is_also_there_without_it(built):
    """The four meter readings appear in the plain table shown without JavaScript."""
    labs, files = built[0], built[3]
    label = labs[site_pages.FEATURED]
    for cid in site_pages.METER_CONDITIONS:
        cost = site_labels.measurement(label, "shrinking_cost", site_labels.builds(label)[1], cid)
        no_js = files["index.html"].split('class="sheet no-js-only"')[1].split("</table>")[0]
        assert site_pages.f(cost["value"], "pts") in no_js, cid


def test_a_footer_without_the_versions_is_caught(built):
    files = dict(built[3])
    files["models/index.html"] = files["models/index.html"].replace("label schema version", "label version")
    assert problems_of(built, files)["every page names commit, brokkr-edge version and schema version"]


def test_the_broken_build_page_opens_with_the_warning(built):
    page = built[3][MNV3S]
    assert page.index('class="alarm"') < page.index('id="facts"')
    assert "Do not use this INT8 build" in page and "This recipe broke the model" in page
    card = built[3]["models/index.html"].split("MobileNetV3-Small")[1].split("</article>")[0]
    assert "INT8 build failed: do not use" in card


def test_int8_slower_and_unstable_speed_rows_are_said_in_words(built):
    regnet = built[3]["models/regnet-y-400mf/int8-percentile-99.99/index.html"]
    assert 'class="callout"' in regnet and "INT8 is slower than FP32 on this laptop CPU" in regnet
    assert "(slower)" in regnet
    convnext = built[3]["models/convnext-tiny/int8-percentile-99.99/index.html"]
    assert convnext.count("unstable: speed varied a lot between repeat runs") == 4


def test_a_user_submitted_label_shows_unverified(built):
    labs, figures = dict(built[0]), built[1]
    fake = copy.deepcopy(labs["resnet18"])  # tests only
    fake["source"] = {
        **fake["source"],
        "kind": "user-submitted",
        "verified": False,
        "submitted_by": "someone",
    }
    labs["resnet18"] = fake
    assert site_labels.label_problems({"resnet18": fake}) == []
    pages, files = site_build.build(labs, figures, COMMIT)
    card = files["models/index.html"].split("ResNet-18")[1].split("</article>")[0]
    assert "User-submitted · unverified" in card
    assert "User-submitted · unverified" in files["models/resnet18/int8-percentile-99.99/index.html"]


def test_a_label_with_an_unknown_schema_version_is_refused(built):
    fake = copy.deepcopy(built[0]["resnet18"])
    fake["schema_version"] = 2
    assert "not known here" in site_labels.label_problems({"resnet18": fake})[0]


def test_two_labels_with_one_address_are_refused(built):
    labs, figures = dict(built[0]), built[1]
    labs["copy"] = copy.deepcopy(labs["resnet18"])
    with pytest.raises(SystemExit, match="same address"):
        site_build.build(labs, figures, COMMIT)
    assert site_checks.check_ids(labs)


def test_build_ids_follow_the_rule_of_section_12d(built):
    ids = {k: site_labels.build_id(lab, site_labels.builds(lab)[1]) for k, lab in built[0].items()}
    assert ids["mobilenet_v3_large"] == "BRK-MNV3L-INT8-P9999"
    assert ids["convnext_tiny"] == "BRK-CNXT-INT8-P9999"
    assert ids["shufflenet_v2_x1_0"] == "BRK-SNV2X10-INT8-P9999"
    assert (
        site_labels.address(built[0]["mobilenet_v3_large"])
        == "/models/mobilenet-v3-large/int8-percentile-99.99/"
    )


def test_the_site_refuses_a_dirty_working_tree():
    assert site_build.dirty_tree_problem("") is None
    assert "uncommitted" in site_build.dirty_tree_problem(" M web/site_build.py\n")


def test_the_built_files_hold_no_path_from_this_machine(built):
    for path, text in built[3].items():
        if isinstance(text, str):
            assert "C:\\" not in text and "C:/" not in text and "Users/" not in text, path


def test_the_landing_finding_is_the_readme_headline_word_for_word(built):
    headline = site_sources.findings()["headline"]
    assert headline.startswith("Consistent with prior work on quantized models")
    assert headline.endswith("and fog hit some models too.")
    page = built[3]["index.html"]
    assert html.escape(headline) in page
    assert (
        "H21" not in page.split('class="finding-text"')[1].split("</p>")[0]
    )  # the H21 detail is on Why labels?


def test_why_labels_holds_the_full_findings(built):
    page = built[3]["why-labels/index.html"]
    fx = site_sources.findings()
    for key in ("headline", "example", "prereg", "explore", "noise_blur"):
        assert html.escape(fx[key]) in page, key
    assert all(h in fx["prereg"] for h in ("H18b", "H20", "H21"))
    for name in ("hero-shrinking-cost-wide", "grid-shrinking-cost", "mnv3l-uncertainty"):
        assert f"{name}-light.svg" in page and f"{name}-dark.svg" in page, name


def test_a_findings_number_that_is_in_no_source_is_caught(built):
    files = dict(built[3])
    assert "10,000 test images" in files["why-labels/index.html"]
    files["why-labels/index.html"] = files["why-labels/index.html"].replace(
        "10,000 test images", "10,500 test images"
    )
    assert any("10,500" in p for p in problems_of(built, files)["numbers come from the labels and figures"])


def test_phone_condition_cards_are_collapsed_with_a_summary_line(built):
    labs, page = built[0], built[3][MNV3L]
    cards = re.findall(r'<details class="cond-card">(.*?)</details>', page, re.S)
    assert len(cards) == len(labs["mobilenet_v3_large"]["conditions"])
    assert '<details class="cond-card" open' not in page
    flagged = labs["mobilenet_v3_large"]["summary"]["large_shrinking_cost"]["count"]
    assert sum("⚠ large" in c.split("</summary>")[0] for c in cards) == flagged


def test_a_failed_build_collapses_where_it_holds_up_to_one_line(built):
    page = built[3][MNV3S]
    env = page.split('id="envelope"')[1].split("</section>")[0]
    assert (
        '<details class="more failed-env"><summary>All <span class="mono">12</span> conditions: '
        "INT8 build failed" in env
    )
    assert '<details class="more failed-env"' not in built[3][MNV3L]
    assert "67.61%" in page.split('id="conditions"')[1]  # FP32 results stay visible


def test_the_compare_table_has_one_sortable_row_per_label(built):
    labs, page = built[0], built[3]["compare/index.html"]
    body = page.split('class="data sortable compare-table"')[1].split("</table>")[0]
    assert body.count("<tr>") == len(labs) + 1  # header row + one per label
    assert "speed-ratio-light.svg" in page and "grid-shrinking-cost-light.svg" in page


ROADMAP_CHECK = "the roadmap shows no dates and no stage or step numbers"
FINDINGS_CHECK = "every findings paragraph equals the README's, character for character"
OFFICIAL_CHECK = 'no sentence says "official" with ImageNet-C or corruption code'


def test_the_roadmap_is_generated_without_dates(built):
    page = built[3]["roadmap/index.html"]
    for phase in ("Built", "Now", "Next", "Later"):
        assert f"<h2>{phase}</h2>" in page
    plan = site_sources.roadmap()
    assert plan["next"] and all(
        html.escape(name) in page for k in ("built", "now", "next", "later") for name in plan[k]
    )
    files = dict(built[3])
    files["roadmap/index.html"] = files["roadmap/index.html"].replace(
        "<h2>Later</h2>", "<h2>Later (by 5 November 2026)</h2>"
    )
    assert problems_of(built, files)[ROADMAP_CHECK]


def test_the_roadmap_shows_only_public_items_with_plain_names(built):
    page = built[3]["roadmap/index.html"]
    plan = site_sources.roadmap()
    assert "The study across ten models" in plan["built"]  # Stage 4, marked done, under its public name
    assert "Core measurement" in plan["built"] and "Stage 1" not in page
    assert "Step " not in page and "summary tables" not in page  # an internal item is left out
    assert "Related work" not in page and "parked research" not in page
    files = dict(built[3])
    files["roadmap/index.html"] = page.replace("<li>Labels</li>", "<li>Step 1 Labels</li>")
    assert problems_of(built, files)[ROADMAP_CHECK]


def test_a_roadmap_item_without_a_mark_or_with_a_numbered_name_stops_the_build():
    text = site_sources.ROADMAP.read_text(encoding="utf-8")
    item = "- The GitHub Action. <!-- public: A GitHub Action that tests a model and attaches its label -->"
    assert item in text
    with pytest.raises(SystemExit, match="no <!-- public -->"):
        site_sources.roadmap(text.replace(item, "- The GitHub Action."))
    mark = "<!-- public: A minimal EEG/EMG pack -->"
    assert mark in text
    with pytest.raises(SystemExit, match="stage, step or task number"):
        site_sources.roadmap(text.replace(mark, "<!-- public: Step 7 EEG/EMG pack -->"))
    assert (
        "A minimal EEG/EMG pack" not in site_sources.roadmap(text.replace(mark, "<!-- internal -->"))["later"]
    )


def test_every_findings_paragraph_on_the_site_is_marked_and_checked(built):
    pages, files = built[2], built[3]
    quoted = {p: [k for k, _ in site_checks.QUOTE.findall(files[p])] for p in pages}
    assert quoted["index.html"] == ["headline"]
    assert quoted["why-labels/index.html"] == ["headline", "example", "prereg", "explore", "noise_blur"]
    assert all(not q for p, q in quoted.items() if p not in ("index.html", "why-labels/index.html"))


@pytest.mark.parametrize(
    "page, old, new",
    [
        ("index.html", "and fog hit some models too.", "and fog hit some models too!"),  # one character
        ("why-labels/index.html", "s5 (H21)", "s5 (H22)"),  # one digit
        ("why-labels/index.html", "which models were hit", "which model were hit"),  # one letter less
    ],
)
def test_a_findings_paragraph_one_character_off_the_readme_is_caught(built, page, old, new):
    files = dict(built[3])
    assert old in files[page]
    files[page] = files[page].replace(old, new, 1)
    problems = problems_of(built, files)[FINDINGS_CHECK]
    assert problems and problems[0].startswith(page) and "differs from the README at character" in problems[0]


def test_a_readme_findings_block_that_changed_is_caught(built):
    pages, files = built[2], built[3]
    readme = site_sources.README.read_text(encoding="utf-8")
    changed = readme.replace("2 of 9 models lost much more", "2 of 9 models lost far more", 1)
    assert changed != readme
    assert site_checks.check_findings(pages, files, readme) == []
    assert any("'prereg'" in p for p in site_checks.check_findings(pages, files, changed))


def test_a_findings_paragraph_without_its_marker_is_caught(built):
    files = dict(built[3])
    files["index.html"] = files["index.html"].replace(
        '<span class="quote" data-findings="headline">', "<span>"
    )
    assert any("shows no marked paragraph" in p for p in problems_of(built, files)[FINDINGS_CHECK])


def test_the_old_official_imagenet_c_wording_is_caught_on_the_site(built):
    old = (
        "The ImageNet-C conditions were made with the official corruption code on these test images, with "
        "fixed seeds and each model's own preprocessing, so they are not directly comparable to published "
        "ImageNet-C results."
    )
    files = dict(built[3])
    files["methods/index.html"] = files["methods/index.html"].replace("</article>", f"<p>{old}</p></article>")
    problems = problems_of(built, files)[OFFICIAL_CHECK]
    assert problems == [f'methods/index.html: "official" with ImageNet-C or corruption code: {old!r}']


def test_speed_times_are_written_as_a_reader_writes_them(built):
    speed = built[3][MNV3L].split('id="speed"')[1].split("</section>")[0]
    assert "Timed on <time" in speed and "3 October 2026, 07:55–08:01 UTC</time>" in speed
    assert "Timed between" not in speed
    assert site_labels.readable_window("2026-10-03T23:58:10+00:00", "2026-10-04T00:03:00+00:00") == (
        "3 October 2026, 23:58 – 4 October 2026, 00:03 UTC"
    )


def test_calibration_thresholds_are_collapsed(built):
    label = built[0]["mobilenet_v3_large"]
    unc = built[3][MNV3L].split('id="uncertainty"')[1].split("</section>")[0]
    shown, collapsed = unc.split("<summary>Calibration thresholds (conformal)</summary>")
    for b in label["builds"]:
        threshold = site_pages.f(label["details"]["conformal"][b["build_id"]]["threshold"], "thr")
        assert threshold not in shown and threshold in collapsed.split("</details>")[0]


def test_the_compare_table_says_it_scrolls_and_keeps_the_model_column(built):
    page = built[3]["compare/index.html"]
    assert '<p class="scroll-hint" aria-hidden="true">scroll →</p>' in page
    css = Path("web/assets/site.css").read_text(encoding="utf-8")
    assert ".compare-table tbody th, .compare-table thead th:first-child { position: sticky; left: 0;" in css


def test_an_identifier_not_on_the_page_list_is_caught(built):
    files = dict(built[3])
    files["methods/index.html"] = files["methods/index.html"].replace("commit a957652", "commit 1234567")
    assert problems_of(built, files)["numbers come from the labels and figures"]


CLAIMS_CHECK = (
    "claims register: every findings paragraph and every Why labels? paragraph has a complete entry"
)


def test_every_why_labels_paragraph_names_a_claim(built):
    page = built[3]["why-labels/index.html"]
    named = re.findall(r'<p\b[^>]*data-claim="([^"]+)"', page)
    assert named == [
        "site-finding",
        "why-example",
        "why-prereg",
        "why-explore",
        "why-noise-blur",
        "fig-uncertainty",
        "why-label-design",
        "imagenet-c-method",
    ]


def test_a_findings_paragraph_without_a_claims_entry_is_caught(built, tmp_path):
    labs, figures, pages, files = built
    claims = json.loads(site_build.CLAIMS.read_text(encoding="utf-8"))
    for c in claims["claims"]:
        if c.get("findings") == "prereg":
            del c["findings"]
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "claims.json").write_text(json.dumps(claims), encoding="utf-8")
    shutil.copy(Path("docs/related_work.md"), tmp_path / "docs" / "related_work.md")
    for script in (
        "scripts/45_readme_findings.py",
        "scripts/47_figures.py",
        "scripts/39_make_labels.py",
        "scripts/28_breadth_sweep.py",
    ):
        (tmp_path / script).parent.mkdir(exist_ok=True)
        (tmp_path / script).write_text("", encoding="utf-8")
    problems = site_checks.check_claims(files, tmp_path / "docs" / "claims.json")
    assert [p for p in problems if "findings paragraph" in p] == [
        "why-labels/index.html: findings paragraph 'prereg' has no claims-register entry"
    ]


def test_a_why_labels_paragraph_without_a_claim_is_caught(built):
    files = dict(built[3])
    files["why-labels/index.html"] = files["why-labels/index.html"].replace(
        '<p data-claim="why-label-design">', "<p>"
    )
    assert any("names no claims-register entry" in p for p in problems_of(built, files)[CLAIMS_CHECK])


def test_a_claim_citing_an_unverified_source_is_caught():
    claim = {k: "x" for k in site_checks.CLAIM_FIELDS} | {
        "id": "made-up",
        "prior_work": ["Someone, arXiv:9999.99999"],
    }
    verified = Path("docs/related_work.md").read_text(encoding="utf-8")
    assert site_checks.claim_problems(claim, Path("."), verified) == [
        "claim made-up: cites arXiv:9999.99999, which is not in docs/related_work.md"
    ]


def test_the_related_work_list_comes_from_the_tracked_doc(built):
    refs = site_sources.related_work()
    assert [c[0] for c in refs["cited"]] == ["xiao", "recti", "kasa", "hendrycks", "michaelis"]
    assert [c[0] for c in refs["also read"]] == ["karimov", "kasa2024", "mitchell"]
    page = built[3]["why-labels/index.html"]
    for key, authors, year, title, where, link in refs["cited"] + refs["also read"]:
        entry = (
            f'<li id="ref-{key}"><a href="{link}">{html.escape(title)}</a>. {html.escape(authors)}. {year} '
        )
        assert entry + f'<span class="muted">({html.escape(where)})</span>.</li>' in page
    assert "EfficientNet-Lite" not in page  # "Site: not shown"
    text = site_sources.RELATED_WORK.read_text(encoding="utf-8")
    broken = text.replace("   Site: key mitchell", "   Notes: key mitchell", 1)
    with pytest.raises(SystemExit, match="has no Site: line"):
        site_sources.related_work(broken)
