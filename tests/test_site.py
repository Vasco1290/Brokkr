"""The website generator (scripts/49_build_site.py, web/site_*.py): a fresh build passes every site check,
and each check fails when the thing it guards is broken (made-up changes, tests only). Needs only committed
files (published/labels/, docs/figures/), so it runs on a fresh clone."""

import copy
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("web").resolve()))
import site_build  # noqa: E402  (web/site_build.py)
import site_checks  # noqa: E402
import site_labels  # noqa: E402
import site_pages  # noqa: E402

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


def test_the_pages_of_slice_a_exist(built):
    labs, _, pages, _ = built
    assert {"index.html", "models/index.html", MNV3L, MNV3S} <= set(pages)
    assert len([p for p in pages if p.startswith("models/") and p.count("/") == 3]) == len(labs)


def test_a_new_page_type_changes_no_existing_page_except_the_navigation(built):
    labs, figures, pages, files = built

    def made_up(site):  # tests only
        return [("made-up/index.html", site_pages.Page("Made up", "<p>Made up.</p>", labels=[]))]

    types = [*site_pages.PAGE_TYPES, site_pages.PageType("made-up", "Made up", "made-up/index.html", made_up)]
    pages2, files2 = site_build.build(labs, figures, COMMIT, types)
    assert set(pages2) == set(pages) | {"made-up/index.html"}

    def without_nav(text):
        return re.sub(r'<nav aria-label="Main">.*?</nav>', "", text, flags=re.S)

    for path in pages:
        assert without_nav(files2[path]) == without_nav(files[path]), path
        assert 'href="' in files2[path] and "Made up</a>" in files2[path], path  # the nav shows the new page


def test_the_navigation_shows_only_pages_that_exist(built):
    nav = re.search(r'<nav aria-label="Main">(.*?)</nav>', built[3]["index.html"], re.S).group(1)
    assert re.findall(r">([^<]+)</a>", nav) == ["Catalog", "GitHub"]
    assert "why-labels" not in built[3]["index.html"]  # slice B


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
