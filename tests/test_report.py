"""Checks for brokkr.report and the paired accuracy difference."""

import json

import numpy as np
import pytest

from brokkr.accuracy import paired_bootstrap_diff
from brokkr.fingerprint import machine_fingerprint
from brokkr.report import accuracy_rows, build_site, render_html


def accuracy_record(precision, predictions, labels, source="brokkr"):
    correct = np.mean(np.array(predictions) == np.array(labels))
    return {
        "schema_version": 1, "kind": "accuracy", "source": source, "model": "m", "precision": precision,
        "settings": {"dataset": "d", "n_images": len(labels), "dataset_licence": "test licence"},
        "metrics": {"top1": correct, "top1_ci95": [correct - 0.1, correct + 0.1], "top5": correct},
        "raw": {"labels": labels, "top5_predictions": [[p, 0, 0, 0, 0] for p in predictions]},
        "machine": machine_fingerprint(),
    }


def test_paired_diff_is_exact_mean_difference():
    a = np.array([1, 1, 1, 0, 0, 1, 0, 1.0])
    b = np.array([1, 0, 1, 0, 0, 1, 0, 0.0])
    diff, lo, hi = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(-0.25)
    assert lo <= diff <= hi


def test_paired_diff_refuses_different_image_sets():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(np.ones(5), np.ones(6))


def test_accuracy_rows_compare_against_fp32_on_same_images():
    labels = [1, 2, 3, 4]
    rows = accuracy_rows([accuracy_record("fp32", [1, 2, 3, 4], labels),
                          accuracy_record("int8", [1, 2, 0, 0], labels)])
    assert [r["precision"] for r in rows] == ["fp32", "int8"]
    assert rows[0]["diff_vs_fp32"] is None
    assert rows[1]["diff_vs_fp32"][0] == pytest.approx(-0.5)


def test_empty_page_says_no_results_instead_of_placeholders():
    page = render_html([], [], [], [], {})
    assert "No results yet" in page


def test_build_site_escapes_text_and_ignores_community_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    ours = accuracy_record("fp32", [1, 2], [1, 2])
    ours["settings"]["dataset"] = "<script>"
    community = accuracy_record("fp16", [1, 2], [1, 2], source="community-submitted")
    (results / "a.json").write_text(json.dumps(ours))
    (results / "b.json").write_text(json.dumps(community))

    page = build_site(results, tmp_path / "no_models", tmp_path / "site").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in page and "<script>" not in page
    assert "fp16" not in page  # community results are not mixed in (hard rule 7)
