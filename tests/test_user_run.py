"""Running a user's model under the conditions (brokkr_edge.user_run; docs/user_models.md, plan of 10 October
2026, tests 9, 12 and 14, and H's decision of 17 November 2026 on models below 32 x 32), on the made-up model
and pictures of tests/user_made_up.py. Nothing here is a result."""

import json
import sys

import numpy as np
import pytest
from user_made_up import CLASSES, PREP, tiny_model, write_images, write_settings

from brokkr_edge import user_run
from brokkr_edge.schema import check_record
from brokkr_edge.user_build import build_int8
from brokkr_edge.user_checks import run
from brokkr_edge.user_plan import check_user_inputs, rebuild_parts
from brokkr_edge.user_settings import UserInputError

PER_CLASS = 390
INT8 = "int8_percentile99.99"


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = tmp_path_factory.mktemp("user_run")
    write_images(d / "images", {c: PER_CLASS for c in CLASSES})
    tiny_model(d / "fp32.onnx")
    tiny_model(d / "small.onnx", size=16)
    tiny_model(d / "probabilities.onnx", probabilities=True)
    return d


def setup(made, out, **changes):
    plan = check_user_inputs(write_settings(made, **changes), made / "images", log=quiet)
    parts = rebuild_parts(plan, made / "images")
    return plan, parts


@pytest.fixture(scope="module")
def full(made, tmp_path_factory):
    """The made-up model, built and run under all 13 conditions."""
    out = tmp_path_factory.mktemp("out")
    plan, parts = setup(made, out)
    build_int8(plan, made / "fp32.onnx", parts, out, log=quiet)
    builds = {"fp32": made / "fp32.onnx", INT8: out / "int8.onnx"}
    paths = user_run.run_conditions(plan, builds, parts, out, log=quiet)
    return plan, parts, out, builds, paths


def records(paths) -> list:
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_every_condition_runs_for_both_builds(full):
    plan, _, _, _, paths = full
    assert len(plan["conditions"]) == 13
    assert len(paths) == 2 * (13 + 1)  # the test part under 13 conditions, and the clean calibration part
    assert all(p.exists() and p.with_suffix(".npz").exists() for p in paths)


def test_every_record_passes_the_checker_as_a_user_record(full):
    plan, parts, _, _, paths = full
    for r in records(paths):
        assert check_record(r) == []
        assert r["source"] == "user-submitted" and r["device"]["label"] == plan["device_kind"]
        assert r["runtime"]["threads"] == 8 and r["runtime"]["spinning"] == "off"  # decision 3
        assert r["settings"]["seed_scheme"] == "content-v1"
        assert r["data"]["n_images"] == len(parts[r["data"]["split"]])


def test_no_record_names_an_image_or_a_path(full, made):
    _, _, out, _, paths = full
    for p in paths:
        text = p.read_text(encoding="utf-8")
        assert ".png" not in text and str(made) not in text and str(out) not in text


def test_the_decoded_pictures_are_deleted(full):
    _, _, out, _, _ = full
    assert not list(out.glob(".brokkr-decoded-*"))
    assert sorted(x.name for x in out.iterdir()) == ["int8.onnx", "int8_build.json", "records"]


def test_a_finished_run_is_not_run_again(full, monkeypatch):
    """Test 14: a stopped run keeps its finished records and does not redo them."""
    plan, parts, out, builds, paths = full
    monkeypatch.setattr(user_run, "decode", lambda *a: pytest.fail("pictures decoded again"))
    assert user_run.run_conditions(plan, builds, parts, out, log=quiet) == paths


def test_a_stopped_run_redoes_only_what_is_missing(full, monkeypatch):
    plan, parts, out, builds, paths = full
    missing = next(p for p in paths if "test_brokkr_fog" in p.name and p.name.startswith(INT8))
    missing.with_suffix(".npz").unlink()
    done, real = [], user_run._save

    def counted(path, *args, **kwargs):
        done.append(path.name)
        real(path, *args, **kwargs)

    monkeypatch.setattr(user_run, "_save", counted)
    user_run.run_conditions(plan, builds, parts, out, log=quiet)
    assert done == [missing.name]


def test_a_crashed_run_leaves_no_decoded_pictures(made, tmp_path, monkeypatch):
    plan, parts = setup(made, tmp_path)
    (tmp_path / ".brokkr-decoded-test.tmp.npy").write_bytes(b"left by a crash")
    monkeypatch.setattr(user_run, "normalised", lambda *a: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        user_run.run_conditions(plan, {"fp32": made / "fp32.onnx"}, parts, tmp_path, log=quiet)
    assert not list(tmp_path.glob(".brokkr-decoded-*"))


def test_a_model_below_32_pixels_runs_five_conditions(made, tmp_path):
    """H, 17 November 2026, option (b): clean and Brokkr's four; the ImageNet-C conditions are not tested."""
    plan, parts = setup(made, tmp_path, fp32="small.onnx",
                        preprocessing={**PREP, "resize": 16, "crop": 16})
    assert [(c["suite"], c["corruption"]) for c in plan["conditions"]] == [
        (None, "clean"), ("brokkr", "fog"), ("brokkr", "darkness"), ("brokkr", "defocus_blur"),
        ("brokkr", "noise")]
    paths = user_run.run_conditions(plan, {"fp32": made / "small.onnx"}, parts, tmp_path, log=quiet)
    assert len(paths) == 5 + 1 and not any("imagenet-c" in p.name for p in paths)


def test_adding_an_image_changes_no_other_images_damage(made, tmp_path):
    """Test 9: content-v1 seeds. The folder gains one picture; every test image the two runs share gets the
    same damaged picture under the random conditions."""
    bigger = tmp_path / "images"
    write_images(bigger, {c: PER_CLASS for c in CLASSES})  # the same pictures (same seed) ...
    write_images(tmp_path / "extra", {"red": 1}, seed=123)
    (tmp_path / "extra" / "red" / "00000.png").replace(bigger / "red" / "zz_added.png")  # ... and one more
    one = rebuild_parts(check_user_inputs(write_settings(made), made / "images", log=quiet), made / "images")
    two = rebuild_parts(check_user_inputs(write_settings(made), bigger, log=quiet), bigger)
    first = {e["sha256"]: (i, e) for i, e in enumerate(one["test"])}
    shared = [(first[e["sha256"]][0], j) for j, e in enumerate(two["test"]) if e["sha256"] in first]
    assert len(shared) > 100
    pictures = [user_run.decode(p["test"], PREP, tmp_path / f"d{k}.npy") for k, p in enumerate((one, two))]
    for cond in ({"suite": "brokkr", "corruption": "fog", "severity": 3},
                 {"suite": "imagenet-c", "corruption": "gaussian_noise", "severity": 5}):
        for i, j in shared[:40]:
            a = user_run.damaged(pictures[0], one["test"], [i], cond)
            b = user_run.damaged(pictures[1], two["test"], [j], cond)
            assert np.array_equal(a, b), (cond, i, j)


def test_probabilities_are_saved_as_logits(made, tmp_path):
    """Test 12: softmax of the saved scores gives the model's own probabilities back."""
    plan, parts = setup(made, tmp_path, fp32="probabilities.onnx", outputs="probabilities")
    paths = user_run.run_conditions(plan, {"fp32": made / "probabilities.onnx"}, parts, tmp_path, log=quiet)
    clean = next(p for p in paths if "test_none_clean" in p.name)
    with np.load(clean.with_suffix(".npz")) as saved:
        logits = saved["logits"]
    from brokkr_edge.benchmark import make_session
    probs = run(make_session(made / "probabilities.onnx", 4), [e["path"] for e in parts["test"]], PREP)
    softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert np.allclose(softmax, probs, atol=1e-6)


def test_missing_imagenet_c_code_stops_before_anything_runs(made, tmp_path, monkeypatch):
    plan, parts = setup(made, tmp_path)
    monkeypatch.setitem(sys.modules, "brokkr_edge.imagenet_c", None)  # an import of it now fails
    with pytest.raises(UserInputError, match="imagenet-c"):
        user_run.run_conditions(plan, {"fp32": made / "fp32.onnx"}, parts, tmp_path, log=quiet)
    assert not list((tmp_path / "records").glob("*.json"))


def test_too_little_disk_space_stops_before_decoding(made, tmp_path, monkeypatch):
    plan, parts = setup(made, tmp_path)
    monkeypatch.setattr(user_run.shutil, "disk_usage", lambda _: type("U", (), {"free": 10})())
    with pytest.raises(UserInputError, match="GB free"):
        user_run.run_conditions(plan, {"fp32": made / "fp32.onnx"}, parts, tmp_path, log=quiet)
