"""`brokkr-edge test --config`, from start to label (brokkr_edge.user_command, brokkr_edge.cli;
docs/user_models.md, plan of 10 October 2026, test 13 and decision 13), on the made-up model and pictures of
tests/user_made_up.py. Nothing here is a result."""

import json
import socket
import subprocess
import sys

import pytest
from user_made_up import CLASSES, ROLLED, quantized, tiny_model, write_images, write_settings

from brokkr_edge import cli, user_command
from brokkr_edge.label_schema import check_label
from brokkr_edge.user_label import LabelError
from brokkr_edge.user_plan import REPOSITORY
from brokkr_edge.user_settings import UserInputError

PER_CLASS = 390


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    d = tmp_path_factory.mktemp("user_command")
    write_images(d / "images", {c: PER_CLASS for c in CLASSES})
    tiny_model(d / "fp32.onnx")
    tiny_model(d / "other.onnx", weights=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])  # another model, same answers
    quantized(tiny_model(d / "rolled.onnx", weights=ROLLED), d / "int8_rolled.onnx")  # agrees on no image
    return d


def command(*args):
    return subprocess.run([sys.executable, "-m", "brokkr_edge.cli", "test", *map(str, args)],
                          capture_output=True, text=True, cwd=REPOSITORY)


@pytest.fixture(scope="module")
def finished(made, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    run = command("--config", write_settings(made), "--images", made / "images", "--out", out)
    return out, run


def test_the_command_makes_a_label_from_start_to_end(finished):
    out, run = finished
    assert run.returncode == 0, run.stderr
    last = run.stdout.splitlines()[-1]
    assert last == ("Label made (user-submitted, UNVERIFIED): FP32 and INT8 under 12 damaged conditions and "
                    "clean images.")
    assert "(0 connection attempts)" in run.stdout and f"Label: {out / 'label.html'}" in run.stdout
    label = json.loads((out / "label.json").read_text(encoding="utf-8"))
    assert check_label(label) == []
    names = sorted(p.name for p in out.iterdir())
    assert names == ["int8.onnx", "int8_build.json", "label.html", "label.json", "label.md", "records",
                     "run_plan.json"]  # decision 13: nothing else, no decoded pictures


def test_the_same_inputs_carry_the_run_on(made, finished):
    out, _ = finished
    again = command("--config", write_settings(made), "--images", made / "images", "--out", out)
    assert again.returncode == 0, again.stderr
    assert "INT8 build: reused" in again.stdout and "skip (complete)" in again.stdout


def test_an_out_folder_with_a_different_run_is_refused(made, finished):
    """Decision 13."""
    out, _ = finished
    settings = write_settings(made, fp32="other.onnx")
    other = command("--config", settings, "--images", made / "images", "--out", out)
    assert other.returncode == 2 and "already holds a different run" in other.stderr


def test_a_failed_supplied_build_says_so_on_the_last_line(made, tmp_path):
    shrunk = {"file": "int8_rolled.onnx", "precision": "int8", "made_by": "made-up test build"}
    run = command("--config", write_settings(made, shrunk_model=shrunk), "--images", made / "images",
                  "--out", tmp_path)
    assert run.returncode == 0, run.stderr
    last = run.stdout.splitlines()[-1]
    assert "the supplied INT8 build FAILED" in last and "FP32 numbers only" in last
    assert not list((tmp_path / "records").glob("int8*"))  # no INT8 run at all


def test_13_the_network_stays_blocked_for_every_step(made, tmp_path, monkeypatch):
    """A connection attempted while the conditions run (not only during the checks) is blocked and stops
    the command, even when the code that tried it hides the error."""
    real = user_command.run_conditions

    def tries_the_network(*args, **kwargs):
        try:
            socket.create_connection(("example.org", 80), timeout=1)
        except OSError:
            pass  # caught and hidden: still counted
        return real(*args, **kwargs)

    monkeypatch.setattr(user_command, "run_conditions", tries_the_network)
    with pytest.raises(RuntimeError, match="network connections were attempted"):
        user_command.run_user_test(write_settings(made), made / "images", None, tmp_path, log=quiet)


def test_a_label_that_cannot_be_made_stops_with_exit_code_3(made, tmp_path, monkeypatch, capsys):
    def refuses(*_):
        raise LabelError("records/x.json fails its check")

    monkeypatch.setattr(user_command, "make_label", refuses)
    code = cli.main(["test", "--config", str(write_settings(made)), "--images", str(made / "images"),
                     "--out", str(tmp_path)])
    assert code == 3 and "the label could not be made" in capsys.readouterr().err


def test_inputs_that_cannot_be_used_stop_before_anything_is_written(made, tmp_path):
    with pytest.raises(UserInputError):
        user_command.run_user_test(write_settings(made, licences={}), made / "images", None, tmp_path / "o",
                                   log=quiet)
    assert not (tmp_path / "o").exists()
