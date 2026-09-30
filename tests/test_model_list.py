"""The PyTorch-free model list: it loads without PyTorch and still matches torchvision."""

import subprocess
import sys

import pytest

from brokkr_edge.model_list import load_model_list


def test_loads_without_pytorch():
    code = ("import sys; from brokkr_edge.model_list import load_model_list; m = load_model_list(); "
            "assert 'torch' not in sys.modules and 'torchvision' not in sys.modules; print(len(m))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert int(out.stdout) == len(load_model_list())


def test_every_model_has_a_licence_and_preprocessing():
    for name, entry in load_model_list().items():
        assert entry["licence"]["code"] and entry["licence"]["weights"], name  # hard rule 8
        p = entry["input"]["preprocessing"]
        assert {"resize", "crop", "interpolation", "mean", "std"} <= set(p), name


def test_matches_torchvision_metadata():
    pytest.importorskip("torchvision")
    result = subprocess.run([sys.executable, "scripts/38_write_model_list.py", "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
