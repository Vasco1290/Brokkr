"""Checks that the brokkr package is installed and importable."""

import brokkr_edge


def test_package_imports_and_has_version():
    assert isinstance(brokkr_edge.__version__, str)
    assert brokkr_edge.__version__ != ""
