"""Checks that the brokkr package is installed and importable."""

import brokkr


def test_package_imports_and_has_version():
    assert isinstance(brokkr.__version__, str)
    assert brokkr.__version__ != ""
