"""Checks for brokkr.fingerprint."""

import json
import re

from brokkr.fingerprint import machine_fingerprint, package_versions


def test_fingerprint_has_required_fields():
    fp = machine_fingerprint()
    for key in ("timestamp_utc", "cpu_model", "cpu_count_logical", "os", "python_version",
                "packages", "git"):
        assert key in fp, f"missing {key}"


def test_fingerprint_is_json_serialisable():
    text = json.dumps(machine_fingerprint())
    assert json.loads(text)["os"] != ""


def test_cpu_model_is_a_real_name():
    cpu = machine_fingerprint()["cpu_model"]
    assert isinstance(cpu, str) and cpu not in ("", "unknown")


def test_git_commit_is_recorded():
    # Tests run inside the repo, so a full 40-character commit hash must be found.
    commit = machine_fingerprint()["git"]["commit"]
    assert commit is not None and re.fullmatch(r"[0-9a-f]{40}", commit)


def test_missing_package_is_none_not_an_error():
    versions = package_versions(("brokkr", "definitely-not-a-real-package-xyz"))
    assert versions["brokkr"] is not None
    assert versions["definitely-not-a-real-package-xyz"] is None
