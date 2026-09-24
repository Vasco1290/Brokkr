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


def test_power_state_has_valid_values():
    power = machine_fingerprint()["power"]
    assert power["on_ac_power"] in (True, False, None)
    assert power["battery_percent"] is None or 0 <= power["battery_percent"] <= 100


def test_missing_package_is_none_not_an_error():
    versions = package_versions(("brokkr", "definitely-not-a-real-package-xyz"))
    assert versions["brokkr"] is not None
    assert versions["definitely-not-a-real-package-xyz"] is None


def test_parse_cpu_list():
    from brokkr.fingerprint import parse_cpu_list
    assert parse_cpu_list("0-3,8\n") == [0, 1, 2, 3, 8]
    assert parse_cpu_list("5") == [5]


def test_core_types_cover_every_cpu_once():
    import os
    types = machine_fingerprint()["core_types"]
    if types is None:  # OS couldn't tell us; nothing more to check
        return
    all_ids = types["performance"] + types["efficiency"]
    assert types["performance"], "there must be at least one performance core"
    assert sorted(all_ids) == list(range(os.cpu_count()))
