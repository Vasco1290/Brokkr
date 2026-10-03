"""Checks for brokkr_edge.fingerprint."""

import json
import re

from brokkr_edge.fingerprint import (
    ARM_FEATURES,
    X86_FEATURES,
    features_from_cpuinfo,
    machine_fingerprint,
    package_versions,
)


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
    versions = package_versions(("brokkr-edge", "definitely-not-a-real-package-xyz"))
    assert versions["brokkr-edge"] is not None
    assert versions["definitely-not-a-real-package-xyz"] is None


def test_parse_cpu_list():
    from brokkr_edge.fingerprint import parse_cpu_list
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


def test_cpu_features_are_recorded():
    features = machine_fingerprint()["cpu_features"]["features"]
    assert set(features) == set(X86_FEATURES + ARM_FEATURES)
    assert all(value in (True, False, None) for value in features.values())


def test_cpu_features_from_linux_cpuinfo():
    pi5 = "processor\t: 0\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics asimddp\n"
    arm = features_from_cpuinfo(pi5)
    assert arm["asimddp"] and not arm["i8mm"] and not arm["avx2"]
    laptop = "flags\t\t: fpu sse4_2 avx avx2 fma avx_vnni\n"
    x86 = features_from_cpuinfo(laptop)
    assert x86["avx2"] and x86["avx_vnni"] and not x86["avx512f"] and not x86["asimd"]
