"""Shared test setup.

Brokkr's seed-sensitivity record (results/checks/seed_sensitivity.json) is gitignored, so a fresh clone does
not have it, and a user label cannot be made without it. Every test therefore uses a made-up stand-in
(tests/user_made_up.py), never the file on this machine; tests/test_user_label.py reads the real one only when
it is present.
"""

import pytest
from user_made_up import made_up_seed_sensitivity


@pytest.fixture(scope="session", autouse=True)
def made_up_seed_record(tmp_path_factory):
    from brokkr_edge import user_label

    path = made_up_seed_sensitivity(tmp_path_factory.mktemp("seed") / "seed_sensitivity.json")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_label, "SEED_SENSITIVITY", path)
        yield path
