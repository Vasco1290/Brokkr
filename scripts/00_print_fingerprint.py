"""Print this machine's fingerprint as JSON, so you can eyeball what gets recorded."""

import json

from brokkr_edge.fingerprint import machine_fingerprint

print(json.dumps(machine_fingerprint(), indent=2))
