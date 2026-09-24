"""Save and load result records as JSON.

Every result file has the same outer shape, so the website and reports can read any of them:

    {
      "schema_version": 1,
      "kind": "speed",                 # what was measured
      "source": "brokkr",              # or "community-submitted" (hard rule 7)
      "model": ..., "precision": ...,
      "settings": {...}, "metrics": {...}, "raw": {...},
      "machine": {...}                 # from brokkr.fingerprint
    }
"""

import json
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED_KEYS = ("schema_version", "kind", "source", "model", "precision", "settings", "metrics",
                 "machine")


def make_record(kind: str, model: str, precision: str, measurement: dict, machine: dict,
                source: str = "brokkr") -> dict:
    """Wrap a measurement (settings/metrics/raw) with what, who, and where."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "source": source,
        "model": model,
        "precision": precision,
        **measurement,
        "machine": machine,
    }


def save_record(record: dict, path) -> Path:
    missing = [k for k in REQUIRED_KEYS if k not in record]
    if missing:
        raise ValueError(f"result record is missing {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path


def load_records(folder) -> list:
    """Load every .json result under `folder` (searching subfolders too)."""
    return [json.loads(p.read_text()) for p in sorted(Path(folder).rglob("*.json"))]
