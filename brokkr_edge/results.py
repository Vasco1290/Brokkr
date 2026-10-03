"""Save and load result records as JSON.

Every result file has the same outer shape, so the website and reports can read any of them:

    {
      "schema_version": 1,
      "kind": "speed",                 # what was measured
      "source": "brokkr",              # or "community-submitted" (hard rule 7)
      "model": ..., "precision": ...,
      "settings": {...}, "metrics": {...}, "raw": {...},
      "machine": {...}                 # from brokkr_edge.fingerprint
    }

Large arrays (e.g. every class score for every image) don't belong in JSON. save_arrays writes
them to a compressed .npz file next to the JSON and records that file's name and SHA-256
checksum in the record, so a missing, swapped, or edited array file is detected on loading.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

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


@contextmanager
def written_atomically(path):
    """Write to <path>.tmp, then rename it to <path> only if the writing finished.

    A crash (or power cut) mid-write leaves at most a .tmp file, never a half-written file under the
    real name, so resume logic that looks for the real name never mistakes a broken file for a done one.

        with written_atomically(out) as tmp:
            tmp.write_text(...)
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
        os.replace(tmp, path)  # atomic on one disk; replaces an old file (a plain rename can't, on Windows)
    finally:
        if tmp.exists():
            tmp.unlink()


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


# Brokkr's one file fingerprint (docs/label_schema.md, second note of 4 October 2026). Text files are hashed
# with Windows line endings (CRLF) turned into LF first, so the same text gives the same fingerprint on every
# OS and after any git checkout; every other file (scores, models, pictures) is hashed as raw bytes.
TEXT_SUFFIXES = frozenset({".json", ".md", ".html", ".txt", ".csv", ".svg", ".py", ".toml", ".yaml", ".yml"})


def sha256_of(path) -> str:
    """The file's fingerprint: SHA-256 of its bytes, with CRLF -> LF first if it is a text file."""
    data = Path(path).read_bytes()
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def save_arrays(record: dict, json_path, **arrays) -> Path:
    """Save arrays to <json_path stem>.npz and note the file and its checksum in `record`.

    Call this before save_record, so the checksum ends up in the saved JSON.
    """
    npz_path = Path(json_path).with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    record["raw_arrays"] = {
        "file": npz_path.name,
        "sha256": sha256_of(npz_path),
        "arrays": {name: {"shape": list(a.shape), "dtype": str(a.dtype)} for name, a in arrays.items()},
    }
    return npz_path


def load_arrays(json_path) -> dict:
    """Load the arrays belonging to a saved result, after checking the checksum."""
    json_path = Path(json_path)
    info = json.loads(json_path.read_text())["raw_arrays"]
    npz_path = json_path.with_name(info["file"])
    if sha256_of(npz_path) != info["sha256"]:
        raise ValueError(f"{npz_path} does not match the checksum in {json_path}")
    with np.load(npz_path) as data:
        return {name: data[name] for name in data.files}
