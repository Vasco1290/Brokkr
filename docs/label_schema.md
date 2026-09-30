# The Brokkr label format (`label.json`), schema version 1

*Written 30 September 2026, before any label is generated; `checks`, `generated`'s last three
fields and speed rows' `hardware_kind` added the same day, while building, before any label was made. The decisions it relies on are in
`docs/hypotheses_stage4.md` (notes of 29–30 September 2026). A validator for this document is part
of step 1 of the platform plan (`ROADMAP.md`).*

A label describes **one shrunk build of one model**, with its full-precision reference beside it: how
big it is, how accurate it is, whether it still knows when it is wrong, how that holds up under
damage, and how fast it runs on named hardware. `label.json` is the only source of a label: the
Markdown model card, the HTML page and the catalog are generated from it and compute nothing.

## Rules every label follows

1. **Every number comes from a checked result file.** Each measured value carries `sources`: the file,
   its SHA-256 and the field it was read from. The label check script recomputes every value from its
   sources and fails on any difference. Nothing is typed in by hand.
2. **Nothing is image-only.** What the model takes in, what the data is and what damage means are
   described by a `modality` and fields that belong to it. Images and signals (EEG/EMG) use the same
   structure; only the modality-specific blocks differ.
3. **Hardware is named, never implied.** Every speed number, and every accuracy number, points to the
   hardware it was measured on. "Raspberry Pi 5" only for a measured Pi 5; cloud ARM stays "cloud ARM".
4. **Who made it is always stated.** `source.kind` is `official` (made by Brokkr from its own runs) or
   `user-submitted`; a user-submitted label is always shown as **unverified**, and the two are never
   mixed without that mark.
5. **Missing is explicit.** Something not measured is present with `"status": "not measured"` (and a
   reason), never left out and never estimated.
6. **Values are stored unrounded.** Rounding happens only when a page is rendered.

## Versioning

- `schema_version` is a whole number. It changes only when a reader written for the old version could
  misread a new label (a field removed, renamed or given a new meaning). Adding an optional field does
  not change it.
- A reader refuses a label whose `schema_version` it does not know, rather than guessing.
- This document is the definition of each version; old versions stay in the document's history.

## Stable IDs

IDs are lowercase, made only of letters, digits, `-`, `_`, `.`, `/`, `:` and `@`, and are built from
facts that do not change when a label is regenerated. The same thing always gets the same ID.

| ID | Built from | Pattern |
|---|---|---|
| `model_id` | publisher, model name, weights name | `<publisher>/<model>@<weights>` |
| `build_id` | model ID, precision, recipe, first 12 hex characters of the model file's SHA-256 | `<model_id>#<precision>:<sha12>` |
| `hardware_id` | device kind, CPU model, operating system | `<device-kind>:<cpu-slug>:<os-slug>` |
| `dataset_id` | publisher, dataset, split | `<publisher>/<dataset>:<split>` |
| `condition_id` | suite, damage type, severity | `<suite>/<damage>/<severity>`, and `clean` |
| `label_id` | the build that is labelled | equal to its `build_id` |

The file hash in `build_id` means a rebuilt file with different bytes is a different build, and gets
its own label. `device-kind` is one of `laptop`, `raspberry-pi-5`, `cloud-arm`, or another kind added
to the schema's list.

## Top level

```
{
  "schema_version": 1,
  "label_id": "<build_id of the labelled build>",
  "source": {...},
  "generated": {...},
  "model": {...},
  "builds": [...],
  "hardware": [...],
  "datasets": [...],
  "conditions": [...],
  "checks": {...},
  "measurements": [...],
  "envelope": {...},
  "summary": {...},
  "speed": [...],
  "details": {...},
  "limits": [...],
  "sources": [...]
}
```

### `source`
```
{"kind": "official" | "user-submitted",
 "verified": true | false,          # true only for "official"
 "submitted_by": null | "<name the submitter chose>",
 "how_made": "brokkr test <version>" | "brokkr catalog <version>" | "hosted run <version>"}
```

### `generated`
Which code made the label (`by`: package and version), git `commit` and `dirty` flag (an official label
must be clean), the UTC time, `ci_level` (the interval level, 0.95), `label_licence` (CC BY 4.0) and
`envelope_rule` (the name of the rule used).

### `checks`
Checks the label depends on, each with its numbers and sources. Version 1 has `fp32_sanity`: the
reference build's clean top-1 (`measured`), the publisher's figure (`published`, with where it comes
from), the `tolerance`, and `pass`. A label whose check fails is not made.

### `model`
```
{"model_id": "...", "name": "...", "publisher": "...", "weights": "...",
 "task": "classification",
 "modality": "image" | "signal",
 "input": {...},                    # modality-specific, below
 "outputs": {"n_classes": <int>, "class_set": "<dataset_id or name of the class list>"},
 "licence": {"code": "...", "weights": "..."}}
```
`input` for `"image"`: shape (channels, height, width), and the preprocessing (resize, crop,
interpolation, mean, std). For `"signal"` (EEG/EMG): channel names and count, sampling rate in Hz,
window length in seconds, units, and any filtering applied before the model. A label for a model with
no licence recorded cannot be made (hard rule 8).

### `builds`
One entry for the reference (full precision) and one for the labelled build:
```
{"build_id": "...", "role": "reference" | "labelled",
 "precision": "fp32" | "fp16" | "int8",
 "recipe": {"method": "...", "calibration": {"dataset_id": "...", "n_items": <int>}, ...,
            "kept_float": null | {...}},
 "file": {"sha256": "...", "size_bytes": <int>},
 "status": "usable" | "failed",
 "failure": null | {"check": "<which build check failed>", "value": <number>, "limit": <number>,
                    "sources": [...]}}
```
A labelled build whose status is `failed` has no measurements of its own; every row that would use it
shows the state **"INT8 build failed"** with `failure`'s check, value and limit (MobileNetV3-Small).

### `hardware`
```
{"hardware_id": "...", "kind": "laptop" | "raspberry-pi-5" | "cloud-arm" | ...,
 "cpu_model": "...", "os": "...", "architecture": "...",
 "cores": {"physical_performance": ..., "hardware_threads_per_core": ..., "efficiency": ...,
           "map": {<logical cpu>: <physical core>}},
 "features": {"avx2": ..., "avx512_vnni": ..., "avx_vnni": ..., "asimddp": ..., ...},
 "runtime": {"name": "onnxruntime", "version": "...", "execution_provider": "..."},
 "fingerprint_source": {...}}       # file and field it was read from
```
A feature the machine could not report is `null` and shown as "unknown".

### `datasets`
```
{"dataset_id": "...", "name": "...", "split": "...", "n_items": <int>,
 "item_kind": "image" | "signal-window", "licence": "...",
 "disjoint_from": ["<dataset_id>", ...]}   # e.g. the calibration split is disjoint from test
```

### `conditions`
```
{"condition_id": "...", "suite": "brokkr" | "imagenet-c" | "<signal suite>" | ...,
 "damage": "clean" | "fog" | "darkness" | "electrode-dropout" | "motion-noise" |
           "power-line-noise" | ...,
 "severity": <int or null>, "modality": "image" | "signal",
 "label": "<human name, always naming its suite>", "description": "..."}
```
Damage types belong to a modality; the schema holds a list of the damage types and suites Brokkr
knows. A condition that was not run is not in this list; the envelope shows it as "not tested".

### `measurements`
One entry per measured number. The same shape serves every modality and every device:
```
{"metric": "top1" | "coverage" | "mean_set_size" | "ece" | "e_aurc" | ...,
 "build_id": "...", "dataset_id": "...", "condition_id": "...", "hardware_id": "...",
 "value": <number>, "ci95": [<low>, <high>] | null, "n_items": <int>, "unit": "fraction" | ...,
 "settings": {...},                 # e.g. the conformal target and threshold, threads, batch size
 "sources": [{"file": "...", "sha256": "...", "field": "..."}]}
```
Derived comparisons are also measurements, with their own metric names and a `paired_with` field:
- `damage_drop`: top-1 under the condition minus clean top-1, same build, same items, paired interval;
- `shrinking_cost`: labelled build minus reference, same condition, same items, paired interval.
`coverage` records its target (e.g. the 90% target) and the calibration dataset in `settings`, and a
`coverage` entry is never rendered without its `mean_set_size` entry.

### `envelope`
The reliability envelope, per build and condition, from the rule named in `generated`:
```
{"rule": "<name of the rule and the note that fixes it>",
 "thresholds": {"coverage_min": ..., "damage_drop_min_points": ...},
 "rows": [{"build_id": "...", "condition_id": "...", "hardware_id": "...",
           "state": "not harmful" | "harmful" | "borderline" | "not tested" | "INT8 build failed",
           "why": ["<which interval cleared or failed which threshold>"],
           "shrinking_cost_flag": null | "large shrinking cost" | "not informative"}]}
```
The thresholds and states are data here, so another modality or a later rule can set its own, stated
in its own dated note.

### `summary`
The block at the top of every rendered label, for the labelled build: one short line per state, each
listing its conditions ("Not harmful in our tests: ...", "Harmful: ...", "Borderline: ...",
"Not tested: ..."). Generated from `envelope`, never written by hand.

### `speed`
One entry per build, hardware and thread count:
```
{"build_id": "...", "hardware_id": "...",
 "status": "measured" | "not measured",
 "reason": null | "<why not>",
 "settings": {"batch": ..., "threads": ..., "intra_op_threads": ..., "inter_op_threads": ...,
              "graph_optimisation": "...", "pinned_cpus": [...], "sessions": ...,
              "warmup_runs": ..., "timed_runs": ..., "input": "random, seed 0",
              "what_is_timed": "model only; excludes loading and pre-processing"},
 "p50_ms": ..., "p95_ms": ..., "p99_ms": ..., "spread_pct": ..., "unstable": true | false,
 "sources": [...]}
```
A Raspberry Pi 5 row exists from the start with `"status": "not measured"` until one is measured. Every
row also names `hardware_kind`; a row for a device Brokkr has not measured yet has `"hardware_id": null`
(no machine to name) and a `reason`.

### `details`
What the main label leaves out: E-AURC, ECE, the conformal thresholds, top-5, the reference build's
own envelope and the full per-condition numbers.

### `limits`
Plain sentences, generated from the facts in the label (e.g. that damage is simulated, that the
ImageNet-C numbers are not directly comparable to published ImageNet-C results, which machine was
used, the "12 conditions" sentence). No limit is written by hand for one label.

### `sources`
Every file the label was made from, with its SHA-256, and for each the check it passed (e.g.
`scripts/22_check_results.py`).

## What renders from what

| Output | Made from | Rule |
|---|---|---|
| Markdown model card | `label.json` only | Hugging Face card metadata at the top; a test fails if any number in it is not in `label.json` |
| HTML page | `label.json` only | the same test |
| Catalog pages, filters, comparisons | the `label.json` files only | the same test; `user-submitted` shows "unverified" everywhere it appears |
