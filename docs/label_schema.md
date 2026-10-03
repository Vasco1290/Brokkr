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
"Not tested: ..."). Generated from `envelope`, never written by hand. *(The groups were replaced
on 30 September 2026: see the note at the end of this document.)*

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

## Note added 30 September 2026, after H's Checkpoint 1 review, before the two labels are regenerated

Decided by H (Checkpoint 1 decisions 1–6). Details marked "(proposed)" are Claude's and await H's
confirmation at the next review. Nothing above is edited except a pointer in `summary`.

**Version.** Schema version 1 is amended, not raised to 2. No label has been published: the two
Checkpoint 1 labels were never released, are gitignored, and are regenerated under this note. No
threshold, envelope state or envelope rule changes.

**1. Display names** (decision 2). Two optional fields, both copied from `brokkr_edge/model_list.json`,
which takes them from `brokkr_edge/export.py` (models) and `brokkr_edge/quantize.py` (INT8 recipes),
where they are typed once:
- `model.display_name`, e.g. "MobileNetV3-Large";
- each build's `display_name`, e.g. "FP32 (full precision)", "INT8 (percentile calibration, 99.99%)".

Every title (page heading, HTML page title, builds table) uses them. The internal names
(`mobilenet_v3_large`, `percentile99.99`) stay in IDs and in `recipe`. A label without them renders
with `model.name` and `precision`.

**2. Summary groups** (decision 3). The summary still describes the labelled build and is still
generated from `envelope`. Each tested condition goes into exactly one group, read from the labelled
build's envelope state and the reference build's state in the same condition:

| Labelled build (INT8) | Reference build (FP32) | Group |
|---|---|---|
| not harmful | any | fine |
| harmful | harmful | too hard for this model |
| harmful | not harmful | hurt by shrinking |
| harmful | borderline | borderline (proposed) |
| borderline | any | borderline |
| INT8 build failed | any | INT8 build failed |

Then "not tested", as before. `summary.lines` entries become `{"group", "count", "conditions"}`
(`group` replaces `state`). The rendered group titles are "Fine (not harmful in our tests)", "Too hard
for this model (FP32 also fails)", "Hurt by shrinking (FP32 copes, INT8 doesn't)" (H's wording), and
"Borderline (too close to a line to call)", "INT8 build failed", "Not tested".
- (proposed) **Harmful INT8 with borderline FP32 goes to "borderline".** FP32 neither clearly copes nor
  clearly fails there, so neither of H's two groups fits, and putting the condition in either would
  claim more than the intervals show. The rendered line names both states for such a condition.
- (proposed) **"Fine" does not depend on FP32**: the summary answers "can I use this INT8 build here?".
- The shrinking-cost column is unchanged, so a condition in "too hard for this model" can still carry
  "large shrinking cost".
- Written after the two Checkpoint 1 labels were seen. It groups existing states only; no threshold or
  state is chosen here.

**3. A failed labelled build** (decision 4). The rendered label opens with one plain sentence, made
from the labelled build's `failure`: that this recipe broke the model and the INT8 build should not be
used, with the check's value, its limit and the number of images it was measured on. `failure` gains
`n_items` and `split` (both read from the build record, with their source).

**4. Explanations** (decision 5). Every rendered label has a section "What the words mean", one plain
line per technical term it uses; in HTML, table headings and summary group titles also carry the same
line as a hover note. The lines are fixed text in `brokkr_edge/label_render.py`, not label data, and
hold no number of their own (so the "every number is in `label.json`" check still covers them).

**5. The full `label.json` stays the only source** (decision 6). Step 2's catalog makes a slim version
from it; the slim version is generated, never edited, and is not a second source.

**6. Command name** (decision 1). The command is `brokkr-edge` (the other PyPI project "brokkr"
installs a `brokkr` command). A label made by the command says
`"how_made": "brokkr-edge test <version>"`. Labels made from the 4.1 records keep
`"scripts/39_make_labels.py, from the 4.1 records"`.

## Note added 30 September 2026, after H's review of the regenerated Checkpoint 1 labels, before any code

Decided by H (review notes 1–4). Where this differs from the note above, it replaces it. Schema
version 1 stays, amended (confirmed by H); no threshold, envelope state or envelope rule changes.

**1. Two-level summary** (replaces the four groups of the note above). The summary describes the shrunk
build, so it groups first by the labelled build's own envelope state, then, for harmful conditions
only, by the cause:
- **not harmful in our tests**; **borderline**; **harmful**; **INT8 build failed**; **not tested**.
- Cause of a harmful condition, from the reference (FP32) build's state in the same condition:
  FP32 harmful → **too hard for this model** (FP32 also fails); FP32 not harmful → **hurt by
  shrinking** (FP32 copes, INT8 doesn't); FP32 borderline → **cause unclear** (FP32 is borderline).
- `summary.lines` entries: `{"state", "cause", "count", "conditions"}`, where `cause` is `null` except
  on harmful lines, and `conditions` lists condition IDs (the renderer shows each condition's `label`).

**2. A large shrinking cost is never hidden.** Every condition whose envelope row carries "large
shrinking cost" is listed in `summary.large_shrinking_cost` (condition IDs, in row order), whatever
its group, and the rendered summary names each one with its shrinking cost in points, e.g. under
"Shrinking made it much worse". A test fails if a flagged condition is missing from the summary.

**3. Which line a harmful row failed.** Each envelope row gains `failed`: the lines whose whole interval
is below the line, a subset of `["damage drop", "coverage"]` in that order (empty unless the state is
harmful). Rendered as "accuracy dropped" (damage drop), "uncertainty signal unreliable (coverage below
<the coverage line>)" (coverage), or both; in the damage table's envelope cell and beside each harmful
condition in the summary. The validator checks that harmful rows, and only they, have a non-empty
`failed`; scripts/40 recomputes it.

*Implementation detail, added before the labels are regenerated:* each harmful summary line also holds
`by_failed`, a list of `{"failed", "count", "conditions"}` (accuracy only, coverage only, both, in that
order, empty ones left out), so every count the rendered summary shows is a `label.json` number.

## Note added 1 October 2026: three lines at the top of the summary, and a suggested next step (before any code)

Decided by H (review item 6, confirmed 1 October 2026). Details marked "(proposed)" are Claude's and
await H's confirmation. Schema version 1 stays, amended; no threshold, state or rule changes.

**1. Three lines open every label's summary, before the groups.** Each number is a `label.json`
number.
- **Shrinking:** the shrinking cost on clean images with its interval (the `shrinking_cost`
  measurement for `clean`), then how many tested conditions carry "large shrinking cost", each named
  with its shrinking cost in points. This line replaces the separate "Shrinking made it much worse"
  line of the note above: every flagged condition is still named in the summary, whatever its group,
  and the test that checks it stays.
- **Harsh conditions:** how many tested conditions are harmful for the reference (FP32) build itself
  ("even at full size"), each named (the name carries the damage type and severity).
- **Uncertainty signal:** how many tested conditions fail the coverage line for the labelled build
  (the whole coverage interval is below the coverage line, i.e. `failed` contains "coverage"), each
  named, with the plain note that the prediction sets were calibrated on clean images and can be
  re-calibrated on the user's own images.
- In `label.json`: `summary.large_shrinking_cost`, `summary.reference_harmful` and
  `summary.coverage_failed`, each `{"count", "conditions"}` (condition IDs, in row order);
  `large_shrinking_cost` changes from a plain list to this shape. The validator checks all three
  against the envelope rows; `scripts/40` recomputes them.
- A label whose labelled build failed shows "not measured: INT8 build failed" on the Shrinking and
  Uncertainty signal lines; the Harsh conditions line is shown as usual (it is about FP32).

**2. A suggested next step on each harmful row** of the labelled build, in the damage table (a new
last column) — fixed text in the renderer, chosen from the row's cause and failed lines, not label
data:
- hurt by shrinking → "try another recipe or model";
- too hard for this model → "consider a stronger model";
- coverage below the line (alone or with either cause) → also "re-calibrate on your own images";
- (proposed) cause unclear (FP32 is borderline) → "try another recipe or a stronger model".

Rows that are not harmful, and rows of a failed build, show no suggestion.
- (proposed) The label says once, under the table and in `limits`, that these are general
  suggestions and were not tested for this model, so a suggestion is never read as a result.

*Confirmed by H on 2 October 2026:* both points marked "(proposed)" in the note above: "try another
recipe or a stronger model" for cause-unclear rows, and the sentence saying the suggested next steps
are general suggestions that were not tested for the model (under the table and in `limits`).

## Note added 2 October 2026: the README's label example is generated from `label.json`

Asked by H on 2 October 2026, so the README cannot drift from the labels. The block between the
`label-example` markers in `README.md` is written by `scripts/41_readme_label_example.py` from two
labels (`mobilenet_v3_large`, and `mobilenet_v3_small` for the "do not use" sentence) with
`brokkr_edge.label_render.readme_example`, which, like the other renderers, reads only `label.json`
and formats every number with `fmt()`. `scripts/40_check_labels.py` fails if the README's block
differs from a fresh render, or if a number in it is not a `label.json` number; a test does the same
whenever the labels are on the machine. Nobody edits the block by hand.

*Added 2 October 2026, before the ten labels are generated:* a build's `recipe` also records
`skip_symbolic_shape` (true for ConvNeXt-Tiny's INT8, false otherwise), copied from its build record
and checked by `scripts/40`; when true, the label's Details say so. A non-default build setting is
never left off a label.

## Note added 3 October 2026: runtimes, both shrinking-cost flags, laptop speed and licences (before any code)

Decided by H on 3 October 2026 (decisions in `docs/hypotheses_stage4.md`, note of 3 October 2026).
Schema version 1 is amended, not raised: no label has been published. No threshold, state or envelope
rule changes. The field details are Claude's (proposed).

**1. Runtime has its own field on every row** (so one device can have several runtimes).
`hardware[].runtime` is removed. A new top-level list `runtimes` holds one entry per runtime set-up:
```
{"runtime_id": "<name>@<version>:<provider slug>:<threads>t:spin-<on|off>",
 "name": "onnxruntime", "version": "...", "execution_provider": "...",
 "threads": <int>, "intra_op_threads": <int or null>, "inter_op_threads": <int or null>,
 "spinning": "<as the record states it>", "graph_optimisation": "<as recorded, or null>",
 "sources": [...]}
```
Every measurement and every measured speed row carries `runtime_id`; a "not measured" speed row has
`"runtime_id": null`. The validator checks that each `runtime_id` names an entry. (Today's labels:
one runtime for the accuracy records, and one each for the 1- and 4-thread latency records.)

**2. Both shrinking-cost flags.** `envelope.rows[].shrinking_cost_flag` (one value) is replaced by
`shrinking_cost_flags`: a list, in the order `["large shrinking cost", "not informative"]`, holding
each flag that applies (empty when neither does). `summary.large_shrinking_cost` counts every row whose
list holds "large shrinking cost".

**3. Laptop speed rows.** A measured row adds, copied from its latency record: `spread_pct`, `unstable`,
`unstable_above_pct`, `sessions`, `warmup_runs`, `timed_runs`, `discarded_sessions` (a count),
`timed_utc` (`first_start`, `last_end`: the first timed run's start and the last one's end, from the
record's `.npz`), `pinning` (`logical_cpus`, `physical_cores`, `core_kind`, `read_back`: false) and
`vnni` (the record's text). The labelled build's measured rows also hold `time_vs_reference`:
`{"ratio_p50": <INT8 p50 / FP32 p50>, "reference_p50_ms": ..., "slower": <ratio above 1>, "sources":
[...]}`. Every number keeps its `sources`.

**4. Licences.** A new top-level field `licences`, generated from the build records (decision 6 of
the 3 October note): `brokkr_code` ("Apache-2.0", fixed), `model_code` and `model_weights` (copied
from the build records, which must agree), `weights_trained_on` ("ImageNet-1k" when the reference
build record's weights name says so, else null), `label_data` ("CC BY 4.0") and `sources`. The
validator refuses an empty field; `scripts/40_check_labels.py` checks it against the build records.
The Markdown model card adds `license_link: "#licences"`, the anchor of its "Licences" section.

**5. Every rendered page states the label schema version** beside the Brokkr version and commit.
