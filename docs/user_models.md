# Testing a user's own model (Platform plan step 3)

*Design note of 6 October 2026, written and committed before any step 3 code. Drafted by Claude and decided
by H on 6 October 2026 (decisions D1–D17, listed at the end with what H chose). Details decided later go in
dated notes at the end of this file; nothing above them is edited. The label-format additions are in
`docs/label_schema.md`, note of 6 October 2026.*

The only numbers below are design settings (counts, lines, seeds) and a few interval widths **worked out from
formulas, not measured**. Nothing in this note has been built or measured yet.

## 0. Scope, and what stays the same

Step 3 turns a user's ONNX image classifier and a folder of the user's own labelled images into a
`label.json`, on the user's own machine. **ONNX only** (H, D15): PyTorch model input moves to after step 4.

What stays exactly as in the study:
- the INT8 recipe (Percentile 99.99, 512 calibration images in groups of 128);
- the 12 damaged conditions plus clean (`brokkr_edge/test_run.py`, the 13 conditions of task 4.1);
- the reliability envelope, version 1 (coverage line 80%, damage-drop line −10 points, large shrinking cost
  below −5 points, near floor when FP32 top-1 is below 10%);
- the bootstrap settings (1,000 resamples, seed 0);
- label schema version 1. The label-format changes are new optional fields only.

Not in step 3: choosing conditions (P4, later versions), trying several recipes (P5), submitting labels and
PyPI (step 4), images without labels (step 4b), any upload (step 5), PyTorch input (after step 4), speed on
user labels (after step 4, section 6).

## 1. Inputs, and what is checked about a supplied pair

**What the user gives.**
- An FP32 ONNX model.
- A folder of labelled images: one subfolder per class, the subfolder name being the class name. Images may
  sit in nested folders inside a class folder.
- Optionally, a folder of **unlabelled** representative images for INT8 calibration (`--calib-images`,
  section 2).
- Optionally, an already-shrunk **INT8** ONNX build of the FP32 model, made by any tool (H, D1: INT8 only.
  The envelope state for a broken shrunk build is named "INT8 build failed"; an FP16 build would need a new
  state, which a reader of the current schema could misread. FP16 pairs are parked).
- A small settings file (JSON, which needs no extra package on Python 3.10).

**The settings file** (`settings_version` 1). Paths in it are relative to the settings file.

| Field | What it holds |
|---|---|
| `name` | the model's name, lowercase letters, digits, `-`, `_`, `.` (used in its IDs) |
| `fp32_model` | the FP32 ONNX file |
| `shrunk_model` | `null`, or `{"file", "precision": "int8", "made_by": "<tool and version>"}` |
| `classes` | the class names, in the model's output order |
| `outputs` | `"logits"` or `"probabilities"` (section 3) |
| `preprocessing` | section 3 |
| `split` | `"brokkr"` (Brokkr splits the folder) or `"own"` (the user's own split, section 2) |
| `expected_accuracy` | `{"top1", "n_images", "measured_on"}`: the FP32 accuracy the user measured themselves (section 3) |
| `reference_predictions` | optional: 8 to 32 `{"file", "top1"}` pairs (section 3) |
| `licences` | `model_code`, `model_weights`, `images`, `images_source` (section 5) |
| `declarations` | `{"images_not_used_to_train_the_model": true}` (section 5) |
| `device_kind` | laptop, desktop, server, raspberry-pi-5, cloud-arm or other (section 6) |
| `submitted_by` | optional name the user chooses, or `null` |

The command is `brokkr-edge test --config <settings file> --images <folder> [--calib-images <folder>]
--out <folder>`, beside today's `brokkr-edge test --model <name>` for the study's models.

**Checks on every model, before any image is tested.** A failure stops the run with a plain message; no label
is made.
- It loads in ONNX Runtime (CPU).
- Exactly one input and one output; the input is float32.
- The input is 4-dimensional, with 3 channels and the stated crop as its height and width, in the stated
  layout (NCHW or NHWC). Its name is read from the model, never assumed.
- The batch size is either free (any number) or fixed at 1; a model with a fixed batch of 1 is run one image
  at a time. Any other fixed batch size stops the run.
- The output has as many classes as `classes` lists.
- The "FP32" model contains no quantized operations and no FP16 weights (if it does, it is not an FP32 model).

**Checks on a supplied pair** (FP32 + a shrunk build made by another tool):
1. **Same inputs and outputs:** the same input name, type and shape, and the same output shape (a dimension
   left free in one counts as matching any size in the other). They must be different files (different
   SHA-256).
2. **Same preprocessing:** there is one preprocessing block, used for both builds, so the two cannot disagree;
   the label states the pair was declared to share it.
3. **Declared precision matches the file:** Brokkr reads the shrunk file and records what it finds (quantized
   operations, 8-bit weight tensors); a file declared INT8 with no quantized operations stops the run.
4. **Plausible agreement:** top-1 agreement with FP32 on the first 256 conformal-calibration images (all of
   them if there are fewer; never test images), with the study's lines:
   - below 20%: the build is marked failed. The label is still made, with the FP32 rows and the
     "INT8 build failed" state with its value, as for MobileNetV3-Small. Its sentence says the build may be
     broken **or not made from this FP32 model** (Brokkr cannot tell which);
   - below 90%: a warning (section 2, `summary.warnings`).
5. When Brokkr builds the INT8 file itself, the same agreement check runs on it. If ONNX Runtime's preparation
   step crashes, Brokkr retries once with `skip_symbolic_shape` and records it in the build's recipe (H, D2;
   the label's Details already show this setting).

## 2. How the user's images are split, minimum counts, and too few images

**No tuning part in step 3** (H, D4). Step 3 chooses no setting (one recipe, no temperature, no method choice),
so a tuning part would only take images from the test. It comes with P5 (`shrink --auto`).

**Listing the folder.**
- Image files are those ending in .jpg, .jpeg, .png, .bmp, .webp, .tif or .tiff (any letter case). Other files
  are not used; their count is reported, never silently dropped.
- The canonical order is the sorted list of relative paths (with `/`). Each image's damage seed is its
  position in that order (the study's rule: the seed is the image's dataset position).
- **Unreadable images stop the run**, listed by name; they are never skipped.
- **Exact duplicates** (identical file bytes, by SHA-256) anywhere in the labelled folder stop the run, listed
  by name, so no image can be in two parts under two names. Near-duplicates (burst shots of one scene) are not
  detected; the label says so (`limits`), and a user's own split (below) is the remedy.
- The folder's subfolders must be exactly the classes in `classes` (in a user's own split: in each part).

**Two ways to split.**
- **Brokkr's split** (`"split": "brokkr"`): the labelled folder is split, stratified by class (each class keeps
  its share) with fixed seeds: first the INT8 calibration images if they come from this folder (below), then
  the rest: **one third conformal calibration, two thirds test**, per class (a third rounded down).
- **The user's own split** (`"split": "own"`, H, D5): the folder holds exactly two subfolders,
  `calibration/<class>/...` and `test/<class>/...`. The user's test images are always the test part; the
  conformal calibration images come from `calibration/`. This is the remedy when photos come in groups, and
  it makes the exact cross-check of section 7 possible. The duplicate check runs across both parts.

**INT8 calibration images** (H, 6 October 2026, changing the draft's D3). INT8 calibration needs no labels:
- With `--calib-images <folder>`: a folder of unlabelled representative images, **at least 512**. If it holds
  more, 512 are chosen with a fixed seed; if exactly 512, all of them, in canonical order. No calibration image
  may also be in the labelled folder (compared by SHA-256): an overlap stops the run, listing the files.
- Without it: the 512 come from the labelled images, stratified and seeded: from the whole folder in Brokkr's
  split, from `calibration/` only in the user's own split (the user's test images are never used for anything
  but testing).
- When the user supplies the shrunk build, there is no INT8 calibration; giving `--calib-images` then stops the
  run (it would not be used).

**Caps:** at most 5,000 conformal-calibration and 10,000 test images (the study's sizes); extra images are left
unused, chosen stratified and seeded, and their count is reported.

**Floors and recommended sizes** (H, 6 October 2026):

| Part | Floor: below it, the run stops | Recommended: below it, the label warns |
|---|---|---|
| Conformal calibration | 200 | 1,000 |
| Test | 200 | 2,000 |
| INT8 calibration (when Brokkr builds INT8) | 512 | 512 |

- Every class needs at least one image in the conformal-calibration part and one in the test part, or the run
  stops.
- Classes with fewer than 20 test images are named in a warning.

**How wide the intervals are at the floor** (worked out, not measured). From the normal approximation to the
binomial, the worst-case 95% half-width of an accuracy (at 50%) is ±6.93 points on 200 test images, ±2.19 on
2,000; for a coverage near 80% it is ±5.54 points on 200 images, ±1.75 on 2,000. And with 200
conformal-calibration images, the coverage the threshold actually gives on new clean images varies from one
calibration draw to another with a standard deviation of 2.11 points (0.95 with 1,000), from the Beta
distribution of split-conformal coverage (target 90%; k = 181 of 200). So on a small test, a row clears or
fails the −10-point line only when it is far from it; most rows come out "borderline", which is the honest
answer.

**What the label says when there are few images** (never a silent pass):
- The intervals are computed as always, from the real counts; wide intervals make rows "borderline" by the
  existing rule.
- `summary.warnings` (a new optional field) has one entry per part below its recommended size, one naming the
  classes with fewer than 20 test images, and one for agreement with FP32 below 90%. It is rendered at the top
  of the label, before the summary lines (e.g. "Few images: N test images (recommended at least 2,000);
  intervals are wide"), with every count taken from `label.json`.
- `limits` gets the same sentences.
- Below a floor there is no label: a message names each count and its floor.

## 3. Preprocessing: how it is stated, and how a mistake shows up

**How it is stated.** Step 3 supports torchvision's standard evaluation pipeline only: `resize` (the shorter
side, in pixels), `crop` (a square centre crop, at most `resize`), `interpolation` (bilinear, bicubic or
nearest), `mean` and `std` (three numbers each, on pixel values 0 to 1, in RGB order), `channel_order` (RGB or
BGR: the order the model receives) and `layout` (NCHW or NHWC). Greyscale and other colour modes are turned
into RGB, as in the study. The photo-rotation tag (EXIF orientation) is not applied; the label says so.
Anything else (letterboxing, non-square inputs, no resize) is not supported yet and stops the run. Damage keeps
the study's place: after resize and crop, before normalisation.

**How a mistake shows up, instead of a quietly wrong label:**
1. **Shape check** (section 1): the stated crop and layout must match the model's input.
2. **Expected accuracy, required** (H, D6). The user states the FP32 top-1 they measured on their own
   validation images, and on how many. Brokkr's clean FP32 top-1 on the test part must be within **5 points**
   of it (compared in whole images: |correct − stated × n| ≤ 0.05 × n), and the lower end of its 95% interval
   must be above chance (1 ÷ the number of classes). Otherwise the run stops: "clean accuracy X% against your
   stated Y%: check the class order, mean and std, channel order and resize". The stated value is recorded as
   `published` in the label's `checks.fp32_sanity`, with the source "stated by the submitter". It is meant to
   catch large mistakes (a wrong class order gives about chance accuracy); a small mismatch (bilinear instead of
   bicubic) may pass it, and `limits` says so.
3. **Logits or probabilities.** If a model already applies softmax and Brokkr applies it again, top-1 is
   unchanged but coverage and ECE come out quietly wrong. Read on the agreement images (section 1):
   - with `"probabilities"`, every FP32 output must be non-negative and every row must sum to 1 (within
     0.001), or the run stops. Probabilities are turned back into logits with log(p) (zero is clipped to the
     smallest float32 number), so softmax gives p back. A shrunk build's outputs are checked only to be finite
     and non-negative: a quantized output's rows need not sum to exactly 1;
   - with `"logits"`, if every FP32 row is non-negative and sums to 1 (within 0.001), the run stops ("these
     look like probabilities").
4. **Class list:** `classes` must name exactly the class folders. A missing or extra folder stops the run;
   a wrong order is caught by check 2.
5. **Reference predictions, optional** (H, D7): 8 to 32 files of the labelled folder with the top-1 class the
   user's own code gave them. Brokkr's FP32 must give the same top-1 on every one, or the run stops.

## 4. Damage conditions, and "not tested"

**The study's 12 damaged conditions plus clean**, with no choice in step 3, so user labels have the same
layout as study labels: fog (Brokkr) s3, darkness (Brokkr) s5, defocus blur (Brokkr) s3, noise (Brokkr) s3, and
fog, contrast, defocus blur and Gaussian noise (ImageNet-C) at s3 and s5. The ImageNet-C conditions need the
optional `imagenet-c` extra; without it the run stops with the install command (no partial runs in step 3).

**Not tested:** everything else (other damage types and severities) is on the summary's "Not tested" line, as
on study labels. Choosing conditions is P4 (later versions).

**Inputs other than 224×224** (H, D8): Brokkr's damage is measured in pixels (blur radius, noise pattern), so
the same severity is relatively stronger on a smaller picture. Damage is applied at the model's own input size,
and when that is not 224×224 a generated `limits` sentence says the severities were designed for 224×224
pictures and are not comparable with Brokkr study labels. Before the damage code is written, a test checks that
each of the 8 ImageNet-C conditions runs on small and large pictures (for example 64×64 and 320×320); this is
not assumed.

## 5. Licences and declarations; nothing leaves the machine

**Required in the settings file** (hard rule 8; an empty value, or "unknown", "none", "n/a" or "?", stops the
run): `licences.model_code`, `licences.model_weights`, `licences.images`, `licences.images_source`, and the
declaration `images_not_used_to_train_the_model: true` (H, D10). Brokkr cannot check the declaration; images
seen in training would make every number look better than it is.

They are recorded verbatim in the label (`licences`, `datasets`), marked "declared by the submitter, not
checked by Brokkr". `brokkr_code` stays "Apache-2.0". `label_data` (H, D9): "chosen by the submitter; labels
submitted to Brokkr's catalog are CC BY 4.0 (step 4)".

**Nothing is uploaded.**
- The command opens no network connection, not even a version check. For the whole run, every connection
  through Python's `socket` module is blocked (it raises an error), and ONNX Runtime's telemetry events are
  switched off at the start (`onnxruntime.disable_telemetry_events()`). A test runs the full check of a
  made-up model under the block and counts connection attempts (none allowed). The block covers Python's
  `socket` module; native code that bypasses it is not intercepted, so the docs say "Brokkr opens no network
  connection", backed by this test, not that it is sandboxed.
- Records and the label go only to `--out`.
- Neither the label nor the run's records hold image file names or absolute paths: only counts, class names,
  and **folder fingerprints** (the SHA-256 of the sorted list of each image's relative path and SHA-256), which
  identify the folder without naming a file. The split is remade from the folder itself (it is fixed by the
  seeds), and the fingerprints show it is the same folder. Class names are printed on the label, and the docs
  say so.

## 6. How user labels are marked and kept apart

- `source.kind` is `"user-submitted"`, `verified` is `false` (the validator already requires this pairing),
  `how_made` is `"brokkr-edge test <version>"`, `submitted_by` is the settings file's (or `null`). The renderer
  already shows "User-submitted label — UNVERIFIED"; the site's display word is "User-submitted".
- **IDs cannot collide** with study labels: the publisher in `model_id` is always `user`:
  `user/<name>@<first 12 hex characters of the FP32 file's SHA-256>`.
- **Storage:** user records and labels are written only to `--out`. The command refuses an `--out` inside this
  repository's `published/labels/`, `labels/` or `results/` folders (released labels, working study labels and
  study records).
- **Not on the site in step 3:** the site, figures, README blocks and Compare read `published/labels/` only.
  User labels reach the catalog only through step 4's submission flow.
- **Hardware:** the user's machine fingerprint, with `device_kind` as the user states it (H, D11): laptop,
  desktop, server, raspberry-pi-5, cloud-arm or other. "raspberry-pi-5" is accepted only on a machine that
  reports itself as a Raspberry Pi 5 (`/proc/device-tree/model`) (hard rule 2).
- **Speed** (H, D12): every speed row of a user label is "not measured", with the reason "speed is not measured
  in step 3 user runs". This is a **known product gap**: the study's latency method relies on checks (power
  mode, pinned cores) Brokkr has only on its Windows laptop. A speed bench for users' machines is planned after
  step 4 (ROADMAP, "Later versions").

## 7. What "done" means for step 3 (each item a test or a script that prints PASS or FAIL)

1. This note and the `docs/label_schema.md` note are committed before any code.
2. **Made-up model and made-up images** (tests only; a tiny ONNX model built in the test, no PyTorch):
   - the split: deterministic, stratified, no image in two parts; the user's own split; the unlabelled
     calibration folder and its overlap check; duplicates and unreadable images stop the run;
   - floors: refusal below a floor; warnings between floor and recommended; classes with fewer than 20 test
     images named;
   - the settings file: a missing licence or declaration stops the run; a class list that does not match the
     folders stops it;
   - pair checks: an input or output mismatch, an identical file and a wrong declared precision each stop the
     run; agreement below 20% gives the failed state, below 90% a warning;
   - outputs and preprocessing: the probabilities-or-logits check; a wrong class order or channel order is
     caught by the expected-accuracy check; reference predictions;
   - privacy: no network (blocked, and no attempts counted); no file names or absolute paths in what is
     written;
   - the label itself: `check_label` passes; `user-submitted` and unverified; the HTML shows the UNVERIFIED
     badge; every rendered number is in `label.json`.
3. **The test user model (section 8), Brokkr shrinks it:** one command gives a schema-valid `label.json` marked
   user-submitted.
4. **The test user model as a supplied pair:** its INT8 build made outside Brokkr's pipeline (a separate script
   calling ONNX Runtime directly, MinMax); the pair passes its checks and gives a valid label. Wording: "a build
   made outside Brokkr's pipeline", not "works with any tool".
5. **One real torchvision model with a folder of images, and the exact cross-check** (H, D14): MobileNetV3-Large
   with folders written from the study's own ImageNet images (original JPEG bytes; local only, never
   committed): `test/` from the test split, `calibration/` from `conformal_calibration`, and `--calib-images`
   from `int8_calibration`, named so the canonical order is the study's order. The user path's top-1 must equal
   the 4.1 records image for image where the pictures are the same (see open point D18 below).
6. Ruff, pytest, `scripts/22`, `scripts/40` and the site build pass, and the site is unchanged; ROADMAP, STATUS
   and the README are updated.

Items 3–5 are long runs, started by H (assistant sessions are cut off after about 10 minutes), after a short
dry run.

**Slices** (H, 6 October 2026): slice 1, the checks (settings file, model and pair checks, split rules with the
user's own split and the unlabelled calibration folder, floors and warnings, the logits-or-probabilities check,
the expected-accuracy check, reference predictions, the no-network block) with their made-up-data tests from
item 2; it writes a run-plan record and runs no damage. Running the conditions and making the user label are a
slice of their own, after H's review of slice 1; the label tests of item 2 ("the label itself") come with it.
Slice 2: the test user model (section 8). Slice 3: the end-to-end runs (items 3–5).

## 8. The test "user" model

**Dataset: EuroSAT, RGB version** (H, D16). Sentinel-2 satellite pictures of 10 land-use classes, 27,000
images. Licence, read on 6 October 2026 from github.com/phelber/EuroSAT: "The dataset is licensed under the MIT
license"; the same page points to the Copernicus Sentinel data terms for the underlying satellite data.
**Before any use** (H): read and record the Copernicus terms (and whether they allow commercial use), confirm
the RGB images' size, and record the download URL, size and SHA-256, all in `brokkr_edge/datasets.py` with its
"use" tag. Not ImageNet, and the model is trained from scratch, so ImageNet's terms appear nowhere in the
chain. It exercises what the study's models never do: a non-224 input and non-ImageNet mean and std.

**The model** (`scripts/51_train_test_user_model.py`, slice 2):
- a seeded, stratified 40% of EuroSAT trains the model (part of it kept back as the training script's own
  validation set); the other 60% is the user's labelled folder for Brokkr;
- torchvision's ResNet-18 architecture (BSD-3-Clause code), random start (`weights=None`), 10 classes, its
  input at the images' own size, seed 0, a fixed thread count; ResNet-18's INT8 build passed every check in
  4.1, so the main path is likely to run end to end (a failed build would still be handled);
- a fixed number of epochs, set before training; nothing is tuned on the 60% given to Brokkr;
- **one timed epoch first** (H); if the whole training is estimated above about 2 hours, H chooses again
  before it runs;
- mean and std computed from the training images and written into the settings file; the training script's
  validation accuracy is the settings file's expected accuracy;
- the model is named `eurosat-resnet18-test`, publisher `user`; its files are never committed (rule 4).

iBean is not used (H, D17).

## Decisions of 6 October 2026 (H)

| | Decision |
|---|---|
| D1 | A supplied shrunk build must be INT8; FP16 pairs parked |
| D2 | Automatic `skip_symbolic_shape` retry, recorded |
| D3 | Changed by H: optional unlabelled `--calib-images` (at least 512, no overlap with the labelled folder); floors 200 test and 200 conformal calibration; recommended 2,000 and 1,000; one image per class per part; fewer than 20 test images per class named |
| D4 | No tuning part in step 3 |
| D5 | Users may bring their own split |
| D6 | Expected accuracy required, 5-point tolerance |
| D7 | Reference predictions optional |
| D8 | Damage at the model's own input size, with a limit sentence when it is not 224×224 |
| D9 | `label_data`: "chosen by the submitter; labels submitted to Brokkr's catalog are CC BY 4.0 (step 4)" |
| D10 | The "not used to train the model" declaration is required |
| D11 | Device kinds: laptop, desktop, server, raspberry-pi-5, cloud-arm, other |
| D12 | Speed on user labels "not measured"; a known product gap, with a bench planned after step 4 |
| D13 | This file, plus a dated note in `docs/label_schema.md` |
| D14 | The exact cross-check is included |
| D15 | PyTorch input deferred until after step 4; step 3 is ONNX only |
| D16 | EuroSAT + ResNet-18 from scratch, after the licence, size and checksum checks and a one-epoch timing |
| D17 | iBean not used |

## Open point for H, found while writing this note

- **D18. The exact cross-check and random damage.** Six of the 12 damaged conditions use random numbers: fog and
  noise (Brokkr) and fog and Gaussian noise (ImageNet-C) at both severities; the other six (darkness and defocus
  blur (Brokkr); contrast and defocus blur (ImageNet-C), both severities) do not. A picture's random pattern
  comes from its seed: in the study, the image's ImageNet dataset position; on the user path, its position in
  the user's folder, which cannot equal it. So item 5 can be exact only for clean, the conformal threshold, the
  INT8 build and the six conditions without random numbers. Claude's proposal: those must match image for
  image (PASS / FAIL); for the six random conditions the difference in top-1 is reported with its paired
  interval (same images, different patterns), not judged. Decide before slice 3.

## Note added 7 October 2026, after H's review of slice 1 (`bed2025`), before any code for it

Decided by H on 7 October 2026. Points marked "(proposed)" are Claude's and wait for H's confirmation. This
note replaces the text above where they differ; the text above is not edited.

**1. The expected-accuracy check measures on the conformal-calibration images, not the test images.** Section
3 said "on the test part". The check can stop a run, and the test part never decides anything, so it now uses
clean FP32 top-1 on every image of the conformal-calibration part. The rule is otherwise unchanged (within 5
points, compared in whole images, and the lower end of the 95% interval above chance). Slice 1 measured it on
the test part; that is corrected in the code, and a test shows which part it uses. At the floor (200
conformal-calibration images) the worst-case 95% half-width is ±6.93 points (section 2, worked out, not
measured). *Raised for H, not changed:* the optional reference predictions may name test images. They compare
Brokkr's FP32 answer with the answer of the user's own code, never with the true class, so no accuracy on test
images is read; but a test image can stop a run through them. H to decide whether they must name
non-test images only.

**2. A failed shrunk build, downstream.** As section 1 (item 4) says: the label shows FP32 numbers only; its
INT8 rows say "INT8 build failed", with the agreement, the 20% line and the number of images; no INT8 number is
shown (as on MobileNetV3-Small's label). The command's last line now says this explicitly when the supplied
build failed.

**3. Wording.** What the run does to the network is stated as "Python-level network connections blocked and
counted; ONNX Runtime telemetry switched off", replacing "for the whole run, every connection through Python's
socket module is blocked" (section 5), in the code, the run plan and STATUS.md as well.

**4. D18, decided.**
- **a. Seeds of a user run: `content-v1`.** A random condition's seed for one image is the first 4 bytes, read
  as a big-endian whole number (0 to 2^32 − 1, the range NumPy's global generator accepts, which the ImageNet-C
  code uses), of the SHA-256 of the UTF-8 text `<the image file's SHA-256 in hex>:<suite>/<damage type>`.
  Adding or removing an image therefore changes no other image's damage. This replaces "each image's damage
  seed is its position in that order" (section 2) for user runs. The study keeps its position seeds
  (`position-study`). The run plan and the label record `"seed_scheme": "content-v1"`.
  - (proposed) The condition name leaves out the severity, keeping the study's rule that one image gets the
    same random pattern at every severity, so severities differ only in strength.
  - (proposed) A label without the field is read as `position-study`. The released study labels gain the field
    the next time they are remade for another reason, not now: remaking them for a field that changes no number
    would be a release with nothing else in it.
  - The six conditions without random numbers (darkness and defocus blur (Brokkr); contrast and defocus blur
    (ImageNet-C), both severities) use no seed, so the scheme does not affect them.
- **b. The exact cross-check** (section 7, item 5) must match the study image for image (PASS or FAIL) for clean,
  the conformal threshold and the six conditions without random numbers, and for the INT8 build: (proposed) the
  user path's INT8 file, built from the same 512 images in the same order, has the study file's SHA-256; if the
  bytes differ, its scores must equal the study build's on every image of every compared record.
- **c. The six random conditions get a tolerance, fixed before slice 3 runs, from a measurement.** Method, fixed
  here before anything is measured:
  - model: MobileNetV3-Large, its FP32 file and its Percentile 99.99 INT8 file (the study's files, the
    cross-check's model);
  - images: all 5,000 `conformal_calibration` images (never test images), the study's preprocessing;
  - conditions: the six random ones: fog (Brokkr) s3, noise (Brokkr) s3, fog (ImageNet-C) s3 and s5, Gaussian
    noise (ImageNet-C) s3 and s5;
  - two seed sets: `position-study` (each image's dataset position) and `content-v1` (from the SHA-256 of each
    image's original JPEG bytes as stored in the dataset, which is what the cross-check's folders will hold).
    This is exactly the change the cross-check meets;
  - measured: for each build and condition (12 cells), top-1 under `content-v1` minus top-1 under
    `position-study` on the same images, in points, counted in whole images, with its paired bootstrap 95%
    interval (1,000 resamples, seed 0);
  - (proposed) the margin rule, fixed now so the margin is not chosen after seeing the numbers: M = the largest
    absolute end of the 12 intervals, rounded up to the next 0.1 point. In the cross-check, each random cell's
    user-path top-1 minus its 4.1 record's top-1 (10,000 test images) must lie within ±M points. The test part
    has twice the images, so the spread from the seed alone is likely smaller there and M errs towards passing:
    a real pipeline difference smaller than M would not be caught by these six cells (clean and the six exact
    cells would still catch it);
  - the script (`scripts/52_seed_sensitivity.py`) is written after H's review of this note; it writes
    `results/checks/seed_sensitivity.json`; its run time is not known yet (if it is over about 10 minutes, H
    starts it). After the run, the proposed M goes in a dated note, and the work stops for H's approval before
    slice 3.

## Note added 7 October 2026 (later the same day), after H's second review, before any code for it

Decided by H on 7 October 2026. Points marked "(proposed)" are Claude's readings of H's words and wait for H's
confirmation; the code follows them meanwhile. Nothing above is edited.

**1. D18: the four "(proposed)" points of the note above are approved**, with one addition to the margin rule,
fixed now, before any measurement: **if M is above 1.5 points, the work stops for review instead of adopting
M** (a margin that large would mean the six random conditions are barely checked). When the released study
labels are remade with `seed_scheme`, that change is listed in `docs/label_changes.json`, so
`scripts/48_compare_labels.py` passes.

**2. Reference predictions use only non-test images** (changes section 3, point 5). An entry naming an image of
the test part is ignored, never run, and the number ignored is reported (run plan and output). If fewer than 20
usable entries remain, the check warns instead of stopping. (proposed) Read as: with 20 or more usable entries,
any mismatch stops the run, as before; with fewer, mismatches give a warning (counts only, no file names) and the
run continues. Since at most 32 entries are allowed, at least 20 must fall outside the test part for this check
to be able to stop a run. A test shows that a test image can never stop the run through this check.

**3. The expected-accuracy check, changed after review, before any user data** (changes D6 and point 1 of the
note above). The run stops only if the gap between measured and stated FP32 top-1 is larger than **both** 5
points **and** the 99% binomial interval half-width for the number of conformal-calibration images used. H's
reason: at the 200-image floor, a correct setup misses 5 points by chance about 1 time in 10; the check is meant
to catch a wrong class order or wrong preprocessing, which cause much larger gaps.
- (proposed) The half-width is the normal approximation at the user's stated accuracy p: 2.576 × √(p(1 − p)/n)
  (the check asks whether the measured accuracy is surprising if the stated one is right). Compared in whole
  images: stop if |correct − p × n| > max(0.05 × n, 2.576 × √(n × p × (1 − p))).
- Worked out from that formula, not measured: at 200 images the half-width is 3.97 points for p = 0.95, 5.46 for
  0.9, 7.29 for 0.8, 9.11 for 0.5; at 1,000 images 1.78, 2.44, 3.26 and 4.07.
- (proposed) The second condition is unchanged: the lower end of the measured 95% interval must be above
  chance (1 ÷ the number of classes).
- Tests: (a) a correct model at 200 images with a 4-point chance gap passes; (b) a shuffled class order stops the
  run.

**4. A standing rule, added to `CLAUDE.md`** (H): any change to `brokkr_edge` code the study uses must pass
`scripts/43_check_reproduction.py` (28 of 28) before merging, with the output pasted; no pytest test covers it.

## Seed-sensitivity outcome (added 7 October 2026, after the run)

Run once with `scripts/52_seed_sensitivity.py` at the clean commit `c8d7c3c`, on mains power, by the method fixed
in the two notes of 7 October 2026 above; no reruns. A dry-run tool check on 64 tuning images (not part of the
measurement; output outside `results/`) ran first and found one bug (the output folder was not created), fixed
before the commit. The check before measuring passed: with position seeds, the seeded damage equals the study's
own on the first batch of every condition. MobileNetV3-Large, 5,000 `conformal_calibration` images. Records:
`results/seed_sensitivity/` (12 schema-2 diagnostic records, with the per-image answers) and
`results/checks/seed_sensitivity.json`; `scripts/22` afterwards: PASS, 1913 of 1913 records.

Top-1 under `content-v1` minus top-1 under `position-study`, same images, points, paired 95% interval:

| Build | Condition | position-study | content-v1 | Difference | 95% interval |
|---|---|---|---|---|---|
| FP32 | fog (Brokkr) s3 | 71.36% | 71.60% | +0.24 (+12 images) | −0.36 to +0.84 |
| INT8 | fog (Brokkr) s3 | 65.36% | 65.40% | +0.04 (+2) | −0.78 to +0.84 |
| FP32 | noise (Brokkr) s3 | 52.26% | 52.28% | +0.02 (+1) | −0.94 to +1.02 |
| INT8 | noise (Brokkr) s3 | 49.66% | 48.66% | −1.00 (−50) | −1.96 to −0.10 |
| FP32 | fog (ImageNet-C) s3 | 63.62% | 63.14% | −0.48 (−24) | −1.24 to +0.40 |
| INT8 | fog (ImageNet-C) s3 | 52.02% | 51.62% | −0.40 (−20) | −1.42 to +0.72 |
| FP32 | fog (ImageNet-C) s5 | 40.68% | 40.90% | +0.22 (+11) | −0.86 to +1.24 |
| INT8 | fog (ImageNet-C) s5 | 29.46% | 29.30% | −0.16 (−8) | −1.26 to +0.88 |
| FP32 | Gaussian noise (ImageNet-C) s3 | 31.40% | 31.78% | +0.38 (+19) | −0.64 to +1.34 |
| INT8 | Gaussian noise (ImageNet-C) s3 | 29.22% | 29.22% | +0.00 (0) | −0.94 to +0.92 |
| FP32 | Gaussian noise (ImageNet-C) s5 | 1.12% | 0.94% | −0.18 (−9) | −0.44 to +0.06 |
| INT8 | Gaussian noise (ImageNet-C) s5 | 1.40% | 1.54% | +0.14 (+7) | −0.18 to +0.48 |

**By the rule: the largest absolute interval end is 1.96 points, so M = 2.0 points, which is above 1.5 points.
M is not adopted; the work stops for H's review** (note of 7 October 2026, later the same day, point 1). No
margin for the six random conditions is fixed, and slice 3 does not start, until H decides.

### Exploratory, after the result (decides nothing)

- The largest end comes from one cell, INT8 noise (Brokkr) s3 (−1.00 points, interval −1.96 to −0.10, the
  only interval of the 12 that excludes zero). With 12 intervals at 95%, about one excluding zero by chance is
  to be expected (likely chance, not tested). Without that cell the largest end would be 1.42 points (INT8 fog
  (ImageNet-C) s3); that is reported, not used: the rule takes all 12 cells.
- Gaussian noise (ImageNet-C) s5 is near floor for both builds (FP32 about 1%), so its cells say little.

## Note added 7 October 2026 (evening), after the seed-sensitivity outcome, before any code for it

Decided by H on 7 October 2026, **after seeing the seed-sensitivity result above**. Points marked "(proposed)"
are Claude's and wait for H's confirmation. Nothing above is edited.

**1. M = 2.0 points is not adopted.** The stop rule did its job; the line is not moved after the data.

**2. The six random conditions are made exact instead of given a margin.** This was decided after seeing the
result, but it does not depend on its numbers: it sets no threshold, margin or tolerance at all. It replaces
the margin of D18 (c) with a reference built to match exactly, so the measured spread no longer enters any rule.

**3. Both "(proposed)" readings of the second note of 7 October 2026 are approved:** with fewer than 20 usable
reference predictions, mismatches only warn; the 99% half-width is worked out at the user's stated accuracy.

**4. The new cross-check plan** (replaces D18 (b) and (c) and section 7, item 5, where they differ):
- **a. The reference for the six random conditions** is the study's code for MobileNetV3-Large (its FP32 file and
  its Percentile 99.99 INT8 file) on the 10,000 test images, with only the seed source swapped to `content-v1`
  (from the SHA-256 of each image's original JPEG bytes, as stored in the dataset). The same check as
  `scripts/52_seed_sensitivity.py` runs first: with position seeds, the damage must equal the study's on the
  first batch of every condition, or the script stops. Script: `scripts/53_content_v1_reference.py`; same
  settings as the 4.1 records (8 threads, thread spinning off, batch 32, the same caches and preprocessing).
- **b. The cross-check (D14) then requires an image-for-image match on all 13 conditions:** clean, the conformal
  threshold, the INT8 build and all 12 damaged conditions (the six without random numbers against the 4.1
  records, the six random ones against these references). No margin anywhere.
- **c. The references are marked so they cannot be mistaken for results:** schema-2 records of kind
  "diagnostic" (never a result; the label maker reads only accuracy records from `results/breadth`), in their
  own folder `results/crosscheck_reference_content_v1/`, every file name ending in
  `_seed-content-v1_crosscheck-reference`, and their settings say `"seed_scheme": "content-v1"` and that they are
  an equality reference for the step 3 cross-check only, never a result and never on a label.
- **d. The test part is used here only as an equality reference:** these records set no number, threshold or
  setting of any kind; they exist only so the user path can be compared with them for exact equality.

**5. A limits sentence for labels with random conditions** (draft; exploratory wording; for H's approval).
(proposed) "Fog and noise damage use a random pattern. In one exploratory check (MobileNetV3-Large, 5,000
images), changing only that pattern moved top-1 by up to 1.0 point; the intervals on this label cover the
choice of images only, not the choice of pattern." The figure is the largest absolute difference of the 12
cells in the table above (INT8 noise (Brokkr) s3, −1.00 points); it is generated from
`results/checks/seed_sensitivity.json`, never typed.

## Content-v1 references (added 10 October 2026, after the run)

Run once by H with `scripts/53_content_v1_reference.py` at the clean commit `206f9e1`, on mains power, by note 4a
above; no reruns. The check before measuring passed (position seeds reproduce the study's damage on the first
batch of every condition). 12 records in `results/crosscheck_reference_content_v1/` (MobileNetV3-Large, FP32 and
Percentile 99.99 INT8, the six random conditions, 10,000 test images); each condition took 3.0 to 3.4 minutes
for both builds (the script's own output). Checked afterwards: all 12 records are kind "diagnostic", record
commit `206f9e1` with a clean tree and `"seed_scheme": "content-v1"`; `scripts/22` PASS (1925 of 1925 records,
159 of them diagnostics); `scripts/40` PASS (10 labels); the site build PASS (17 pages, 61 files), and no file
in `labels/`, `published/` or the built site mentions them.

## Note added 10 October 2026: the limits sentence for random conditions, approved

Decided by H on 10 October 2026: the draft sentence of note 5 above is approved, with one addition. Nothing
above is edited. The sentence, as approved:

"Fog and noise damage use a random pattern. In one exploratory check (MobileNetV3-Large, 5,000 images), changing
only that pattern moved top-1 by up to 1.0 point; on other models, or with fewer images, it may be larger. The
intervals on this label cover the choice of images only, not the choice of pattern."

- **The figure is generated, never typed:** the largest absolute `difference` of the 12 cells in
  `results/checks/seed_sensitivity.json` (`raw.cells`), in points. Today that is INT8 noise (Brokkr) s3, −1.00
  points (−50 of 5,000 images), read from the record on 10 October 2026.
- (proposed) The model name and the image count ("5,000") are read from the same record (`model`,
  `settings.n_images`), so no number in the sentence is typed. The figure is shown with one decimal, rounded up
  (never down), so the sentence never understates the measured spread.
- It is added to `limits` only on labels that tested at least one random condition (fog and noise (Brokkr), fog
  and Gaussian noise (ImageNet-C)). It is built with the slice that makes user labels; whether the released study
  labels gain it is decided when they are next remade, and would be listed in `docs/label_changes.json`.

*Confirmed by H on 10 October 2026:* both "(proposed)" points of the note above: the model name and the image
count in the limits sentence are read from `results/checks/seed_sensitivity.json` (`model`, `settings.n_images`),
so no number in the sentence is typed; the figure is shown with one decimal, rounded up, never down.

## Note added 10 October 2026: plan for the slice that runs the conditions and makes the user label (before any code)

Asked by H on 10 October 2026: plan first, no code. Nothing below is built. Points marked **(question for H)**
wait for H's answer; points marked "(proposed)" are Claude's and are followed unless H says otherwise. The only
numbers here are design settings, counts read from files on 10 October 2026, and run-time **estimates**, each
with what it is based on.

**What the slice does.** After slice 1's checks pass, the same command (`brokkr-edge test --config <settings>
--images <folder> [--calib-images <folder>] --out <folder>`) goes on to: (1) build the INT8 file, unless the
submitter supplied one; (2) run FP32 and INT8 on the clean conformal-calibration part and on the test part under
the 13 conditions, saving each run's scores; (3) work out the reliability numbers; (4) write `label.json`,
`label.md` and `label.html` to `--out`. The cross-check folder script and the cross-check come after this slice
(STATUS.md, order of 10 October 2026).

### a. Reuse or copy: one implementation for the study and the user path

**Moves out of `scripts/39_make_labels.py` into a new module `brokkr_edge/label_build.py`** (functions that take
numbers and arrays and never read files; `scripts/39` keeps only finding, checking and reading the study's
records, and calls these):
- `runtime_entry` (today `scripts/39` lines 135–150), unchanged;
- the conditions block (each condition's ID, suite, damage, severity, modality, label);
- the measurements: top-1, top-5, coverage with set size, ECE and E-AURC copied in, and `damage_drop` and
  `shrinking_cost` computed with `brokkr_edge.label.paired` (unchanged);
- the envelope rows, including the "INT8 build failed" rows of a failed build;
- the failure block of a failed build;
- the `limits` sentences: the study's sentences word for word, plus the user sentences of section b;
- the label skeleton (the top-level fields in schema order).

**Stays in `scripts/39` (study only):** the laptop latency rows, the FP32 sanity check against torchvision's
published top-1, and the licences read from the study's build records.

**Used as they are, not changed:** `brokkr_edge/label.py` (envelope rules, summary), `brokkr_edge/sweep.py`,
`brokkr_edge/seeds.py`, `brokkr_edge/accuracy.py`, `brokkr_edge/shift/*`, `brokkr_edge/judge.py`, and from
`brokkr_edge/test_run.py` the helpers `complete` and `keep_awake` (imported, unchanged). `test_run.run_model`
stays the study's: it reads ImageNet from Parquet and uses position seeds, the input name "images" and
ImageNet's mean and std.

**New, user path only:**
- `brokkr_edge/user_run.py`: decodes each user image once with the functions the study's picture caches use
  (`accuracy.open_image`, `accuracy.resize_and_crop`), at the model's own crop size; applies the damage with
  `seeds.damaged_batch_seeded` and `seeds.content_seed` (the code `scripts/52` and `scripts/53` use); normalises
  with the user's mean, std, channel order and layout (`user_checks.prepare` is split into its two halves, so
  slice 1 and this slice use the same code); runs with the model's own input name, in batches of 32 (1 for a
  model with a fixed batch of 1); turns probabilities into logits with `user_checks.as_logits` before any
  reliability number.
- The INT8 build: `quantize.to_int8` with the study's recipe (Percentile 99.99, the 512 INT8-calibration images in
  canonical order, batches of 32 in groups of 4), on the user's preprocessing. Build checks: finite outputs,
  weights per channel (`quantize.weight_quantization`, unchanged) and `user_checks.agreement` on the first 256
  conformal-calibration images (the same check as for a supplied pair). `quantize.check_int8_build` is not used:
  it expects the input name "images" and 1,000 classes. If ONNX Runtime's preparation step crashes, one retry
  with `skip_symbolic_shape`, recorded (D2).
- The reliability numbers: see question 2 below.

**Every shared function that changes** (code the study uses; each change only adds, so study outputs stay the
same):

| Where | Change | How "study unchanged" is shown |
|---|---|---|
| `scripts/39_make_labels.py` | assembly moved to `brokkr_edge/label_build.py` | remake the ten labels from a clean commit into a scratch folder; `scripts/48_compare_labels.py published/labels <scratch folder>` must PASS with no listed change (only commit and time differ); `scripts/40` PASS |
| `brokkr_edge/quantize.py`, `to_int8` | new optional `input_name`, passed to `ImageBatches` (default `"images"`, so every study call is unchanged) | a test: building the made-up model with and without the argument gives byte-identical files; `scripts/43` |
| `brokkr_edge/label_schema.py`, `check_label` | accepts the D11 device kinds and checks the user-label rules of section b; study rules unchanged | `tests/test_published_labels.py` and `scripts/40` PASS on the released labels |
| `brokkr_edge/label_render.py` | warnings block, declarations, submitter-declared licences, the supplied-build failure sentence, the "stated by the submitter" check line; each shown only when its field exists | `tests/test_published_labels.py`: every released `label.html` still renders byte for byte; `scripts/48` |
| `brokkr_edge/schema.py` | only if H chooses option (a) of question 1 | `scripts/22` PASS, same record count |

**`scripts/43` before merging** (rule of 7 October 2026), from the clean commit at the end of the slice, on mains
power, output outside `results/`:

`.venv/Scripts/python.exe scripts/43_check_reproduction.py --out data/checks/reproduction_<date>`

Run time: the run of 7 October 2026 finished its repeatability step at 05:13:51 UTC and wrote its 28 records
between 05:17:27 and 06:23:24 UTC (timestamps read from its records on 10 October 2026; its start time is not in
the files), so **about 70 minutes or a little more**. H runs it. Afterwards `scripts/22` must still PASS.

### b. Everything the user label holds, and the decision behind each part

| Part | What it holds | Decided in |
|---|---|---|
| `source` | `kind` "user-submitted", `verified` false, `how_made` "brokkr-edge test <version>", `submitted_by` from the settings file or null | section 6; `docs/label_schema.md` note of 6 Oct, 1 |
| Badge | "User-submitted label — UNVERIFIED" at the top of `label.md` and `label.html` (already in the renderer) | section 6 |
| IDs | `model_id` `user/<name>@<first 12 hex of the FP32 file's SHA-256>`, `model.publisher` "user"; build IDs come from it, so no ID can equal a study label's | section 6; label note 1 |
| Where it is written | only to `--out`; an `--out` inside `published/labels/`, `labels/` or `results/` is refused (built in slice 1); the site reads `published/labels/` only | section 6 |
| `model` | name, input (crop, layout, the stated preprocessing), outputs (number of classes, and the class names, which are printed on the label), declared code and weights licences | sections 3, 5 |
| `builds` | FP32 reference; the labelled INT8 build either made by Brokkr (the study's recipe fields, calibration dataset `user/<name>:int8_calibration`, `skip_symbolic_shape` when the retry ran) or supplied (`method` "supplied by the submitter", `made_by`, `found`) | D1, D2; label note 4 |
| Failed INT8 build | agreement below 20%: status "failed", a failure with the check, value, the 20% line, image count and part; FP32 measurements only; every INT8 envelope row "INT8 build failed"; the summary's Shrinking and Uncertainty lines "not measured: INT8 build failed"; for a supplied build, the sentence says it may be broken or not made from this FP32 model; the command's last line says so (built in slice 1) | section 1 item 4; label note 4; note of 7 Oct, 2 |
| `seed_scheme` | "content-v1" (where it sits in the label: question 4) | D18 (a) |
| `hardware` | this machine's fingerprint, `kind` as the submitter states it; "raspberry-pi-5" only on a machine that reports itself as one (built in slice 1) | D11; section 6 |
| `runtimes` | one entry: ONNX Runtime version, CPU provider, threads, spinning | label note of 3 Oct, 1 |
| `datasets` | `user/<name>:test`, `:conformal_calibration`, and `:int8_calibration` when Brokkr builds INT8; each with the declared licence and source, `split_mode`, the `folder_fingerprint` of that part, `labelled: false` for a `--calib-images` part, `disjoint_from` | label note 5; section 5 |
| `conditions` | the study's 13, same IDs and names | section 4 |
| `checks` | `fp32_sanity` from the expected-accuracy check (question 5); `outputs`; `reference_predictions` (counts only); `agreement_with_fp32` for the labelled build | label note 3; notes of 7 Oct |
| `measurements`, `envelope`, `summary` | as on study labels: the test part; the conformal threshold from each build's clean conformal-calibration part; envelope version 1; the same summary | section 0 |
| `summary.warnings` | "few images", "classes with few test images", "low agreement with FP32", and "few usable reference predictions" (question 9), rendered at the top, before the summary lines, every count a `label.json` number | section 2; label note 2; second note of 7 Oct, 2 |
| `speed` | every row "not measured", reason "speed is not measured in step 3 user runs", `runtime_id` null (question 8 on the Raspberry Pi row) | D12; label note 7 |
| `licences` | `brokkr_code` "Apache-2.0"; `model_code`, `model_weights`, `images` verbatim with `declared_by_submitter: true` ("not checked by Brokkr" when rendered); `label_data` D9's text; source: `run_plan.json` (question 7 on the wording) | D9; section 5; label note 6 |
| `declarations` | `images_not_used_to_train_the_model: true` | D10; label note 6 |
| `limits` | the study's general sentences (simulated damage; the ImageNet-C package sentence; one machine; chance with 12 conditions; coverage tuned on clean images; next steps are general suggestions; question 6 on "laptop"); and, only when they apply: one per warning; input not 224×224 (D8); licences and declarations not checked by Brokkr; near-duplicates not detected; the photo-rotation tag not applied; the expected-accuracy check may pass a small mistake; the random-condition sentence, only when a random condition ran (always, while all 13 conditions run; the code keeps the condition for P4) | label note 8; D8; note of 10 Oct |
| No file names or paths | the label, its renders, the records and the run plan hold counts, class names and fingerprints only; scores are stored by canonical index; every `sources` file is a path relative to `--out` | section 5 |
| `sources` | every record file in `--out`, with its SHA-256 and the check it passed | label schema, `sources` |

### c. Tests

The four label tests deferred from section 7 item 2, on the made-up model and images (`tests/user_made_up.py`),
both when Brokkr builds INT8 and for a supplied pair:
1. `check_label` passes on the label the command writes;
2. `label.html` and `label.md` show the UNVERIFIED badge;
3. every number in `label.md` and `label.html` is a `label.json` number (`unexplained_numbers` is empty);
4. no image file name, file stem, absolute path or home folder appears in `label.json`, `label.md`,
   `label.html`, `run_plan.json` or any record in `--out`.

New tests:
5. **Written before the damage code** (section 4): each of the 12 damaged conditions runs on pictures of 64×64
   and 320×320, and on the made-up model's own size. If a condition cannot run at the made-up model's size, the
   made-up model's size is raised in the tests to one where all 13 run (tests only), and that is reported.
6. Failed builds: a supplied build below 20% agreement gives a label with FP32 rows only, "INT8 build failed"
   rows, no INT8 measurement and the "may be broken or not made from this FP32 model" sentence.
7. Brokkr's own INT8 build of the made-up model: the recipe fields; the `skip_symbolic_shape` retry recorded
   when the first attempt crashes (the crash simulated in the test).
8. `to_int8` with and without `input_name="images"` gives byte-identical files.
9. Seeds: the user run uses `content_seed`; adding one image to the folder leaves every other test image's
   damaged picture unchanged.
10. The random-condition limits sentence appears only when a random condition ran; its figure, model and image
    count come from a made-up `seed_sensitivity.json` in the test, and the figure is rounded up.
11. Each warning appears in `summary.warnings` and as a `limits` sentence, with its counts from `label.json`.
12. A probabilities model: coverage and ECE are worked out from `as_logits`, so softmax gives the model's own
    probabilities back.
13. No network for the whole command (slice 1's test, extended to the full run).
14. Resume: a stopped run keeps its finished records and does not redo them.
15. Study unchanged: `tests/test_published_labels.py` (existing) passes; the `label_build` functions do what
    `scripts/39` did, shown by the remake and `scripts/48` (section a).

Before merging: ruff, pytest, `scripts/22`, `scripts/40`, the site build (still 17 pages and 61 files),
`scripts/48` on the remade labels, and `scripts/43` (H).

### d. Steps longer than about 10 minutes (H runs them)
- **`scripts/43`**: about 70 minutes (section a).
- **The cross-check** (after this slice and the folder script): MobileNetV3-Large through the user path on
  10,000 test images under 13 conditions and 5,000 clean conformal-calibration images, two builds, plus the INT8
  build. **Estimate, not measured:** H's `scripts/53` run of 10 October took 3.0 to 3.4 minutes per condition
  for two builds on 10,000 pictures already decoded; 13 conditions at that rate is about 40 to 45 minutes, plus
  decoding the images and building INT8, so probably about an hour.
- **Remaking the ten labels** for the `scripts/48` comparison: not timed. It reads saved scores and runs paired
  bootstraps; if it passes 10 minutes in a session, it stops and H runs it.

Everything else runs in a session: pytest on made-up models, ruff, `scripts/22`, `scripts/40`, the site build,
`scripts/48`, and dry runs on 64 images or fewer, with output outside `results/`.

### e. Questions for H (nothing below is decided)
1. **Where user runs' records are checked.** `brokkr_edge/schema.py` accepts only the sources "brokkr" and
   "community-submitted" and the device labels laptop, raspberry-pi-5 and cloud-arm, so a user run's records do
   not pass `check_record`. (a) Add the source "user-submitted" and the device kinds of D11 to `schema.py`
   (study code; additive; user records never go in `results/`, which slice 1 already refuses); or (b) give user
   records their own small checker and leave `schema.py` alone. Claude would choose (a): one checker for every
   record.
2. **The reliability numbers.** `scripts/31` is one short loop (its lines 92–150) around the shared, tested functions in
   `brokkr_edge/shift/` (ECE, conformal threshold and sets, E-AURC). (i) Move those lines into the package and
   make `scripts/31` call them, shown unchanged by re-running it into a scratch folder and comparing every metric
   with `results/breadth_reliability/` (run time not measured; possibly H's); or (ii) the user path calls the
   same `brokkr_edge/shift/` functions directly and `scripts/31` stays as it is (its records are final). Claude
   would choose (ii).
3. **Threads.** The study ran with 8 threads, thread spinning off. Use the same fixed setting for user runs,
   recorded on the label? The cross-check needs it to compare scores exactly: INT8 was shown repeatable bit for
   bit at a fixed thread count, not across thread counts.
4. **Where `seed_scheme` sits in the label.** (proposed) `generated.seed_scheme`, one value for the whole label (a
   label without it is read as "position-study", note of 7 October 2026).
5. **`checks.fp32_sanity` on a user label.** The label note of 6 October says `tolerance` 0.05; the rule became
   "larger than both 5 points and the 99% half-width", measured on the conformal-calibration part (notes of 7
   October). (proposed) Store `measured` from the conformal-calibration part with its dataset ID, `published`
   (the stated figure, "stated by the submitter", its image count and where it was measured), `tolerance` 0.05,
   `half_width_99`, `allowed_gap`, `chance` and `pass`; the validator recomputes `pass`.
6. **Study sentences that name the laptop.** "Accuracy was measured on the laptop; it has not been checked on
   other hardware." (proposed) On user labels: "Accuracy was measured on this machine only; it has not been
   checked on other hardware." The other general sentences are kept word for word.
7. **The licence lines.** The study line "This label's own numbers and text are Brokkr output, licensed
   {label_data}" reads wrongly with D9's text. (proposed) On user labels: "This label's licence is chosen by the
   submitter; labels submitted to Brokkr's catalog are CC BY 4.0 (step 4)", and "declared by the submitter, not
   checked by Brokkr" after the model's code, weights and images licences.
8. **The Raspberry Pi 5 speed row.** Study labels carry one ("no Raspberry Pi 5 yet"). (proposed) User labels have
   one "not measured" row per build for the submitter's own machine only, and no Raspberry Pi row.
9. **A warning kind not in the label note.** Slice 1 added "few usable reference predictions" (second note of 7
   October, 2); the label note of 6 October lists three kinds. (proposed) Add it as a fourth allowed kind.
10. **Brokkr's own INT8 build crashes even after the retry.** (a) Stop the run with no label, or (b) make the
    label with FP32 rows and "INT8 build failed" (check "the INT8 build crashed, also with skip_symbolic_shape",
    no value or line). Claude would choose (b), as for MobileNetV3-Small, but it needs the validator and the
    renderer to accept a failure with no number.
11. **The decoded pictures.** Decoding every image once for all 13 conditions means keeping the pictures on disk
    during the run (crop × crop × 3 bytes each; at 224×224 and 10,000 test images about 1.5 GB, worked out, not
    measured). (proposed) A memory-mapped file in `--out`, deleted once the label is written; a resumed run
    decodes again.
12. **Commit on a user label.** Outside a git checkout (after `pip install`, step 4) there is no commit.
    (proposed) `generated.commit` and `generated.dirty` are null there, with the brokkr-edge version; inside a
    checkout they are recorded as on study labels, and a dirty checkout is recorded, not refused (user labels are
    unverified anyway).
13. **What `--out` holds.** (proposed) `run_plan.json`, `int8_build.json` (and the INT8 file when Brokkr builds
    it), `records/` (the score records), and `label.json`, `label.md`, `label.html`.

**Order inside the slice, each step ending with its check:** the size test (c5); the move out of `scripts/39`,
with the remake and `scripts/48`; the schema, validator and renderer additions, with the released labels still
byte-identical; `to_int8`'s `input_name` and the user INT8 build; running the conditions; the reliability numbers
and the label; the command's last steps. Then all checks, `scripts/43` (H), and a stop for H's review before the
cross-check folder script.
