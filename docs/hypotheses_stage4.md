# Stage 4 hypotheses

Written **before** any Stage 4 accuracy or reliability measurement. Git history dates this file, so
it provably came first. What came before it, and is described below: speed profiling, the INT8 builds
with their sanity checks (agreement with FP32 on 256 tuning images), and diagnostics of the failed
builds. All used tuning or calibration images only; none touched the test split or computed an
accuracy. As in Stages 2 and 3, each prediction has a threshold, a way to judge it, and a confidence
level.
Outcomes will be added below without changing anything above. Details decided later go in dated
notes, each committed before the measurement it governs.

## Design rules, fixed in advance

- **Stage 3's rules carry over unchanged** (`docs/hypotheses_stage3.md`, "Design rules"): no setting
  is tuned on the test split; settings come from `tuning`, `conformal_calibration` or
  `int8_calibration`; test-split reruns only for technical failure, logged with the reason.
- **Differences are "new minus old"**, with paired bootstrap 95% intervals: 1,000 resamples of the
  same images for both sides, seed 0 (`brokkr.accuracy.paired_bootstrap_diff`). For accuracy and
  coverage, positive = better; for E-AURC and ECE, positive = worse.
- **Words used below.**
  - *Compression-caused gap* at a condition: INT8 top-1 minus FP32 top-1, same model, same images
    (negative = INT8 is worse).
  - *Extra gap* at a condition: the compression-caused gap there minus the same model's clean
    compression-caused gap (negative = quantization hurts more under that damage than on clean
    images). Its interval is paired over the test images, since the same images appear in both
    conditions.
  - *Absolute weakness*: FP32 top-1 under the condition, and FP32 minus its own clean top-1.
  - *Near floor*: FP32 top-1 under the condition below 10%. There, any gap is at most 10 points, so a
    small gap is not evidence of robustness. Near-floor cells are shown, flagged, and left out of
    every judged prediction below.
- **Conditions are always named with their suite**: "fog (Brokkr) s3" and "fog (ImageNet-C) s3" are
  different conditions made by different code.
- **Result format**: every Stage 4 *result* (accuracy, reliability, speed, levels) is saved in schema
  version 2 (`brokkr/schema.py`); planning profiles and diagnostics are labelled records that are never
  results. Diagnostics that a write-up will cite (such as the MobileNetV3-Small layer check below) are
  schema-2 records of kind "diagnostic", which name the script, split, image count, model and code
  version. `scripts/22_check_results.py` must pass: it checks every result record, and every model
  build record (clean commit, licence, the build's own sanity check, any non-default build setting
  stated), and fails if a result uses a model whose build failed or was never checked.

## Task 4.0: the mechanism test

**Question.** Report 1 (section 5) gave a *likely*, untested explanation for why darkness broke
default INT8: MinMax calibration stretches each layer's 256 levels to cover rare extreme values, so
the smaller values of dim or low-contrast images fall into only a few levels. Percentile calibration
ignores the rarest values and spends the levels where values actually are. This test counts the
levels directly.

**Design, fixed now.**
- Models: MobileNetV3-Large default INT8 (MinMax) and Percentile 99.99 INT8, the Stage 3 files.
- Images: the first 500 images of the `tuning` split, in split order (not test images: this test
  sets nothing and reports no accuracy, but the test split is kept for results).
- Conditions: clean; darkness (Brokkr) s5; fog (Brokkr) s3. Damage as in Stage 2: applied to the
  224x224 picture, pattern seed = the image's dataset position.
- What is counted: each of the 142 activation `QuantizeLinear` outputs in the model (the same 142
  tensors in both models; they include the input image and the final scores). For each image and
  each tensor: the number of distinct 8-bit levels used (0–256). Per image, the summary is the
  **median over the 142 tensors**. The per-tensor table is saved too (reported, not judged).
- How the levels are read: the 142 tensors are added as extra model outputs. Because that can change
  how ONNX Runtime fuses operations, a check runs first: the modified model's top answers must match
  the unmodified model's on all 500 clean images. If they don't, the test stops and a dated note
  decides what to do; nothing is judged.
- Comparisons (paired over the same 500 images, 1,000 resamples, seed 0):
  - (a) damaged minus clean, default INT8 (negative = damaged images use fewer levels);
  - (b) Percentile minus default, on the damaged images (positive = Percentile uses more levels).

**M1. Low-contrast images use fewer levels, and Percentile gives them more.** Judged separately for
darkness (Brokkr) s5 and fog (Brokkr) s3. *Confidence: medium for darkness, low for fog* (fog lifts
values towards light grey rather than towards zero; after the first layers that may or may not
narrow the range).
- **Supports** the explanation if both hold: (a) is below zero with its interval below zero, and the
  reduction is at least 20% of the clean images' value; and (b) is above zero with its interval
  above zero.
- **Rejects** it if (a)'s interval includes zero or lies above zero: damaged images do not use
  fewer levels, so the squeeze the explanation needs is not there.
- **Inconclusive** otherwise (for example fewer levels, but less than 20% fewer; or (a) holds but
  (b) does not). Reported as such, never rounded up to "supports".
- The 20% bar is set so that "supports" needs a reduction big enough to matter, not just a
  measurable one: a small squeeze is unlikely to explain a collapse from 60% to 20% top-1.
- Even "supports" shows only that the squeeze happens. It does not prove that the squeeze *causes*
  the accuracy loss; the write-up will say "consistent with", not "proves".

## Task 4.1: the breadth study

**Question.** Does clean accuracy predict robustness after quantization? Which weakness is absolute
(FP32 also fails) and which is caused by compression (INT8 minus FP32)?

### Models

Candidates (torchvision, default weights, each read from torchvision so nothing is typed by hand):

| Model | Weights | Resize / crop / interpolation (torchvision's own) |
|---|---|---|
| MobileNetV3-Large | IMAGENET1K_V2 | 232 / 224 / bilinear |
| MobileNetV3-Small | IMAGENET1K_V1 | 256 / 224 / bilinear |
| MobileNetV2 | IMAGENET1K_V2 | 232 / 224 / bilinear |
| EfficientNet-B0 | IMAGENET1K_V1 | 256 / 224 / bicubic |
| ShuffleNetV2 x1.0 | IMAGENET1K_V1 | 256 / 224 / bilinear |
| MNASNet 1.0 | IMAGENET1K_V1 | 256 / 224 / bilinear |
| RegNetY-400MF | IMAGENET1K_V2 | 232 / 224 / bilinear |
| ResNet-18 | IMAGENET1K_V1 | 256 / 224 / bilinear |
| ResNet-50 | IMAGENET1K_V2 | 232 / 224 / bilinear |
| ConvNeXt-Tiny | IMAGENET1K_V1 | 236 / 224 / bilinear |

- **Licences.** torchvision's code is BSD-3-Clause. None of these weights carries a licence entry in
  torchvision's metadata (the only ones that do, the SWAG weights for RegNet and ViT, are under a
  non-commercial licence and are excluded). Each model is recorded in Brokkr's model list with the
  same wording as MobileNetV3-Large: no separate weights licence stated by torchvision; ImageNet's
  terms are non-commercial research. A model is not run until its entry exists (hard rule 8).
- **Weights files** were downloaded on 26 September 2026 from download.pytorch.org, and each file's
  SHA-256 matches the hash prefix in its file name.
- **Trimming rule, fixed before profiling:** the estimated 4.1 run time must fit in 30 hours of
  laptop time; if it doesn't, the slowest candidates are dropped first until it fits; if fewer than 8
  models fit, stop and ask. A model is never dropped because of an accuracy result.
- **Profiling (26 September 2026; speed only, no accuracy read).** Each candidate was profiled on
  1,000 tuning images with the accuracy-run settings (4 threads, batch 32, not pinned), plugged in,
  back to back after the INT8 builds (so the laptop was already warm; `results/profile/*_all10.json`,
  ConvNeXt-Tiny's INT8 in `*_int8.json`). Estimated 4.1 run time, from these measured rates: about
  14.5 hours (inference 13.2 h, of which ConvNeXt-Tiny 5.4 h; damage and normalising 1.2 h for four
  preprocessing groups; caches 7 minutes). This is under 30 hours, so **no candidate is trimmed**.
- **INT8 build results** (Percentile 99.99, the recipe below; built from clean commits `74e63b7`, and
  `8267797` / `e33195e` for the two rebuilds below):
  - 7 new builds passed every check at the first attempt. EfficientNet-B0's agreement with FP32 on
    256 tuning images (83.2%) is below the 90% warning line but above the 20% failure line; it stays
    in.
  - **MobileNetV3-Small: INT8 failed.** Rebuilt from the clean commit `8267797` (the first build's
    record was marked dirty): again top-1 agreement with FP32 on 256 tuning images of 2.7%, below the
    20% failure line (the model gave one class for about half the images). Checks made (diagnostics,
    not results):
    - per-channel weights are on for every model (`per_channel=True` in `brokkr/quantize.py`, and
      every INT8 file stores all its weight tensors per-channel, none per-tensor);
    - the FP32 file agrees with PyTorch on real images (100% top-1), so the pipeline is not at fault;
    - a MinMax build (not kept) also collapsed, so the calibration method is not the cause;
    - layer by layer (`scripts/26_int8_layer_divergence.py`, ONNX Runtime's `qdq_loss_debug`, 32
      tuning images; diagnostic record `results/checks/mobilenet_v3_small_int8_layer_divergence.json`),
      INT8 first drifts sharply from FP32 **inside the first block**: its depthwise
      convolution's output falls to 7.9 dB signal-to-noise (MobileNetV3-Large at the same place:
      20.0 dB), the largest single drop is at the first squeeze-and-excitation multiply
      (53.8 -> 10.2 dB), and the block's output is at 1.0 dB. MobileNetV3-Small has
      squeeze-and-excitation in its first block; MobileNetV3-Large's first comes later. A likely
      explanation, not a tested one.
    - Handling: reported as **"INT8 failed"**, with no model-specific settings; its **FP32 is still
      run** on all 13 conditions for absolute weakness. It is not in any INT8-based prediction.
  - **ConvNeXt-Tiny: built with one recorded exception.** The first attempt failed before
    calibrating: ONNX Runtime's preparation step (`quant_pre_process`) crashed in its symbolic shape
    inference on a `Range` operation. The build was retried with that shape inference switched off
    (`skip_symbolic_shape=True`, stored in the model's record). Checked first on MobileNetV3-Large:
    with the switch, all 670 stored tensors (weights, scales, zero-points) and all 554 operations are
    identical to the Stage 3 model, and so are the scores on 256 tuning images; only the shape
    annotations differ (metadata, not used in any calculation). ConvNeXt-Tiny then built and passed
    every check (agreement with FP32 93.4% on 256 tuning images). No other model uses the switch.
- **Final list for the predictions below: the models whose INT8 passed its checks**: MobileNetV3-Large,
  MobileNetV2, EfficientNet-B0, ShuffleNetV2 x1.0, MNASNet 1.0, RegNetY-400MF, ResNet-18, ResNet-50,
  ConvNeXt-Tiny (9 models; "half of the models" below means at least 5 of them). MobileNetV3-Small is
  reported as "INT8 failed" and measured in FP32 only; it is never silently dropped.
- **Preprocessing**: each model uses torchvision's resize size, crop and interpolation for its
  weights; damage is applied after that model's own resize and crop, as in Stage 2. So different
  models see damaged versions of slightly different pictures: comparisons within a model are paired,
  and across models the *gaps* are compared, not pixel-identical images.

### FP32 sanity check, per model

Clean test-split FP32 top-1 must be within **±1.0 point** of torchvision's published top-1 for those
weights (read from torchvision's metadata). **Why this tolerance:** we evaluate on 10,000 images,
torchvision on all 50,000. The binomial standard error of an accuracy in the 65–85% range on 10,000
images is under half a point, so a 95% sampling range stays under ±1 point; a larger miss points to
a preprocessing or export mistake, not chance. A model that fails is left out of every 4.1 result
until the cause is found; the fix and the rerun are logged as a technical failure.

### INT8

Percentile 99.99, Stage 3's choice, applied as it is (not re-chosen per model): the 512
`int8_calibration` images in groups of 128, per-channel int8 weights, per-tensor uint8 activations,
ONNX Runtime static quantization (QDQ). The Stage 3 build checks apply (loads, finite scores,
per-channel weights). A model whose INT8 build fails a check is reported as failed, not dropped.

### Conditions (test split, 10,000 images, 13 in all)

| Suite | Corruption | Severities |
|---|---|---|
| — | clean | — |
| Brokkr | fog | 3 |
| Brokkr | darkness | 5 |
| Brokkr | defocus blur | 3 |
| Brokkr | noise | 3 |
| ImageNet-C | fog | 3, 5 |
| ImageNet-C | contrast | 3, 5 |
| ImageNet-C | defocus blur | 3, 5 |
| ImageNet-C | Gaussian noise | 3, 5 |

- **ImageNet-C** conditions are made with the official corruption code: `imagecorruptions` 1.1.2
  (Apache-2.0), vendored in `brokkr/third_party/imagecorruptions` with one change for NumPy 2
  (`np.float_` -> `np.float64` in fog's `plasma_fractal`, an alias NumPy 2 removed; the diff is in the
  repo, and a test shows the vendored fog equals the original pixel for pixel). The damage is applied
  to the same 224x224 picture as Brokkr's own, before normalisation.
- **Seeds.** Every damaged picture, Brokkr's or ImageNet-C's, uses one fixed seed per image: its
  dataset position. The seed is **shared across all conditions and severities** (Stage 2's rule: the
  same image gets the same random pattern at every severity, so severities differ only in strength).
  Because it is Stage 2's rule, Stage 3's damaged pictures are **reproduced exactly** (tested in
  `tests/test_sweep.py`, and checked on real scores at the start of every sweep). The ImageNet-C
  code draws from NumPy's global generator, so `brokkr.imagenet_c.damage` seeds it for each call
  and restores it afterwards; a test shows the same batch made twice has identical pixels, whatever
  else used the generator in between.
- **Pre-4.1 checks (done 26 September 2026):** licences of the package and its dependencies recorded
  (`pyproject.toml`); installs cleanly with NumPy 2.2.6 and scikit-image 0.25.2 except fog (fixed as
  above; glass blur and Gaussian blur also fail, unused); outputs on 5 sample images: see the
  picture sheet `results/samples/imagenet_c_check.png` (for viewing only, not a measurement).
- **How our ImageNet-C differs from the released files** (from the original generation script,
  `hendrycks/robustness`, `make_imagenet_c.py`): the released images were made from ImageNet
  validation photos resized to 256 and centre-cropped to 224 for every model, with unseeded random
  numbers, and saved as JPEG at quality 85. Ours use each model's own resize, fixed seeds, no JPEG
  re-save, and only our 10,000 test images. Our numbers are therefore **not directly comparable** to
  published ImageNet-C results, and are never presented as if they were.
- **Darkness stays Brokkr's own**: ImageNet-C has no darkening corruption.

### How the runs are organised (the approved restructure)

- Models with the same preprocessing share one cache of clean pictures, keyed by everything that made
  it (split, positions, source files, resize/crop/interpolation, code and package versions; a stale
  key means a rebuild). Each damaged batch is made once and fed to every model and precision sharing
  that preprocessing. Damaged batches are regenerated, never cached.
- Before its first condition, every sweep checks a sample of cached pictures against freshly made
  ones, and (for MobileNetV3-Large) that its clean and Brokkr-damaged scores reproduce the Stage 3
  saved scores exactly on a sample. A mismatch stops the run.
- Preprocessing groups: 232 bilinear (MobileNetV3-Large, MobileNetV2, RegNetY-400MF, ResNet-50);
  256 bilinear (ShuffleNetV2, MNASNet, ResNet-18, MobileNetV3-Small FP32); 256 bicubic
  (EfficientNet-B0); 236 bilinear (ConvNeXt-Tiny).
- The 4.1 compute runs and 4.2's never overlap.

### Metrics

For every (model, precision, condition): top-1 (with top-5 as recorded), ECE, conformal coverage with
average set size (threshold from the same model and precision's clean `conformal_calibration`
images, 5,000), and E-AURC, all from saved logits. The alarm needs tuning-split windows for each
model, so it is measured in 4.2, not here.

### Predictions

Judged per model only for models that pass the FP32 sanity check and the INT8 build checks. The 12
damaged conditions are the 13 above minus clean; near-floor cells are left out of every count.

**H18. A leaderboard tells you which model is more accurate under damage, but not which one
quantization hurts.** Across models, Spearman rank correlation, with a 95% bootstrap interval over
models (1,000 resamples of the model list, seed 0; a resample with fewer than 4 distinct models is
redrawn):
- **H18a.** Clean FP32 top-1 vs INT8 top-1 under the condition: interval above zero in at least 8 of
  the 12 damaged conditions. *Confidence: medium-high.*
- **H18b.** Clean FP32 top-1 vs the compression-caused gap under the condition: interval includes
  zero in at least 8 of the 12. *Confidence: medium.* **This is a weak test**: with 9 models only
  a strong relation makes an interval exclude zero, so a PASS means "no strong relation found", not
  "no relation".

**H19. A model's clean INT8 loss predicts its INT8 loss under damage.** Spearman rank correlation
across models between the clean compression-caused gap and the compression-caused gap under the
condition: interval above zero in at least 6 of the 12 damaged conditions. *Confidence: low-medium.*
If this holds, testing INT8 on clean images at least ranks models correctly, even when it
underestimates the damage.

**H20. INT8's extra gap is large under ImageNet-C contrast**, as the level-wasting mechanism
predicts (contrast squeezes values into a narrow range, like fog). Contrast (ImageNet-C) s3: extra
gap at most −5.0 points with its paired interval below zero, for at least half of the models (5 of
the 9). Severity 5 is reported, not judged (more likely near floor). *Confidence: medium.*

**H21. The darkness result generalises.** Darkness (Brokkr) s5: extra gap at most −5.0 points with
its paired interval below zero, for at least half of the models (5 of the 9). *Confidence: medium.*
Stage 3 found this for one model only.

**H22. The control: noise does not squeeze values, so its extra gap is smaller than contrast's.**
For each model, contrast (ImageNet-C) s3 extra gap minus Gaussian noise (ImageNet-C) s3 extra gap:
below zero with its paired interval below zero, for at least half of the models (5 of the 9).
*Confidence: low-medium.* If H20 passes and H22 fails, "damage in general hurts INT8 more" explains the data
as well as the mechanism does.

- **Why −5.0 points** (H20, H21): large enough to change which model a user should pick, and many
  times the build-to-build range measured in Stage 3 (0.28 points of clean top-1 across four
  calibration sets).
- **Absolute weakness is reported next to every compression-caused number**, in points and as a
  fraction of FP32 (relative gap), but not judged.

## Task 4.2: depth (the Stage 3 pipeline on two more models)

- **Models:** EfficientNet-B0 and ResNet-18.
- **Why ResNet-18** (recorded before its runs): cheap to run; a different design family (plain
  convolutions, no depthwise layers), unlike MobileNetV3 and EfficientNet-B0; and the most-studied
  model in quantization papers, so results can be compared with published work.
- **Phase A (every model):** tuning and `conformal_calibration` outputs, clean and damaged (as 3.1);
  INT8 method choice on tuning (as 3.2); temperature (3.5); robust conformal and alarm (3.6); one
  final test run from a tagged commit (3.7); ImageNetV2 (3.9).
- **Phase B (only where INT8 degrades):** damaged-image calibration, leave-one-out (3.3), and
  unrounded output (3.4).
- **"INT8 degrades", fixed now, judged on the tuning split** right after Phase A's method choice,
  using the chosen INT8 method and the Stage 3 conditions (Brokkr's 25 damaged conditions plus
  clean). Phase B runs if **either** holds:
  - clean: chosen INT8 minus FP32 top-1 is −1.0 point or lower, with its paired interval below zero;
  - damaged: in any of the 25 damaged conditions, the compression-caused gap is −5.0 points or
    lower, with its paired interval below zero.
  - If neither holds, Phase B is skipped for that model, and the numbers that decided it are
    recorded.
- **Why these numbers:** 1.0 point is several times the measured build-to-build range (0.28 points);
  −5.0 points matches "large" in H20 and H21.
- **4.2 predictions** (Stage 3's H10–H17, restated for each new model) are added in a dated note
  before any 4.2 measurement.

## Later tasks

The exact reliability-envelope wording (task 4.3) and the laptop-vs-Pi agreement rule (task 4.4) are
added as dated notes before those tasks measure anything.

## What would surprise us most

- M1 rejected for darkness: dark images use as many levels as clean ones. The Report 1 explanation
  would then be wrong, and the darkness collapse would need another cause.
- H18a failing: if clean accuracy did not even rank models under damage, leaderboards would be of
  little use for choosing edge models.
- ResNet-18 or ResNet-50 showing a large extra gap under contrast: plain-convolution models are
  usually reported to quantize well on clean images.

## Note added 26 September 2026, before the mechanism test (task 4.0) is run: how M1 is computed

Implementation details only; the design, thresholds and verdict rules above are unchanged.
- **Tensors:** in each model file, the outputs of the `QuantizeLinear` nodes whose input is not a stored
  constant (so weights are excluded). Both files must have the same 142 tensor names, else the test
  stops.
- **Levels:** for one image and one tensor, the number of distinct 8-bit values among all of that
  tensor's elements for that image (all channels and positions).
- **Per image:** the median over the 142 tensors, as written above. **Comparisons (a) and (b)** use
  the mean over the 500 images of these per-image values, with `brokkr.accuracy.paired_bootstrap_diff`
  (new minus old; 1,000 resamples, seed 0): (a) damaged minus clean, default INT8; (b) Percentile
  minus default, on the same damaged images.
- **"At least 20% of the clean images' value":** (a)'s difference is at most −0.20 times the mean over
  the 500 clean images of default INT8's per-image value.
- **Reading the levels:** each of the 142 outputs is added as an extra model output. The check before
  anything is counted: on all 500 clean images, the modified model's top answer equals the
  unmodified model's for every image (and the largest score difference is reported).
- **Pictures:** MobileNetV3-Large's own preprocessing (232 / 224 / bilinear); damage by
  `brokkr.sweep.damaged_batch` (seed = the image's dataset position).
- **Reported, not judged:** Percentile INT8's own damaged-minus-clean difference, and the mean levels
  of every tensor.
- **Records:** six schema-2 `levels` records (2 models x 3 conditions; tuning split, 500 images), each
  with the mean per-image value and its 95% bootstrap interval; the verdicts in
  `results/final/mobilenet_v3_large_m1_verdict.json`. Script: `scripts/27_mechanism_levels.py`.

## M1 outcome (added 26 September 2026, after the mechanism test)

Run once, at commit `4c0944f` with no uncommitted changes (`scripts/27_mechanism_levels.py`); no
reruns. The check before counting passed for both models: same top answer as the unmodified model on
all 500 clean tuning images, largest score difference 0.

Mean over the 500 tuning images of each image's median levels used (95% interval):

| Model | Clean | Darkness (Brokkr) s5 | Fog (Brokkr) s3 |
|---|---|---|---|
| Default INT8 (MinMax) | 67.07 (66.84 to 67.31) | 59.33 (59.13 to 59.53) | 61.27 (61.01 to 61.53) |
| Percentile 99.99 INT8 | 127.93 (127.60 to 128.28) | 114.98 (114.64 to 115.35) | 121.75 (121.28 to 122.24) |

- **Darkness (Brokkr) s5: INCONCLUSIVE.** (a) damaged minus clean, default INT8: −7.74 levels (−7.95
  to −7.57), which is −11.5% of the clean value; "supports" needed −13.41 or lower (−20%). (b)
  Percentile minus default on the dark images: +55.66 (+55.36 to +55.95), as required.
- **Fog (Brokkr) s3: INCONCLUSIVE.** (a) −5.80 (−6.00 to −5.60), −8.7% of the clean value; needed
  −13.41 or lower. (b) +60.48 (+60.11 to +60.87), as required.
- By the rule: damaged images do use fewer levels (so M1 is not rejected), but by less than the 20%
  fixed in advance, so the test neither supports nor rejects the explanation.
- Reported, not judged: Percentile INT8's own damaged minus clean: darkness −12.95 (−13.24 to
  −12.62), fog −6.18 (−6.55 to −5.79).

### Exploratory, after the verdict (not part of M1; changes no verdict)

From the saved per-tensor means (`results/levels/*_levels.json`, field `raw`):
- The reduction is concentrated at the start of the network. The quantized input image uses on
  average 241.0 levels when clean and 64.2 at darkness s5 (149.5 at fog s3), in both models. Across
  all 142 tensors the median relative change is much smaller (default INT8: −5.5% at darkness, −4.3%
  at fog). A likely reason the pre-registered summary (a median over all 142 tensors) found less than
  20%: it weighs the few early tensors where the squeeze is large the same as the many later ones
  where it is small. This is an observation after the verdict, not a tested claim.
- Percentile INT8 uses about twice as many levels as default INT8 on clean images too (127.93 vs
  67.07), so (b) shows that Percentile spends more levels in general, not specifically on damaged
  images.

## M2: where does INT8's extra error under darkness and fog arise?

Added 26 September 2026, before any 4.1 result exists.

Written before any 4.1 result exists; **M2 runs after the 4.1 sweep**. M1 (MobileNetV3-Large) is
accepted as recorded and is not re-run with another summary. M2 asks a follow-up question on models
M1 never used.

**Question.** For darkness (Brokkr) s5 and fog (Brokkr) s3, in which layers does INT8 add more
rounding error than it does on clean images?

**Design, fixed now.**
- **Models (8):** MobileNetV2, EfficientNet-B0, ShuffleNetV2 x1.0, MNASNet 1.0, RegNetY-400MF,
  ResNet-18, ResNet-50, ConvNeXt-Tiny. Not MobileNetV3-Large (used by M1) and not MobileNetV3-Small
  (its INT8 build failed).
- **Precisions:** Percentile 99.99 INT8 (the 4.1 builds) and default INT8 (MinMax, Stage 1's recipe:
  the 512 `int8_calibration` images; ConvNeXt-Tiny with the same recorded `skip_symbolic_shape`
  switch). The MinMax builds were made on 26 September 2026, before this section was committed:
  rebuilt the same way, MobileNetV3-Large's MinMax model has exactly the file bytes of its Stage 1
  default INT8, so the recipe is Stage 1's. Seven passed every check; **EfficientNet-B0's MinMax build
  failed** (top-1 agreement with FP32 on 256 tuning images 17.6%, below the 20% line), so it is
  reported and left out of the default-INT8 verdicts. So the default-INT8 verdicts count **7 models**
  and the Percentile verdicts **8 models**.
- **Images:** 128 tuning images, positions 500–627 of the tuning split in split order (images 501–628;
  none used by M1). Tuning only, never test images. They are processed in batches of 8 and every
  number is computed per image before any summary, so memory is bounded by one batch whatever the
  model. Measured on 26 September 2026 on the heaviest model (ConvNeXt-Tiny FP32, all 217 tensors
  exposed, 8 images): peak memory 2.87 GB, well within this laptop's 15.7 GB.
- **Conditions:** clean; darkness (Brokkr) s5; fog (Brokkr) s3; each model's own preprocessing;
  damage seed = the image's dataset position.
- **Tensors:** each activation `QuantizeLinear` in the INT8 model, in graph order, matched by name to
  the same tensor in the FP32 model after ONNX Runtime's preparation step. Tensor #0 is the input
  image. Tensors that cannot be matched are counted and listed, and left out.

**Main measure: local rounding error.** For one image and one tensor: take the FP32 model's values
there, quantize them with that tensor's INT8 scale and zero-point (from the INT8 model file): divide,
round halves to even, add the zero-point, **clip to the 8-bit range**, then turn them back into numbers
(`brokkr.levels.fake_quantize`). The clip matters: Percentile's error from cutting off rare large
values is part of the measure. A test shows this equals ONNX Runtime's own QuantizeLinear followed by
DequantizeLinear exactly, on made-up values beyond the range and on every quantized tensor of
MobileNetV3-Large's Percentile INT8, some of which are clipped (`tests/test_levels.py`). Then,
and compare with the unrounded values as a signal-to-noise ratio,
SQNR = 20·log10(‖x‖ / ‖x − x̂‖) in dB (ONNX Runtime's formula). This is the error INT8 adds *at that
tensor alone*, given perfect inputs; unlike the cumulative measure below, it contains no error
carried forward from earlier layers, so it can show where extra error arises.
- **Per image first:** SQNR is computed for each image separately; summaries are over images.
- **Extra error at a tensor:** E(t) = the mean over images of [SQNR_clean(t) − SQNR_damaged(t)] for
  the same image, in dB (positive = the damage makes INT8 round that tensor worse).
- **Early layers:** the first 10% of a model's matched tensors after the input (tensors #1 to
  #ceil(N/10), where N is the number of matched tensors after #0). **Rest:** every tensor after the
  early block. The input tensor (#0) is reported separately and belongs to neither.
- **Summaries per model, precision and condition:** E_early = the median of E(t) over the early
  block; E_rest = the median over the rest. 95% intervals by resampling the 128 images (1,000
  resamples, seed 0) and recomputing E(t), E_early and E_rest from the per-image SQNRs.
- **Reported, not judged:** E_early with 5% and 20% cut-offs instead of 10%; the cumulative SQNR of
  `scripts/26_int8_layer_divergence.py` (INT8 model vs FP32 model at each tensor, error carried
  forward included), also per image.

**Input images before quantization (M2b).** For each image and each colour channel: V_pre = the
number of distinct values in that channel of the normalised input (what the input quantizer
receives), V_q = the number of distinct 8-bit levels in that channel after the input quantizer.
Per image, R = the average over the three channels of V_q / min(V_pre, 256).

**M2a. The extra rounding error arises early.** Four verdicts: {Percentile, default} x {darkness s5,
fog s3}. Judged on the local measure only.
- **Supports**, for that precision and condition, if in **at least 6 of the models** (6 of 8 for
  Percentile, 6 of 7 for default):
  E_early ≥ 3.0 dB with its interval above 0, and E_early > E_rest with the interval of
  (E_early − E_rest) above 0.
- **Rejects** if in **more than half of the models** (5 of 8 for Percentile, 4 of 7 for default)
  E_early ≤ E_rest (the point estimate), or E_early's
  interval includes 0: the extra error is not concentrated early.
- **Inconclusive** otherwise. Models whose E(t) never reaches 1.0 dB at any tensor are listed as "no
  extra error"; they count against "supports".
- *Confidence:* medium for default INT8 under darkness; low for Percentile (it may have removed most
  of the early squeeze) and for fog.

**M2b. At the input, the loss is the image, not INT8.** Percentile INT8 judged; default reported, not
judged. For darkness s5 and fog s3 separately:
- **Supports** if in **at least 6 of the 8 models** the 95% interval of the mean of
  R(damaged) − R(clean) (paired over images) lies inside −0.05 to +0.05.
- **Rejects** if in **at least 5 of the 8 models** that interval lies entirely below −0.05 (INT8's
  input quantizer itself drops many of the values a damaged image still has).
- **Inconclusive** otherwise. *Confidence:* medium.

- **Why 3.0 dB:** 3 dB means the rounding error's power relative to the signal doubles.
- **Why 10% for "early":** in MobileNetV3-Small the collapse was inside the first block, under 10%
  of its tensors; one cut-off is fixed for every model so that no boundary is chosen per model. The
  5% and 20% cut-offs are reported to show how much the answer depends on it, without judging them.
- **Why these counts:** "supports" needs at least three quarters of the models (6 of 8; 6 of 7,
  rounded up), as in H18's 8 of 12; "rejects" needs more than half (5 of 8; 4 of 7). The two cannot
  both hold.
- **What M2 cannot show:** that extra early rounding error *causes* the accuracy loss. Even
  "supports" will be written as "consistent with".
- **Check before judging:** the script's cumulative SQNR, pooled over images as ONNX Runtime pools
  them, must reproduce `scripts/26_int8_layer_divergence.py`'s saved values for MobileNetV3-Small
  (32 images) to 0.01 dB. If it does not, M2 stops and a dated note decides what to do.
- **Records:** schema-2 diagnostic records per model and precision (script, split, images, model,
  commit), and the verdicts in `results/final/`; the MinMax builds are build records checked by
  `scripts/22_check_results.py`.

## Exploratory note (26 September 2026): squeeze-and-excitation and weak INT8 builds

**Not a prediction and not judged.** A possible later test, recorded now because it was noticed
before any 4.1 result exists.

All three models with weak or failed INT8 builds contain squeeze-and-excitation blocks:
MobileNetV3-Small (Percentile: failed, 2.7% agreement with FP32 on 256 tuning images; a diagnostic
MinMax build, not kept, also collapsed at 1.2%), EfficientNet-B0 (MinMax: failed, 17.6%) and
MobileNetV3-Large (MinMax: 63.7% on 256 tuning images in the 26 September rebuild check). RegNetY-400MF
also contains squeeze-and-excitation blocks and passed both builds (MinMax 92.2%, Percentile 94.5%).

Checked from the model code (`torchvision.ops.misc.SqueezeExcitation` modules): MobileNetV3-Large 8,
MobileNetV3-Small 9, EfficientNet-B0 16, RegNetY-400MF 16; the other six models none. Also noticed:
the two Percentile builds below the 90% warning line (MobileNetV3-Large 86.3%, EfficientNet-B0 83.2%)
are squeeze-and-excitation models too, and every build of a model without such blocks agreed with FP32
on at least 92.2% of the 256 tuning images. Four such models are too few to separate this from
other differences between the models.

## Note added 27 September 2026, before H18–H22 are judged: how the 4.1 rules are computed

Written after the 4.1 sweep finished and before any 4.1 test-split number beyond the FP32 sanity
checks was computed. Implementation details only; the predictions and thresholds above are unchanged.
The first four points were left open by the rules above and were decided by H (the owner) before any
4.1 test-split result was computed.
- **Near floor in a correlation (H18, H19):** a near-floor cell (FP32 top-1 below 10% for one model
  at one condition) is left out by dropping that model from that condition's correlation; the other
  models stay. If fewer than 4 models remain, that condition is "not judged".
- **Pass counts stay absolute:** H18a and H18b need 8 conditions, H19 needs 6, H20–H22 need 5 models,
  whatever is left out. A condition not judged, or a model left out, never counts towards a PASS.
- **Rank ties:** Spearman correlation uses average ranks for tied values (standard; equal to scipy's
  default). In the bootstrap over models, a resample with fewer than 4 distinct models, or where
  either variable is constant (so the correlation is undefined), is redrawn; redraws do not count
  towards the 1,000.
- **H22 near floor:** a model is left out if FP32 is near floor at contrast (ImageNet-C) s3 OR at
  Gaussian noise (ImageNet-C) s3.
- **Conventions reused from Stage 3 (`brokkr.judge`):** top-1 breaks score ties towards the lower
  class number, and must equal each record's saved top-1 exactly; paired intervals resample the same
  image positions for every side (1,000 resamples, seed 0, a fresh generator for each interval);
  intervals are percentile intervals (2.5% and 97.5%); an interval is "below zero" only if its upper
  end is below zero and "above zero" only if its lower end is above zero, so an interval ending
  exactly at zero "includes zero".
- **"At most −5.0 points"** (H20, H21) is compared in whole images, extra-gap count × 100 ≤ −5 × n,
  so floating-point rounding cannot move a value across the line.
- **Correlation inputs:** each model's top-1 values (point estimates) on the test split; the
  correlation interval resamples only the model list (1,000 resamples, seed 0, one fresh generator
  per condition and prediction).
- **Judged models:** FP32 sanity check passed (recomputed from the clean test record, same rule and
  tolerance as the sweep) and a usable INT8 build (`brokkr.schema.check_build_record`), with all 13
  conditions present at both precisions. Every record must hold the same images in the same order.
- **Code:** `brokkr/judge_breadth.py` (tested in `tests/test_judge_breadth.py`) and
  `scripts/32_judge_breadth.py`, which saves `results/final/breadth_4.1_verdicts.json` and prints,
  after the verdicts, absolute weakness next to every compression-caused number (reported, not judged).
- **Reliability** (reported, not judged): `scripts/31_breadth_reliability.py` computes ECE, conformal
  coverage with average set size, and E-AURC for every test-split record with Stage 2–3's functions;
  the conformal threshold comes from the same model and precision's clean `conformal_calibration`
  images (5,000).
- **Tool check before the real run:** both scripts were run on the 64-image tuning dry-run records
  (made at `4e4c4e5`); the conformal thresholds came from the real `conformal_calibration` records.
  These dry-run numbers test the code only and are never results.

## H18–H22 outcome (added 27 September 2026, after judging)

Run once on the 4.1 test-split results (10,000 images per record), at commit `1bcc63b` with no
uncommitted changes (`scripts/32_judge_breadth.py`); no reruns. Verdicts:
`results/final/breadth_4.1_verdicts.json`. All 10 FP32 sanity checks passed; 9 models judged
(MobileNetV3-Small left out: INT8 build failed). Gaussian noise (ImageNet-C) s5 is not judged in
H18–H19: only ConvNeXt-Tiny is above the floor there (fewer than 4 models).

| | Verdict | Count |
|---|---|---|
| H18a | **FAIL** | 3 conditions hold, 8 needed (11 of 12 judged) |
| H18b | **PASS** | 8 hold, 8 needed (11 of 12 judged) |
| H19 | **PASS** | 9 hold, 6 needed (11 of 12 judged) |
| H20 | **FAIL** | 2 models hold, 5 needed (9 of 9 judged) |
| H21 | **FAIL** | 2 hold, 5 needed (9 of 9 judged) |
| H22 | **FAIL** | 4 hold, 5 needed (8 of 9 judged) |

- **H18a.** Holds for fog (Brokkr) s3 (rho +0.767, 95% interval +0.123 to +1.000), noise (Brokkr) s3
  (+0.983, +0.739 to +1.000) and Gaussian noise (ImageNet-C) s3 (+0.929, +0.615 to +1.000; 8 models).
  The other 8 judged conditions: rho +0.350 to +0.600, every interval includes zero.
- **H18b.** The interval excludes zero (below zero) for noise (Brokkr) s3 (rho −0.883, −1.000 to
  −0.459), fog (ImageNet-C) s5 (−0.733, −1.000 to −0.030) and Gaussian noise (ImageNet-C) s3 (−0.810,
  −1.000 to −0.215; 8 models); it includes zero for the other 8 judged conditions. As stated above,
  a PASS of this test means "no strong relation found", not "no relation".
- **H19.** The interval includes zero for contrast (ImageNet-C) s3 (rho +0.644, −0.244 to +0.983) and
  s5 (+0.543, −0.333 to +1.000; 6 models); above zero for the other 9 judged conditions (lowest lower
  end: +0.115).
- **H20** (contrast (ImageNet-C) s3, extra gap in points, paired 95% interval): holds for
  MobileNetV3-Large −13.23 (−14.20 to −12.28) and EfficientNet-B0 −37.87 (−39.09 to −36.58). Not for
  ConvNeXt-Tiny −4.19 (−4.77 to −3.55), MobileNetV2 −1.98 (−2.58 to −1.40), RegNetY-400MF −1.03
  (−1.65 to −0.41), ResNet-50 −0.71 (−1.25 to −0.19), ResNet-18 −0.60 (−1.07 to −0.14), ShuffleNetV2
  −0.42 (−1.02 to +0.19), MNASNet +0.98 (+0.35 to +1.63).
- **H21** (darkness (Brokkr) s5): holds for MobileNetV3-Large −11.60 (−12.53 to −10.76) and
  EfficientNet-B0 −38.94 (−40.13 to −37.78). Not for ShuffleNetV2 −1.75 (−2.32 to −1.15), ConvNeXt-Tiny
  −1.21 (−1.71 to −0.67), MNASNet −1.05 (−1.67 to −0.41), MobileNetV2 −1.01 (−1.63 to −0.42),
  RegNetY-400MF −0.89 (−1.47 to −0.33), ResNet-50 −0.40 (−0.90 to +0.15), ResNet-18 −0.24 (−0.72 to
  +0.22).
- **H22** (contrast s3 extra gap minus Gaussian noise s3 extra gap): holds for MobileNetV3-Large −12.55
  (−13.64 to −11.54), MobileNetV2 −2.38 (−2.98 to −1.81), EfficientNet-B0 −40.16 (−41.33 to −39.01),
  ResNet-18 −0.66 (−1.13 to −0.20). Not for RegNetY-400MF −0.64 (−1.33 to +0.03), ConvNeXt-Tiny −0.29
  (−1.08 to +0.47), ResNet-50 +0.44 (−0.19 to +1.06), MNASNet +0.95 (+0.31 to +1.57). ShuffleNetV2
  left out (near floor: FP32 5.44% at Gaussian noise (ImageNet-C) s3).
- Check: MobileNetV3-Large reproduces Stage 3 (FP32 clean 75.58%, Percentile INT8 73.60%; darkness
  (Brokkr) s5 73.73% and 60.15%).
- Reliability (reported, not judged): `scripts/31_breadth_reliability.py` at `1bcc63b`, 741 records
  (ECE, conformal coverage with set size, E-AURC) from 247 test results, all checks PASS; records in
  `results/breadth_reliability/`.

### Exploratory, after the verdicts (not part of H18–H22; changes no verdict)

- H22 depends on the near-floor rule fixed in the note above: ShuffleNetV2's difference, −1.31
  (−1.80 to −0.78), would have been a fifth holding model had only the contrast cell been checked.
  The rule was fixed before any test-split result was computed, so the verdict stands.
- The only large extra gaps in H20 and H21 are MobileNetV3-Large and EfficientNet-B0, both
  squeeze-and-excitation models (see the exploratory note of 26 September); RegNetY-400MF also has
  such blocks and its extra gaps are small (−1.03, −0.89). Consistent with "a few models collapse"
  rather than "INT8 generally collapses", and not tested.
- H18a: with 9 models, correlations of about +0.35 to +0.6 give intervals that include zero; likely
  a small-sample limit, as the plan anticipated.

## Note added 27 September 2026: at least 6 models in any correlation from now on

Decided by H after the exploratory H19 check above, where a correlation with 4 models had an
interval of zero width (with 4 models, the only resamples that are not redrawn contain every model
once, so every resample gives the same value).
- From now on, **any correlation across models needs at least 6 distinct models** after near-floor
  models are left out; with fewer, that correlation is "not judged" (or, in a reported analysis,
  "not computed").
- The bootstrap redraw rule is unchanged: a resample with fewer than 4 distinct models (or an
  undefined correlation) is redrawn. (With a redraw threshold of 6, a 6-model correlation would again
  have an interval of zero width.)
- This changes no existing verdict. H18a, H18b and H19 were judged with the minimum of 4 fixed before
  they ran; the smallest judged condition had 6 models (contrast (ImageNet-C) s5), and Gaussian noise
  (ImageNet-C) s5, not judged, had 1.

## Note added 27 September 2026, before M2 runs: how M2 is computed

Written before any M2 number is computed. The design, thresholds and counts in the M2 section above are
unchanged. The first three points were left open by the M2 rules and were decided by H before running.
- **M2a "rejects", per model:** a model counts towards "rejects" if E_early ≤ E_rest (point estimates)
  OR its E_early interval is not above zero; "rejects" if that count reaches 5 of 8 (Percentile) or 4
  of 7 (default).
- **"E_early's interval includes 0" is read as "not above zero":** an interval entirely below zero
  (damage makes early rounding error smaller) also counts towards "rejects".
- **M2b boundaries:** "inside −0.05 to +0.05" includes the ends (lower ≥ −0.05 and upper ≤ +0.05);
  "entirely below −0.05" is strict (upper < −0.05).
- **Verdict order:** SUPPORTS if at least 6 models support; otherwise REJECTS if the reject count is
  reached; otherwise INCONCLUSIVE (the section above shows the two cannot both hold).
- **Reading the tensors:** both the FP32 model (after `quant_pre_process`, with the build's recorded
  `skip_symbolic_shape`) and the INT8 model get ONNX Runtime's own augmentation
  (`qdq_loss_debug.modify_model_output_intermediate_tensors`), as in `scripts/26`; 4 threads; batches
  of 8 images. A tensor is matched if its name is saved by the FP32 model and its DequantizeLinear
  output is saved by the INT8 model; each QuantizeLinear input is listed once, in graph order.
- **Local measure:** `brokkr.m2.local_rounding` (FP32 values, `brokkr.levels.fake_quantize` with the
  tensor's scale and zero-point from the INT8 file). **Cumulative measure** (reported only): FP32 values
  vs the INT8 model's DequantizeLinear output. SQNR uses ONNX Runtime's guard (each norm at least the
  machine epsilon).
- **Check before judging:** the pooled cumulative SQNR (sums of squared norms over the 32 images, ONNX
  Runtime's pooling) must be within 0.01 dB of `scripts/26`'s saved value at every one of its tensors
  (those were saved rounded to 0.01 dB). If not, M2 stops.
- **Zero signal or zero error:** if any (image, tensor) has a signal or local error whose norm is at
  most the machine epsilon (so the guard, not the data, would set its SQNR), no verdict is computed and
  a dated note decides.
- **"No extra error":** the largest E(t) over all matched tensors, including the input (#0), is below
  1.0 dB. (Such a model cannot meet E_early ≥ 3.0 dB, so this label cannot change a verdict.)
- **Intervals:** 1,000 resamples of the 128 images, seed 0, a fresh generator for each (model,
  precision, condition); the E_early, E_rest and E_early − E_rest intervals come from the same
  resamples; percentile intervals (2.5%, 97.5%); "above zero" means the lower end is above zero.
- **M2b's R** uses each precision's own input quantizer (tensor #0); per image, the mean over the three
  channels of V_q / min(V_pre, 256).
- **Default INT8 models:** those whose MinMax build record is "usable" (7; EfficientNet-B0's failed).
- **Code and records:** `brokkr/m2.py` (tested in `tests/test_m2.py`) and
  `scripts/35_m2_rounding_error.py`; per-image numbers in `results/m2/*.npz`, schema-2 diagnostic
  records in `results/m2/`, verdicts in `results/final/m2_verdicts.json`.

## Note added 27 September 2026, before M2 is measured: a technical failure and how INT8 values are read

- **First run (commit `b1397aa`) crashed before measuring anything:** the image positions were passed
  as a Python list where the dataset reader needs an array. Technical failure; fixed, nothing was
  computed.
- **A dry run was then added and used** (`--dry-run`: 3 models, 16 tuning images at positions
  628–643, outside M2's images; output outside `results/`; a tool check, never a result). It found
  that ONNX Runtime's augmentation tool makes ConvNeXt-Tiny's INT8 graph invalid (it tries to save an
  int32 bias zero-point as a float tensor).
- **Change:** the INT8 model is read with ONNX Runtime's tool wherever that works (as in `scripts/26`,
  the path the check validates: largest difference 0.0050 dB); only where the tool makes the graph
  invalid are the DequantizeLinear outputs exposed directly, and the record says which was used.
  Exposing only those outputs lets ONNX Runtime fuse operations differently: in the dry run, using it
  for MobileNetV3-Small moved the pooled cumulative SQNR by up to 0.0758 dB, so it is a fallback
  only. This affects only the cumulative measure (reported, never judged); the judged local measure
  uses the FP32 values and the INT8 file's scales and zero-points and never runs the INT8 model.
- **Unmatched tensor, as the rule above says (counted, listed, left out):** in every dry-run model
  the final output's quantizer input, `logits_QuantizeLinear_Input`, has no FP32 tensor of that name
  (the INT8 build renames the model output); `scripts/26` has no such row either.

## M2 outcome (added 27 September 2026, after the run)

Run once to completion at commit `ed4f920` with no uncommitted changes
(`scripts/35_m2_rounding_error.py`); the earlier start at `b1397aa` crashed before measuring anything
(note above). Tuning split, positions 500–627 (128 images). Verdicts: `results/final/m2_verdicts.json`;
per-image numbers and schema-2 diagnostic records: `results/m2/`.
- **Check before judging: PASS.** Pooled cumulative SQNR for MobileNetV3-Small on 32 tuning images: 126
  tensors, largest difference from `scripts/26`'s saved values 0.0050 dB (tolerance 0.01).
- Default INT8 usable for 7 of 8 models (EfficientNet-B0's MinMax build failed). Every model had one
  unmatched tensor, `logits_QuantizeLinear_Input` (left out). ConvNeXt-Tiny's INT8 values (cumulative
  measure only) were read with directly exposed outputs, the other models' with ONNX Runtime's tool.
- (image, tensor) pairs with a zero signal or zero local error: 0.

| Verdict | Result | Counts |
|---|---|---|
| M2a Percentile, darkness (Brokkr) s5 | **REJECTS** | supporting 0 (6 needed), rejecting 6 (5 needed), 8 models |
| M2a Percentile, fog (Brokkr) s3 | **REJECTS** | supporting 0, rejecting 6 (5 needed), 8 models |
| M2a default, darkness (Brokkr) s5 | **INCONCLUSIVE** | supporting 0, rejecting 0 (4 needed), 7 models |
| M2a default, fog (Brokkr) s3 | **INCONCLUSIVE** | supporting 0, rejecting 1 (4 needed), 7 models |
| M2b Percentile, darkness (Brokkr) s5 | **SUPPORTS** | supporting 8 (6 needed), rejecting 0, 8 models |
| M2b Percentile, fog (Brokkr) s3 | **SUPPORTS** | supporting 8, rejecting 0, 8 models |

E_early, in dB, with its 95% interval (10% block):
- **Percentile, darkness s5:** MobileNetV2 +1.40 (+1.34 to +1.47), EfficientNet-B0 −0.34 (−0.51 to
  −0.29), ShuffleNetV2 −1.85 (−2.26 to −1.12), MNASNet −1.26 (−1.94 to −0.70), RegNetY-400MF +0.41
  (−0.07 to +0.72), ResNet-18 −2.17 (−3.15 to −1.43), ResNet-50 −0.09 (−0.41 to −0.04), ConvNeXt-Tiny
  +1.54 (+1.29 to +1.70). Rejecting: all but MobileNetV2 and ConvNeXt-Tiny.
- **Percentile, fog s3:** MobileNetV2 +0.79 (+0.75 to +0.83), EfficientNet-B0 −0.40 (−0.59 to −0.22),
  ShuffleNetV2 −1.29 (−1.80 to −0.84), MNASNet −2.31 (−2.92 to −1.62), RegNetY-400MF −0.58 (−1.00 to
  −0.19), ResNet-18 −2.37 (−3.36 to −1.64), ResNet-50 −0.20 (−0.54 to +0.10), ConvNeXt-Tiny +0.53 (+0.38
  to +0.65). Rejecting: all but MobileNetV2 and ConvNeXt-Tiny.
- **Default, darkness s5:** MobileNetV2 +1.43 (+1.38 to +1.49), ShuffleNetV2 +1.06 (+0.94 to +1.20),
  MNASNet +2.20 (+2.05 to +2.35), RegNetY-400MF +1.68 (+1.60 to +1.81), ResNet-18 +1.75 (+1.56 to +1.95),
  ResNet-50 +0.64 (+0.56 to +0.72), ConvNeXt-Tiny +2.01 (+1.91 to +2.08). None reaches 3.0 dB.
- **Default, fog s3:** MobileNetV2 +0.80, ShuffleNetV2 +1.43, MNASNet +1.49, RegNetY-400MF +0.69,
  ResNet-18 +1.39, ResNet-50 +0.32, ConvNeXt-Tiny +0.96 (every interval above zero). ResNet-50 rejects:
  E_early − E_rest −0.11 (−0.15 to −0.06).
- **M2b, Percentile:** mean R(damaged) − R(clean), darkness s5: +0.0047 to +0.0055 for all 8 models,
  every interval inside ±0.05; fog s3: −0.0002 to +0.0003, every interval inside ±0.05.
- Per-model E_rest, E_early − E_rest, the 5% and 20% blocks, M2b for default INT8 and the cumulative
  SQNR are in `results/final/m2_console.txt` and the records (reported, not judged).

### Exploratory, after the verdicts (not part of M2; changes no verdict)

- Under Percentile INT8, E_early is negative for most models (damage makes the early tensors' local
  rounding error smaller). A likely reason, not tested: Percentile clips rare large values, and dark
  or foggy images have fewer of them to clip.
- The largest single-tensor E(t) values (up to +12.10 dB, RegNetY-400MF, Percentile, darkness s5) are
  not in the early block; where they sit is not yet described.

## Squeeze-and-excitation diagnostic (SE1): design and pass rule, added 27 September 2026

Written after M2 and its exploratory analysis (`scripts/36_m2_top_tensors.py`), before anything below
is built or measured. A diagnostic: it changes no earlier result, and the official 4.1 builds and
records stay as they are. Pass rule set by H; the reading of its details marked "(proposed)" is to be
confirmed by H before the run.

**Question.** Does keeping the squeeze-and-excitation blocks in float remove most of INT8's extra gap
under darkness? Squeeze-and-excitation blocks (`torchvision.ops.misc.SqueezeExcitation`) are the small
side branches that rescale each channel. In 4.1 the two models with large extra gaps, EfficientNet-B0
(darkness (Brokkr) s5 −38.94 points, contrast (ImageNet-C) s3 −37.87) and MobileNetV3-Large (−11.60,
−13.23), both have them; so does MobileNetV3-Small, whose INT8 build failed.

**Models.**
- Judged: EfficientNet-B0 and MobileNetV3-Large.
- Control, judged: RegNetY-400MF. It has squeeze-and-excitation blocks and the largest
  squeeze-and-excitation E(t) in M2's exploratory analysis, but small 4.1 extra gaps (−0.89 darkness
  s5, −1.03 contrast s3).
- Reported, not judged: MobileNetV3-Small (third fragile model; its Percentile INT8 failed its build
  check, 2.7% agreement with FP32 on 256 tuning images).

**Builds.** For each of the four models, two new INT8 variants, made exactly like its 4.1 Percentile
99.99 build (the same 512 `int8_calibration` images in groups of 128, per-channel int8 weights,
per-tensor uint8 activations, the same recorded `skip_symbolic_shape` setting) except:
- **SE-all:** every node inside a squeeze-and-excitation module stays in float (not quantized);
- **SE-output:** only the squeeze-and-excitation output path stays in float: the second 1×1
  convolution (`fc2`), the scale activation (`scale_activation`) and the multiply that rescales the
  block's input.
- Nodes are chosen from the FP32 export's own module records (`pkg.torch.onnx.class_hierarchy` and
  `name_scopes`, as in `scripts/36`), never from tensor names, and matched to the prepared model by
  output tensor name. The number of nodes kept in float, and the file size, are recorded per build.
- Every variant gets the usual INT8 build checks (loads, finite scores, per-channel weights,
  agreement with FP32 on 256 tuning images). A variant below the 20% agreement line is reported as
  "failed the build check" and still measured (for MobileNetV3-Small that agreement is itself the
  question). The comparison baselines are the existing Percentile builds (the Stage 3 file for
  MobileNetV3-Large, the 4.1 files for the others); MobileNetV3-Small's failed build is measured too,
  as its baseline.

**Images and conditions.** Tuning split only, all 5,000 images, in split order; never the test
split. Each model's own preprocessing; damage seed = the image's dataset position (as in 4.1).
Conditions: clean; darkness (Brokkr) s5 (judged); contrast (ImageNet-C) s3 and fog (Brokkr) s3
(reported the same way, not judged). Precisions per model: FP32, the baseline Percentile INT8,
SE-all, SE-output.

**Measures.** Per model, build and damaged condition, on the same images:
- extra gap = (INT8 − FP32 top-1 under the condition) − (INT8 − FP32 top-1 on clean images), as in 4.1;
- change = extra gap of the variant minus extra gap of the baseline build (new minus old; positive =
  the variant loses less to the damage), with a paired bootstrap 95% interval (1,000 resamples of
  the 5,000 images, seed 0). Per image this is (variant damaged − variant clean) − (baseline damaged −
  baseline clean), so FP32 cancels;
- recovered share = change / (−baseline extra gap).
- Also reported: clean top-1 of every build, agreement with FP32, file size, and near-floor flags
  (FP32 below 10%).

**SE1 pass rule (judged on SE-all, darkness (Brokkr) s5).** PASS only if all three hold:
1. EfficientNet-B0: recovered share ≥ 50%, and the change's paired interval is above zero;
2. MobileNetV3-Large: the same;
3. RegNetY-400MF: the change is smaller than 2.0 points in size, |change| < 2.0 points.
Otherwise FAIL.
- (proposed) "At least 50%" and "less than 2 points" are compared on the point estimates, in whole
  images (no rounding can move a value across a line); the interval condition is on the change in
  points, not on the share.
- (proposed) If a judged model's baseline extra gap on these tuning images is not below zero, its
  share is undefined and that model's part fails.
- (proposed) If a judged model's SE-all build fails a build check, or a run fails technically,
  SE1 is NOT JUDGED and a dated note decides.
- SE-output, contrast s3, fog s3 and MobileNetV3-Small are reported with the same numbers, not judged.
- *Confidence (proposed): low.* The 4.1 extra gaps come from the test split; on tuning images they
  are measured afresh, and M2 found the largest EfficientNet-B0 E(t) under darkness outside the
  squeeze-and-excitation blocks.
- Even a PASS shows only that the squeeze-and-excitation path is involved ("consistent with"), not
  that it causes the collapse.

**What stays planned.** The diagnostic-only MobileNetV3-Small rebuild with only its first
squeeze-and-excitation multiply in float is not part of SE1 and is still to be decided.

**Code (to be written; details decided later go in a dated note before the run).** A build option
for nodes kept in float (stated in every later build record, like `skip_symbolic_shape`), node
selection from the exporter's module records, a build script, and an evaluation-and-judging script
that prints the verdict with the numbers that decided it. Tried first on a dry run outside `results/`.

## Note added 27 September 2026: SE1 details confirmed and changed by H (before any SE1 build)

This note replaces the points marked "(proposed)" in the SE1 section above; nothing has been built or
measured.
- **Confirmed:** "at least 50%" and "less than 2 points" are compared on the point estimates, in whole
  images, and the interval condition is on the change in points, not on the share; if a judged
  model's SE-all build fails a build check, or a run fails technically, SE1 is NOT JUDGED and a dated
  note decides; confidence: low.
- **Changed: a judged model must first have a baseline loss.** EfficientNet-B0 or MobileNetV3-Large
  "has a baseline loss" if its existing Percentile build's darkness (Brokkr) s5 extra gap on these
  tuning images is −2.0 points or lower (in whole images) with its paired 95% interval below zero.
  If either judged model has no baseline loss, **SE1 is NOT JUDGED**: there is nothing to recover. (This
  replaces "its share is undefined and that model's part fails".)
- **Added to the limits:** RegNetY-400MF's 4.1 extra gaps are about 1 point (−0.89 darkness s5, −1.03
  contrast s3), so its "change under 2 points" control is close to automatic and carries little
  evidence.
- **Dropped:** the separate MobileNetV3-Small rebuild with only its first squeeze-and-excitation
  multiply in float; SE-all and SE-output cover it.
- **Not done for now:** M2's layer measurement for MobileNetV3-Large.
