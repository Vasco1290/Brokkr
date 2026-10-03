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

## Note added 27 September 2026, before the SE1 builds: how SE1 is computed

Implementation details only; the design, the pass rule and H's confirmations above are unchanged.
- **Choosing the nodes** (`brokkr/se_float.py`, tested): from the module records the exporter wrote on
  each node of the FP32 file. SE-all: every node whose module classes include
  `torchvision.ops.misc.SqueezeExcitation`. SE-output: inside such a module, the nodes of its `fc2` and
  `scale_activation` submodules, and the multiply in the module's own forward. Counts read from the
  four FP32 files (squeeze-and-excitation modules; nodes SE-all / SE-output): EfficientNet-B0 16;
  112 / 48. MobileNetV3-Large 8; 48 / 24. RegNetY-400MF 16; 96 / 48. MobileNetV3-Small 9; 54 / 27.
- **Keeping them in float** (`brokkr.quantize.to_int8(keep_float_outputs=...)`, tested): the chosen
  nodes' output tensor names are looked up in the prepared model (after `quant_pre_process`) and those
  nodes are passed to ONNX Runtime as `nodes_to_exclude`. The build stops if any name is missing or its
  node is unnamed. In a check before the build code was written, every chosen output of all four
  models was found on a named node of the prepared model.
- **Builds** (`scripts/24_build_models.py --keep-float se-all|se-output`): saved as
  `models/<model>_int8_percentile99.99_seall.onnx` / `_seoutput.onnx`. Their records state
  `kept_float` (selection, node count, block count); every later build record states it (null for
  none), and `scripts/22_check_results.py` requires that. One extra build check: no tensor inside the
  kept path (a chosen output whose consumers in the export are all chosen nodes) is the input of a
  QuantizeLinear in the built model.
- **"A build fails a build check"** means its record is not "usable" by
  `brokkr.schema.check_build_record`. Added here: SE1 is also NOT JUDGED if the control's SE-all build
  or any judged model's baseline build fails its check (the rules above name only the judged models'
  SE-all builds).
- **Measuring** (`scripts/37_se1.py`): the 5,000 tuning images in split order, batches of 32, 8
  threads, thread spinning off; top-1 ties go to the lower class number. Records are schema-2
  "diagnostic" records in `results/se1/` (with the scores), since SE1 is a diagnostic and some builds
  (MobileNetV3-Small) may have failed their check; the verdict goes to `results/final/se1_verdict.json`.
  MobileNetV3-Large's baseline is its Stage 3 Percentile file.
- **Rule code:** `brokkr/se1.py` (tested): baseline loss, change, recovered share, control, verdict.
- **Dry run first:** the first 64 tuning images, all output outside `results/`; a tool check, never a
  result.

## Note added 27 September 2026, before the SE1 run: images, and one confirmation

- **Images changed to 4,936.** The SE1 dry run (`scripts/37_se1.py --dry-run`, at `aa81196`) used the
  first 64 tuning images, which were also part of the planned 5,000, and its printed numbers were seen.
  Nothing was tuned on them (the pass rule was fixed and committed before any build), but so that no
  image whose SE1 numbers were already seen enters the run, H decided that SE1 runs on the tuning
  split without its first 64 images: positions 64 to 4,999 in split order (4,936 images). The dry-run
  numbers are not reported and not used. (`scripts/37_se1.py` changed at `cb7555a`.)
- **Confirmed by H:** SE1 is also NOT JUDGED if the control's SE-all build or any baseline build fails
  its build check (the addition in the implementation note above).

## SE1 outcome (added 27 September 2026, after the run)

Run once at commit `126af6a` with no uncommitted changes (`scripts/37_se1.py`); no reruns. Tuning
split, positions 64–4,999 (4,936 images). Verdict: `results/final/se1_verdict.json`; records and scores:
`results/se1/` (64 diagnostic records). Builds made at `aa81196`; MobileNetV3-Small's SE-all and
SE-output builds failed their agreement check (3.1% and 2.7% on 256 tuning images), like its baseline.

**SE1: FAIL.** Both judged models have a baseline loss, so SE1 is judged; neither recovers half of it.

Darkness (Brokkr) s5, points, paired 95% intervals (change = variant extra gap minus baseline extra
gap; positive = loses less):

| Model | Role | Baseline extra gap | SE-all change (share) | SE-output change (share) |
|---|---|---|---|---|
| EfficientNet-B0 | judged | −37.82 (−39.45 to −36.14) | +0.08 (−0.71 to +0.83) (0%) | +0.14 (−0.71 to +0.93) (0%) |
| MobileNetV3-Large | judged | −11.79 (−13.05 to −10.47) | +0.59 (−0.14 to +1.32) (5%) | +0.47 (−0.30 to +1.13) (4%) |
| RegNetY-400MF | control | −0.97 (−1.74 to −0.18) | +0.10 (−0.47 to +0.67) | +0.55 (+0.02 to +1.07) |
| MobileNetV3-Small | reported | +6.54 (+5.31 to +7.70) | +0.10 (−0.12 to +0.32) | +0.12 (−0.10 to +0.34) |

- Parts: EfficientNet-B0 recovers at least 50% with the interval above zero: does not hold.
  MobileNetV3-Large: does not hold. Control changes by less than 2.0 points: holds.
- Reported, not judged, SE-all change (share): contrast (ImageNet-C) s3: EfficientNet-B0 +0.26 (−0.53
  to +0.99) (1%), MobileNetV3-Large +0.26 (−0.41 to +1.03) (2%), RegNetY-400MF −0.02; fog (Brokkr) s3:
  EfficientNet-B0 +0.49 (−0.32 to +1.32) (5%), MobileNetV3-Large +0.75 (+0.16 to +1.44) (17%),
  RegNetY-400MF +0.12. Baseline extra gaps, contrast s3: −36.35, −13.86, −1.60; fog s3: −8.95, −4.48,
  −0.28.
- Clean top-1 on these images (FP32 / baseline / SE-all / SE-output): EfficientNet-B0 76.28 / 71.15 /
  71.15 / 71.31; MobileNetV3-Large 73.76 / 72.00 / 71.62 / 71.39; RegNetY-400MF 74.70 / 74.19 / 74.17 /
  73.74; MobileNetV3-Small 66.55 / 3.38 / 3.28 / 3.30.
- File sizes (baseline / SE-all / SE-output, MB): EfficientNet-B0 6.50 / 8.20 / 7.30;
  MobileNetV3-Large 5.92 / 10.36 / 8.12; RegNetY-400MF 5.21 / 7.11 / 6.12; MobileNetV3-Small 2.83 / 4.15
  / 3.48.

### Exploratory, after the verdict (not part of SE1; changes no verdict)

- Keeping the squeeze-and-excitation blocks in float changed the darkness extra gap by less than one
  point in every model, so quantizing these blocks is likely not what the collapse depends on in
  EfficientNet-B0 and MobileNetV3-Large. Where it does come from remains open (parked, research
  freeze).
- MobileNetV3-Small's positive "extra gap" is a floor effect: its INT8 builds are near chance (about
  3% clean), so damage cannot make them much worse; its builds stay failed with the blocks in float.

## Note added 29 September 2026, before any label is generated: the reliability envelope (P1)

The envelope's three states were fixed by H; details marked "(proposed)" await H's confirmation.
Nothing has been generated.
- **Harm thresholds** (the proposed definition of the 3.9 note, used unchanged): a condition is harmful
  for a build if its clean-tuned 90% conformal coverage falls below 80%, or its top-1 is more than 10
  points below the same build's clean top-1.
- **The two intervals, per build and tested condition:** coverage with its 95% bootstrap interval (the
  4.1 conformal record: threshold from the same build's clean `conformal_calibration` images, 1,000
  resamples, seed 0); and the drop = top-1 under the condition minus the same build's clean top-1,
  with a paired 95% interval over the same test images (1,000 resamples, seed 0).
- **States:**
  - **not harmful:** the whole coverage interval is at or above 80% (lower end ≥ 80%) and the whole
    drop interval is at or above −10.0 points (lower end ≥ −10.0);
  - **harmful:** the whole coverage interval is below 80% (upper end < 80%), or the whole drop interval
    is below −10.0 points (upper end < −10.0);
  - **borderline:** neither: an interval straddles a threshold and no threshold is wholly failed;
  - **not tested:** any condition without a checked record for that build.
- (proposed) **Ends:** "at or above" includes the threshold (so ≥ is "not harmful" and a lower end of
  exactly 80.0% or −10.0 points still clears); "below" is strict.
- (proposed) **Which build:** the states are computed for the shrunk build and for FP32 beside it; the
  summary block at the top describes the shrunk build.
- (proposed) **Which conditions:** the 12 damaged 4.1 conditions, each named with its suite; clean is
  shown as the reference row and is not part of the envelope. Rows where FP32 is near floor (below
  10%) are flagged; their state is computed the same way.
- (proposed) **No correction for 12 conditions tested at once:** the label says the intervals are per
  condition.
- **Wording on the label:** "not harmful in our tests", "harmful", "borderline", "not tested"; never
  "safe", "robust" or "guaranteed". The summary block has one short line per state, listing its
  conditions.
- **Inputs:** only records that pass `scripts/22_check_results.py`: the 4.1 test-split accuracy
  records (with their scores) and the 4.1 conformal records.

### Revised 30 September 2026 (envelope; before any label is generated)

Appended to the 29 September envelope note; nothing above is changed. Where the two differ, this
section replaces it. Decided by H on 30 September unless marked "(proposed)".

**Confirmed from the note above:** interval ends count as "not harmful" ("below" is strict); the summary
block describes the shrunk build; intervals are per condition, with no multiple-testing correction, and
the methods page says: "With 12 conditions, an occasional result may cross a line by chance."; the
label wording ("not harmful in our tests"; never "safe", "robust" or "guaranteed").

**1. Two drops, by name.**
- **Damage drop** = top-1 under the condition minus clean top-1, same build, same test images; paired
  95% interval (resample the images once, compute both on each resample; 1,000 resamples, seed 0).
- **Shrinking cost** = INT8 top-1 minus FP32 top-1, same condition, same images; paired 95% interval
  in the same way.
- **The envelope of every build, FP32 and INT8 alike, is judged on that build's own damage drop and
  coverage** (the two thresholds of the 3.9 harm definition). Reason: the label answers "can I use
  this build in this condition?", and a build that is poor because FP32 is poor is still poor.
  - FP32 row: its envelope comes from FP32's damage drop and coverage; it has no shrinking cost (it is
    the reference).
  - INT8 row: its envelope comes from INT8's damage drop and coverage; beside it, as its own column,
    the shrinking cost with its interval, flagged "large shrinking cost" when the whole interval is
    below −5.0 points (upper end < −5.0).
- Not the same quantity as H20 and H21: those judged the *extra* gap (shrinking cost under damage
  minus shrinking cost on clean images); this flag uses the plain shrinking cost in that condition.
- **FLAGGED, not resolved (conflict with committed text):** the 3.9 note (`621cc43`) defines harm
  "measured on the tuning split before any alarm result is looked at"; the label, like the 29
  September note and ROADMAP 4.3, applies the same two thresholds to the 4.1 **test**-split records.
  Awaiting H's decision.

**2. What coverage measures.** For one build and condition: the share of test images whose prediction
set contains the true class. A prediction set is every class whose probability is at least 1 − q,
where q is the conformal threshold (LAC method) computed from the same build's clean
`conformal_calibration` images (5,000) for a **target coverage of 90%**. The 90% promise holds only for
images like the calibration images (clean); under damage there is no promise, and the label shows the
coverage measured. Coverage comes from prediction sets, so it is always shown with the average set
size (and its interval) beside it. 95% bootstrap intervals: 1,000 resamples, seed 0.

**3. New state "not informative"** (rule proposed; H to confirm).
- (proposed) A condition is **near floor** when FP32 top-1 under that condition is below 10%. Both
  rows of that condition (FP32 and INT8) then get the state "not informative" instead of being judged.
- Reason: it is the near-floor rule fixed for 4.1 before any 4.1 result existed (`1682b01`, 26
  September 09:30, before the sweep started at `5be2cb5`), so it was not chosen by looking at labels;
  below 10%, FP32 gets fewer than one image in ten right, so no build of the model is usable there and
  the shrinking cost cannot be large (at most 10 points), so it says little about shrinking.
- **FLAGGED:** judged normally, such a row would likely come out "harmful" (the damage drop from a
  usable clean score to below 10% is large), which is a true answer to "can I use this build here?".
  "Not informative" fits the shrinking-cost column most exactly. Alternative for H: keep the envelope
  state judged and mark only the shrinking cost "not informative".

**4. New state "INT8 build failed".** When a model's INT8 build record fails its build check
(`brokkr.schema.check_build_record`), the INT8 row shows "INT8 build failed" with the reason read from
the build record (which check failed, and its value); no INT8 accuracy, coverage or shrinking cost is
shown. The FP32 row is labelled as usual. MobileNetV3-Small gets such a label, so P1 produces 10
labels (ROADMAP P1 updated).
- **FLAGGED:** the latency method H confirmed covers FP32 and INT8 of the 9 models (18 builds);
  MobileNetV3-Small's FP32 is not in it, so its label would show "Laptop latency: not measured" unless
  H adds it.

**5. Where the two thresholds come from** (git history, checked 30 September 2026).
- **80% coverage:** first committed in `237effa` (25 September 2026, 10:15 +0530), the Stage 3
  pre-registration, where it picked the "harmful" conditions from the Stage 2 baselines (FP32, test
  split): it was chosen with the Stage 2 results known. It was carried into the 3.9 harm proposal
  (`621cc43`).
- **−10.0 points:** first committed in `621cc43` (26 September 2026, 02:57 +0530), the 3.9 note, after
  Stage 3's final results (it names best INT8 in darkness as its motivation).
- **Both were committed before any 4.1 result existed:** before the 4.1 hypotheses (`1682b01`, 26
  September 09:30) and before the 4.1 sweep started (`5be2cb5`, 13:33; every 4.1 record is later). They
  were not chosen by looking at 4.1 data.
- **Reasons that do not depend on 4.1 data** (written 30 September):
  - 80% coverage: the prediction sets are built to miss at most 10% of images; below 80% they miss at
    least twice that, so the "I'm not sure" promise is clearly broken, not just slightly under target.
  - −10.0 points: a fall of ten percentage points means that at least one image in ten that the build
    got right on clean images is now wrong: a change a user would notice and should be told about, and
    one far larger than the chance variation of a paired comparison on thousands of images.

## Note added 29 September 2026, before any latency is measured: the laptop latency method (P1)

Proposed by Claude, to be confirmed by H before any timing. Built on `brokkr.benchmark` (Stage 1's
tested code).
- **What is timed:** one image at a time (batch 1, 3×224×224, random input with seed 0: speed does
  not depend on the picture), ONNX Runtime CPU execution provider, for FP32 and the Percentile INT8
  build of each of the 9 models.
- **Machine checks, at the start and end of every model (the run stops if one fails):** plugged in;
  Windows power mode "best performance"; battery saver off; and, before starting, a 10-second idle
  check of total CPU use, which must be below 10% (other programs closed). All of it is recorded.
- **Cores and threads:** pinned to the performance cores this laptop's fingerprint lists (logical CPUs
  0–3), because unpinned Windows moves the work between fast and slow cores (Stage 1). Two thread
  counts, 1 and 4, as in Stage 1; the label's main line shows 4 threads, the details show both.
  Thread spinning left at ONNX Runtime's default, since one session runs at a time.
- **Runs:** 10 sessions per build. In each session, a model's FP32 and INT8 builds are timed back to
  back, in alternating order (FP32 first in odd sessions, INT8 first in even ones), so slow drift
  affects both alike. Each timing: 20 warm-up runs (not counted), then 300 timed runs (the rules ask
  for at least 100; 300 puts 3 runs above p99).
- **Cool-down:** 60 seconds of rest before each model's first session, and 30 seconds between
  sessions.
- **Reported, per build and thread count:** p50, p95 and p99 in milliseconds, each the median across
  the 10 sessions; the spread = the interquartile range of the 10 session p50s as a percentage of
  their median, and the fastest and slowest session p50. A spread above 10% is flagged "unstable"
  (Stage 1's rule), shown on the label, and not hidden.
- **Recorded with every result:** the machine fingerprint (CPU, OS, core types, power state), the
  pinned CPUs, thread count, spinning, ONNX Runtime version, sessions, warm-up and timed runs,
  cool-down, the idle check's reading, and the power state at start and end.
- **On the label:** "Laptop latency (Intel Core i5-1235U, Windows 11, batch 1, 4 threads on
  performance cores)" with p50 / p95 / p99 and the spread; "Raspberry Pi 5: not measured". Laptop
  latency is never presented as the speed of an edge device.
- Stage 1's speed records are not reused (different sessions and settings).

### Revised 30 September 2026 (latency; before any latency is measured)

Appended to the 29 September latency note; nothing above is changed.

**Confirmed by H:** the stop conditions; pinning to logical CPUs 0–3; thread counts 1 and 4 (the main
line shows 4); 10 sessions; 20 warm-up and 300 timed runs; the cool-downs; p50, p95 and p99 as medians
across sessions; the spread and the "unstable" flag above 10%.

**Added:**
- **Recorded in every latency record and shown on the label:** ONNX Runtime version, execution
  provider, intra-op and inter-op thread counts, graph optimisation level, CPU model, and whether VNNI
  is available. `brokkr.benchmark.make_session` sets intra-op threads to the thread count, inter-op
  threads to 1, sequential execution, and leaves graph optimisation at ONNX Runtime's default, which
  for the installed version (1.23.2, read 30 September) is `ORT_ENABLE_ALL`; the record states the
  level explicitly rather than relying on the default.
- **VNNI (FLAGGED: method to be confirmed):** on Windows, Brokkr's fingerprint records both VNNI
  flavours as unknown (no standard-library way to read them). Read on 30 September: NumPy 2.2.6's
  CPU-feature list reports AVX512_VNNI as not available (and AVX-512F as not available), but has no
  entry for the other flavour, AVX-VNNI. Proposed: read both from the CPU itself (CPUID leaf 7:
  sub-leaf 0, ECX bit 11 = AVX512_VNNI; sub-leaf 1, EAX bit 4 = AVX-VNNI), with a small new helper or
  the py-cpuinfo package (its licence checked and recorded first). Until then the record and label say
  "VNNI: unknown (not yet readable on Windows by Brokkr)".
- **Which physical cores logical CPUs 0–3 are** (read on 30 September from Windows'
  `GetSystemCpuSetInformation`): logical CPUs 0 and 1 are the two hardware threads of one physical
  performance core (core index 0), and 2 and 3 those of the other (core index 2); logical CPUs 4–11
  are eight efficiency cores, one thread each. So "4 threads on the performance cores" means 2
  physical cores with 2 hardware threads each; with 1 thread, Windows picks one of the four. The map
  is read at run time and recorded in every latency record (never typed in), and the label says
  "4 threads (2 performance cores, 2 threads each)".
- **On the label:** "Timing is model only; excludes image loading and pre-processing. Random input
  (seed 0)."
- **The run refuses to start if task 4.2 or any other Brokkr job is running.** Before starting, and
  before each model, it lists the running processes and stops if any other process runs a Python
  script from this repository's `scripts/` folder or a `brokkr` command. (proposed method: Windows'
  process list with command lines, through PowerShell's `Get-CimInstance Win32_Process`; `psutil` is
  not installed.)

## Note added 30 September 2026, before any P1 run: the reproduction rule for `brokkr test`

Fixed by H before `brokkr test` exists or runs.
- **Target: MobileNetV3-Large** (not ResNet-18), FP32 and its Percentile INT8: the 13 test conditions
  and clean `conformal_calibration`, i.e. its 28 records in `results/breadth/`. Read on 30 September:
  all 28 come from the clean commit `fc965cd`, with 8 threads, thread spinning off and batch 32, and
  the two model files' SHA-256 equal those the records name. `brokkr test` runs these existing model
  files; nothing is rebuilt.
- **Step 1, is INT8 inference on this laptop repeatable?** Before anything is compared with 4.1, the
  INT8 model runs twice on the same 64 tuning images (positions 0–63: tuning, never test), in two
  separate sessions, with 4.1's settings (8 threads, spinning off, batch 32). Repeatable = the two sets
  of scores are identical bit for bit (`np.array_equal` on the float32 scores). The result is printed
  and recorded before step 2.
- **Step 2, the pass rule, chosen by step 1:**
  - **If repeatable:** PASS = in every one of the 28 (precision, condition) records, every image's top-1
    prediction (ties to the lower class number, as everywhere) is identical to the one from the 4.1
    record's saved scores. Reported, not judged: whether the scores are identical bit for bit, and the
    largest score difference.
  - **If not repeatable:** PASS = in every record, top-1 within 0.1 points of the 4.1 record's
    (compared in whole images: at most 10 of the 10,000 test images, 5 of the 5,000 calibration images)
    **and** identical build settings: the same model files (SHA-256 equal to the records'), and the same
    preprocessing, damage seeds, batch size and thread count as the 4.1 records state.
- **Which applies is decided by step 1, not assumed.** Recorded evidence points to "repeatable" (at
  both starts of the 4.1 sweep, MobileNetV3-Large reproduced Stage 3's scores exactly on 64 test
  images: `results/breadth/run_log.txt`), but only step 1 decides.
- **How it is checked:** a script prints step 1's result, which rule applies, PASS / FAIL for each of
  the 28 records, and PASS only if all 28 pass.

## Note added 30 September 2026: H's decisions on the flagged points (before any P1 code)

These decide the points flagged in the two "Revised 30 September 2026" sections and the reproduction
note above; where they differ, this note replaces them. Nothing has been built or measured.

**Envelope and label**
1. **Near floor:** the envelope is judged normally for every row, near floor or not. Only the
   shrinking-cost column says "not informative" when FP32 top-1 under that condition is below 10%
   (then the shrinking cost can be at most 10 points and says little about shrinking). This replaces
   the proposed "not informative" state for whole rows.
2. **Split:** labels use the **test** split: the same harm definition as `621cc43`, applied to the test
   split. The conformal thresholds come from the `conformal_calibration` split, which shares no image
   with the test split. Checked on 30 September: by the split definition (`brokkr.datasets.make_splits`:
   10,000 test and 5,000 calibration images, 0 shared), and on the image positions saved in the score
   files of all 19 builds of 4.1 (0 builds with shared or unexpected positions).
3. **"Large shrinking cost"** is the name of the label's flag (whole interval of the shrinking cost
   below −5.0 points), so it cannot be confused with H20 and H21's −5.0 on the extra gap. The label never
   calls it an "extra gap".

**Latency**
4. **VNNI:** read with py-cpuinfo 9.0.0 (MIT: licence field on PyPI and the LICENSE file in its wheel,
   checked 30 September; installed into `.venv` on 30 September, `py_cpuinfo-9.0.0-py3-none-any.whl`).
   Read on 30 September: py-cpuinfo reports AVX512_VNNI as not available on this CPU (NumPy's feature
   list agrees), and has no check at all for AVX-VNNI (its source reads AVX512_VNNI from CPUID leaf 7
   but never the sub-leaf that holds AVX-VNNI). So records and the label say "AVX512-VNNI: no;
   AVX-VNNI: unknown (not reported by py-cpuinfo)". No custom CPUID helper for v0.
5. **Threads on the label:** "2 physical cores × 2 hardware threads". The timing plan is unchanged.
6. **MobileNetV3-Small's FP32 is added to the latency plan:** 19 builds (FP32 of all 10 models, INT8 of
   the 9 whose build passed).
7. **Other Brokkr jobs are detected with psutil** (BSD-3-Clause: licence field on PyPI, checked 30
   September; not yet installed), because the same check must work on Linux (Raspberry Pi 5), not
   through PowerShell.

**Methods page (for the website)**
- It states: the 80% coverage threshold was set after the Stage 2 results were known (`237effa`), and
  the −10.0-point threshold after the Stage 3 results (`621cc43`); both were set before any 4.1 result
  existed. It gives their data-independent reasons (from the envelope revision above): below 80% the
  sets miss at least twice the 10% they are built to miss; a fall of ten points means at least one
  image in ten that was right on clean images is now wrong. And: "With 12 conditions, an occasional
  result may cross a line by chance."

## Note added 30 September 2026: the package was renamed

The import package `brokkr` was renamed `brokkr_edge` (published as `brokkr-edge`), because the PyPI
name "brokkr" belongs to an unrelated project (commit `758c383`). Paths and module names in the notes
above (`brokkr/...`, `brokkr.schema`, ...) are as they were when written; the same files now live under
`brokkr_edge/`. Result records made earlier keep the names they were made with. The word "brokkr" as
a value (a record's `source`, the name of Brokkr's own damage suite) is unchanged.

## Note added 1 October 2026, before any latency is measured: a session interrupted by sleep or a pause is discarded

Required by H (1 October 2026): the run must detect that the laptop slept or paused mid-run (any gap
between timed runs far above normal) and discard that session instead of recording it. Appended to
the latency notes of 29–30 September; nothing above is changed. The numbers marked "(proposed)" are
Claude's and await H's confirmation before any timing; no latency has been measured.

- **What is watched.** For every timing (20 warm-up runs, then 300 timed runs), the script records,
  with a clock that keeps counting while the machine sleeps (`time.time`, the wall clock), the start
  and end of each timed run. Two things are checked: the duration of each timed run, and the gap
  between the end of one timed run and the start of the next.
- **(proposed) The rule.** A timing is **interrupted** if any timed run, or any gap between two timed
  runs, is longer than the larger of 1.0 second and 20 times that timing's own median run time. A
  second sign is also recorded: the wall clock and the timer used for the latencies
  (`time.perf_counter`) disagreeing by more than 1.0 second over the timing.
- **What is discarded.** A session holds one model's FP32 and INT8 timings back to back; if either is
  interrupted, the **whole session** (both builds, both thread counts timed in it) is discarded, so
  FP32 and INT8 always come from the same sessions. Nothing from a discarded session enters p50, p95,
  p99 or the spread.
- **(proposed) What happens next.** The discarded session is repeated after the usual 30-second
  cool-down, with the same FP32/INT8 order it had. At most 3 sessions may be discarded per model; a
  fourth stops the run for that model with a clear FAIL, and no latency record is written for it.
- **What is recorded.** Every latency record states the rule and its numbers, the count of discarded
  sessions, and for each one the time, which build was being timed, and the longest run or gap seen. A
  discard is logged, never hidden; the label shows the count when it is not zero.
- **Why 1.0 second and 20 times (data-independent; written before any timing).** A sleep or a
  suspended process lasts seconds or more, while one image through these models takes milliseconds to
  a fraction of a second, so both bounds sit far above a normal run and far below any real sleep. An
  ordinary slow run (the operating system briefly busy) stays below them and is kept: it is real
  latency and belongs in p99.
- **How it is checked.** A test feeds the detector made-up run times with one long gap, with one long
  run, and with none, and checks that the first two are flagged and the third is not.

*Confirmed by H on 2 October 2026:* the numbers marked "(proposed)" in the note above: a timing is
interrupted if any timed run or gap is longer than the larger of 1.0 second and 20 times that timing's
median run time; at most 3 discarded sessions per model, and a fourth is a FAIL for that model.

## Note added 3 October 2026, before `brokkr-edge test` exists or runs: how the reproduction is carried out

Implementation details of the reproduction rule above (30 September); the rule itself is unchanged.
- **The command.** `brokkr-edge test --model <name>` runs the 13 test conditions and clean
  `conformal_calibration` for the model's FP32 build and, if its build record is usable, its Percentile
  INT8 build, and writes schema-2 accuracy records with their scores. It needs no PyTorch (model facts
  come from `brokkr_edge/model_list.json`). It uses the code the 4.1 sweep used to make the pictures
  (`brokkr_edge.sweep`: the same caches, damage seed = dataset position) and 4.1's settings: 8 threads,
  thread spinning off, batch 32. It writes to `results/test_runs/<model>` unless told otherwise, and
  never into `results/breadth`. `--split tuning --limit N` is a dry run on the first N tuning images.
- **Step 1 (repeatability).** The command is run twice, each time in its own process (so also in its
  own ONNX Runtime session), with `--split tuning --limit 64` (tuning positions 0–63). The decision
  uses the INT8 scores on clean images only, as the rule says: repeatable = identical bit for bit.
  The same comparison for the other conditions and for FP32 is printed, not judged.
- **Step 2.** One full run into `results/reproduction/mobilenet_v3_large`; each of its 28 records is
  compared with the record of the same name in `results/breadth`.
  - If repeatable: PASS = the same images in the same order and identical top-1 predictions (the
    class with the highest score, ties to the lower class number). Printed, not judged: whether the
    scores are identical bit for bit, and the largest score difference.
  - If not repeatable: PASS = the number of correct top-1 answers differs by at most 10 (test) or 5
    (calibration), and the two records state the same model file SHA-256, preprocessing, damage seed
    rule, batch size and thread count.
- **When it may run.** Only from a clean commit (the records state it) and only on mains power (H: "run
  it when I'm plugged in"): the script stops if the laptop is on battery.
- **What is written.** `results/reproduction/repeatability.json` before step 2 starts, and
  `results/reproduction/reproduction_check.json` with every record's result and the verdict.
  `scripts/43_check_reproduction.py` prints PASS only if all 28 pass.

## Note added 3 October 2026, before any latency is measured: how the latency script carries out the method

Implementation details of the latency notes above (29–30 September, 1–2 October); the method is
unchanged. No latency has been measured: on 3 October the script was run once in its `--smoke` mode
(one model, 2 sessions, 2-second cool-downs, written outside `results/`, every record marked
`smoke_test`) to check that it works. That run is not a measurement and none of its timings is used.
- **Script and rules:** `scripts/44_laptop_latency.py`; the rules live in `brokkr_edge/latency.py` and
  are tested on made-up timings (`tests/test_latency.py`).
- **Inside a session:** the model's builds are timed back to back at 1 thread, then again at 4 threads,
  in the session's order (FP32 first in odd sessions, INT8 first in even ones). A fresh ONNX Runtime
  session is opened for every timing. MobileNetV3-Small has FP32 only.
- **Records:** one schema-2 `speed` record per build and thread count (38 files for the 19 builds),
  `results/latency/<model>_<precision>_laptop_<threads>threads.json`, each with a `.npz` holding every
  timed run of every kept session and their wall-clock start and end times.
- **Machine checks:** all of them (with the 10-second idle reading) before each model; the power state
  again after each model. A failed check stops the run (exit code 2) before anything is written for
  that model. Finished models are skipped when the same command is run again.
- **Other Brokkr jobs** are found with psutil 7.2.2 (BSD-3-Clause: the licence field and the LICENSE
  file in its wheel, read 3 October 2026): any other process whose command line names a `.py` file
  inside this repository's `scripts/` folder, `brokkr-edge`, or `-m brokkr_edge.cli`. The script's own
  process, its parents and its children are not counted.
- **Clocks:** each run's latency is read from `time.perf_counter_ns`; its start and end are also read
  from `time.time` for the sleep/pause rule. The "second sign" of the 1 October note is recorded, not
  judged: every record stores `clock_disagreement_s`, the largest difference, over its kept timings,
  between the time a timing took by the wall clock and by the precise timer.
- **Which physical core each pinned CPU is** is read at run time (`brokkr_edge.fingerprint.physical_cores`)
  and stored in every record.
- **Started by H only.** The real run is started by H, at the desk, overnight.

## Reproduction outcome (added 3 October 2026, after the run)

`scripts/43_check_reproduction.py` at the clean commit `0aa9e0d`, on mains power. The first attempt
was cut off by the assistant session's time limit after step 1 and 4 of the 28 records; H ran the same
command again in a terminal, which kept those and made the rest (the script is resumable; no record
was redone).
- **Step 1: repeatable.** INT8 on clean tuning images 0–63, two separate runs: identical bit for bit
  (and all 26 score files of the dry run, every condition and both builds). So the rule for step 2 was
  "identical top-1 predictions on every image".
- **Step 2: PASS, 28 of 28 records.** Every record has the same images in the same order and identical
  top-1 predictions as its 4.1 record. Reported, not judged: all 28 score files are identical bit for
  bit (largest score difference 0). All 28 new records pass the schema check, come from the clean
  commit and state the same build settings. Record: `results/reproduction/reproduction_check.json`.
- `scripts/22_check_results.py` afterwards: PASS, 1863 of 1863 result records (the 54 new ones
  included).

## Note added 3 October 2026, before any code: H's fix list, latency on the labels, and two past decisions

Decided by H on 3 October 2026 **after seeing the ten labels and the 38 latency records**, so these
are choices made with the data known. They change wording, advice and what a label displays only: no
verdict, threshold, envelope state or envelope rule changes, and no label has been published. Points
marked "(proposed)" are Claude's and await H's confirmation. The label format changes they need are in
`docs/label_schema.md`, note of 3 October 2026.

**1. The suggested next step follows the line that actually failed** (replaces the choice and order of
suggestions in `docs/label_schema.md`, note of 1 October 2026, point 2). Rule: every suggestion must
plausibly fix the line that failed. Re-calibrating fixes coverage; a stronger model fixes accuracy;
another recipe fixes harm caused by shrinking (either line). The cause-based advice is unchanged
("hurt by shrinking" → "try another recipe or model"; "too hard for this model" → "consider a stronger
model"; "cause unclear" → "try another recipe or a stronger model").
- Fails both lines: the cause-based advice first, "re-calibrate on your own images" second.
- Fails accuracy (damage drop) only: the cause-based advice only.
- Fails coverage only: "re-calibrate on your own images" first; "try another recipe" second only where
  shrinking is involved ("hurt by shrinking", "cause unclear"). Never "consider a stronger model" on a
  coverage-only row: FP32's accuracy holds there, so it would point at the wrong problem.
- A general rule with a test for each case on made-up rows, not a special case for the row that raised
  it (ConvNeXt-Tiny, fog (ImageNet-C) s3). Only coverage-only rows change.

**2. "Not informative" never hides a large shrinking cost.** Until now the flag was one value and "not
informative" (FP32 top-1 below 10% in that condition) took priority. From now on both are shown when
both apply, and the summary's "large shrinking cost" count includes such a row. The rule for each flag
is unchanged. A test uses a made-up row where both apply. (In the ten labels made at `753cdf1` no such
row exists.)

**3. Threshold wording and a correction to the 30 September envelope note.**
- The README and labels describe the two lines as "a line whose value was written down before these
  results existed and adopted for the labels afterwards, unchanged" (wording approved by H).
- **Correction** to "Where the two thresholds come from" (envelope note, revised 30 September) and to
  the methods-page point of H's 30 September decisions, which both say the 80% coverage line was first
  committed in `237effa`: **80% first appears in `a957652`** (24 September 2026, 22:57 +0530), the
  Stage 2 pre-registration, in H6 ("FP32 coverage is below 80% for at least 3 of the 5 corruptions"),
  before any Stage 2 result; `237effa` (25 September, 10:15 +0530) is where it first became the harm
  line, with the Stage 2 results known. The −10.0-point history is unchanged. The methods page states
  both commits.

**4. Latency on the labels** (H's decisions a, b, c and e, and the speed line, 3 October 2026).
- **Source.** Each laptop speed row is filled from its latency record,
  `results/latency/<model>_<precision>_laptop_<threads>threads.json`, which must pass
  `brokkr_edge.schema.check_record` and come from a clean commit, or the label is refused. p50, p95,
  p99, the spread and the "unstable" flag are copied, never recomputed or rounded. MobileNetV3-Small's
  INT8 rows stay "not measured: INT8 build failed"; the Raspberry Pi 5 rows stay "not measured".
- **Spread and "unstable" on every row** (required by the 29 September method note): every laptop row
  shows its spread; an unstable row (spread above the line its record states) also says, in plain
  words, "unstable: speed varied a lot between repeat runs (spread X%); treat as rough", with X from
  the record, and the label's glossary explains "Spread" and "Unstable". No claim about why a record is
  unstable. (proposed) "spread X%" instead of H's example "up to X%": the record's spread is the
  interquartile range of the 10 session p50s as a share of their median, not a maximum.
- **Relative speed line** (shown, never hidden): for each thread count, N = the labelled (INT8) build's
  p50 divided by the reference (FP32) build's p50, both the medians across sessions from the latency
  records; "INT8 takes N× the time of FP32", with "(slower)" added when N is above 1. N is stored in
  `label.json` and recomputed by `scripts/40_check_labels.py`. When either row is unstable the line is
  still shown, with the flag and the plain-words note.
- **Where INT8 is slower:** the plain sentence "INT8 is slower than FP32 on this laptop CPU (relative
  comparison only)." Never a negative "speed-up", and no wording that assumes INT8 is faster.
  (proposed) If INT8 is slower at only some thread counts, the sentence names them ("at 1 thread").
- **Cores (decision c), on the label:** "Pinned to the laptop's 2 performance cores (4 hardware threads,
  as reported by Windows). The 4-thread setting therefore runs on 2 physical cores." The counts come
  from the record (`pinned_cpus` and `physical_core_of_each_cpu`), never typed in. A limitation: "The
  CPU pin was not read back after it was set" (`brokkr_edge.benchmark.pin_to_cpus` stops only if
  Windows refuses it).
- **When the timing ran (decision e).** The run started 13:24:39 and finished 15:02:18 local time on 3
  October 2026 (`results/latency/run_log.txt`), in the daytime rather than overnight as planned; H
  accepted it with no rerun. Each speed row stores its own timed window (first timed run's start, last
  timed run's end, in UTC) from its record's `.npz` wall-clock times, and the label shows the run's
  window in UTC.
- **No rerun and no new rerun rule** for the 7 unstable records: a rule made now would be chosen after
  seeing the data.

**5. A past decision, recorded here for the first time: labels live outside `results/`.** Decided by
H on 30 September 2026 to fix the Checkpoint 1 failure: the label files then written to
`results/labels/` carry `"schema_version": 1`, so the results loader (`brokkr_edge.schema`) took them
for Stage 1–3 records and `scripts/22_check_results.py` stopped ("two schema-1 files share a name").
Fix: labels move to the gitignored top-level `labels/` folder, and a test fails if a label appears
under `results/` (commit `94f1b97`, `tests/test_label_location.py`). Rejected alternative: make the
results loader skip label files.

**6. The model card's licence** (decided by H, 3 October 2026; a Checkpoint 1 point never recorded
before). The Hugging Face metadata keeps `license: other` and `license_name: see-label-licences` and
adds `license_link`, pointing to the label's own "Licences" section. That section is generated from
the build records, with no hand-typed licence text beyond fixed wording: Brokkr's code is Apache-2.0;
the model's code and weights keep their original licences, copied from the build records; weights
trained on ImageNet-1k (read from the build record's weights name) carry ImageNet's non-commercial
terms of access; the label's own numbers and text are Brokkr output, licensed CC BY 4.0. A check fails
if a label's licence section is empty or differs from its build records.

**7. Parked** (research freeze; STATUS.md): "Add severity 1 to label conditions (Phase B severity
menu) so labels show where models still work"; "Why is RegNetY-400MF's INT8 slower than FP32 on this
laptop?". Suggestion parked for later: "Add a pin read-back check to the latency script."

## Note added 3 October 2026 (later the same day), before any code: next-step advice line by line

Decided by H on 3 October 2026 **after seeing the regenerated labels** (made at `1e7b054`), so this is
a choice made with the data known. It changes the suggested-next-step text only; no verdict, envelope
state, cause group, threshold or rule changes. It replaces point 1 of the note above. Points marked
"(proposed)" are Claude's and await H's confirmation.

**The case that raised it:** ConvNeXt-Tiny, fog (ImageNet-C) s3. INT8 fails both lines; FP32 is
harmful there only because it fails the coverage line, while its accuracy holds. Advice chosen from
the cause group ("too hard for this model") said "consider a stronger model" first, which points at
accuracy, where FP32 copes.

**The rule: advice is chosen line by line, comparing INT8 with FP32 on each line separately.** For each
line the labelled build failed (its whole interval below the line), FP32's interval on the same line
is read: it *fails* (whole interval below the line), *copes* (whole interval at or above it), or
*straddles* it.
- **INT8 fails accuracy (the damage-drop line):** FP32 also fails accuracy → "consider a stronger
  model"; FP32 copes on accuracy → "try another recipe" (shrinking caused it). (proposed) FP32
  straddles → "try another recipe or a stronger model".
- **INT8 fails coverage:** "re-calibrate on your own images"; "try another recipe" is added if FP32
  copes on coverage. (proposed) FP32 straddles coverage → nothing added (H's rule adds it only when FP32
  copes).
- **Order:** the accuracy advice first when accuracy failed, the coverage advice second. (proposed) A
  suggestion is shown once: if both lines give "try another recipe", it appears once, in the first
  place. *Added before the labels are regenerated:* this includes the straddle advice "try another
  recipe or a stronger model", which already names another recipe (two rows of the `1e7b054` labels
  would otherwise repeat it).
- Rows that are not harmful, and rows of a failed build, get no suggestion. Tests on made-up rows for
  each case, including the ConvNeXt-Tiny fog (ImageNet-C) s3 pattern.
- To read FP32's state on each line, every envelope row stores its per-line states
  (`docs/label_schema.md`, note of 3 October 2026, later addition).

**H's confirmations of the "(proposed)" points of the note above (3 October 2026):** "spread X%" (with a
plain glossary definition); naming the thread counts where INT8 is slower; the runtime field design
(`docs/label_schema.md`). Also a standing rule: **a value a record does not hold is always shown as "not
recorded", never guessed.**
