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
