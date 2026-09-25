# Brokkr Technical Report 1

**Stages 1–3: what quantization does to a vision model's accuracy and reliability, and what fixes it**

Code: [github.com/Vasco1290/Brokkr](https://github.com/Vasco1290/Brokkr). Every Stage 3 test-split result
comes from the commit tagged `stage3-final-run` (ImageNetV2, task 3.9: commit `e7f5b9b`).

Written 26 September 2026. Every number below comes from running Brokkr's code on one laptop (Intel
Core i5-1235U, Windows 11, CPU only, ONNX Runtime 1.23.2) and can be traced to a saved result file
(listed at the end). Predictions were committed to git before measuring
(`docs/hypotheses.md` for Stage 2, `docs/hypotheses_stage3.md` for Stage 3).

## The two-minute version

We took one image classifier (MobileNetV3-Large, torchvision `IMAGENET1K_V2` weights), made it smaller
("quantized" it from 32-bit to 8-bit numbers), and tested it on clean and deliberately damaged photos.

- **Out of the box, 8-bit (INT8) quantization broke the model.** It became 3.75x smaller but lost 15.4
  points of accuracy on clean images, and far more on damaged ones. At darkness severity 5 it fell to
  20.5% while the original stayed at 73.7%.
- **One setting fixed most of it.** Changing how INT8 picks its number ranges (Percentile instead of
  MinMax calibration, chosen on separate tuning images) gives a model of the same size that is
  **2.0 points below the original on clean images, but still 13.6 points below at darkness severity 5**
  (the default INT8 was 53 points below there) and 21.5 points below at fog severity 5.
- **Knowing when it is wrong is a separate problem.** Under blur and noise, the model's "90% sure the
  answer is in this set" promise quietly fails. Calibrating that promise on damaged images restores
  much of the coverage in our tests (+12.8 points), at the price of sets about four times larger
  (7.7 instead of 2.3 classes on clean images). For FP32, a simple alarm that watches the model's
  average confidence fired on every batch from the 12 damage conditions where its clean-tuned
  coverage had fallen below 80%, and on no clean batch.
- **Real new photos cause the same silent failure.** On ImageNetV2 (Recht et al., 2019), 10,000 photos
  collected the same way years later, FP32 falls from 75.6% to 62.1%, and its 90% promise from 90.6%
  to 80.7%, while its sets grow only from 2.3 to 2.6 classes. Best INT8 stays about 2 points behind
  FP32 there too (59.8%).
- **Two predicted fixes did nothing.** Keeping the last layer's output unrounded removed tied scores
  but did not improve how well the model ranks its own mistakes (H11). Calibrating INT8 on damaged
  images did not help with darkness it had not seen (H12), likely because Percentile calibration had
  already done most of that.

## 1. The question

When a vision model is quantized (FP32 -> FP16 -> INT8) for small devices, what happens to its size,
accuracy, and reliability, especially when conditions get bad (fog, blur, noise, darkness)? And can
the damage be fixed without cheating?

"Reliability" here means three measurable things:
- **Calibration:** when the model says 80% sure, is it right 80% of the time? Measured by ECE
  (expected calibration error, 15 equal-width bins; 0 = perfectly honest).
- **Conformal prediction sets:** instead of one answer, a set of classes promised to contain the right
  one 90% of the time. Measured by coverage (how often the promise holds) and set size (how big the
  sets get). **Coverage is always reported with set size here**: a promise kept by listing half the
  classes is worthless.
- **Selective prediction:** does the model's confidence put its mistakes at the bottom, so it could
  skip them? Measured by E-AURC (lower is better).

**Differences are always "new minus old"** (the fix minus what it replaces), with a paired bootstrap
95% interval: the same resampled test images for both sides. For accuracy and coverage, positive means
the fix is better; for E-AURC and ECE, positive means the fix is *worse*.

## 2. Related work

*To be written by H.*

## 3. Setup

- **Data:** ImageNet-1k validation set (50,000 images; non-commercial research terms). Split once,
  in code, into non-overlapping parts: `test` (10,000; every reported number), `tuning` (5,000;
  choosing Stage 3 settings), `conformal_calibration` (5,000), `int8_calibration` (512). A test checks
  that no image is in two splits.
- **Correctness check:** our FP32 pipeline scores 75.26% on all 50,000 images; torchvision publishes
  75.27% for these weights.
- **Real-photo shift:** ImageNetV2 matched-frequency (Recht et al., "Do ImageNet Classifiers Generalize
  to ImageNet?", ICML 2019): 10,000 new photos, 10 per class, collected by repeating ImageNet's
  process. Licence as the sources state it: the dataset card says MIT; the authors say that licence
  "does not apply to the actual image data. The images come from Flickr which provides corresponding
  license information." Used for evaluation only; no image is shown or redistributed.
- **Damage:** five simulated camera problems (fog, defocus blur, motion blur, noise, darkness) at
  severities 1–5, applied to the 224x224 picture the model sees. Our own short implementations in the
  style of ImageNet-C, so not directly comparable to published ImageNet-C numbers.
- **Models:** FP32 (22.2 MB), FP16 (11.3 MB), default INT8 (5.9 MB, ONNX Runtime static quantization,
  MinMax calibration, per-channel int8 weights, per-tensor uint8 activations), and the Stage 3 INT8
  variants (all 5.9 MB).

### How Stage 3 avoided marking its own exam

- **No Stage 3 setting was tuned on the test split.** INT8 method, temperature and alarm threshold
  were chosen on `tuning`; conformal thresholds on `conformal_calibration`.
- **Leave one damage type out:** any fix that learns from damaged images was built five times, each
  time without one damage type, and judged only on the type it never saw.
- **Predictions and judging rules were committed before measuring.** Every implementation detail
  decided later (group sizes, seeds, exact windows) was added as a dated note *before* it was run.
- **The test split was used once**, in one run from a tagged commit (`stage3-final-run`), with a rule
  that reruns are allowed only for technical failures. There were none.
- **Build-to-build luck was measured:** four INT8 builds from different random calibration images
  differ by up to 0.28 points in top-1 and 0.0026 in E-AURC. Differences smaller than that are
  labelled "within noise".

## 4. Results

### Accuracy (top-1, 10,000 test images)

| Condition | FP32 | Default INT8 | Best INT8 (Percentile 99.99) |
|---|---|---|---|
| Clean | 75.58% | 60.15% | **73.60%** |
| Fog, severity 3 / 5 | 72.2% / 52.8% | 40.0% / 10.8% | 66.4% / 31.3% |
| Defocus blur, 3 / 5 | 51.3% / 22.4% | 19.2% / 4.9% | 46.4% / 19.9% |
| Motion blur, 3 / 5 | 47.5% / 18.5% | 19.4% / 5.0% | 43.6% / 16.5% |
| Noise, 3 / 5 | 51.8% / 11.1% | 33.4% / 3.5% | 49.2% / 11.4% |
| Darkness, 3 / 5 | 74.5% / 73.7% | 39.7% / 20.5% | 68.7% / 60.2% |

FP16 matches FP32 everywhere (largest difference 0.25 points in Stage 2). Best INT8 was chosen on the
tuning split from MinMax (58.46% tuning top-1), Percentile 99.99 (72.10%), Percentile 99.999 (71.06%)
and Entropy (56.12%).

### The Stage 3 predictions

| | Prediction | Verdict | Measured (paired 95% intervals) |
|---|---|---|---|
| H10 | Better calibration recovers half the INT8 loss (>= 67.9%) | **PASS** | 73.60% |
| H11 | Unrounded final output: 0 ties, E-AURC >= 10% lower | **FAIL** (within noise) | 0 ties, but E-AURC: best INT8 0.0510 -> best INT8 unrounded 0.0524; difference +0.0014 (+0.0001 to +0.0030), 2.7% *worse* |
| H12 | Calibration on damaged images fixes unseen darkness (+10 points at s5) | **FAIL** | At darkness 5: best INT8 60.15% -> INT8 calibrated without darkness 59.48%; difference −0.67 points (−1.41 to +0.06) |
| H13 | Damaged calibration costs < 1 point on clean images | **PASS** (within noise) | Clean: best INT8 73.60% -> average of the five leave-one-out models 73.85%; difference +0.25 points (−0.12 to +0.59) |
| H14 | Temperature scaling fixes clean calibration but backfires under damage | **PASS**, mixed | Clean ECE 0.018; worse under blur and noise, *better* under fog and darkness |
| H15 | Robust conformal: +10 points coverage, clean sets at least 2x | **PASS** | +12.8 points; clean sets 2.28 -> 7.72 classes |
| H16 | Confidence alarm catches >= 90% of harmful batches, <= 2% clean | **PASS** | 100% of 1,200 harmful windows; 0 of 100 clean |
| H17 | Real new photos (ImageNetV2): FP32 top-1 60–68%, coverage interval below 88% | **PASS** | Top-1 62.10% (61.16 to 62.99); coverage 80.72% (79.93 to 81.44) with set size 2.64 |

**The failures, as plainly as the passes:**
- **H11.** Leaving the last layer's output unrounded removed every tie (110 of 10,000 test images had
  tied top scores before), and the file stayed the same size. But the model did *not* get better at
  ranking its own mistakes; E-AURC got slightly worse. The change is smaller than build-to-build luck
  and under 15%, so it is within about three times a noise range that comes from only four builds.
- **H12.** Calibrating INT8 on a half-damaged image set, without darkness, did not help in the dark.
  Plain Percentile calibration had already lifted darkness severity 5 from 20.48% (default INT8) to
  60.15% (best INT8). (That 60.15% is best INT8 *at darkness 5*; it only happens to equal default
  INT8's *clean* score, 60.15%.)
- **H14 is mixed.** The prediction passed on its pre-set rule (at least 3 of 5 corruptions), but
  sharpening the model's confidence helped calibration under fog and darkness while hurting it under
  blur and noise:

| ECE at severity 5 | Fog | Defocus blur | Motion blur | Noise | Darkness |
|---|---|---|---|---|---|
| Raw scores | 0.142 | 0.022 | 0.009 | 0.026 | 0.175 |
| Temperature-scaled (T = 0.743) | 0.054 | 0.145 | 0.155 | 0.165 | 0.019 |

A likely explanation, not tested: temperature makes the model more confident. That helped where the
raw model stayed under-confident under damage (fog and darkness, whose raw ECE stayed high, 0.14 and
0.18) and hurt where damage had already removed the under-confidence (blur and noise, raw ECE 0.01 to
0.03), pushing the model into over-confidence.

### "I'm not sure": prediction sets, with their size

FP32 at severity 3, each corruption held out from the robust calibration. Coverage (average set size):

| Held-out corruption | Clean-tuned threshold | Robust threshold |
|---|---|---|
| Fog | 88.4% (2.4 classes) | 96.4% (10.3) |
| Defocus blur | 72.6% (2.8) | 88.0% (12.6) |
| Motion blur | 66.6% (2.9) | 84.6% (13.6) |
| Noise | 72.3% (3.0) | 87.6% (12.0) |
| Darkness | 89.8% (2.3) | 97.2% (9.8) |
| **Clean images** | 90.6% (2.28) | 96.8% (7.72), average over the five thresholds* |

\* Computed after the verdicts from saved scores (`scripts/20_robust_conformal_clean.py`); H15 judged
only the clean set size.

This is **improved coverage in our tests, not a guarantee**: conformal prediction promises nothing for
damage it was not calibrated on, and under three of the five corruptions coverage is still below 90%.
The cost is sets about four times larger, on clean images too, where the robust thresholds now
over-deliver (96.8% instead of the 90% asked for). For best INT8 (extra analysis, not a prediction)
the pattern is the same: +14.9 points coverage at severity 3; on clean images 90.2% (2.63 classes)
with its clean-tuned threshold and 97.1% (10.32) with the robust ones*.

### The confidence alarm

The alarm averages the model's confidence over a window of 100 images and fires when that average is
below the 1st percentile of clean tuning windows. For FP32 it fired on all 1,200 windows of the 12
conditions where its clean-tuned coverage had fallen below 80%, and on none of 100 clean windows.

An exploratory check, computed after the verdicts from the same saved scores, shows the alarm on every
condition. It tracks harm rather than damage as such: FP32 barely loses accuracy in the dark, and its
alarm fires in at most 1% of dark windows; best INT8 does lose accuracy there, and its alarm fires in
66–96% of windows at darkness 4–5. It also fires in some milder conditions (FP32: 94% at noise 2, 56%
at defocus blur 1); whether that is a useful early warning or a false alarm depends on how much those
conditions cost, which was not defined in advance.

### Real new photos (ImageNetV2, task 3.9)

Each model uses its conformal threshold tuned on clean ImageNet images. Top-1; coverage (average set
size):

| Model | ImageNet test | ImageNetV2 |
|---|---|---|
| FP32 (H17) | 75.58%; 90.6% (2.28) | 62.10%; 80.7% (2.64) |
| Best INT8 (extra analysis, not a prediction) | 73.60%; 90.2% (2.63) | 59.83%; 80.6% (3.03) |

The 13.5-point drop is within the roughly 10–15 points that H17 took from published results for many
ImageNet models. The more important part
is the sets: they barely grow while the promise falls by ten points, the same silent failure Stage 2
found under simulated blur and noise. Best INT8 is 2.27 points below FP32 on these photos (1.98 on
ImageNet test), so real photo shift does not noticeably widen the INT8 gap here.

## 5. Why did darkness break default INT8?

A **likely explanation, not a proven one:** MinMax calibration stretches each layer's 256 available
levels to cover the most extreme value it ever saw. A few rare, extreme values then leave too few
levels for ordinary values, and dim images, whose values are small, fall into just a handful of
levels. Percentile calibration ignores the rarest 0.01% of values and spends the levels where the
values actually are. The evidence fits this: Percentile fixed most of the darkness collapse, and
adding dark-ish damaged images to the calibration set (H12) added nothing.

The pattern of what is left is consistent with this, though it does not prove it. At severity 5, best
INT8's remaining gaps to FP32 are large only for fog (21.5 points) and darkness (13.6 points), the two
corruptions that squeeze image values into a narrow range (fog blends every pixel towards light grey;
darkness scales them towards zero). Under blur and noise, which do not, the gaps are about 2.5 points
or less (defocus blur 2.5, motion blur 2.0, noise −0.3). We did not test the mechanism directly (for
example by counting how many levels dim or foggy images actually use per layer).

## 6. Other observations (from tuning-split checks)

- **The same accuracy does not mean the same model.** Building Percentile 99.99 with calibration
  groups of 64 instead of 128 images left tuning top-1 unchanged (72.00% vs 72.10%, interval −0.58 to
  +0.34 points), yet the two models gave the same top answer on only 95.2% of images.
- **Calibration luck moves reliability more than accuracy.** Four builds from different random
  calibration images spread by 0.28 points in top-1 (about 0.4% of 72%) but by 0.0026 in E-AURC (about
  5% of its value). Small E-AURC differences between INT8 builds should be read with care.

## 7. Practical takeaways (one model; starting points, not rules)

- **Prefer Percentile calibration over MinMax** for INT8. Here it recovered 13.5 of the 15.4 lost
  points at no cost in size.
- **Test under low-contrast damage** such as fog and darkness, not only clean images: that is where
  the remaining INT8 gap was largest, and where default INT8 failed worst.
- **Always report coverage with set size.** A coverage gain here cost sets about four times larger.
- **A confidence alarm is a cheap first layer.** Averaging confidence over 100 images needs no extra
  model and, for FP32, separated clean from badly damaged batches completely in our tests.

## 8. Limitations

- **One model** (MobileNetV3-Large). Other architectures may quantize differently.
- **One dataset** (ImageNet validation; 10,000 test images). The tuning split turned out 1.7 points
  harder than the test split (interval −3.20 to −0.14), a chance draw that may have set the alarm
  threshold and temperature slightly differently than test-like images would.
- **Synthetic damage.** Our own simulations of fog, blur, noise and darkness are not real weather or
  real cameras. The one real-photo check, ImageNetV2, is a different kind of shift (new photos of the
  same classes, not bad conditions). It showed the same silent coverage failure, but it says nothing
  about real fog or darkness, and the Stage 3 fixes (robust conformal, alarm) were not tested on it.
- **ImageNetV2's image licences** are per-image Flickr licences rather than one stated licence, so it
  is used for evaluation only and its images are never shown or shared.
- **Laptop only.** All measurements are on one Windows laptop's CPU. Stage 3 measured no speed; the
  rough Stage 1 speed numbers are laptop-only, and nothing here is a Raspberry Pi or edge-device result.
- **The noise floor comes from only 4 builds**, measured on tuning images and applied to test images.
- **Alarm windows hold one condition each.** How fast the alarm reacts when conditions change, and how
  it behaves on mixed batches, was not measured.
- **Robust conformal** was calibrated for FP32, FP16, default INT8 and best INT8 only.
- **Judging:** H13 was judged on its measured difference, because its "the interval must exclude zero"
  rule cannot apply to a "no big difference" claim (decided before the run). The judging script's
  interval check was made direction-aware *after* the run; a re-check gave identical verdicts
  (`docs/stage3_final_run_log.md`).

## 9. Reproduce

From the tag `stage3-final-run`, with the data and models in place (see `README.md`):

```bash
python scripts/run_stage3_final.py
python scripts/18_judge_stage3.py
```

ImageNetV2 (task 3.9): `python scripts/03_evaluate_accuracy.py --dataset imagenetv2-matched-frequency
--split all --precision fp32` (and `--precision int8_percentile99.99`), then
`python scripts/18_judge_stage3.py --out results/final/mobilenet_v3_large_stage3_verdicts_with_h17.json`
and `python scripts/21_imagenetv2_summary.py`.

The settings it uses were produced by scripts 10–17 (INT8 candidates and choice, grouping and
calibration-luck checks, damaged calibration, unrounded output, temperatures, shift-aware thresholds).

**Where the numbers come from:** `results/final/mobilenet_v3_large_stage3_verdicts.json` (verdicts and
their measurements), `results/sweep/` and `results/accuracy/` (every model's scores on every test
condition), `results/choices/` (Stage 3 settings), `results/checks/` (grouping and calibration-luck
checks), `results/final/mobilenet_v3_large_alarm_all_conditions_exploratory.json` (the exploratory
alarm table), `results/final/mobilenet_v3_large_stage3_verdicts_with_h17.json` and
`results/final/mobilenet_v3_large_imagenetv2_summary.json` (ImageNetV2). The results folder is not in git; every file can be regenerated by the scripts above.
