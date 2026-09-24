# Stage 2 hypotheses

Written **before** any corrupted-image measurement was made. Git history dates this file, so
it provably came first. Each hypothesis says what we expect, how we will judge it, and how
confident we are. Wrong predictions will be reported as wrong, not quietly rewritten.

## Setup these predictions refer to

- Model: MobileNetV3-Large (torchvision weights), as FP32, FP16, and INT8.
  INT8 uses ONNX Runtime's default static quantization (MinMax calibration on 512 clean images).
- Test images: the fixed 10,000-image ImageNet validation subset from Stage 1.
- Corruptions: fog, defocus blur, motion blur, noise, darkness, each at severities 1 (mild) to 5 (severe).
- Conformal prediction: the standard "LAC" method with a 90% target, tuned separately for each
  precision on 5,000 clean calibration images that never overlap the test images.
- "Clearly" means the 95% confidence interval excludes the comparison value.

What we already know from Stage 1 (clean images, 10,000 test images):
FP32 top-1 75.58%, FP16 75.63% (paired change +0.05 points, CI includes 0), INT8 60.33% (−15.25 points).

## Accuracy

**H1. Accuracy falls as damage gets stronger.** For every precision and corruption, top-1 accuracy
at severity 5 is clearly below severity 1. *Confidence: high.* This is mostly a sanity check that
the corruptions do what they claim.

**H2. FP16 behaves like FP32 under damage.** For every corruption and severity, the paired
FP16 − FP32 top-1 difference stays within ±0.5 points. *Confidence: high.* FP16 only rounds
weights slightly, and on clean images the difference was indistinguishable from zero.

**H3. Default INT8 falls apart faster than FP32 under damage.** At severity 3, INT8 keeps a smaller
share of FP32's accuracy than it does on clean images (clean: 60.33 / 75.58 = 0.80), for at
least 3 of the 5 corruptions. *Confidence: medium.* INT8's value ranges were set on clean images;
damaged images push values outside those ranges, where they get clipped. But when both models are
near-useless at high severity, the gap could also shrink, so this is genuinely uncertain.

**H4. Noise is the hardest corruption for INT8.** Of the five corruptions, noise gives the lowest
INT8/FP32 accuracy ratio at severity 3. *Confidence: low.* Noise adds many extreme pixel values,
exactly what MinMax calibration handles badly (Stage 1 diagnostic). A guess worth testing.

## Knowing when it's wrong

**H5. The 90% promise holds on clean images.** On clean test images, conformal coverage is between
88.5% and 91.5% for all three precisions. *Confidence: high.* This is what conformal prediction
guarantees when test images look like calibration images, so failing it would mean a bug.

**H6. The 90% promise breaks under damage.** Coverage falls as severity rises, and at severity 5
FP32 coverage is below 80% for at least 3 of the 5 corruptions. *Confidence: medium-high.* The
guarantee assumes test images resemble calibration images, and damaged images don't.

**H7. The model partly notices it's unsure.** Average prediction-set size at severity 5 is more
than twice the clean average, for every corruption (FP32). *Confidence: medium.* Bigger sets mean
the model spreads its bets, but models are often confidently wrong on unfamiliar inputs, so
set size may not grow enough to keep coverage.

**H8. Damage makes the model overconfident.** FP32 expected calibration error (ECE) at severity 5
is clearly higher than on clean images, for every corruption. *Confidence: medium-high.*

**H9. Default INT8 is worse at knowing when it's wrong.** INT8's AURC (error when skipping the
least-confident images; lower is better) is clearly higher than FP32's on clean images and at
every severity. *Confidence: medium-high.* It is less accurate, so its error is higher at most
coverage levels, but this also tests whether its confidence ranking still works.

## What would surprise us most

- FP16 differing from FP32 by more than 0.5 points anywhere (H2 false) — likely a bug.
- Clean coverage outside 88.5–91.5% (H5 false) — almost certainly a bug.
- INT8 holding up *better* than FP32 relative to its clean accuracy (H3 reversed).
