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

---

# Results (added after measuring; the predictions above are unchanged)

Measured on 25 September 2026: 3 precisions x (clean + 5 corruptions x 5 severities) on the fixed
10,000-image test split. Model outputs from commit `404a68c` (`scripts/08_corruption_sweep.py`),
reliability numbers from commit `62c47f4` (`scripts/07_reliability.py`). "Clearly" = the 95%
bootstrap interval excludes the comparison value, as defined above.

**Score: 6 confirmed (one with an exception), 3 rejected.**

| | Prediction | Outcome |
|---|---|---|
| H1 | Accuracy falls with severity | **Confirmed.** Severity 5 clearly below severity 1 in all 15 precision x corruption cases (paired intervals). |
| H2 | FP16 within ±0.5 points of FP32 | **Confirmed.** Largest paired difference −0.25 points (fog, severity 5; interval −0.37 to −0.13). |
| H3 | Default INT8 keeps a smaller share of FP32's accuracy at severity 3 than clean (0.80), for >= 3 corruptions | **Confirmed for 5 of 5.** INT8/FP32 at severity 3: fog 0.55, defocus blur 0.38, motion blur 0.41, noise 0.65, darkness 0.53. |
| H4 | Noise is INT8's worst corruption at severity 3 | **Rejected.** Noise was INT8's *least* bad (ratio 0.65); defocus blur was the worst (0.38). |
| H5 | Clean coverage in 88.5–91.5% | **Confirmed.** FP32 90.58%, FP16 90.54%, INT8 89.83%. |
| H6 | FP32 severity-5 coverage clearly below 80% for >= 3 corruptions | **Confirmed for 4 of 5:** fog 72.4%, defocus blur 36.7%, motion blur 30.1%, noise 19.6%. **Exception: darkness**, 89.2%. |
| H7 | FP32 average set size at severity 5 more than twice the clean 2.28, for every corruption | **Rejected.** Severity 5: fog 2.82, defocus blur 2.66, motion blur 2.51, noise 2.13, darkness 2.36. |
| H8 | FP32 ECE at severity 5 clearly above clean (0.177) for every corruption | **Rejected.** ECE fell or stayed level: fog 0.142, defocus blur 0.022, motion blur 0.009, noise 0.026, darkness 0.175. |
| H9 | INT8 AURC clearly above FP32 clean and at every severity | **Confirmed** at all 26 conditions (intervals never overlap). |

## What the rejections teach

- **H7: the model doesn't notice it is failing.** Under blur and noise, FP32's 90% coverage promise
  falls to 20–37% at severity 5, while its prediction sets stay about the same size. Conformal
  prediction tuned on clean images gives no protection, and no warning, under these shifts.
- **H8: the prediction assumed an over-confident model.** Task 2.4 showed this model is
  *under*-confident on clean images (confidence 57.9% vs accuracy 75.6%). Damage lowered accuracy
  towards its confidence, so ECE improved while accuracy collapsed (only noise at severity 5 became
  slightly over-confident, +2.6 points). ECE alone would have given a falsely reassuring picture.
  Small ECE values are also biased upwards (see `brokkr/shift/reliability.py`).
- **H4: the reasoning about extreme pixel values didn't transfer to accuracy.** A likely reason
  (not tested): noise hurts FP32 so badly (11.1% at severity 5) that there is little left for INT8
  to lose relatively.

## Findings not predicted

- **Darkness harms only INT8.** FP32 barely changes (75.6% clean, 73.7% at severity 5, coverage
  holds at 89.2%), but default INT8 falls from 60.2% to 20.5%. A likely explanation, not yet
  verified: INT8's value ranges were set on normally lit images, so dark images use only a few of
  its 256 levels. Stage 3's "INT8 calibration on corrupted images" experiment tests this.
- **Default INT8 needs much bigger sets to keep its promise:** 8.3 classes vs 2.3 on clean images,
  and its confidence ranks its own mistakes worse (E-AURC 0.085 vs 0.049).
