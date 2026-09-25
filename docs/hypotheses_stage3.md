# Stage 3 hypotheses

Written **before** any Stage 3 measurement. Git history dates this file, so it provably came first.
As in Stage 2, each prediction has a threshold, a way to judge it, and a confidence level, and the
outcomes will be added below without changing anything above.

## Design rules, fixed in advance

- **No Stage 3 setting is tuned on the test split; the test split was used for Stage 2 baselines.**
  Stage 3 measures its fixes on the test split in the final run (task 3.7). Every setting is chosen
  on other splits:
  - INT8 calibration method, temperature, and alarm threshold: the **tuning** split (5,000 images).
  - Conformal thresholds: the **conformal_calibration** split (5,000 images).
  - INT8 value ranges: the **int8_calibration** split (512 images).
- **Leave-one-corruption-out.** A fix that learns from damaged images is built five times, each time
  without one corruption type, and each version is tested only on the corruption it never saw
  (plus clean images). Reported numbers for these fixes are always on unseen corruption types.
- **INT8 calibration method (3.2):** MinMax, Percentile (99.99 and 99.999), and Entropy are compared;
  the one with the highest clean top-1 on the tuning split is "best INT8". Only that one goes on.
- **INT8 weights stay per-channel (fixed, not a candidate).** Every Stage 1–3 INT8 model uses
  per-channel int8 weights and per-tensor uint8 activations (`brokkr/quantize.py`). Per-tensor weights
  are excluded in advance: in the Stage 1 diagnostic they agreed with FP32 on only 28.5% of images,
  and MobileNetV3's depthwise layers have very different value ranges per channel.
- **Damaged INT8 calibration (3.3):** best method; the 512 calibration images are half clean, half
  damaged with a random corruption (from the four allowed) at a random severity 1–5.
- **Unrounded output (3.4):** best INT8 with the final layer's output left unquantized
  (ONNX Runtime `OpTypesToExcludeOutputQuantization = ["Gemm"]`); its weights stay 8-bit.
- **Temperature scaling (3.5):** one number T per model, fitted to minimise negative log-likelihood
  on clean tuning images. Divides the scores before softmax; it never changes the top answer.
- **Robust conformal (3.6):** the 90% threshold is tuned on conformal_calibration images that are
  one-third clean and two-thirds damaged (the four allowed corruptions, severities 1–5 equally).
- **Alarm (3.6):** looks at the average confidence (probability of the top answer) of a window of
  100 images, and fires when that average is below a threshold.
  - *Threshold:* draw 10,000 random windows of 100 different images from the clean tuning split
    (seed 4); the threshold is the 1st percentile of their average confidences (about 1% false alarms
    on clean images by construction). Chosen separately for each model.
  - *Test windows:* for each of the 26 test conditions, put the 10,000 test images in one fixed random
    order (seed 3, the same order for every condition and model) and cut it into 100 consecutive,
    non-overlapping windows of 100 images. Every window holds a single condition; switching between
    conditions (how fast the alarm reacts to a change) is not measured.
  - *"Harmful" conditions,* fixed now from the Stage 2 baselines (FP32 clean-tuned coverage below 80%
    on the test split), 12 of 25: defocus blur s2–s5, motion blur s2–s5, noise s3–s5, fog s5.
- **How comparisons are judged.** A threshold on a single number ("at least 67.9%") refers to the
  measured value. Wherever a prediction compares two models or settings, it also requires the
  **paired bootstrap 95% interval of the difference to exclude zero**: 1,000 resamples of the same
  test images for both sides, seed 0, as in `brokkr.accuracy.paired_bootstrap_diff`.

## Predictions

Stage 2 baselines (10,000 test images): FP32 top-1 75.58%, default INT8 60.15%; default INT8 at
darkness severity 5: 20.5% (FP32 73.7%); FP32 conformal coverage at severity 5: fog 72.4%, defocus blur
36.7%, motion blur 30.1%, noise 19.6%, darkness 89.2%; default INT8 E-AURC 0.085 (FP32 0.049).

**H10. Better calibration recovers most of INT8's loss.** Best INT8 recovers at least half of the
clean top-1 gap to FP32 (test top-1 >= 67.9%). *Confidence: medium.* The Stage 1 diagnostic
(Percentile agreement with FP32 88% vs MinMax 71%) points this way, but agreement is not accuracy.

**H11. Unrounded output removes ties and improves confidence ranking.** Best INT8 with unrounded
output has 0 tied test images and an E-AURC at least 10% lower (relative) than best INT8 without it,
with the paired 95% interval of the difference excluding zero, and a file size increase under 5%. *Confidence: medium.* Ties can only hurt ranking, but they affected under 3% of
images, so the effect may be small.

**H12. Damaged calibration fixes darkness, even unseen.** INT8 calibrated without darkness images
(leave-one-out) scores at least 10 points higher at darkness severity 5 than best INT8 calibrated on
clean images only, with the paired 95% interval of the difference excluding zero. *Confidence: low.* It helps only if any damage widens INT8's value ranges in a
way that also suits dark images; that is a guess.

**H13. Damaged calibration costs little on clean images.** Averaged over the five leave-one-out INT8
models, clean test top-1 is within 1 point of best INT8 calibrated on clean images. *Confidence: medium.*

**H14. Temperature scaling sharpens, fixes clean calibration, but backfires under damage.** For FP32:
fitted T < 1 (the model is made more confident); clean ECE falls below 0.03; but at severity 5, ECE
with scaling is at least 0.02 higher than without, with the paired 95% interval of the difference
excluding zero, for at least 3 of 5 corruptions. *Confidence: medium-high.*
Stage 2 showed damage erases the under-confidence, so sharpening should overshoot into overconfidence.

**H15. Robust conformal trades clean efficiency for coverage under unseen damage.** For FP32, averaged
over the five held-out corruptions at severity 3, coverage rises by at least 10 points over the
clean-tuned threshold; clean average set size at least doubles (from 2.28). *Confidence: medium.*

**H16. The alarm catches the damage that matters.** For FP32, over the 12 harmful conditions listed
in the design rules (1,200 test windows), the alarm fires in at least 90% of windows; on clean test
images it fires in at most 2 of 100 windows. *Confidence: medium.* Stage 2 showed confidence drops as accuracy
drops (ECE stayed low), so falling confidence should track the harm.

**H17. Real photos from a new collection also shift the model (ImageNetV2).** FP32 top-1 on ImageNetV2
("matched frequency", 10,000 images) is between 60% and 68%, and the upper end of the bootstrap 95%
interval of clean-tuned conformal coverage is below 88%. *Confidence: medium.* Published results report accuracy drops of roughly 10-15
points on ImageNetV2 for many ImageNet models; the coverage part is our own guess.

## What would surprise us most

- Best INT8 still below 62% clean (calibration method barely matters).
- Temperature scaling making damaged-image calibration better (H14 reversed).
- The alarm firing often on clean images despite being tuned for 1% (would suggest the clean tuning
  and test images differ more than expected).

## Note added 25 September 2026, before any task 3.2–3.6 measurement

Nothing above has been changed. This records a fact learned in task 3.1, before any Stage 3 setting
was chosen or any Stage 3 fix was measured.

**The tuning split is harder than the test split.** FP32 clean top-1: tuning 73.90% (5,000 images),
test 75.58% (10,000 images). Tuning minus test: −1.68 points, 95% interval −3.20 to −0.14 (unpaired
bootstrap, 1,000 resamples, seed 0; `scripts/09_compare_splits.py`). Conformal_calibration minus test:
−0.30 points (−1.68 to +1.15), no measurable difference. The splits are drawn at random with fixed
seeds and are not stratified by class, so this is a chance draw.

**What it could affect, stated in advance:**
- *INT8 method choice (3.2):* not affected; methods are compared on the same tuning images.
- *Temperature (3.5):* T is fitted on clean tuning images, so its value may differ slightly from a
  fit on images like the test split.
- *Alarm (3.6, H16):* the threshold comes from average confidence on clean tuning images. If harder
  images mean lower confidence, the threshold is lower and the alarm fires less often on test images:
  fewer false alarms on clean test windows, and fewer detections in harmful windows. When H16 is
  judged, this direction will be reported alongside the outcome.

**Decision: all design rules stay exactly as committed.** Changing them after seeing tuning data is
what pre-registration is meant to prevent.

## Note added 25 September 2026, before any task 3.2 measurement: how the INT8 candidates are built

Nothing above has been changed; no design rule changes. These details were fixed while writing the
code, before any candidate was built or measured.

- **Calibration images are fed in groups of 128** (4 batches of 32, in the split's fixed order;
  onnxruntime's `CalibStridedMinMax` option). The Percentile and Entropy methods otherwise keep every
  layer's output for all 512 images in memory at once, about 22 GB, and this laptop has 15.7 GB.
  MinMax gives exactly the same model either way (checked by `scripts/10_int8_methods.py` and a test).
  For Percentile and Entropy the first group sets the histogram's bin width, so the group size can
  shift the ranges slightly; it is fixed here and will not be tuned.
- **Histogram settings are onnxruntime 1.23.2's defaults** (`quantize_static` cannot change them):
  Percentile uses 2,048 bins on absolute values, with the range clipped to the smallest and largest
  value seen; Entropy uses 128 bins and 128 quantized bins.
- **The MinMax candidate is the existing default INT8 model** (`models/*_int8.onnx`).
- **The build sanity check uses 256 tuning images**, not test images.
- **An exact tie in tuning top-1 is not covered by the rule**: `scripts/11_choose_int8.py` stops,
  and the decision will be written here before going on.

## Note added 25 September 2026, before the grouping checks and before any task 3.3 build

Nothing above has been changed; no design rule changes.

- **Group size 128 is a fixed part of the INT8 method** for all of Stage 3: calibration images are
  fed in 4 batches of 32 per group, in the calibration split's fixed order. The 3.3 models use it
  exactly as the 3.2 candidates did. It will not be changed, whatever the checks below show.
- **Grouping checks for the histogram method** (`scripts/12_int8_grouping_check.py`). The MinMax
  check does not cover Percentile or Entropy, whose histograms are re-binned as groups arrive.
  - (a) *Repeatability:* Percentile 99.99 is rebuilt with groups of 128 a second time. Expected: the
    same model (every stored number equal) and identical scores on all 5,000 tuning images. If not,
    stop and report before going on.
  - (b) *Sensitivity:* Percentile 99.99 is built once with groups of 64. Its clean tuning top-1 and
    the paired difference from the group-128 model are reported. This is information only: it does
    not change the choice of method or the group size.
- **Weights are per-channel, as the design rule says, and this was checked in the files.** In the
  chosen Percentile 99.99 model, all 64 int8 weight tensors have one scale per output channel and
  none has a single scale; all 142 activation quantizers have one scale each and use uint8. This is
  recorded in `brokkr/quantize.py` (`INT8_SETTINGS`, `per_channel=True`), in every INT8 model's
  `.json` record, and in the design rules above; every 3.3 model is checked the same way
  (`brokkr.quantize.weight_quantization`).
- **3.3 calibration images, exactly** (the design rule says "half clean, half damaged with a random
  corruption (from the four allowed) at a random severity 1–5"): exactly 256 of the 512 images,
  chosen at random, are damaged; each gets a corruption drawn uniformly from the four allowed and a
  severity drawn uniformly from 1–5, all from one random generator with seed 5
  (`brokkr.quantize.damaged_calibration_plan`). The five leave-one-out models damage the same images
  at the same severities; only the list of allowed corruptions differs. Each image's damage pattern
  is seeded by its dataset position, as in the sweep, and damage is applied to the 224x224 picture
  before normalisation, as everywhere.
- **3.3 build safety:** each model is written to a temporary file and only kept if it loads, gives
  finite scores of the right shape on 256 clean tuning images, agrees with FP32's top answer on at
  least 20% of them, and has only per-channel weights. The run stops at the first failure.

## Note added 25 September 2026, before the calibration-luck analysis and before any task 3.4 build

Nothing above has been changed; no prediction and no design rule changes.

- **Extra analysis: calibration luck (noise floor).** Rebuilding with groups of 64 instead of 128
  left tuning top-1 unchanged but gave the same top answer on only 95.2% of images, so part of any
  INT8 model is luck in how it was calibrated. To measure that luck
  (`scripts/14_int8_calibration_luck.py`):
  - Percentile 99.99 is built 3 more times, each from a different random set of 512 calibration
    images, seeds 6, 7 and 8. The sets are drawn (without replacement) from the 29,488 images that
    belong to no split (`brokkr.datasets.unassigned`), so they never touch test, tuning or
    conformal-calibration images. Everything else is unchanged (method, groups of 128, per-channel
    weights), and each model passes the same build checks as the 3.3 models.
  - For these 3 models plus the original (calibrated on the int8_calibration split), clean tuning
    top-1 and E-AURC (confidence = probability of the top answer) are reported, with their spread:
    the range (largest minus smallest of the 4) and the standard deviation.
  - **Use in the final run (3.7):** when two INT8 variants are compared (top-1 or E-AURC), a
    difference whose size is smaller than this range is also labelled "within noise". The verdicts
    of the predictions are unchanged; this label is added next to them.
  - Limitation, stated in advance: the range is measured on the 5,000 tuning images and applied to
    the 10,000 test images, and 4 builds give only a rough range. It measures build luck only; the
    paired bootstrap intervals already cover the luck of which images were tested.
- **H11 already has numeric thresholds** (checked, nothing added): E-AURC at least 10% lower,
  relative to best INT8 without unrounded output, with the paired 95% interval of the difference
  excluding zero; 0 tied test images; file size increase under 5%.
- **CPU details in every result from now on:** the machine record (`brokkr.fingerprint`) now also
  lists instruction-set features, for the later laptop-vs-Raspberry-Pi comparison. On this laptop
  Windows reports AVX yes, AVX2 yes, AVX-512F no; Windows gives no standard-library way to read VNNI,
  so it is recorded as unknown rather than guessed. On Linux (e.g. the Pi) all are read from
  `/proc/cpuinfo`.
- **3.4 build, exactly** (`scripts/15_int8_unrounded_output.py`): Percentile 99.99, the same 512
  int8_calibration images, groups of 128, per-channel int8 weights, with onnxruntime's
  `OpTypesToExcludeOutputQuantization = ["Gemm"]` so the final layer's output stays in float; its
  weights stay int8. It passes the same build checks, plus a check that the final output really is
  not rounded. Tied top scores on clean tuning images are reported as a tool check; H11 is judged
  on the test split in 3.7.
