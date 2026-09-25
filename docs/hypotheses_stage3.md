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

## Note added 25 September 2026, before any temperature is fitted (task 3.5)

Nothing above has been changed; no prediction and no design rule changes.

- **Task 3.6 uses raw scores, not temperature-scaled scores, for every model** (robust conformal
  thresholds, their clean-tuned baselines, and the confidence alarm). Decided now, before any T is
  fitted. Reasons: (1) H15 and H16 and their baselines (clean set size 2.28, the 12 "harmful"
  conditions, Stage 2 coverage) were all set on raw scores, so switching would change what the
  predictions are about; (2) one change at a time: if 3.6 used scaled scores, its results would mix
  the effect of the 3.6 method with the effect of temperature, and neither could be read on its own;
  (3) H14 predicts that temperature makes the model over-confident under damage, which would work
  directly against the confidence alarm, so the alarm is judged on the model's own scores.
  Temperature-scaled scores are used only for the calibration numbers of task 3.5 and H14.
- **Which models get a temperature:** every model in the final run, each its own T: FP32, FP16,
  default INT8, best INT8 (Percentile 99.99), best INT8 with unrounded output, and the five
  leave-one-out INT8 models. Not the check-only models (group-64, calibration-luck seeds).
- **How T is fitted:** minimise the average negative log-likelihood (NLL) of the true class on the
  clean tuning split (5,000 images; logits from `scripts/03_evaluate_accuracy.py --split tuning`),
  scores divided by T before softmax, computed in float64.
  - *Search:* golden-section search over log T in the range **T = 0.1 to 10**, stopping when the
    interval is narrower than 0.0001 in log T (T known to about 0.01%). NLL is convex in 1/T, so it
    has a single minimum and the search cannot get stuck in a wrong dip.
  - *Edge check:* if a fitted T lies within 1% of either end of the range (below 0.101 or above
    9.9), the run stops and reports it instead of using it.
  - *Sanity checks:* NLL at the fitted T is not above NLL at T = 1, and the top answer of every image
    is unchanged.
- **ECE uses exactly Stage 2's binning:** `brokkr.shift.reliability.ece`, 15 equal-width bins on
  the probability of the top answer, bin k = (k/15, (k+1)/15]. That code is unchanged since the
  Stage 2 results (no commit to `brokkr/shift/` or `scripts/07_reliability.py` after `1ed5d65`).
- **Reporting rule for H11:** if best INT8 with unrounded output improves E-AURC by less than 15%
  (relative), the report adds that this is within about 3 times the build-to-build range (0.0026,
  about 5% of best INT8's tuning E-AURC of 0.0539), and that the range comes from only 4 builds.
- **CPU features (optional check done):** py-cpuinfo 9.0.0 (MIT) was tried on this laptop. It reads
  the AVX-512 VNNI flag (absent, as expected without AVX-512) but not AVX-VNNI, the variant this CPU
  family would have, so VNNI stays recorded as unknown and py-cpuinfo is not added.

## Note added 25 September 2026, before any task 3.6 threshold is computed

Nothing above has been changed; no prediction changes. This note fills in details the design rules
left open, before anything is computed.

- **H16 test windows: already fully specified above, confirmed.** Per condition: the 10,000 test
  images in one fixed random order (seed 3; the same order for every condition and every model),
  cut into 100 consecutive, non-overlapping windows of 100 images. Every window holds one condition
  only (no mixing); 26 conditions, so 100 clean windows and 1,200 windows over the 12 harmful
  conditions. Details now fixed as well:
  - The order is `numpy.random.default_rng(3).permutation(10000)`, applied to the test split's
    sorted image order (the order of every saved result file); window w holds places 100w to 100w+99.
  - Confidence is the raw softmax probability of the top answer (raw, per the 3.5 note).
  - The alarm fires when a window's average confidence is strictly below the threshold.
- **Alarm threshold details (tuning split):** one random generator with seed 4 draws 10,000 windows
  one after another; each window is 100 different images (drawn without replacement), and windows
  are drawn independently, so two windows may share images. The threshold is
  `numpy.percentile(window averages, 1)` (NumPy's default linear interpolation). One threshold for
  every final-run model (the ten listed in the 3.5 note); H16 is judged for FP32.
- **Known before computing: the tuning split is harder than the test split** (FP32 clean top-1 73.90%
  vs 75.58%). If harder images also mean lower confidence, the threshold set on tuning sits slightly
  low, so on test the alarm should give fewer clean false alarms but also be less sensitive. This
  direction will be reported next to the H16 outcome.
- **Conformal score: identical to Stage 2, unchanged since the Stage 2 results.** Score = 1 minus
  the raw softmax probability of the true class (LAC); threshold = the exact k-th smallest score,
  k = ceil((n + 1) x 0.9) (`brokkr/shift/conformal.py`, last changed in task 2.5, `2a77a66`). The
  softmax it uses (`brokkr/shift/reliability.py`) is unchanged: since Stage 2 that file has only had
  lines added (temperature functions), none changed or removed. The clean-tuned baseline threshold
  is recomputed and must equal Stage 2's saved one.
- **Robust conformal calibration mix, exactly** (the design rule says one-third clean, two-thirds
  damaged, the four allowed corruptions and severities 1–5 equally): the 5,000 conformal_calibration
  images are shuffled once (seed 9; the same shuffle for every held-out corruption and every model).
  The first 1,667 stay clean. The other 3,333 are damaged; the k-th of them gets the (k mod 20)-th of
  the 20 (corruption, severity) pairs, the four allowed corruptions in Brokkr's order times
  severities 1 to 5, so each pair gets 166 or 167 images. Every image is used once, in one version.
  The damaged scores come from the task 3.1 sweep (damage pattern seeded by dataset position).
  n = 5,000, so k = 4,501.
- **Which models get robust conformal thresholds:** FP32, FP16 and default INT8, the three models
  with damaged conformal_calibration outputs from task 3.1. H15 is about FP32. The best-INT8
  variants are not included here (they would need new damage sweeps on conformal_calibration).
- **Final-report wording:** robust conformal has no coverage guarantee under corruptions it was not
  calibrated on. Results are described as "improved coverage in our tests", never "guaranteed".

## Note added 25 September 2026, before the final run (task 3.7)

Nothing above has been changed; no prediction and no threshold changes. This note fixes how the
final run is done and judged, before any Stage 3 model has seen a test image.

- **H17 is postponed to task 3.9, not part of 3.7.** Reasons: ImageNetV2 is not downloaded (1.26 GB,
  which needs your go-ahead), and its licence is not recorded yet, so hard rule 8 forbids using it.
  H17 involves only FP32 and the existing clean-tuned conformal threshold, so running it later does
  not reuse the ImageNet test images. The judging script already contains H17's rule; it reports
  "NOT RUN" until the ImageNetV2 result exists (FP32 on all 10,000 matched-frequency images, saved
  as `results/accuracy/mobilenet_v3_large_fp32_imagenetv2-matched-frequency_all.json`).
- **Extra analysis (not a prediction): robust conformal for best INT8.** Before the final run,
  Percentile 99.99 is measured on the conformal_calibration split, clean and damaged (the task 3.1
  sweep, same settings), and its robust thresholds are computed exactly as for FP32 (dated 3.6 note).
  In the final run its coverage and set size on the held-out corruptions are reported, labelled as
  an extra analysis.
- **What the final run measures on the test split (10,000 images).** Best INT8 and best INT8 with
  unrounded output: clean and all 25 damaged conditions. Each leave-one-out INT8 model: clean and
  its held-out corruption at severities 1–5 only (the design rule). FP32, FP16 and default INT8 are
  not rerun: their Stage 2 test results are the baselines.
- **How each prediction is judged** (`scripts/18_judge_stage3.py`, written and tested on fake data
  before the run). "Paired CI" = paired bootstrap 95% interval of the difference, 1,000 resamples
  of the same test images for both sides, seed 0.
  - H10: best INT8 clean test top-1 >= 67.9%.
  - H11 (clean test): unrounded has 0 tied images; E-AURC of best INT8 minus E-AURC of unrounded is
    at least 10% of best INT8's E-AURC, with its paired CI excluding zero; file size increase < 5%.
  - H12: at darkness severity 5, top-1 of the model calibrated without darkness minus top-1 of best
    INT8 >= 10 points, with its paired CI excluding zero.
  - H13: the average over the five leave-one-out models of clean test top-1, minus best INT8's,
    is between −1 and +1 point. *Interpretation fixed now:* H13 claims the two are close, so the
    rule "the paired CI must exclude zero" cannot sensibly apply (it would need a real difference
    to confirm "no big difference"). H13 is judged on the measured value; its paired CI is reported.
  - H14 (FP32): fitted T < 1 (from 3.5); clean test ECE with scaling < 0.03; at severity 5, ECE
    with scaling minus ECE without >= 0.02 with its paired CI excluding zero, for at least 3 of 5
    corruptions.
  - H15 (FP32): for each held-out corruption at severity 3, coverage with its robust threshold minus
    coverage with the clean-tuned threshold; averaged over the five: >= 10 points, paired CI (of the
    per-image average) excluding zero. Clean test average set size with the robust thresholds
    (averaged over the five) at least 2x that with the clean-tuned threshold, paired CI of the
    difference excluding zero. Both parts must hold.
  - H16 (FP32): the alarm fires in >= 90% of the 1,200 harmful-condition windows and in at most 2 of
    the 100 clean windows.
  - H17 (FP32, ImageNetV2): top-1 between 60% and 68%, and the upper end of the bootstrap 95%
    interval of clean-tuned coverage below 88%.
  - Verdicts: PASS (every part holds), FAIL, or NOT RUN. **"Within noise"** is added next to the
    verdict when an INT8-vs-INT8 difference (H11 E-AURC, H12 and H13 top-1) is smaller than the
    calibration-luck range (top-1 0.28 points, E-AURC 0.0026). **H11 wording:** if the relative
    E-AURC improvement is under 15%, the output adds that it is within about 3x the build-to-build
    range, from only 4 builds.
- **Coverage is never shown without set size.** Every coverage number in the judging output and the
  write-up is shown next to its average set size, clean and damaged. A coverage gain is never called
  a win without its set size.
- **Rerun rule.** A test-split measurement may be rerun only for a technical failure (crash,
  corrupted or incomplete output file, power loss), never because of its result. Every rerun is
  logged with its reason in `docs/stage3_final_run_log.md`.
- **Running it.** The run starts from a commit tagged `stage3-final-run`, with no uncommitted
  changes. The laptop is kept awake by the run script itself (Windows' keep-awake request, which ends
  when the script ends; no system setting is changed). The run is resumable (finished steps are
  skipped), and at the end every expected result file is checked to exist and be complete (it
  loads, its checksum matches, and it has scores for all 10,000 images).

## Outcomes (added 26 September 2026, after the final run)

Measured once on the 10,000 test images, from the tag `stage3-final-run` (`54026c9`), with no reruns
(`docs/stage3_final_run_log.md`). Judged by `scripts/18_judge_stage3.py` using only the rules above
(`results/final/mobilenet_v3_large_stage3_verdicts.json`). Intervals are paired bootstrap 95%.
"Within noise": the INT8-vs-INT8 difference is smaller than the calibration-luck range (top-1 0.28
points, E-AURC 0.0026). Nothing above this section was changed.

| | Prediction | Verdict | Measured |
|---|---|---|---|
| H10 | Best INT8 recovers half the clean gap (>= 67.9%) | **PASS** | 73.60% (FP32 75.58%, default INT8 60.15%): the gap shrinks from 15.4 to 2.0 points. |
| H11 | Unrounded output: 0 ties, E-AURC >= 10% lower, file < 5% bigger | **FAIL** (within noise) | 0 tied images and the same file size, but E-AURC went from 0.0510 to 0.0524, 2.7% *worse*: best minus unrounded −0.0014 (−0.0030 to −0.0001). Under 15%: within about 3x the build-to-build range, which comes from only 4 builds. |
| H12 | Calibrated without darkness: >= 10 points better at darkness s5 | **FAIL** | 60.15% for best INT8, 59.48% calibrated without darkness: −0.67 points (−1.41 to +0.06). |
| H13 | Damaged calibration within 1 point on clean images | **PASS** (within noise) | Average of the five leave-one-out models 73.85% vs 73.60%: +0.25 points (−0.12 to +0.59). |
| H14 | Temperature: T < 1, clean ECE < 0.03, backfires at s5 for >= 3 of 5 | **PASS** | T 0.743; clean ECE 0.018. At severity 5, scaled minus raw ECE: defocus blur +0.123, motion blur +0.147, noise +0.140 (backfires), but fog −0.088 and darkness −0.156 (scaling *helped*). |
| H15 | Robust conformal: +10 points coverage at s3 (held-out), clean sets >= 2x | **PASS** | Coverage +12.8 points (12.4 to 13.2); clean average set size 2.28 -> 7.72 classes (3.4x; difference 5.3 to 5.6). |
| H16 | Alarm fires in >= 90% of harmful windows, <= 2 of 100 clean | **PASS** | 1,200 of 1,200 harmful windows (100%); 0 of 100 clean windows. |
| H17 | ImageNetV2 | **NOT RUN** | Postponed to task 3.9 (dataset not downloaded, licence not recorded). |

Coverage with average set size, FP32 at severity 3, clean-tuned threshold -> robust threshold (H15):
fog 88.4% (2.4) -> 96.4% (10.3); defocus blur 72.6% (2.8) -> 88.0% (12.6); motion blur 66.6% (2.9)
-> 84.6% (13.6); noise 72.3% (3.0) -> 87.6% (12.0); darkness 89.8% (2.3) -> 97.2% (9.8). This is
improved coverage in our tests, not a guarantee: under three of the five held-out corruptions it is
still below 90%, and the price is sets about four times larger.

Extra analysis (not a prediction), best INT8 with robust conformal: coverage at severity 3 +14.9
points (14.5 to 15.3); clean average set size 2.63 -> 10.32 classes.

### What the failures and the mixed result teach

- **H12: there was little left to fix.** Percentile calibration on clean images already took best
  INT8 at darkness severity 5 from default INT8's 20.48% to 60.15% (FP32 73.73%). Adding damaged
  images to the calibration set gave nothing more for unseen darkness. The prediction's reasoning
  ("value ranges fitted only to well-lit images") was probably the wrong mechanism; a likely, not
  proven, explanation is that MinMax spent its 256 levels on rare extreme values, which Percentile
  ignores.
- **H11: removing ties did not improve confidence ranking.** Every tie disappeared, but E-AURC got
  slightly worse, by less than the build-to-build luck. Ties were already rarer with Percentile
  calibration (110 of the 10,000 clean test images, vs 262 for default INT8), so there was little to
  gain.
- **H14 passed on its 3-of-5 rule, but the picture is mixed.** Sharpening made confidence less
  honest under blur and noise, as predicted, but more honest under fog and darkness, where the model
  stays under-confident. "Temperature scaling backfires under damage" holds only for some damage.

### Findings not predicted

- **Best INT8's remaining gap is largest under heavy fog and darkness:** 2.0 points below FP32 on
  clean images, but 13.6 points below at darkness severity 5 (default INT8: 53 points below) and
  21.5 points below at fog severity 5 (default INT8: 42.0 points below). Under severity-5 blur and
  noise it is within 2.5 points of FP32.
- **The alarm separated clean from harmful conditions completely** (100% vs 0%), even with its
  threshold set on the harder tuning split. Only single-condition windows were tested; how fast it
  reacts when conditions change was not measured.

## Note added 26 September 2026, before any ImageNetV2 measurement (task 3.9)

Nothing above has been changed; H17's prediction and thresholds are unchanged.

- **Data and licence.** ImageNetV2 matched-frequency (Recht et al., ICML 2019), 10,000 images, 10
  per class, downloaded from the authors' Hugging Face page (1,264,079,360 bytes; SHA-256 checked
  against the published value). Licence recorded in `brokkr/datasets.py` exactly as the sources state
  it: the Hugging Face card says "mit"; the authors' README says the licence file "does not apply to
  the actual image data. The images come from Flickr which provides corresponding license
  information." Used for evaluation only: the images are never shown, committed or redistributed,
  and the dataset paper is cited wherever its results appear. Labels are the folder names, as in the
  authors' own loader.
- **H17 run.** FP32 on all 10,000 images (`scripts/03_evaluate_accuracy.py --dataset
  imagenetv2-matched-frequency --split all`), judged by `scripts/18_judge_stage3.py` (the
  direction-aware version) with FP32's clean-tuned threshold (0.963033). The verdicts are saved to a
  separate file; the original 3.7 verdict record is kept unchanged. Same rerun rule as 3.7: technical
  failures only, each logged with its reason in `docs/stage3_final_run_log.md`.
- **Extra analysis (not a prediction): best INT8 on ImageNetV2.** Percentile 99.99, same images,
  reported next to FP32: top-1 and coverage with average set size, using its own clean-tuned
  threshold (0.968429) (`scripts/21_imagenetv2_summary.py`).
- **Proposed definition of "harm" for future stages (proposal only; not applied to Stage 3).** A
  condition is *harmful* for a model if, measured on the tuning split before any alarm result is
  looked at, either (a) its clean-tuned 90% conformal coverage falls below 80%, or (b) its top-1 is
  more than 10 points below the same model's clean top-1. An alarm firing on a harmful condition is a
  catch; firing on a condition that is neither harmful nor clean is reported separately as an "early
  warning", not counted as a false alarm; firing on clean images is a false alarm. Reason: (a) is
  Stage 3's rule; (b) adds accuracy loss that coverage can miss (for example best INT8 in darkness);
  defining it on tuning keeps the test split untouched.
