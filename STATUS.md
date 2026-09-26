# Brokkr status

Snapshot as of **27 September 2026**. **Stages 1, 2 and 3 are complete** and merged into `main`
(merge commit `be01206`, tagged `report-1`: Brokkr Technical Report 1, `docs/writeup.md`). **Stage 4
is in progress** on branch `stage-4` (created from `main` at `be01206`): task 4.0 done, the task 4.1
breadth sweep has run; its analysis has not started.

## 0. Start here (a new session needs nothing else)

### Where things stand
- **Stage 3 is done** (tasks 3.0–3.9). Predictions and outcomes: `docs/hypotheses_stage3.md`. Final-run
  record: `docs/stage3_final_run_log.md` (no reruns). Verdicts:
  `results/final/mobilenet_v3_large_stage3_verdicts.json` (3.7) and `..._verdicts_with_h17.json` (3.9).
- **Brokkr Technical Report 1** (`docs/writeup.md`, task 3.8) is approved and committed. Its "Related
  work" section is a placeholder marked "To be written by H"; do not draft it.
- **Stage 4 is in progress** (plan: `ROADMAP.md`; predictions and outcomes: `docs/hypotheses_stage4.md`).
  Task 4.0 is done; the task 4.1 sweep finished on 27 September 2026 (see "Stage 4 progress" below).
  Next: the reliability script, then judging H18–H22, then M2.
- Always use `.venv/Scripts/python.exe` (the system Python lacks the packages). Tests: `pytest`; style:
  `ruff check .`.

### Stage 3 headline results (10,000 ImageNet test images unless stated; MobileNetV3-Large, laptop CPU)
- **Default INT8 broke the model:** 60.15% clean vs FP32 75.58%; 20.48% at darkness severity 5 vs FP32
  73.73%.
- **Percentile 99.99 calibration ("best INT8", chosen on the tuning split) fixed most of it:** 73.60%
  clean (2.0 points below FP32), same 5.9 MB size; 60.15% at darkness 5 (13.6 below FP32; default INT8
  was 53 below); 31.3% at fog 5 (21.5 below FP32).
- **Verdicts:** H10, H13, H14, H15, H16, H17 PASS; H11, H12 FAIL.
  - H11: unrounded output removed all ties but E-AURC got 2.7% worse (within build-to-build noise).
  - H12: calibrating on damaged images did not help unseen darkness (−0.67 points).
  - H14 passed on its 3-of-5 rule but is mixed: temperature hurt calibration under blur and noise,
    helped under fog and darkness.
  - H15: robust conformal +12.8 points coverage at severity 3, with clean sets 2.28 -> 7.72 classes
    (clean coverage 96.8%).
  - H16: the FP32 alarm fired on 1,200 of 1,200 harmful windows and 0 of 100 clean ones.
  - H17 (ImageNetV2, real new photos): FP32 62.10%, clean-tuned coverage 80.72% with set size 2.64
    (ImageNet test: 90.58% with 2.28). Best INT8 (extra): 59.83%, 80.64% with 3.03.
- **Main lesson:** the model's "I'm not sure" signal fails silently under shift (coverage falls, sets
  barely grow), on simulated damage and on real new photos alike.

### Standing conventions (also in `CLAUDE.md`)
- **Differences are "new minus old"** (the fix minus what it replaces), with paired bootstrap 95%
  intervals (same resampled images, 1,000 resamples, seed 0). State the direction: for accuracy and
  coverage positive = better; for E-AURC and ECE positive = worse.
- **Coverage is never reported without its average set size** (clean and damaged). A coverage gain is
  never called a win without its set size.
- **No example numbers unless measured.** Docs, plans and comments contain no illustrative numbers
  that could be mistaken for results.
- **Hypotheses are committed before measuring.** Every later implementation detail is added as a dated
  note, committed before it is run. Outcomes are appended; nothing above them is edited.
- **No setting is tuned on the test split.** Settings come from `tuning`, `conformal_calibration` or
  `int8_calibration`; fixes that learn from damage are judged leave-one-corruption-out.
- **Test-split reruns only for technical failure** (crash, corrupted/incomplete file, power loss),
  never because of a result; every rerun is logged with its reason.
- **Label what came later:** analyses done after the verdicts are marked "after the verdicts" or
  "exploratory"; likely explanations are marked "likely", never stated as proven.
- **Report absolute and compression-caused weakness separately** (see below).

### Absolute weakness vs compression-caused weakness
- **Absolute weakness:** FP32 itself fails, so every precision inherits it. Example (measured): at
  defocus blur severity 5 FP32 scores 22.4%; best INT8 is only 2.5 points lower. Quantization is not
  the problem there.
- **Compression-caused weakness:** INT8 minus FP32 on the same images. Example (measured): at darkness
  severity 5 FP32 holds 73.7% but best INT8 is 13.6 points lower (default INT8: 53 lower). Fog 5 has
  both: FP32 drops 22.8 points from clean, and best INT8 loses another 21.5.
- Reports and nutrition labels should show both, so a user can tell "this model is weak here" from
  "compressing it made it weak here".

### Proposed definition of "harm" (proposal only; not applied to Stage 3)
From the dated 3.9 note in `docs/hypotheses_stage3.md`: a condition is *harmful* for a model if,
measured on the tuning split before any alarm result is looked at, (a) its clean-tuned 90% conformal
coverage falls below 80%, or (b) its top-1 is more than 10 points below the same model's clean top-1.
Alarm firing on a harmful condition = catch; on a condition that is neither harmful nor clean = "early
warning" (reported separately, not a false alarm); on clean images = false alarm.

### "Reliability envelope": direction agreed, exact wording not yet fixed
Agreed in the Stage 4 plan (26 September 2026): a model's reliability envelope is the set of *tested*
conditions (damage type x severity) in which it is not harmful by the proposed harm definition above,
measured on a named split and machine; untested conditions are shown as "not tested". The exact
wording, and how intervals are handled, are fixed in `docs/hypotheses_stage4.md` before task 4.3 runs.

### Stage 4 progress (27 September 2026)
Plan: `ROADMAP.md` (order 4.0 -> 4.1 -> 4.3 -> 4.5, 4.2 alongside; 4.4 when the Pi 5 arrives).
Predictions (M1, H18–H22, M2, 4.2 rules) and outcomes: `docs/hypotheses_stage4.md`, committed before
measuring.

**Task 4.0 (done).** Result format schema 2 (`brokkr/schema.py`) and a checker for results and model
build records (`scripts/22_check_results.py`). Mechanism test M1 (`scripts/27_mechanism_levels.py`, 500
tuning images): **INCONCLUSIVE** for darkness and fog by the pre-set rule; accepted as recorded, never
re-run with another summary (outcome and exploratory notes in the hypotheses file).

**Task 4.1 models.** 10 torchvision models; weights hash-checked; licences in `brokkr/export.py`.
Percentile 99.99 INT8 usable for 9; **MobileNetV3-Small INT8 failed** (2.7% agreement with FP32 on 256
tuning images; FP32 only). ConvNeXt-Tiny's INT8 was built with `skip_symbolic_shape` (recorded; checked
to leave MobileNetV3-Large's model unchanged). Default MinMax INT8 was built for M2 for 8 models;
**EfficientNet-B0's MinMax failed** (17.6%).

**Task 4.1 sweep (finished).** `scripts/28_breadth_sweep.py`: 13 test conditions (Brokkr's own and
ImageNet-C, labelled separately) plus clean `conformal_calibration`; 8 threads; 6 passes, smallest
models first.
- Commits: `5be2cb5` (first 30 records, thread spinning on); stopped by hand after 15:03:24 on 26
  September because steps ran about 3x slower than estimated (idle sessions' threads spinning);
  spinning switched off after checks (`4e4c4e5`, `fc965cd`: 2.46x faster, every score bit-identical;
  a full tuning dry run gave 247 of 247 score files bit-identical); resumed at `fc965cd` for every
  later record. The stop, reason and both commits are in `results/breadth/run_log.txt`.
- Both pre-flight checks passed at both starts: cached pictures equal fresh ones; MobileNetV3-Large
  reproduces Stage 3's scores exactly on 64 test images.
- Finished 2026-09-27 04:22:41, exit code 0, no model excluded. **266 of 266 records** (247 test, 19
  conformal_calibration), no leftover `.tmp` files. 9 slow-step warnings (1.5–1.8x the estimate; none
  at 2x); no disk stop.
- FP32 sanity check (clean test top-1 within 1.0 point of torchvision's published top-1), all PASS:

```
sanity check PASS (mobilenet_v3_small): FP32 clean top-1 0.6761, torchvision 0.6767, tolerance 0.01
sanity check PASS (shufflenet_v2_x1_0): FP32 clean top-1 0.6984, torchvision 0.6936, tolerance 0.01
sanity check PASS (mnasnet1_0): FP32 clean top-1 0.7386, torchvision 0.7346, tolerance 0.01
sanity check PASS (mobilenet_v2): FP32 clean top-1 0.7269, torchvision 0.7215, tolerance 0.01
sanity check PASS (mobilenet_v3_large): FP32 clean top-1 0.7558, torchvision 0.7527, tolerance 0.01
sanity check PASS (regnet_y_400mf): FP32 clean top-1 0.7611, torchvision 0.7580, tolerance 0.01
sanity check PASS (efficientnet_b0): FP32 clean top-1 0.7798, torchvision 0.7769, tolerance 0.01
sanity check PASS (resnet18): FP32 clean top-1 0.6987, torchvision 0.6976, tolerance 0.01
sanity check PASS (resnet50): FP32 clean top-1 0.8118, torchvision 0.8086, tolerance 0.01
sanity check PASS (convnext_tiny): FP32 clean top-1 0.8279, torchvision 0.8252, tolerance 0.01
```

- Checker after the sweep: `PASS: 909 of 909 result records pass the schema check; 42 of 42 build
  records acceptable; 0 results use an unusable build`. Free disk after: 16.4 GB.
- No accuracy or reliability result beyond these sanity checks has been computed or looked at yet.

**Next, in order.** (1) Reliability from the saved logits (ECE, conformal coverage with set size,
E-AURC): script not yet written. (2) Judge H18–H22 by the committed rules: script not yet written.
(3) M2 (design committed at `54c476d`): first step reproduces `scripts/26`'s MobileNetV3-Small values
to 0.01 dB; default-INT8 verdicts count 7 models. (4) Diagnostic-only MobileNetV3-Small rebuild with
the first squeeze-and-excitation multiply unquantized (tuning only; official result stays "INT8
failed"). Handoff details: `results/handoff_4.1.md` (gitignored).

### Suggestions parked for later (not decided)
- ROADMAP "before going public": add a commercial-use check (data and model licences), and tag each
  dataset "research only" / "commercial use allowed" in `brokkr/datasets.py`.
- Robust conformal for more INT8 variants (needs damage sweeps on `conformal_calibration`).
- Alarm behaviour when conditions switch mid-stream (only single-condition windows were tested).
- Charts for the write-up, built into the local results page (`site/`, not committed; rule 4).
- ImageNetV2: evaluation only; never show or redistribute its images; cite Recht et al., ICML 2019.

This file is a snapshot. [ROADMAP.md](ROADMAP.md) is the live plan, [README.md](README.md) the public
summary, and [docs/hypotheses.md](docs/hypotheses.md) the Stage 2 predictions and outcomes.

---

## 1. What Stages 1–2 found

Model: MobileNetV3-Large (torchvision `IMAGENET1K_V2` weights), run with ONNX Runtime 1.23.2 on an
Intel Core i5-1235U laptop (Windows 11, CPU only). Accuracy and reliability are measured on a fixed
10,000-image test split of the ImageNet-1k validation set, with 95% bootstrap confidence intervals.

### Stage 1: size, accuracy, speed on clean images

| | FP32 | FP16 | Default INT8 |
|---|---|---|---|
| File size | 22.2 MB | 11.3 MB | 5.9 MB |
| Top-1 accuracy | 75.58% (74.70–76.48) | 75.63% (74.78–76.53) | 60.15% (59.14–61.11) |
| Change vs FP32 (same images) | — | +0.05 pts (−0.02 to +0.13) | −15.43 pts (−16.30 to −14.59) |

- **Correctness check:** on all 50,000 validation images our FP32 pipeline scored 75.26%;
  torchvision publishes 75.27%. This proves export, preprocessing, labels, and scoring are right.
- **FP16** halves the size with no measurable accuracy change.
- **Default INT8** (ONNX Runtime static quantization, MinMax calibration on 512 images) is 3.75x
  smaller but 15 points less accurate. Its outputs take only 237 distinct values, so on 262 test
  images two classes tie exactly for first place.
- **Speed (rough, laptop only):** the CPU has fast "performance" and slow "efficiency" cores; unpinned,
  Windows moved the benchmark between them. What held up across runs: one performance core is about
  2x as fast as one efficiency core (FP32 7.50 vs 13.74 ms/image); FP16 is never faster; default INT8
  is not reliably faster. Proper speed measurement waits for the Raspberry Pi 5 (task 4.4).

### Stage 2: damaged images, and whether the model knows when it's wrong

Five simulated corruptions (fog, defocus blur, motion blur, noise, darkness) at severities 1–5.
Nine predictions were committed before measuring: **6 confirmed, 3 rejected**.

- **FP16 = FP32 everywhere:** largest difference in 26 conditions was 0.25 points.
- **Default INT8 falls apart faster** under every corruption (keeps 38–65% of FP32's accuracy at
  severity 3, vs 80% clean).
- **Darkness hurts only INT8:** FP32 75.6% -> 73.7% at severity 5; INT8 60.2% -> 20.5%.
- **The model is under-confident on clean images:** average confidence 57.9% vs accuracy 75.6%.
- **Conformal prediction (90% promise):** holds on clean images (FP32 90.6%, FP16 90.5%, INT8 89.8%),
  but breaks under blur and noise (FP32 at severity 5: 20–37%), **while prediction sets barely grow**
  (2.3 classes clean, 2.1–2.8 at severity 5). The model gives no warning that it's failing.
- **ECE can mislead:** damage lowered accuracy toward the model's low confidence, so calibration error
  *fell* (0.177 -> 0.009–0.026 under blur/noise) while accuracy collapsed.
- **Default INT8 is worse at knowing when it's wrong:** sets of 8.3 classes vs 2.3 to keep the 90%
  promise; error when answering only its most confident half 16.2% vs 5.2%; E-AURC 0.085 vs 0.049.

---

## 2. Every design decision so far, and why

### Project rules (from `CLAUDE.md`)
| Decision | Why |
|---|---|
| No fake or placeholder numbers; every number from running code | Credibility; rule 1 |
| Report machine, settings, and uncertainty with every number | No overclaiming; rule 2 |
| Only permissive code and models; licences recorded for every model and dataset | Rules 3 and 8 |
| Models, data, and results never committed to git | Large and regenerable; rule 4 |
| Work stays inside the current stage | Keeps the project finishable; rule 5 |
| Results saved as JSON first; page, charts, README tables generated from them | One source of truth |

### Measurement
| Decision | Why |
|---|---|
| Speed: >= 20 warm-up runs, >= 100 timed runs, enforced in code | Measurement rules can't be skipped by accident |
| Speed: p50/p95/p99, median across 10 interleaved sessions | One bad session can't distort results; interleaving keeps comparisons fair if the laptop drifts |
| Speed: stability judged by the IQR of session medians (flag above 10%) | Fastest-vs-slowest spread is ruined by one hot session |
| Speed: pinned to one core type (`--cores`, default performance) | Hybrid CPU made timings jump between two speeds |
| Every result records CPU, OS, core types, power state, library versions, git commit + "dirty" flag | Any number can be traced to exact code and conditions |
| Accuracy: bootstrap 95% CIs; precision comparisons as *paired* differences on the same images | Paired intervals are several times tighter and fair |
| FP32 correctness check against torchvision's published accuracy | Proves the pipeline before trusting any comparison |
| Score ties broken toward the lower class number everywhere; tie count and range recorded | INT8 ties made accuracy depend on sorting luck |
| All 1,000 logits per image saved (float32 `.npz`, SHA-256 checksum in the JSON) | Every metric can be recomputed without rerunning models; float16 changed 1 in 1,000 answers |
| Predictions written and committed before measuring (`docs/hypotheses.md`) | Git history proves they came first; outcomes appended, never edited |

### Data
| Decision | Why |
|---|---|
| ImageNet-1k validation set (gated; you accepted the terms yourself) | Allows the correctness check against torchvision |
| Imagenette rejected | Most of its validation images come from ImageNet's *training* set, so accuracy would be inflated |
| All splits defined once in `brokkr.datasets.make_splits` | No script can pick overlapping images; tests enforce it |

### Quantization
| Decision | Why |
|---|---|
| FP16 via `onnxconverter-common` (MIT), inputs/outputs kept FP32 | Rest of the pipeline unchanged |
| INT8 kept at ONNX Runtime defaults in Stage 1–2 | Measure what a typical user gets; fixes belong to Stage 3 with a proper tuning split |
| INT8 sanity check warns below 90% agreement with FP32, fails below 20% | A big drop is a finding, not a broken conversion |

### Corruptions and reliability (`brokkr/shift`, shared with Argos)
| Decision | Why |
|---|---|
| Own short implementations in the style of ImageNet-C | Readable and explainable; not directly comparable to published ImageNet-C numbers |
| Applied to the 224x224 picture after resize/crop, before normalisation | Damages what the model actually sees |
| Random pattern depends on seed = image's dataset position, not on severity | Same image, same pattern at every severity and for every precision |
| Darkness strengthened to 1–5 stops after viewing the sample sheets | Was much milder than the other corruptions |
| `brokkr/shift` imports nothing from Brokkr, only NumPy and Pillow (test enforces) | Reuse in Argos |
| ECE: 15 equal-width bins; its interval is biased upward when ECE is tiny, documented and tested | Standard method; honest about its limitation |
| Conformal: LAC method, threshold = exact k-th smallest score, k = ceil((n+1) x 0.9) | Exact finite-sample guarantee; `np.quantile(method="higher")` was one rank off |
| Conformal thresholds always tuned on *clean* calibration images | Tests whether the promise survives shift it wasn't tuned for |
| Selective prediction: AURC and E-AURC; tied confidences averaged over all orders | INT8's ties would otherwise make results depend on sorting |

### Engineering and process
| Decision | Why |
|---|---|
| Sweep caches decoded test images once; each image damaged once and fed to all precisions | Faster and paired; resumable after interruptions |
| Sweep self-check: its clean condition must reproduce validated logits exactly (difference was 0.00) | Proves the fast path equals the checked path |
| Charts written as plain SVG (`brokkr/charts.py`), no plotting library | No dependency; light/dark colours; tooltips; table view for every chart |
| Colour palette validated for colour-blind safety; marker shapes as a second cue | Accessibility |
| ruff + pytest in GitHub Actions; `.gitattributes` for LF endings; `requirements-lock.txt` | Code quality and reproducibility |
| One branch per stage, merged to `main` when the stage is done | `main` always holds finished work |
| No AI co-author lines in commits; commits use your GitHub noreply email | Your preference; privacy before going public |
| Repository stays private for now; results page built locally | Your decision; publishing is on the "before going public" checklist |

---

## 3. Where everything lives

### Code (`brokkr/`)
| File | What it does |
|---|---|
| `fingerprint.py` | Records machine, OS, core types, power, versions, git commit |
| `export.py` | Model list with licences; PyTorch -> ONNX export and check |
| `quantize.py` | FP16 conversion and INT8 static quantization |
| `benchmark.py` | Speed measurement, sessions, CPU pinning |
| `datasets.py` | Dataset list with licence, Parquet reader, **the data splits** |
| `accuracy.py` | Preprocessing, running a model, top-1/top-5, bootstrap intervals, paired differences |
| `results.py` | Standard JSON record format; `.npz` arrays with checksums |
| `report.py` | Builds the results page from the JSON files |
| `judge.py` | The Stage 3 judging rules (thresholds from the hypotheses file only) |
| `charts.py` | SVG line charts for the page |
| `shift/corruptions.py` | The five corruptions (shared with Argos) |
| `shift/reliability.py` | Softmax, ECE, reliability-diagram data |
| `shift/conformal.py` | Conformal thresholds, sets, coverage |
| `shift/selective.py` | Risk-coverage curves, AURC, E-AURC |
| `shift/robust_conformal.py` | The one-third clean, two-thirds damaged calibration mix (Stage 3) |
| `shift/alarm.py` | Confidence alarm: window averages, threshold, fires (Stage 3) |

### Scripts (`scripts/`, run from the project folder)
| Script | Produces |
|---|---|
| `00_print_fingerprint.py` | Prints this machine's fingerprint |
| `01_export_model.py` | `models/*_fp32.onnx` + record |
| `02_benchmark_speed.py` | `results/speed/*.json` |
| `03_evaluate_accuracy.py --split test\|conformal_calibration\|tuning\|all` | `results/accuracy/*_<split>.json` + `.npz` logits |
| `04_quantize.py` | `models/*_fp16.onnx`, `*_int8.onnx` + records |
| `05_build_site.py` | `site/index.html` |
| `06_corruption_samples.py` | `results/samples/*.png` picture sheets |
| `07_reliability.py` | `results/reliability/*_{calibration,conformal,selective}.json` |
| `08_corruption_sweep.py --split test\|tuning\|conformal_calibration` | `results/sweep/*_<split>_<corruption>_s<severity>.json` + `.npz` |
| `09_compare_splits.py` | Prints FP32 clean accuracy per split, differences vs test, images per class |
| `10_int8_methods.py` | `models/*_int8_<method>.onnx` + records (Percentile 99.99/99.999, Entropy) |
| `11_choose_int8.py` | `results/choices/*_int8_method.json` (the pre-registered choice rule) |
| `12_int8_grouping_check.py` | `results/checks/*_grouping.json`: repeat build + group-64 comparison |
| `13_int8_damaged_calibration.py` | `models/*_mixed_without_<corruption>.onnx` + records (task 3.3) |
| `14_int8_calibration_luck.py` | `models/*_calibseed<seed>.onnx`, `results/checks/*_calibration_luck.json` |
| `15_int8_unrounded_output.py` | `models/*_unrounded.onnx` + record (task 3.4) |
| `16_temperature.py` | `results/choices/*_temperatures.json` (task 3.5) |
| `17_shift_aware.py` | `results/choices/*_shift_aware.json`: robust conformal and alarm thresholds (3.6) |
| `run_stage3_final.py` | The final run on the test split (3.7): resumable, keeps Windows awake, checks files |
| `18_judge_stage3.py` | `results/final/*_stage3_verdicts.json`: PASS / FAIL / NOT RUN per prediction |
| `19_alarm_all_conditions.py` | Exploratory, after the verdicts: alarm firing rate for every test condition |
| `20_robust_conformal_clean.py` | After the verdicts: robust thresholds' coverage and set size on clean test images |
| `21_imagenetv2_summary.py` | ImageNetV2 (3.9): top-1 and coverage with set size, FP32 and best INT8 |
| `run_stage1.py` | Reruns all of Stage 1 in one command |

Tests: `tests/` (137 tests, run with `pytest`; style check `ruff check .`). One test uses the real
ImageNet data (skipped where it isn't downloaded): all split images are validation images, by their
original ImageNet file names, and no two splits share an image.
Docs: `docs/hypotheses.md` (Stage 2 predictions and outcomes), `docs/hypotheses_stage3.md` (Stage 3
design rules and predictions, committed before measuring).

### Data and results on disk (all gitignored)
| Location | Size | Contents | Made by commit |
|---|---|---|---|
| `data/imagenet-1k/` | 6.5 GB | ImageNet validation set, 14 Parquet files, 50,000 images | downloaded |
| `data/imagenetv2/` | 2.4 GB | ImageNetV2 matched-frequency, 10,000 images (archive + extracted) | downloaded (3.9) |
| `data/cache/` | 2.9 GB | Test, tuning and conformal-calibration images, resized and cropped, plus labels | sweep script |
| `data/old_results_stage1/` | 98 MB | Superseded results, kept for the record | various |
| `data/old_results_3.1_dirty/` | 3.8 GB | First 3.1 run (records say `dirty`); identical to the clean rerun, safe to delete | `237effa` + uncommitted |
| `models/` | 112 MB | FP32, FP16, default INT8 `.onnx` + records | `run_stage1` |
| | | INT8 candidates: Percentile 99.99, Percentile 99.999, Entropy | `9c3ae84` (task 3.2) |
| | | Group-64 Percentile 99.99 (sensitivity check only); five 3.3 models | `6844c3a` |
| | | Three calibration-luck models (seeds 6–8); the 3.4 unrounded-output model | `3771a0b` |
| `results/choices/` | 4 KB | Which INT8 method was chosen, and by how much | `9c3ae84` (task 3.2) |
| | | One temperature per final-run model | `bfc623b` (task 3.5) |
| | | Robust conformal and alarm thresholds | `88703f7` (task 3.6) |
| `results/checks/` | 23 MB | Grouping check (with the group-64 tuning scores) | `6844c3a` |
| | | Calibration-luck check (with the three new models' tuning scores) | `3771a0b` |
| `results/accuracy/` | 352 MB | Test and conformal-calibration results, 3 precisions, with logits | `2a77a66` |
| | | Tuning results (3 precisions) and the FP32 50,000-image check | task 3.1, rerun at `096461e` (clean) |
| `results/sweep/` | 4.4 GB | Test split: 78 results (3 precisions x 26 conditions), with logits | `404a68c` |
| | | Tuning and conformal-calibration splits: 78 results each, same layout | task 3.1, rerun at `096461e` (clean) |
| `results/reliability/` | 1.8 MB | 243 calibration/conformal/selective results | `62c47f4` |
| `results/speed/` | 0.5 MB | 21 pinned speed results (performance and efficiency cores) | `b42db22` |
| `results/samples/` | 7 MB | Corruption picture sheets (for viewing only, not measurements) | task 2.2, before its commit |
| `site/index.html` | ~0.1 MB | Results page (local only) | built from the above |

### The data splits (ImageNet-1k validation, 50,000 images)
Defined in `brokkr/datasets.py` (`make_splits`). No image is in two splits (tested).

| Split | Images | How chosen | Used for | Used so far? |
|---|---|---|---|---|
| `test` | 10,000 | random, seed 0 | Every reported accuracy/reliability number | Yes: all Stage 1–2 results |
| `int8_calibration` | 512 | random from the rest, seed 1 | Setting INT8's value ranges | Yes: default INT8 |
| `conformal_calibration` | 5,000 | shuffled remainder, seed 2 | Tuning conformal thresholds | Yes: Stage 2 thresholds (clean only); damaged outputs saved in 3.1 |
| `tuning` | 5,000 | shuffled remainder, seed 2 | Choosing Stage 3 settings | Outputs saved in 3.1 (clean and damaged); no setting chosen from them yet |
| (unassigned) | 29,488 | — | Nothing yet | — |

The `all` option in `03_evaluate_accuracy.py` uses all 50,000 images; it overlaps the INT8 calibration
images, so it is only used for the FP32 correctness check.

---

## 4. Known gaps and loose ends

1. ~~The 50,000-image correctness result isn't in `results/`.~~ Fixed 25 September 2026: rerun gave
   75.26% (95% CI 74.88–75.61%), torchvision publishes 75.27%, PASS; back on the results page.
2. **Speed numbers are laptop-only and rough.** Real speed study is task 4.4 (Raspberry Pi 5).
3. **The results page isn't published** (repository is private). See ROADMAP's "before going public".
4. **Raspberry Pi readiness:** the model list lives in `export.py`, which imports PyTorch, so the
   accuracy script needs PyTorch installed. Move the model list to its own file (planned in task 4.3).
5. ~~ImageNetV2 not downloaded yet.~~ Done in task 3.9 (licence recorded as the sources state it).
6. **Stage 3 scripts 12, 14 and 15 need `scripts/08_corruption_sweep.py --split tuning` run first.**
   The old unkeyed caches in `data/cache/` were deleted on 26 September 2026 to free disk space (with
   `data/old_results_3.1_dirty/`); those scripts read the old tuning cache and stop with "not found"
   until it is rebuilt.

---

## 5. Stage 3 record ("fix what broke"), complete

Detailed results per task, in the order they were done. Section 0 has the summary.

### Task 3.1 results (inputs for choosing settings, not findings)

Machine: the same i5-1235U laptop. All three runs printed PASS.
- Clean tuning split (5,000 images): FP32 73.90% (95% CI 72.76–75.16%), FP16 73.96%, default INT8
  58.46%.
- **The tuning split is measurably harder than the test split** (`scripts/09_compare_splits.py`):
  FP32 tuning minus test = −1.68 points, 95% CI −3.20 to −0.14 (unpaired bootstrap: different
  images, each side resampled on its own). Conformal_calibration minus test = −0.30 points
  (−1.68 to +1.15), no measurable difference. Splits are drawn at random with fixed seeds, not
  stratified by class (tuning: 0–12 images per class, 4 classes absent), so this is a chance draw,
  but a real one. Comparing options on the same tuning images (3.2) is unaffected; settings whose
  *level* comes from clean tuning images (temperature, alarm threshold) may be shifted. No design
  rule has been changed because of this; a dated note in `docs/hypotheses_stage3.md` records it.
- **Clean rerun:** all 3.1 results were rerun from commit `096461e` with no uncommitted changes. All
  160 result files (logits, labels, images, metrics) and the rebuilt image caches are identical to
  the first run.
- Damage sweeps on `tuning` and `conformal_calibration` (26 conditions x 3 precisions each): the
  clean condition reproduced the validated logits exactly (largest difference 0.00) for all
  precisions. Runtimes 90 and 69 minutes (the first minutes were on battery, which is slower).
- The results page shows only test-split sweeps (`brokkr/report.py`, tested), so tuning outputs can
  never appear there as results.

### Task 3.2 results (a choice made on the tuning split, not a finding)

All candidates use the same 512 calibration images (in groups of 128, see the dated note in
`docs/hypotheses_stage3.md`), per-channel int8 weights and per-tensor uint8 activations; all are
5.9 MB. Clean tuning split, 5,000 images; paired differences on the same images:

| Candidate | Tuning top-1 | vs MinMax (paired 95% CI) |
|---|---|---|
| MinMax (default INT8) | 58.46% | — |
| **Percentile 99.99** | **72.10%** | +13.64 points (+12.60 to +14.76) |
| Percentile 99.999 | 71.06% | +12.60 points (+11.58 to +13.70) |
| Entropy | 56.12% | −2.34 points (−3.42 to −1.24) |

- **Chosen: Percentile 99.99** (the rule: highest clean tuning top-1). Margin over Percentile
  99.999: +1.04 points (paired 95% CI +0.30 to +1.76), so the choice is not a coin flip.
- On tuning images it is 1.80 points below FP32 (73.90%), where MinMax was 15.44 below.
- **H10 is not judged yet.** It is about the test split and is measured once, in the final run (3.7).
- Entropy was worse than MinMax. Only onnxruntime's default Entropy settings (128 bins) were tried;
  per the design rules nothing was tuned after seeing this.
- Checks: MinMax built in groups gave exactly the default INT8's outputs; every record is from the
  clean commit `9c3ae84`.

### Task 3.3 results (models built; judged on the test split in 3.7)

- **Grouping check first** (`scripts/12`): rebuilding Percentile 99.99 with groups of 128 gave the
  same model (every stored number, and scores on all 5,000 tuning images). Groups of 64: tuning top-1
  72.00% vs 72.10%, paired difference −0.10 points (95% CI −0.58 to +0.34), but the same top answer
  on only 95.2% of images, so group size changes individual answers. It stays fixed at 128.
- **Five models built** (`scripts/13`), one per held-out corruption, each from Percentile 99.99 with
  256 of 512 calibration images damaged (seed 5; the same images and severities for all five). Every
  model passed its checks before being kept: loads, finite scores, per-channel weights (64 of 64),
  top-1 agreement with FP32 on 256 clean tuning images 84.8–86.7% (clean-calibrated: 86.3%).
- H12 (darkness) and H13 (clean cost) are about the test split and are judged in the final run (3.7).
- All records are from the clean commit `6844c3a`. The weights were confirmed per-channel from the
  model files, and recorded in the hypotheses file's dated note.

### Calibration luck: the noise floor for INT8 comparisons (extra analysis, before 3.4)

Percentile 99.99 rebuilt from three more random sets of 512 calibration images (seeds 6, 7, 8, drawn
from the 29,488 images in no split). Clean tuning split, 5,000 images:

| Calibration images | Top-1 | E-AURC |
|---|---|---|
| int8_calibration split (the original) | 72.10% | 0.0539 |
| unassigned, seed 6 | 71.90% | 0.0514 |
| unassigned, seed 7 | 71.82% | 0.0535 |
| unassigned, seed 8 | 72.10% | 0.0528 |
| **Range (largest − smallest)** | **0.28 points** | **0.0026** |
| Standard deviation | 0.14 points | 0.0011 |

In 3.7, INT8 differences smaller than the range are also labelled "within noise" (dated note in the
hypotheses file). The E-AURC range is about 5% of its value; H11 asks for at least 10%.

### Task 3.4 results (model built; H11 judged on the test split in 3.7)

- Percentile 99.99 with the final layer's output left in float (`OpTypesToExcludeOutputQuantization
  = ["Gemm"]`); weights still int8 per-channel (64 of 64); same calibration images and groups.
- Checks passed, including that the final output is really not rounded. Tool check on clean tuning
  images: tied top scores 58 with rounding, 0 without. File size 5.925 MB in both (−0.01%: removing
  the output's rounding step saves a few bytes).
- Every result record now also lists CPU instruction-set features. This laptop (Windows): AVX yes,
  AVX2 yes, AVX-512F no; VNNI recorded as unknown (Windows has no standard-library way to read it).

### Task 3.5 results (temperatures fitted on the tuning split; H14 judged on test in 3.7)

Rules fixed in a dated note before fitting: NLL on clean tuning images, golden-section search over
T = 0.1 to 10, stop if T is within 1% of an end; 3.6 uses raw scores for every model. All ten T are
far from the ends, NLL fell for every model, and no top answer changed. Clean tuning, 5,000 images
(ECE: Stage 2's 15 bins; a tool check, not a result):

| Model | T | ECE before | ECE after |
|---|---|---|---|
| FP32 | 0.743 | 0.169 | 0.025 |
| FP16 | 0.743 | 0.169 | 0.025 |
| Default INT8 | 0.726 | 0.181 | 0.034 |
| Best INT8 (Percentile 99.99) | 0.729 | 0.188 | 0.023 |
| Best INT8, unrounded output | 0.730 | 0.186 | 0.024 |
| Leave-one-out INT8 (five models) | 0.729–0.734 | 0.179–0.190 | 0.022–0.028 |

- Every T is below 1: all models are made more confident, as H14 expects for FP32.
- Clean tuning top-1 of the new models (scores from `scripts/03`, for the fits): unrounded 72.12%;
  leave-one-out 71.80–72.34% (best INT8 72.10%). H11 and H13 are judged on the test split in 3.7.
- All records are from the clean commit `bfc623b`.

### Task 3.6 results (thresholds set on calibration/tuning images; H15, H16 judged on test in 3.7)

Details fixed in a dated note before computing: raw scores; H16 test windows (seed-3 order, 100
non-overlapping single-condition windows per condition, fires if strictly below); the exact
robust mix (seed 9: 1,667 clean, 3,333 damaged over 20 balanced corruption/severity pairs);
robust conformal for FP32, FP16 and default INT8 (the models with damaged calibration outputs).

- **Check passed:** the clean-only conformal thresholds recomputed here equal Stage 2's exactly
  (FP32 0.963033, FP16 0.962827, INT8 0.991209).
- **Robust thresholds** (one per held-out corruption): FP32 0.9932–0.9960, FP16 0.9931–0.9960,
  default INT8 0.9988–0.9991. On their own calibration mix they give 90.0% coverage, as built.
  Set sizes on that mix (mostly damaged images): FP32 8.7–15.3 classes, default INT8 91–127.
  These are calibration images; clean and held-out-corruption test numbers come in 3.7.
- **Alarm thresholds** (raw confidence, clean tuning): FP32 0.512 (mean confidence 0.570), FP16
  0.512, default INT8 0.345, best INT8 0.477, unrounded 0.479, leave-one-out 0.475–0.482. Exactly
  1.00% of the clean tuning windows fire for every model, as built.
- Final-report wording fixed in advance: "improved coverage in our tests", never "guaranteed".
- The record is from the clean commit `88703f7`.
- **Extra analysis (not a prediction), added before the final run:** best INT8 on
  conformal_calibration, clean and damaged (sweep PASS, largest logit difference 0.00). Its
  clean-only threshold is 0.9684; its robust thresholds 0.9948–0.9966, set sizes on their own
  calibration mix 12.6–20.2 classes. Rerunning the script left every earlier threshold identical.

### Task 3.7 results: the final run on the test split (10,000 images)

Run once from the tag `stage3-final-run` (`54026c9`); no reruns (`docs/stage3_final_run_log.md`).
Judged by `scripts/18_judge_stage3.py` with the rules fixed before the run. Paired 95% intervals.

| | Prediction | Verdict | What was measured |
|---|---|---|---|
| H10 | Best INT8 top-1 >= 67.9% | **PASS** | 73.60% (FP32 75.58%, default INT8 60.15%) |
| H11 | Unrounded output: 0 ties, E-AURC >= 10% lower | **FAIL** (within noise) | 0 ties, same size; but E-AURC 0.0510 -> 0.0524, 2.7% *worse* (interval −0.0030 to −0.0001); smaller than build-to-build luck (0.0026) |
| H12 | Calibrated without darkness: +10 points at darkness s5 | **FAIL** | 60.15% -> 59.48%, −0.67 points (−1.41 to +0.06) |
| H13 | Leave-one-out models within 1 point on clean | **PASS** (within noise) | +0.25 points (−0.12 to +0.59) |
| H14 | Temperature fixes clean ECE, backfires at s5 for >= 3 of 5 | **PASS** | T 0.743; clean ECE 0.018; worse for defocus/motion blur/noise (+0.12 to +0.15), *better* for fog (−0.09) and darkness (−0.16) |
| H15 | Robust conformal: +10 points coverage at s3, clean sets >= 2x | **PASS** | +12.8 points (12.4 to 13.2); clean sets 2.28 -> 7.72 classes (3.4x) |
| H16 | Alarm: >= 90% of harmful windows, <= 2 of 100 clean | **PASS** | 1,200 of 1,200 harmful windows; 0 of 100 clean |
| H17 | ImageNetV2 | NOT RUN | Postponed to 3.9 (download and licence first) |

Coverage with set size, FP32 at severity 3 (clean-tuned -> robust threshold): fog 88.4% (2.4) -> 96.4%
(10.3); defocus blur 72.6% (2.8) -> 88.0% (12.6); motion blur 66.6% (2.9) -> 84.6% (13.6); noise
72.3% (3.0) -> 87.6% (12.0); darkness 89.8% (2.3) -> 97.2% (9.8). Improved coverage in our tests, not
guaranteed: under three of the five held-out corruptions it is still below 90%.

Extra analysis (no prediction), best INT8 robust conformal: coverage at s3 +14.9 points; clean sets
2.63 -> 10.32 classes.

Not a prediction, but visible in the run: best INT8 (Percentile 99.99, clean calibration) scores
60.15% at darkness s5, where default INT8 scored 20.48% (FP32 73.73%).

### Task 3.9 results: ImageNetV2 (real new photos)

Run at `e7f5b9b`, no reruns. **H17 PASS:** FP32 top-1 62.10% (61.16 to 62.99); clean-tuned coverage
80.72% (79.93 to 81.44) with average set size 2.64 (ImageNet test: 75.58%; 90.58% with 2.28). Extra
analysis, best INT8: 59.83%; coverage 80.64% with set size 3.03. A proposed definition of "harm" for
future alarm checks is in the dated 3.9 note (not applied to Stage 3).

### Stage 3 in plain words: each step, how, and why

**The two rules behind everything**
- *No Stage 3 setting is tuned on the test split; the test split was used for Stage 2 baselines.*
  Every Stage 3 fix has knobs to set (which INT8 method, what temperature, what alarm level). If we set
  knobs by looking at test results, we'd be marking our own exam, and the results would look better
  than reality. So knobs are set on other splits (mainly `tuning`), and the fixes are measured on the
  test split in the final run (3.7).
- *Leave one damage type out.* A fix that learns from damaged images might only work on the damage it
  practised on. So each such fix is built five times, each time hiding one damage type, and tested
  only on the hidden type. That tells us whether it helps with *new* kinds of trouble, which is what a
  real camera will meet.

**3.1 Tuning-split tooling.** *What:* run the models on the untouched `tuning` split (clean and
damaged), and the `conformal_calibration` split damaged, saving logits as in Stage 2. *How:* extend
`03`/`08` to accept those splits (the sweep currently only does `test`). *Why:* every later step needs
somewhere other than the test set to choose its settings.

**3.2 Better INT8 settings.** *What:* build INT8 three ways (MinMax = today's default; Percentile =
ignores the most extreme 0.01% or 0.001% of values; Entropy = keeps the ranges that lose least
information) and keep the one with the best clean accuracy on `tuning`. *Why:* default INT8 loses 15
points, and a Stage 1 diagnostic hinted that rare extreme values are the cause. *Prediction H10:*
recovers at least half the loss (>= 67.9%).

**3.3 INT8 calibrated on damaged images.** *What:* build INT8 with the best method, but half of its 512
calibration images damaged (four damage types, one left out). *Why:* INT8 collapses in the dark while
FP32 doesn't; the suspected cause is that INT8's value ranges were only fitted to well-lit images.
*Predictions H12–H13:* helps darkness by >= 10 points even when darkness is the hidden type (a real
guess), and costs < 1 point on clean images.

**3.4 Unrounded final output ("mixed precision").** *What:* best INT8, but its last layer's *output*
stays in normal decimals (weights stay 8-bit). *Why:* INT8's rounded output causes exact ties (262 test
images), which blunt its ability to rank how sure it is. *Prediction H11:* no ties, better confidence
ranking (E-AURC), file under 5% bigger.

**3.5 Temperature scaling.** *What:* divide all of a model's scores by one number T before turning them
into probabilities, with T fitted on clean `tuning` images. It never changes the answer, only how sure
the model sounds. *Why:* the model says 58% when it's right 76% of the time. *Prediction H14:* fixes clean
calibration but makes the model over-confident under heavy damage, a warning about "fixing"
calibration on clean data only.

**3.6 Shift-aware "I'm not sure".** Two tools.
- *Robust conformal:* tune the 90% threshold on a mix of clean and damaged `conformal_calibration`
  images (hidden damage type left out). *Why:* Stage 2's promise broke under damage it wasn't tuned
  for. *Prediction H15:* coverage on the hidden damage rises by >= 10 points, but clean sets at least
  double; we measure that trade-off.
- *Alarm:* watch the average confidence of the last 100 images and raise a warning when it drops below
  what 99% of clean windows look like. *Why:* sets don't grow under damage, but confidence does fall,
  so falling confidence might be a usable "conditions changed" signal. *Prediction H16:* catches >= 90%
  of harmful conditions with <= 2% false alarms on clean images.

**3.7 The final run.** All chosen fixes measured once on the 10,000 test images under all 26 conditions,
using the Stage 2 sweep (one overnight run).

**3.8 Write-up.** `docs/writeup.md`: question, method, findings, what worked and what didn't, and
limitations (simulated damage isn't real weather; one model; one laptop).

**3.9 Real-world check (ImageNetV2).** 10,000 new photos collected years after ImageNet: naturally
"shifted" data rather than simulated damage. *Why:* the obvious weakness of Stage 2 is that fog and
blur are simulated. *Prediction H17:* FP32 drops to 60–68%, and the upper end of the 95% interval of the
clean-tuned 90% promise's coverage is below 88%; we then check whether the Stage 3 fixes help on real shift too.
