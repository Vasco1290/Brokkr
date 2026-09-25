# Brokkr status

Snapshot as of **25 September 2026**. Branch `stage-3` (from `main` at `5da3bde`, "Merge Stage 2").
Stages 1 and 2 are complete and merged. Stage 3 has started: its task list and its pre-registered
predictions (`docs/hypotheses_stage3.md`, task 3.0) are committed. Task 3.1 is done: model outputs
on the clean and damaged `tuning` and `conformal_calibration` images are saved. No Stage 3 setting
was chosen from them in 3.1. Task 3.2 is done: Percentile 99.99 was chosen as "best INT8" on the
tuning split. No Stage 3 fix has been measured on the test split yet.

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
  is not reliably faster. Proper speed measurement waits for the Raspberry Pi 5 (Stage 5).

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
| `charts.py` | SVG line charts for the page |
| `shift/corruptions.py` | The five corruptions (shared with Argos) |
| `shift/reliability.py` | Softmax, ECE, reliability-diagram data |
| `shift/conformal.py` | Conformal thresholds, sets, coverage |
| `shift/selective.py` | Risk-coverage curves, AURC, E-AURC |

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
| `run_stage1.py` | Reruns all of Stage 1 in one command |

Tests: `tests/` (106 tests, run with `pytest`; style check `ruff check .`). One test uses the real
ImageNet data (skipped where it isn't downloaded): all split images are validation images, by their
original ImageNet file names, and no two splits share an image.
Docs: `docs/hypotheses.md` (Stage 2 predictions and outcomes), `docs/hypotheses_stage3.md` (Stage 3
design rules and predictions, committed before measuring).

### Data and results on disk (all gitignored)
| Location | Size | Contents | Made by commit |
|---|---|---|---|
| `data/imagenet-1k/` | 6.5 GB | ImageNet validation set, 14 Parquet files, 50,000 images | downloaded |
| `data/cache/` | 2.9 GB | Test, tuning and conformal-calibration images, resized and cropped, plus labels | sweep script |
| `data/old_results_stage1/` | 98 MB | Superseded results, kept for the record | various |
| `data/old_results_3.1_dirty/` | 3.8 GB | First 3.1 run (records say `dirty`); identical to the clean rerun, safe to delete | `237effa` + uncommitted |
| `models/` | 55 MB | FP32, FP16, default INT8 `.onnx` + records | `run_stage1` |
| | | INT8 candidates: Percentile 99.99, Percentile 99.999, Entropy | `9c3ae84` (task 3.2) |
| `results/choices/` | 4 KB | Which INT8 method was chosen, and by how much | `9c3ae84` (task 3.2) |
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
2. **Speed numbers are laptop-only and rough.** Real speed study is Stage 5 (Raspberry Pi 5).
3. **The results page isn't published** (repository is private). See ROADMAP's "before going public".
4. **Raspberry Pi readiness:** the model list lives in `export.py`, which imports PyTorch, so the
   accuracy script needs PyTorch installed. Move the model list to its own file before Stage 5.
5. **ImageNetV2 not downloaded yet** (planned for task 3.9, 1.26 GB; to be confirmed before download).

---

## 5. What happens next: Stage 3 ("fix what broke")

### The very next steps

1. **Task 3.3, INT8 calibrated on damaged images** (explained below), built with Percentile 99.99.
   Tasks 3.0–3.2 are done (see "Task 3.1 results" and "Task 3.2 results" below), and
   `docs/hypotheses_stage3.md` is committed. Its **design rules** fix, in advance, how every Stage 3 setting will be chosen
   (including per-channel INT8 weights as a fixed setting, the exact alarm windows, and the 12
   "harmful" conditions), and **predictions H10–H17** state what we expect, with numeric thresholds and
   paired 95% intervals for every comparison.

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
