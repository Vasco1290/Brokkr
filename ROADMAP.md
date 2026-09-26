# Brokkr roadmap

Brokkr is built from the inside out. Each stage must work on its own before the next one starts,
so stopping at any stage still leaves something finished.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done.

The website grows with every stage as a **walking skeleton**: the thinnest end-to-end version
(results JSON -> HTML page) exists from Stage 1, and each stage adds only what it actually measured.

Languages: Python for the factory, test lab, and site builder; HTML/CSS (+ a little JavaScript for
search) for the website; a shell script for the installer. The device runner starts in Python and moves
to C++ only if measurements show Python overhead matters.

## Stage 1 — Core measurement `[x]`

One model, three precisions, measured honestly on the laptop.

- [x] 1.1 Project skeleton: package, `pyproject.toml`, first test, CI on every push
- [x] 1.2 Machine fingerprint: CPU, OS, package versions, git commit attached to every result
  (the thread count is a benchmark setting, recorded by the benchmark in 1.4)
- [x] 1.3 Export MobileNetV3 (torchvision) to ONNX FP32
- [x] 1.4 Speed benchmark: fixed threads, >=20 warm-up, >=100 timed runs, p50/p95/p99 -> JSON;
  5 interleaved sessions (median + spread); power state recorded
  - [x] Run plugged in, "Best performance" mode. Remaining instability traced to hybrid cores
    (fixed by pinning, `--cores`) and run-to-run drift of 10-20% on this laptop; see README
- [x] 1.5 Accuracy on a fixed, seeded image set (split + image count recorded) -> JSON
  (ImageNet-1k validation; correctness check against torchvision's published top-1 passes)
- [x] 1.6 FP16 and INT8 versions, same measurements (INT8 with ONNX Runtime default settings)
- [x] 1.7 Walking-skeleton website: a Python script turns results JSON into one plain HTML page
  (`scripts/05_build_site.py` -> `site/index.html`). Real numbers only, no design yet.
  - [ ] Publish on GitHub Pages (gh-pages branch) — waiting until the repository is made public

**Done when:** one command produces real speed/size/accuracy numbers for FP32/FP16/INT8, they appear on
the results page, and tests pass.

Hardening after Stage 1: one-command runner (`scripts/run_stage1.py`), all results regenerated from
committed code, speed benchmark pinned to one core type with IQR-based stability flags, ruff in CI,
`.gitattributes` (LF line endings), `requirements-lock.txt`, README with real results.

### Before making the repository public

- [x] Switch local git author email to the GitHub noreply address and rewrite unmerged commits that
  contain the personal email
- [ ] Publish the results page to GitHub Pages
- [ ] Re-read README and ROADMAP for anything that overclaims

## Stage 2 — Stress test `[x]`

How much worse does each precision get on damaged photos, and does it still know when it's wrong?

- [x] 2.0 Hypotheses written down *before* measuring (`docs/hypotheses.md`), committed first
- [x] 2.1 Fixed, non-overlapping image splits: test 10,000 / conformal calibration 5,000 /
  tuning 5,000 (kept for Stage 3) / INT8 calibration 512
- [x] 2.2 Corruptions: fog, defocus blur, motion blur, noise, darkness at 5 severities —
  self-contained module shared with Argos (no Brokkr imports), plus a sample image sheet
- [x] 2.3 Save every class score (logits, float32 .npz with checksum) so every reliability number
  can be recomputed
- [x] 2.4 Calibration: expected calibration error (ECE) and reliability diagram data
- [x] 2.5 Conformal prediction: 90% sets tuned on clean calibration images; coverage and set
  size on clean and corrupted test images
- [x] 2.6 Selective prediction: risk–coverage curves and AURC
- [x] 2.7 Full sweep: 3 precisions x (clean + 5 corruptions x 5 severities) on the 10,000 test
  images (overnight, plugged in)
- [x] 2.8 Headline chart and a robustness section on the results page, generated from JSON;
  outcomes recorded in `docs/hypotheses.md` (6 confirmed, 3 rejected)

Bootstrap confidence intervals on every accuracy/reliability number.

**Done when:** the headline chart (precision × corruption × reliability) is generated from JSON.

## Stage 3 — Fixes and the study `[x]`

Can each Stage 2 problem be fixed, and at what cost? No Stage 3 setting is tuned on the test split;
the test split was used for Stage 2 baselines. Settings are chosen on the tuning and calibration splits. Fixes that learn from damaged images are judged
leave-one-corruption-out: tuned on four corruption types, tested on the fifth.

- [x] 3.0 Hypotheses written down before measuring (`docs/hypotheses_stage3.md`), committed first
- [x] 3.1 Tuning-split tooling: model outputs on clean and damaged tuning images, for choosing settings
- [x] 3.2 INT8 calibration methods: MinMax vs Percentile vs Entropy, chosen on the tuning split.
  Motivation: in task 1.6 default (MinMax) INT8 disagreed with FP32 on about a third of images; a
  quick diagnostic (on test images, so not a result) suggested outlier-robust calibration helps.
  Chosen: Percentile 99.99 (tuning top-1 72.10% vs MinMax 58.46%); H10 is judged on test in 3.7
- [x] 3.3 INT8 calibrated on clean + damaged images. Motivation from Stage 2: darkness barely affects
  FP32 (75.6% -> 73.7% at severity 5) but drops default INT8 from 60.2% to 20.5%
- [x] 3.4 INT8 with the final layer's output kept unrounded (mixed precision), to remove exact score
  ties (262 of 10,000 test images) that hurt its ability to rank its own confidence
- [x] 3.5 Temperature scaling for the model's under-confidence (58% confidence vs 76% accuracy)
- [x] 3.6 Shift-aware "I'm not sure": conformal thresholds tuned on clean + damaged images, and an
  alarm that watches average confidence. Motivation from Stage 2: FP32 coverage fell to 20-37%
  under severe blur and noise while set sizes barely grew
- [x] 3.7 Final run: chosen fixes measured once on the 10,000 test images under all 26 conditions
- [x] 3.8 Write-up (`docs/writeup.md`) with a limitations section
- [x] 3.9 Real-world shift check on ImageNetV2 (10,000 new photos, natural rather than simulated shift)

**Done when:** one finding can be explained in two minutes, with the numbers behind it, and the
write-up exists.

## Stage 4 — Breadth, depth, labels and recommendations `[ ]`

Stage 3 studied one model on one laptop. Stage 4 asks whether its findings hold across models, tests
the likely explanation for INT8's darkness collapse, and turns results files into labels and
recommendations. Plan revised 26 September 2026; nothing built yet.

**Rules for the whole stage** (Stage 3's rules carry over unchanged):
- Differences are "new minus old" only, with paired bootstrap 95% intervals and the direction stated
  (accuracy/coverage: positive = better; E-AURC/ECE: positive = worse).
- The metric set is Stage 3's: top-1, ECE, conformal coverage with average set size, E-AURC, alarm
  firing, plus size and speed (p50/p95/p99). A new metric is added only with the question it answers
  written next to it. Planned additions: levels used per layer (4.0), rank correlation across models
  (4.1), laptop-vs-Pi agreement (4.4).
- No example numbers in docs unless measured. Compute times below are **estimates**, scaled from
  measured Stage 3 run times, and are replaced by measured times as each task runs.
- No setting is tuned on the test split. Hypotheses are committed before measuring; later details go
  in dated notes; outcomes are appended.
- Every model, dataset and code package has its licence recorded before use (rules 3 and 8).

**Compute reference** (measured in Stage 3, this laptop, MobileNetV3-Large, 10,000 test images): one
model on one condition took about 2–3 minutes (clean runs 2.3–3.3 min; two INT8 models x 26
conditions, 110 min). Each saved run (logits + record) is about 38 MB on disk.

**Order:** 4.0 -> 4.1 -> 4.3 -> 4.5, with 4.2 running alongside (its compute queued so it never
overlaps 4.1's); 4.4 when the Pi 5 arrives (expected in a few weeks).

**Naming:** Brokkr's own corruptions and ImageNet-C's are labelled separately everywhere (records,
tables, labels), e.g. "fog (Brokkr)" and "fog (ImageNet-C)".

### 4.0 Foundations: result schema, hypotheses, mechanism test `[ ]`

*Scope*
- **Unified result schema:** one versioned JSON format for every Stage 4 result, checked by a
  validator. Fields: schema version; git commit and "dirty" flag; model (name, weights, licence);
  precision and how it was made (e.g. INT8 method); runtime (ONNX Runtime version, execution
  provider, threads); device (the existing fingerprint); dataset, split, number of images; condition
  (corruption, its source — Brokkr's own or ImageNet-C — and severity); metrics, each with its 95%
  interval; saved-logits path and checksum; `source: brokkr` (so community results, Stage 7, can
  never be mixed in unlabelled).
- A converter reads the Stage 1–3 records into the schema (no reruns), so 4.3 and 4.5 can use them.
- **`docs/hypotheses_stage4.md`**, committed before any Stage 4 measurement: predictions and judging
  rules for the mechanism test and 4.1 (including the ImageNet-C contrast prediction); the final
  model and condition lists; the rule for trimming the model list if it is too slow (fixed before any
  accuracy is seen); the rule for "INT8 degrades" used in 4.2.
- **Mechanism test.** Question: does darkness (and fog) break default INT8 because the image's
  values squeeze into only a few of the 256 quantization levels? (Report 1, section 5, calls this
  the likely explanation, untested.) For default and Percentile INT8 MobileNetV3-Large, count how
  many levels each quantized layer's activations actually use on clean, dark (Brokkr darkness,
  severity 5) and foggy (Brokkr fog, severity 3) images. Tuning-split images, not test; image count
  fixed in the hypotheses file, which also states, before running, which outcome supports the
  "low-contrast images use fewer levels" explanation and which rejects it.

*Done when:* the validator passes on every new record and on the converted Stage 1–3 records; the
hypotheses file is committed; the mechanism test writes a results JSON and prints its verdict;
`pytest` passes.

*Depends on:* nothing.

*Compute (estimate):* schema and converter, negligible. Mechanism test: minutes (a small sample of
tuning images, two models, three conditions).

### 4.1 Breadth: does clean accuracy predict robustness after quantization? `[ ]`

*Scope*
- **Models:** 8–10 torchvision ImageNet classifiers, MobileNetV3-Large included. Candidates, before
  licence check and timing probe: MobileNetV3-Small, MobileNetV2, EfficientNet-B0, ShuffleNetV2,
  MNASNet, a RegNet, ResNet-18, ResNet-50, ConvNeXt-Tiny. Weights under non-commercial licences
  (e.g. torchvision's SWAG weights) are excluded. The final list goes in the hypotheses file.
- **Each model:** FP32 sanity check before any of its numbers are trusted: clean test-split top-1
  within ±1.0 point of torchvision's published top-1. The tolerance allows for sampling error, since
  we evaluate on 10,000 images and torchvision on 50,000; tolerance and reason are in the hypotheses
  file. Each model uses its own torchvision preprocessing; damage is applied after its own resize and
  crop (as in Stage 2).
- **Precisions:** FP32 and Percentile 99.99 INT8 only. The Stage 3 method is applied as it is, not
  re-tuned per model, with the same 512 calibration images. A model whose INT8 build fails its checks
  is reported as failed, not dropped silently.
- **Conditions** (test split, 10,000 images, 13 in all): clean; Brokkr's own fog 3, darkness 5,
  defocus blur 3, noise 3; ImageNet-C fog, contrast, defocus blur and Gaussian noise at severities 3
  and 5.
- **ImageNet-C option:** generated on our test split with the official corruption code (the
  `imagecorruptions` package). The 8 conditions are listed in the hypotheses file. **Before 4.1
  starts:** confirm the package's licence (and its dependencies'), that it installs cleanly with our
  numpy / scikit-image versions, and that its outputs look correct on 5 sample images. Any difference
  from the released ImageNet-C files (for example how they were saved) is recorded, and our numbers
  are not called directly comparable to published ImageNet-C results unless they are. Darkness stays
  Brokkr's own, since ImageNet-C has no darkening corruption.
- **Metrics:** the Stage 3 set; conformal thresholds tuned on clean `conformal_calibration` images
  for each model and precision.
- **Analysis:**
  - Absolute weakness (FP32 under the condition, and FP32 minus its own clean score), reported
    separately from compression-caused weakness (INT8 minus FP32 on the same images, paired).
  - Absolute gaps in points and relative gaps (INT8 as a fraction of FP32).
  - Floor effects noted: where FP32 is already near chance, the gap cannot be large, so a small gap
    there is not evidence of robustness.
  - The main question is answered by a rank correlation across models: clean FP32 top-1 against
    INT8 minus FP32 under each condition, with a bootstrap interval over models. With 8–10 models this
    is a small sample and is reported as such.
- **Hypotheses** (in the file, before measuring) include: INT8's extra gap (INT8 minus FP32) is large
  under ImageNet-C contrast, as the level-wasting mechanism predicts. "Large" is defined in the file.

*Done when:* every listed model has a schema-valid record for each (precision, condition); the
judging script prints PASS / FAIL per 4.1 hypothesis; a summary script prints the absolute and
compression-caused tables.

*Depends on:* 4.0 (schema; hypotheses file committed).

*Compute (estimate):* first a timing probe for each candidate on tuning images (speed only; no
accuracy is read). A MobileNetV3-sized model needs about 1–1.5 hours (26 runs at 2–3 min, plus
calibration images). Larger models take longer, in proportion to their probe speed; the total is
fixed after the probe. Disk: about 1 GB per model.

### 4.2 Depth: the full Stage 3 pipeline on two more models `[ ]`

*Scope*
- **Models:** EfficientNet-B0 and ResNet-18. Reason for ResNet-18, recorded in the hypotheses file
  before its runs: cheap, a different design family (plain convolutions, no depthwise layers), and
  the most-studied model in quantization papers, so results can be compared with published work.
- **Phase A (every model):** tuning and `conformal_calibration` outputs, clean and damaged (as 3.1);
  INT8 method choice on tuning (3.2); temperature (3.5); robust conformal and alarm (3.6); one final
  run on the test split from a tagged commit (3.7); ImageNetV2 (3.9).
- **Phase B (only where INT8 degrades):** damaged-image calibration, leave-one-out (3.3), and
  unrounded output (3.4). "INT8 degrades" is judged on the tuning split after Phase A's method choice,
  by a numeric threshold written in the hypotheses file before any 4.2 run.
- Same rules as Stage 3: settings from tuning/calibration splits only; leave one corruption out; one
  final test run; reruns only for technical failure, logged with the reason.

*Done when:* each model has a verdict JSON from the judging script and a final-run log; Phase B was
either run or skipped, with the rule's output recorded.

*Depends on:* 4.0; reuses 4.1's EfficientNet-B0 and ResNet-18 exports and sanity checks.

*Compute (estimate):* MobileNetV3's Stage 3 pipeline took about 6 hours of logged run time (tuning
sweeps 90 + 69 min, final run 3 h 17 min), plus builds and fits. Each 4.2 model: that, scaled by its
probe speed. Phase B adds the leave-one-out sweeps (for MobileNetV3, 10–20 min each, five of them).
Overnight runs.

### 4.3 `brokkr shrink` / `brokkr test` and the label, for any ONNX model `[ ]`

*Scope*
- **Input:** an FP32 ONNX ImageNet-1k classifier (1,000 outputs) and a small config: preprocessing
  (resize, crop, mean, std) and its licence. The commands refuse to run without a licence (rule 8).
- **`brokkr shrink`:** FP16 and Percentile 99.99 INT8 (Stage 3's method), standard 512 calibration
  images, the existing INT8 sanity checks.
- **`brokkr test`:** the 4.1 condition set on the test split, conformal thresholds from clean
  `conformal_calibration` images, schema records saved.
- **Label generator:** reads only results JSON. Shows size, accuracy with interval, coverage with set
  size, absolute vs compression-caused weakness, the machine, split and image count, and the
  reliability envelope. No hand-typed values: a test checks that every number on the label is in
  the JSON.
- **Reliability envelope:** the tested conditions in which the model is *not harmful* by the proposed
  harm definition (clean-tuned 90% coverage at least 80%, and top-1 no more than 10 points below its
  own clean top-1), applied to the label's own measurements. Untested conditions are shown as "not
  tested", never assumed. Exact wording, and how intervals are handled, are fixed in the hypotheses
  file before 4.3 runs.
- The model list moves out of `export.py`, so these commands run without PyTorch.

*Done when:* on one 4.1 model exported and treated as a user-supplied file, shrink -> test -> label
runs end to end, and the label's numbers match that model's 4.1 records (check script prints pass /
fail).

*Depends on:* 4.0; 4.1 (condition set and reference records).

*Compute (estimate):* one model, about as long as one 4.1 model (about 1.5 hours if MobileNetV3-sized).
Label generation: seconds.

### 4.4 Raspberry Pi 5 target over SSH `[ ]`

*Scope*
- **Target interface:** "where a model runs" is pluggable. The laptop (local) and a Pi 5 (over SSH)
  run the same testing code. Deploy = copy the ONNX files and runner, run, fetch the results JSON.
- **Pi fingerprint:** board, OS, ONNX Runtime version, CPU governor, temperature and throttling state
  before and after each run.
- **Speed** on the Pi for the 4.1 models, FP32 and Percentile INT8, with the same benchmark code as
  the laptop speed runs in 4.5: fixed threads, >= 20 warm-up and >= 100 timed runs, p50/p95/p99.
- **Laptop-vs-Pi agreement.** Question: does the same file give the same answers on ARM? (INT8
  kernels differ between x86 and ARM.) Same ONNX files, same preprocessed inputs sent from the laptop
  (so preprocessing cannot differ): top-1 agreement and largest logit difference. Image count fixed
  in the hypotheses file. Accuracy on labels stays laptop-measured unless agreement shows it
  transfers.
- Results are labelled "Raspberry Pi 5" only when measured on one; cloud ARM stays "cloud ARM".

*Done when:* one command runs the same speed benchmark and agreement check on the laptop and the Pi and
writes schema records for both; the agreement check prints pass / fail.

*Depends on:* 4.0; 4.3's PyTorch-free model list; a Pi 5 with SSH access and cooling (arriving in a
few weeks).

*Compute (estimate):* laptop side, minutes (agreement inputs). Pi: unknown until measured; the first
step is a timing probe on one model.

### 4.5 `brokkr recommend` `[ ]`

*Scope*
- `brokkr recommend --task classification --device ... --min-fps ... --max-size ... --condition ...`,
  reading only results JSON.
- **Built on laptop results.** Laptop speed for the 4.1 models (FP32 and Percentile INT8; pinned,
  fixed threads, >= 20 warm-up, >= 100 timed runs, p50/p95/p99) is measured here and always labelled
  "laptop latency". Pi rows show "not measured" until 4.4 exists.
- **Laptop latency needs its own fixed method, decided before 4.5** (note added 26 September 2026):
  plugged in, a fixed power mode, a cool-down before each measurement, repeated runs reported as the
  median and spread, and the whole setup recorded. Reason: the i5-1235U mixes fast and slow cores and
  slows down when hot. Two back-to-back pipeline profiles on 26 September differed by 25–45% on most
  steps, so a single run is not a valid latency number.
- **Constraint filtering:** only options with a measurement for that device and condition. Anything
  unmeasured is reported as "not measured", never estimated.
- **Pareto frontier:** options that no other option beats on all three of accuracy under the
  condition, speed (p95) on the device, and size.
- **"Why this":** for each recommended option, its measured numbers and the result files they come
  from.
- **"Within measured noise" flag:** two options are marked as not distinguishable when their paired
  difference interval includes zero, or the difference is smaller than the measured build-to-build
  noise.
- Condition names map only to measured conditions (e.g. `--condition night` -> darkness at the tested
  severity), and the output says which.

*Done when:* tests cover filtering, the frontier, noise flags and "not measured" (with small hand-made
inputs, used in tests only and never shown as results); a run on the real Stage 4 results prints
recommendations whose every number traces to a file (check script prints pass / fail).

*Depends on:* 4.0, 4.1 (accuracy), 4.3 (labels). Not on 4.4: Pi rows fill in once 4.4 exists.

*Compute (estimate):* laptop speed runs, minutes per model; recommendations, seconds.

**Stage 4 done when:** every 4.x "done when" is met, and labels and recommendations come straight from
results files.

### Later (not Stage 4)

- Object detection (permissive models only, e.g. YOLOX or torchvision detection)
- Runtime monitoring
- Adaptation

## Stage 5 — Energy and heat on real devices `[ ]`

The Raspberry Pi 5 SSH target moved to task 4.4.

- Energy per inference (USB power meter) and thermal throttling over sustained runs, using the 4.4
  target

**Done when:** energy and sustained-run results for the Pi come from the same testing code as 4.4.

## Stage 6 — Website + installer `[ ]`

- Static catalog site (GitHub Pages) built from JSON: problem-based search, model pages with labels
- Install script: detect device, pick model, install a camera runner that can say "not sure"

**Done when:** a stranger can go from search to a running camera demo.

## Stage 7 — Community lab + collections `[ ]`

- Public benchmark script; results submitted as pull requests with automatic checks
- Community results always labelled "community-submitted"
- Niche collections (e.g. factory defects, road scenes, farm pests), licences checked

**Done when:** a result from someone else's device appears on the site, correctly labelled.

## Later, only if real users need it

A real backend server, accounts, hosted uploads.
