# Brokkr — project brief

## What Brokkr is

Brokkr is a platform that makes it easier and faster to put quantized AI vision models on edge hardware.
It shrinks models, stress-tests them under real-world conditions (fog, blur, noise, darkness, camera shake),
and deploys them so they can say "I'm not sure" instead of guessing wrong.

Every model gets a **nutrition label**: size, speed, and power on real devices, plus how accuracy and
reliability (calibration + conformal coverage) hold up when conditions get bad.

Core research question: when a model is quantized (FP32 -> FP16 -> INT8), what happens to its speed, size,
accuracy, AND its reliability under distribution shift?

The platform is built in stages (see `ROADMAP.md`). The rigorous study (stages 1-3) comes first; everything
else is built on its measurements.

## Owner context

* The owner is learning. Explain what you did and why in plain, simple words after each task. Define any
  technical term the first time you use it.
* Prefer small, readable code over clever code. The owner must be able to explain every file in an interview.

## Hard rules (never break these)

1. NO mocked, fake, hard-coded, or placeholder metrics. Every number must come from actually running the
   code. If something can't be measured yet, say so.
2. NO overclaiming in README, docs, website, or comments. Report what was measured, on which machine, with
   which settings. Cloud ARM results are labelled "cloud ARM", never "Raspberry Pi".
3. NEVER copy code from or ship AGPL/GPL projects or models (e.g. Ultralytics YOLO). Only permissively
   licensed code and models (Apache-2.0, MIT, BSD). YOLOX (Apache-2.0) and torchvision models are fine.
4. Model files, datasets, and generated results are NOT committed to git. Anything re-downloadable or
   regenerable stays out (see `.gitignore`).
5. Stay inside the CURRENT STAGE's scope (see `ROADMAP.md`; from 30 September 2026, the current step of
   the Platform plan). Do not pull in work from later stages or add
   unrequested features. Suggest them at the end instead. The website is a walking skeleton: each stage may
   extend it, but only to show results that stage actually produced.
6. Every task ends with a way to verify it works (a test or a script that prints clear pass/fail).
7. Community-submitted results are always labelled "community-submitted" and are never mixed with
   Brokkr's own measurements without that flag.
8. Every model and dataset used or listed has its licence recorded. No licence recorded = not used.

## Measurement rules

* Speed: fix the thread count, warm up (>=20 runs), then time (>=100 runs). Report p50/p95/p99, not just
  the mean. Always record the machine (CPU model, OS, thread count).
* Accuracy/reliability: always report which dataset split and how many images. Add 95% bootstrap
  confidence intervals.
* Seeds: set random seeds so results are reproducible.

## Reporting and study rules (standing, from Stage 3)

* Differences are "new minus old" (the fix minus what it replaces), with paired bootstrap 95% intervals;
  always state the direction (accuracy/coverage: positive = better; E-AURC/ECE: positive = worse).
* Coverage is never reported without its average set size.
* No example numbers unless measured: no illustrative figures that could be mistaken for results.
* Hypotheses are committed before measuring; later implementation details go in a dated note committed
  before running; outcomes are appended, never edited above.
* No setting is tuned on the test split. Test-split reruns only for technical failure, logged with the
  reason; never because of a result.
* Analyses done after the verdicts are labelled "after the verdicts" or "exploratory"; likely
  explanations are labelled "likely".
* Report absolute weakness (FP32 also fails) separately from compression-caused weakness (INT8 minus
  FP32 on the same images).
* Every number in a report (test counts, timings, accuracies, file counts) must come from a command
  run in the current session. If a number wasn't just measured, say so or don't state it. (Added
  26 September 2026, after a Stage 4 report stated "210 tests" that no command had produced.)

## Research freeze (from 27 September 2026, until website v0 ships)

* No new research questions, diagnostics or hypotheses until website v0 ships. If a result raises a
  new question, add it to the "Parked questions" list in STATUS.md and move on.
* Task 4.2 may still run, overnight only, and only as pre-registered (in `docs/hypotheses_stage4.md`)
  before it starts. H23 is parked (in STATUS.md) and is not run during the freeze.
* Work follows the Platform plan in `ROADMAP.md` (set by H, 30 September 2026), strictly in this order:
  1 labels (by 5 Oct), 2 catalog site and going public (10 Oct), 3 testing a user's model (16 Oct),
  4 label submission, PyPI and quick start (19 Oct), 4b unlabelled mode (23 Oct),
  5 hosted upload on Hugging Face Spaces (29 Oct),
  7 EEG/EMG pack (5 Nov); 6 Raspberry Pi 5 when the board arrives (dates revised by H, 1 October
  2026). This replaces the earlier priority
  line (4.3, website v0, 4.5).

## Architecture principles

* "Where a model runs" is a pluggable target (laptop now, Raspberry Pi 5 over SSH later, community devices
  after that). Testing logic must not depend on a specific device.
* Results are saved as JSON first. Charts, HTML reports, nutrition labels, search/recommendations, and the
  website are generated from the JSON, never computed separately.
* The image-corruption + measurement code is shared with the Argos project, so keep it self-contained with
  no Brokkr-specific assumptions.
* The website is a static site (GitHub Pages).
* **Hosted upload (decided by H, 30 September 2026):** the earlier rule "no backend server until real
  users need one" is overridden for the hosted upload on Hugging Face Spaces (Platform plan step 5),
  because Brokkr is now a platform. The Space never publishes users' models or images, only the
  resulting labels.

## Tech stack

* Python 3.10+, PyTorch (CPU), torchvision, ONNX, ONNX Runtime, NumPy
* Website: static HTML/CSS generated by a Python script (JavaScript only when search needs it)
* Installer: POSIX shell script. Device runner: Python first; C++ only if measurements show Python overhead matters
* Dev machine: Windows laptop, CPU only, no GPU
* Licences: Apache-2.0 for the code; CC BY 4.0 for published labels (decided 30 September 2026)

## Layout

* `brokkr_edge/` reusable package code, published on PyPI as `brokkr-edge` (renamed from `brokkr/` on
  30 September 2026: the PyPI name "brokkr" belongs to an unrelated project)
* `scripts/` quick experiments, numbered in order (00_, 01_, ...)
* `tests/` automatic checks (run with `pytest`)
* `models/`, `data/`, `results/` gitignored
