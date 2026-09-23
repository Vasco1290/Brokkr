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
5. Stay inside the CURRENT STAGE's scope (see `ROADMAP.md`). Do not pull in work from later stages or add
   unrequested features. Suggest them at the end instead.
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

## Architecture principles

* "Where a model runs" is a pluggable target (laptop now, Raspberry Pi 5 over SSH later, community devices
  after that). Testing logic must not depend on a specific device.
* Results are saved as JSON first. Charts, HTML reports, nutrition labels, search/recommendations, and the
  website are generated from the JSON, never computed separately.
* The image-corruption + measurement code is shared with the Argos project, so keep it self-contained with
  no Brokkr-specific assumptions.
* The website starts as a static site (GitHub Pages). No backend server until real users need one.

## Tech stack

* Python 3.10+, PyTorch (CPU), torchvision, ONNX, ONNX Runtime, NumPy
* Dev machine: Windows laptop, CPU only, no GPU
* Licence: Apache-2.0

## Layout

* `brokkr/` reusable package code (becomes `pip install brokkr` later)
* `scripts/` quick experiments, numbered in order (00_, 01_, ...)
* `tests/` automatic checks (run with `pytest`)
* `models/`, `data/`, `results/` gitignored
