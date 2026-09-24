# Brokkr roadmap

Brokkr is built from the inside out. Each stage must work on its own before the next one starts,
so stopping at any stage still leaves something finished.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done.

The website grows with every stage as a **walking skeleton**: the thinnest end-to-end version
(results JSON -> HTML page) exists from Stage 1, and each stage adds only what it actually measured.

Languages: Python for the factory, test lab, and site builder; HTML/CSS (+ a little JavaScript for
search) for the website; a shell script for the installer. The device runner starts in Python and moves
to C++ only if measurements show Python overhead matters.

## Stage 1 — Core measurement `[~]`

One model, three precisions, measured honestly on the laptop.

- [x] 1.1 Project skeleton: package, `pyproject.toml`, first test, CI on every push
- [x] 1.2 Machine fingerprint: CPU, OS, package versions, git commit attached to every result
  (the thread count is a benchmark setting, recorded by the benchmark in 1.4)
- [x] 1.3 Export MobileNetV3 (torchvision) to ONNX FP32
- [x] 1.4 Speed benchmark: fixed threads, >=20 warm-up, >=100 timed runs, p50/p95/p99 -> JSON;
  5 interleaved sessions (median + spread); power state recorded
  - [ ] Run plugged in, "Best performance" mode, to compare with the battery runs
    (battery runs so far: session-to-session spread 24% to over 1000%)
- [ ] 1.5 Accuracy on a fixed, seeded image set (split + image count recorded) -> JSON
- [ ] 1.6 FP16 and INT8 versions, same measurements
- [ ] 1.7 Walking-skeleton website: a Python script turns results JSON into one plain HTML page,
  published on GitHub Pages. Real numbers only, no design yet.

**Done when:** one command produces real speed/size/accuracy numbers for FP32/FP16/INT8, they appear on
the live page, and tests pass.

## Stage 2 — Stress test `[ ]`

- Corruptions (fog, blur, noise, darkness, shake) at 5 severities — self-contained module shared with Argos
- Reliability: calibration error (ECE), conformal prediction coverage and set size, risk–coverage curves
- Bootstrap confidence intervals on every accuracy/reliability number

**Done when:** the headline chart (precision × corruption × reliability) is generated from JSON.

## Stage 3 — Fixes and the study `[ ]`

- Hypotheses written down *before* measuring (`docs/hypotheses.md`)
- Fixes: recalibration after quantization, mixed precision, shift-aware conformal
- Experiment: INT8 calibration on clean vs corrupted images
- Write-up with a limitations section

**Done when:** one finding can be explained in two minutes, with the numbers behind it.

## Stage 4 — Nutrition label + search `[ ]`

- Standard label per (model, precision, device), generated from JSON
- `brokkr recommend --task ... --device ... --min-fps ... --condition night`
- Add object detection (permissive models only, e.g. YOLOX or torchvision detection)

**Done when:** labels and recommendations come straight from results files.

## Stage 5 — Real devices `[ ]`

- Raspberry Pi 5 target over SSH, using the same testing code
- Energy per inference (USB power meter) and thermal throttling over sustained runs

**Done when:** the same test runs on laptop and Pi without changing testing logic.

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
