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

## Stage 2 — Stress test `[~]`

How much worse does each precision get on damaged photos, and does it still know when it's wrong?

- [ ] 2.0 Hypotheses written down *before* measuring (`docs/hypotheses.md`), committed first
- [ ] 2.1 Fixed, non-overlapping image splits: test 10,000 / conformal calibration 5,000 /
  tuning 5,000 (kept for Stage 3) / INT8 calibration 512
- [ ] 2.2 Corruptions: fog, defocus blur, motion blur, noise, darkness at 5 severities —
  self-contained module shared with Argos (no Brokkr imports), plus a sample image sheet
- [ ] 2.3 Save compact per-image confidence data so every reliability number can be recomputed
- [ ] 2.4 Calibration: expected calibration error (ECE) and reliability diagram data
- [ ] 2.5 Conformal prediction: 90% sets tuned on clean calibration images; coverage and set
  size on clean and corrupted test images
- [ ] 2.6 Selective prediction: risk–coverage curves and AURC
- [ ] 2.7 Full sweep: 3 precisions x (clean + 5 corruptions x 5 severities) on the 10,000 test
  images (overnight, plugged in)
- [ ] 2.8 Headline chart and a robustness section on the results page, generated from JSON

Bootstrap confidence intervals on every accuracy/reliability number.

**Done when:** the headline chart (precision × corruption × reliability) is generated from JSON.

## Stage 3 — Fixes and the study `[ ]`

- Fixes: recalibration after quantization, mixed precision, shift-aware conformal
- INT8 calibration study: MinMax vs Percentile vs Entropy, chosen on a tuning set disjoint from both
  calibration and test images, then measured on the test set. Motivation: in task 1.6, ONNX Runtime's
  default (MinMax) INT8 MobileNetV3 disagreed with FP32 on about a third of images; a quick diagnostic
  (on test images, so not a result) suggested outlier-robust calibration recovers much of it.
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
