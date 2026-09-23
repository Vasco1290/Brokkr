# Brokkr roadmap

Brokkr is built from the inside out. Each stage must work on its own before the next one starts,
so stopping at any stage still leaves something finished.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done.

## Stage 1 — Core measurement `[~]`

One model, three precisions, measured honestly on the laptop.

- [x] 1.1 Project skeleton: package, `pyproject.toml`, first test, CI on every push
- [ ] 1.2 Machine fingerprint: CPU, OS, threads, package versions, git commit attached to every result
- [ ] 1.3 Export MobileNetV3 (torchvision) to ONNX FP32
- [ ] 1.4 Speed benchmark: fixed threads, >=20 warm-up, >=100 timed runs, p50/p95/p99 -> JSON
- [ ] 1.5 Accuracy on a fixed, seeded image set (split + image count recorded) -> JSON
- [ ] 1.6 FP16 and INT8 versions, same measurements

**Done when:** one command produces real speed/size/accuracy numbers for FP32/FP16/INT8, and tests pass.

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
