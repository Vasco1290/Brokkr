<p align="center">
  <img src="docs/assets/brokkr-banner.jpg" alt="Brokkr: a small anvil fused onto a computer chip" width="100%">
</p>

# Brokkr

**Small models you can actually trust on the edge.**

Brokkr shrinks AI vision models so they fit on small devices, stress-tests them on damaged photos (fog,
darkness, blur, noise, low contrast), and writes a **label** for each one: how big it is, how accurate
it is, and whether it still knows when it is wrong once conditions get bad.

The question behind it: when a model is shrunk from full precision (FP32) to small whole numbers
(INT8), what happens to its accuracy **and** to its "I'm not sure" signal, on clean photos and on
damaged ones?

> **Status, 2 October 2026.** The study (Stages 1–3) and the ten-model breadth study (Stage 4) are
> done. Labels exist for all ten models and pass their checks, but none is published yet. There is
> **no** published package, command-line tool or website yet, and nothing has been measured on a
> Raspberry Pi. Every number below was measured on one Windows laptop (Intel Core i5-1235U, CPU only).

## Contents

- [What works today](#what-works-today)
- [What a label says](#what-a-label-says)
- [What we have found so far](#what-we-have-found-so-far)
- [How we measure](#how-we-measure)
- [Set up and run](#set-up-and-run)
- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)
- [Licence](#licence)

## What works today

| | State |
|---|---|
| Shrinking a model to FP16 and INT8 (four INT8 recipes) | works (`brokkr_edge/quantize.py`) |
| Accuracy with 95% intervals, on clean and damaged images | works (`brokkr_edge/accuracy.py`, `brokkr_edge/shift/`) |
| "I'm not sure": calibration, prediction sets with their size, confidence ranking | works (`brokkr_edge/shift/`) |
| Ten models, thirteen conditions, every result checked against a schema | done (`scripts/28`, `scripts/22`) |
| Labels: `label.json`, a Markdown model card and an HTML page, checked against their sources | made for all ten models, not published yet (`scripts/39`, `scripts/40`) |
| Speed (latency) by a fixed method for 19 builds | **not measured yet** (rough Stage 1 timings only, see [STATUS.md](STATUS.md)) |
| `brokkr-edge` command, PyPI package | **not built yet** |
| Catalog website, testing your own model, hosted upload | **not built yet** (see [Roadmap](#roadmap)) |
| Raspberry Pi 5 | **not measured**; waiting for the board |

## What a label says

One label describes one shrunk build of one model, with the full-precision original beside it. No
label is published yet: the catalog comes with the website (step 2 of the roadmap).

<!-- label-example:start (written by scripts/41_readme_label_example.py; do not edit by hand) -->
This is the top of the label for **MobileNetV3-Large · INT8 (percentile calibration, 99.99%)**, measured on 10000 ImageNet-1k validation images (the test split):

|  | FP32 (full precision) | INT8 (percentile calibration, 99.99%) |
|---|---|---|
| File size | 22.22 MB | 5.92 MB |
| Top-1 accuracy, clean images | 75.58% (74.70% to 76.48%) | 73.60% (72.75% to 74.52%) |
| Coverage of its prediction sets, clean images | 90.58% (90.01% to 91.16%), set size 2.28 | 90.22% (89.62% to 90.85%), set size 2.63 |

Shrinking cost on clean images: -1.98 (-2.47 to -1.47) points (INT8 minus FP32 on the same images, so negative means shrinking lost accuracy).

For the 12 kinds of damage tested, the label opens with three lines:

- **Shrinking** made it much worse (a large shrinking cost) in 6 of 12 conditions; the largest is contrast (ImageNet-C) s5, -37.20 points.
- **Harsh conditions:** the FP32 original is itself harmful, even at full size, in 8 of 12.
- **Uncertainty signal:** unreliable (coverage below 80%) in 10 of 12. The prediction sets were calibrated on clean images and can be re-calibrated on your own images.

Then it sorts each condition by what happened to the shrunk model, and why:

- **Not harmful in our tests** (1 of 12): fog (Brokkr) s3
- **Harmful, too hard for this model (FP32 also fails)** (8 of 12)
- **Harmful, hurt by shrinking (FP32 copes, INT8 doesn't)** (2 of 12): darkness (Brokkr) s5, contrast (ImageNet-C) s3
- **Harmful, cause unclear (FP32 is borderline)** (1 of 12): fog (ImageNet-C) s3

"Harmful" means a whole 95% interval is below a line fixed before these results existed: coverage below 80%, or a damage drop (accuracy under the damage minus clean accuracy) below -10 points. "Not harmful in our tests" is not a guarantee.

A recipe that breaks a model is labelled as such: MobileNetV3-Small's INT8 build matched FP32's first answer on only 2.73% of 256 tuning images (a usable build needs at least 20%), so its label opens with "do not use this INT8 build".
<!-- label-example:end -->

Every harmful row of a label says which line it failed (accuracy, coverage or both) and carries a
suggested next step, such as "try another recipe or model". The suggestions are general; they were not
tested for the model.

The format is defined in [docs/label_schema.md](docs/label_schema.md). Every number in a label names
the result file it came from, and `scripts/40_check_labels.py` recomputes each one.

## What we have found so far

**One model in depth (Stages 1–3, MobileNetV3-Large).** Full write-up:
[Brokkr Technical Report 1](docs/writeup.md).

| Build | File size | Top-1, clean | Top-1, darkness s5 |
|---|---|---|---|
| FP32 | 22.22 MB | 75.58% | 73.73% |
| FP16 | 11.28 MB | 75.63% | 73.71% |
| INT8, default recipe (min-max) | 5.92 MB | 60.15% | 20.48% |
| INT8, percentile 99.99% recipe | 5.92 MB | 73.60% | 60.15% |

- The default INT8 recipe broke this model; a different recipe, chosen without touching the test
  images, recovered most of the clean accuracy at the same file size.
- Darkness barely bothers the full-precision model but still costs the better INT8 build about 13.6
  points.
- The "I'm not sure" signal fails quietly under damage: the share of images whose prediction set holds
  the right answer falls, but the sets do not grow to warn you. For the shrunk build under contrast
  (ImageNet-C) s5, coverage is 12.63% with a set size of 0.99 (on clean images: 90.22% with 2.63).
- Eight predictions were written down before measuring: 6 passed, 2 failed
  ([docs/hypotheses_stage3.md](docs/hypotheses_stage3.md)).

**Ten models (Stage 4).** Plain-language summary: [docs/stage4_story.md](docs/stage4_story.md);
predictions and outcomes: [docs/hypotheses_stage4.md](docs/hypotheses_stage4.md).

- Ten torchvision models, 13 conditions, 266 checked result records. All ten full-precision models
  scored within one point of their published accuracy. Nine INT8 builds were usable; one
  (MobileNetV3-Small) broke.
- Of six predictions committed before measuring, 2 passed and 4 failed. The darkness collapse is not
  common: only 2 of the 9 models lose much more to darkness when shrunk (we had predicted at least 5).
- The tests of *why* it happens have not found the cause: each was inconclusive or rejected our
  guess. The question is still open, and research is paused until the labels and the website ship.

## How we measure

- **No made-up numbers.** Every number comes from running the code, is saved as JSON first, and records
  the machine, library versions and git commit it came from. Charts, labels and pages are generated
  from that JSON.
- **Predictions first.** Hypotheses and pass rules are committed before measuring; outcomes are
  appended, never edited.
- **No tuning on the test images.** Settings are chosen on separate tuning and calibration images.
- **Intervals everywhere.** Accuracy and coverage carry 95% bootstrap intervals; differences are
  paired on the same images. Coverage is never shown without its average set size.
- **Named hardware.** Laptop results are called laptop results. Nothing here is a Raspberry Pi number.

The full rules are in [CLAUDE.md](CLAUDE.md).

## Set up and run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (on Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
ruff check .
```

To recreate the exact package versions used for the results, see `requirements-lock.txt`.

**Test data.** Accuracy is measured on the ImageNet-1k validation set (50,000 images, about 6.7 GB).
Its terms allow non-commercial research and educational use only, and each user must accept them:

1. Log in at huggingface.co and accept the terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. Create a "Read" access token, then run `hf auth login` in a regular terminal and paste it
3. Download the validation files into `data/` (gitignored):

```bash
pip install -e ".[data]"
hf download ILSVRC/imagenet-1k --repo-type dataset --include "data/validation-*" --local-dir data/imagenet-1k
```

**Reproduce Stage 1** (export, FP16 and INT8 builds, speed, size, accuracy, results page):

```bash
python scripts/run_stage1.py          # add --full to also score FP32 on all 50,000 images
```

**Make and check labels** (needs the Stage 4 result files in `results/`, which are not in git):

```bash
python scripts/39_make_labels.py --models mobilenet_v3_large mobilenet_v3_small
python scripts/40_check_labels.py     # prints PASS only if every number matches its source
python scripts/22_check_results.py    # checks every result record
```

Reproduction steps for Stages 2–3 are in section 9 of the [technical report](docs/writeup.md).

## Roadmap

The platform plan, in order (details and "done when" lines in [ROADMAP.md](ROADMAP.md)):

| Step | What | Planned |
|---|---|---|
| 1 | Labels for all ten models, the `brokkr-edge test` command, laptop latency | 7 October 2026 |
| 2 | Catalog website and making this repository public | 10 October 2026 |
| 3 | Testing your own ONNX or PyTorch model with your own labelled images | 16 October 2026 |
| 4 | Label submission, PyPI package (`brokkr-edge`), quick start | 19 October 2026 |
| 4b | Testing with images that have no labels | 23 October 2026 |
| 5 | Hosted upload on Hugging Face Spaces | 29 October 2026 |
| 7 | EEG/EMG signals: one dataset, one model, three damage types | 5 November 2026 |
| 6 | Raspberry Pi 5 latency and labels | when the board arrives |

## Repository layout

| Path | What is there |
|---|---|
| `brokkr_edge/` | the package: shrinking, damage, measurement, result schema, label rules and renderer |
| `scripts/` | numbered scripts, in the order the work was done |
| `tests/` | automatic checks (`pytest`) |
| `docs/` | the technical report, hypotheses and outcomes, the label format |
| `STATUS.md` | where everything stands, in detail |
| `models/`, `data/`, `results/`, `labels/` | not in git: re-downloadable or regenerated |

## Licence

Code: Apache-2.0 ([LICENSE](LICENSE)). Published labels: CC BY 4.0. Model weights and datasets keep
their own terms, recorded with every result and on every label: the torchvision weights used here
were trained on ImageNet-1k, whose terms allow non-commercial research use only.
