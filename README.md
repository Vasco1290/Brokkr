# Brokkr
Shrink AI models for edge hardware, stress-test them in real-world conditions, and deploy them to know when they're unsure.

**Status:** Stages 1–2 of 7 complete (core measurement; damaged images and reliability). Next: Stage 3,
fixing what broke. See [ROADMAP.md](ROADMAP.md).

## What works now

One command reproduces every Stage 1 result: it exports a model to ONNX, makes FP16 and INT8 versions,
measures their speed, size, and accuracy, and builds a results page from the saved JSON.

```bash
python scripts/run_stage1.py          # add --full to also score FP32 on all 50,000 images
```

Every result file records the machine, power state, CPU cores used, library versions, and the git
commit it came from.

## Stage 1 results

MobileNetV3-Large (torchvision weights) run with ONNX Runtime 1.23.2 on a laptop
(Intel Core i5-1235U, Windows 11, CPU only).

**Size and accuracy** on a fixed 10,000-image subset of the ImageNet-1k validation split.
Brackets are 95% bootstrap confidence intervals; "vs FP32" is the paired difference on the same images.

| Precision | File size | Top-1 accuracy | Top-1 vs FP32 | Top-5 |
|---|---|---|---|---|
| FP32 | 22.2 MB | 75.58% (74.70–76.48) | — | 92.70% |
| FP16 | 11.3 MB | 75.63% (74.78–76.53) | +0.05 pts (−0.02 to +0.13) | 92.71% |
| INT8 | 5.9 MB | 60.15% (59.14–61.11)\* | −15.43 pts (−16.30 to −14.59) | 83.37% |

\* INT8 outputs take only 237 distinct values, so on 262 of the 10,000 images two classes tie
exactly for the top score. Ties go to the lower class number; any other tie-break would give a top-1
between 59.62% and 61.01%.

Correctness check: on all 50,000 validation images our FP32 pipeline scores 75.26% (74.88–75.61);
torchvision publishes 75.27% for these weights.

**What this shows**

- FP16 halves the file size with no measurable accuracy change.
- INT8 made with ONNX Runtime's default static quantization (MinMax calibration, 512 images) is 3.75x
  smaller but loses about 15 points of top-1 accuracy, and its rounded outputs cause exact ties.
  Improving this is planned for Stage 3.

**Speed on this laptop: rough, development machine only.** The i5-1235U mixes fast "performance" and
slow "efficiency" cores. Unpinned, Windows moved the benchmark between them, so timings jumped between
two speeds; the benchmark now pins to one core type and records it. Even pinned, repeated runs on this
laptop disagreed by 10–20%, so only these conclusions held up (plugged in, "Best performance" mode,
10 interleaved sessions per setting, median time per image):

- One thread on a performance core is about twice as fast as on an efficiency core
  (FP32: 7.50 ms vs 13.74 ms).
- FP16 gives no speed benefit on this CPU.
- Default INT8 is not reliably faster. On 4 efficiency cores it was slower than FP32 (6.53 ms vs 5.42 ms,
  both stable); on one performance core it looked faster (6.37 ms vs 7.50 ms), but that measurement was
  unstable (sessions varied by 33%).

Speed will be measured properly on a Raspberry Pi 5 in Stage 5.

## Stage 2 results: damaged images and "knowing when it's wrong"

The same 10,000 test images, damaged by fog, defocus blur, motion blur, noise, and darkness at
severities 1–5 (`brokkr/shift`, our own implementations in the style of ImageNet-C). Nine predictions
were written down and committed before measuring; full outcomes are in
[docs/hypotheses.md](docs/hypotheses.md): 6 confirmed, 3 rejected. The results page
(`scripts/05_build_site.py`) has the charts and every number.

- **FP16 behaves like FP32 everywhere:** the largest accuracy difference in any of 26 conditions was
  0.25 points.
- **Default INT8 falls apart faster than FP32 under every kind of damage.** At severity 3 it keeps
  38–65% of FP32's accuracy, against 80% on clean images.
- **Darkness hurts only INT8.** FP32 goes from 75.6% (clean) to 73.7% at severity 5; default INT8
  from 60.2% to 20.5%.
- **The 90% conformal promise holds on clean images** (FP32 90.6%, FP16 90.5%, INT8 89.8%) **but
  breaks under blur and noise** (FP32 at severity 5: 20–37%), **while prediction sets barely grow**
  (2.3 classes clean, 2.1–2.8 at severity 5). The model gives no warning that it is failing.
- **Calibration numbers can mislead.** This model is under-confident on clean images (average
  confidence 57.9% vs accuracy 75.6%); damage lowered accuracy toward its confidence, so ECE
  *improved* while accuracy collapsed.
- **Default INT8 is worse at knowing when it's wrong:** it needs sets of 8.3 classes (vs 2.3) to keep
  the 90% promise on clean images, and its error when answering its most confident half is 16.2%
  (vs 5.2% for FP32).

To reproduce (about 2.5 hours on this laptop, plus calibration-split runs of `scripts/03`):

```bash
python scripts/03_evaluate_accuracy.py --precision fp32 --split conformal_calibration   # and fp16, int8
python scripts/08_corruption_sweep.py
python scripts/07_reliability.py
```

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (on Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
ruff check .
```

To recreate the exact package versions used for the results above, see `requirements-lock.txt`.

## Test data

Accuracy is measured on the ImageNet-1k validation set (50,000 images, about 6.7 GB). Its terms allow
non-commercial research and educational use only, and each user must accept them:

1. Log in at huggingface.co and accept the terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
2. Create a "Read" access token, then run `hf auth login` in a regular terminal and paste it
3. Download the validation files into `data/` (gitignored):

```bash
pip install -e ".[data]"
hf download ILSVRC/imagenet-1k --repo-type dataset --include "data/validation-*" --local-dir data/imagenet-1k
```

## Licence

Apache-2.0. Model weights and datasets keep their own terms (recorded with every result): the torchvision
MobileNetV3 weights were trained on ImageNet-1k, whose terms allow non-commercial research use only.
