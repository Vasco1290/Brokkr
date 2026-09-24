# Brokkr
Shrink AI models for edge hardware, stress-test them in real-world conditions, and deploy them to know when they're unsure.

**Status:** Stage 1 of 7 complete (core measurement). Stage 2 (damaged images and reliability) is in
progress. See [ROADMAP.md](ROADMAP.md).

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
