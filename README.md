# Brokkr
Shrink AI models for edge hardware, stress-test them in real-world conditions, and deploy them to know when they're unsure.

**Status:** early development (Stage 1 of 7). There are no results yet. See [ROADMAP.md](ROADMAP.md).

## Development setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (on Linux/macOS: source .venv/bin/activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

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

Apache-2.0
