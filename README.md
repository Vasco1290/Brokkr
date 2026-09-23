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

## Licence

Apache-2.0
