"""Measure how often an ONNX image classifier gets the right answer.

Works with any stream of (image, true_label) pairs, so it doesn't depend on a
particular dataset. Only needs Pillow, NumPy and ONNX Runtime (no PyTorch), so the
same code can run on small devices.
"""

import io

import numpy as np
from PIL import Image

from brokkr_edge.benchmark import make_session

# ImageNet colour statistics that torchvision's pretrained models expect.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INTERPOLATIONS = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "nearest": Image.NEAREST}


def preprocess(image: Image.Image, resize_size: int = 232, crop_size: int = 224,
               interpolation: str = "bilinear") -> np.ndarray:
    """Turn a picture into the (3, 224, 224) array the model expects.

    Same steps as torchvision's transforms for the weights: shrink so the shorter side is
    `resize_size` pixels (232 for MobileNetV3-Large), cut out the central 224x224 square, scale to
    0-1, normalise colours. Each model's own settings come from brokkr_edge.export.preprocessing.
    """
    return normalize(resize_and_crop(image, resize_size, crop_size, interpolation))


def resize_and_crop(image: Image.Image, resize_size: int = 232, crop_size: int = 224,
                    interpolation: str = "bilinear") -> np.ndarray:
    """First half of preprocessing: the (224, 224, 3) picture as uint8 pixels (0-255).

    Image corruptions are applied here, to the picture the model will actually see.
    """
    image = image.convert("RGB")  # some photos are greyscale or CMYK
    w, h = image.size
    if w <= h:
        new_w, new_h = resize_size, int(resize_size * h / w)
    else:
        new_w, new_h = int(resize_size * w / h), resize_size
    image = image.resize((new_w, new_h), INTERPOLATIONS[interpolation])

    left = int(round((new_w - crop_size) / 2.0))
    top = int(round((new_h - crop_size) / 2.0))
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return np.asarray(image, dtype=np.uint8)


def normalize(pixels: np.ndarray) -> np.ndarray:
    """Second half of preprocessing: uint8 (224, 224, 3) -> normalised float (3, 224, 224)."""
    pixels = pixels.astype(np.float32) / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return pixels.transpose(2, 0, 1)  # channels first


def topk_correct(logits: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """For each image, 1 if the true label is among the model's k highest scores, else 0."""
    topk = np.argsort(-logits, axis=1)[:, :k]
    return (topk == labels[:, None]).any(axis=1).astype(np.float64)


def bootstrap_ci(correct: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> tuple:
    """95% confidence interval for accuracy, by bootstrap resampling.

    We pretend to re-draw the test set many times (sampling images with replacement)
    and see how much the accuracy moves. The middle 95% of those accuracies is the interval.
    """
    rng = np.random.default_rng(seed)
    n = len(correct)
    resampled = [correct[rng.integers(0, n, n)].mean() for _ in range(n_resamples)]
    low, high = np.percentile(resampled, [2.5, 97.5])
    return float(low), float(high)


def open_image(source) -> Image.Image:
    """Open an image from a file path or from raw file bytes."""
    return Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)


def paired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray, n_resamples: int = 1000,
                          seed: int = 0) -> tuple:
    """Accuracy of B minus accuracy of A, with a 95% CI, when both saw the SAME images.

    Resampling the same image positions for both models cancels out "this image is just hard",
    so the difference is measured much more precisely than by comparing two separate intervals.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("both models must be evaluated on the same images")
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs.append(correct_b[idx].mean() - correct_a[idx].mean())
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(correct_b.mean() - correct_a.mean()), float(low), float(high)


def unpaired_bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray, n_resamples: int = 1000,
                            seed: int = 0) -> tuple:
    """Accuracy of B minus accuracy of A, with a 95% CI, when A and B are DIFFERENT images.

    Used to compare two splits (e.g. tuning vs test). Each side is resampled on its own, so the
    interval includes the chance variation of both image sets. Wider than a paired interval.
    """
    rng = np.random.default_rng(seed)
    na, nb = len(correct_a), len(correct_b)
    diffs = [correct_b[rng.integers(0, nb, nb)].mean() - correct_a[rng.integers(0, na, na)].mean()
             for _ in range(n_resamples)]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(correct_b.mean() - correct_a.mean()), float(low), float(high)


def predict_logits(onnx_path, samples, batch_size: int = 32, num_threads: int = 4) -> tuple:
    """Run the model on (image, label) samples. Returns (logits, labels).

    logits: float32 array (n_images, n_classes), the model's raw scores before softmax.
    `samples` can be a list or a generator; each image is a file path or raw file bytes.
    Images are processed a batch at a time, so large datasets don't fill up memory.
    """
    session = make_session(onnx_path, num_threads)
    input_name = session.get_inputs()[0].name

    labels, logits, batch = [], [], []
    for image, label in samples:
        batch.append(preprocess(open_image(image)))
        labels.append(label)
        if len(batch) == batch_size:
            logits.append(session.run(None, {input_name: np.stack(batch)})[0])
            batch.clear()
    if batch:
        logits.append(session.run(None, {input_name: np.stack(batch)})[0])
    return np.concatenate(logits).astype(np.float32), np.array(labels)


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """Top-1/top-5 accuracy with bootstrap intervals, computed from saved logits.

    Ties: INT8 models output only a few hundred distinct score values, so two classes can tie
    exactly for first place. Ties are always broken the same way, towards the lower class
    number (as np.argmax does), and the number of tied images plus the lowest and highest
    top-1 accuracy any tie-break could give are recorded.
    """
    top5 = np.argsort(-logits, axis=1, kind="stable")[:, :5]  # stable: lower class wins ties
    top1_correct = (top5[:, 0] == labels).astype(np.float64)
    top5_correct = (top5 == labels[:, None]).any(axis=1).astype(np.float64)

    best = logits.max(axis=1)
    tied = (logits == best[:, None]).sum(axis=1) > 1
    label_among_best = logits[np.arange(len(labels)), labels] == best
    return {
        "metrics": {
            "top1": float(top1_correct.mean()),
            "top1_ci95": list(bootstrap_ci(top1_correct, seed=seed)),  # list, as JSON stores it
            "top5": float(top5_correct.mean()),
            "top5_ci95": list(bootstrap_ci(top5_correct, seed=seed)),
            "top1_tied_images": int(tied.sum()),
            # Worst case: every tie resolved wrongly. Best case: every tie containing the label won.
            "top1_range_over_tie_breaks": [float((top1_correct * ~tied).mean()),
                                           float(((top1_correct > 0) | (tied & label_among_best)).mean())],
        },
        "raw": {"labels": labels.tolist(), "top5_predictions": top5.tolist()},
    }


def evaluate(onnx_path, samples, batch_size: int = 32, num_threads: int = 4, seed: int = 0) -> tuple:
    """Run the model and score it. Returns (result, logits).

    result has settings/metrics/raw for the JSON record; logits (all class scores for every
    image) are too big for JSON and are saved separately with results.save_arrays.
    """
    logits, labels = predict_logits(onnx_path, samples, batch_size, num_threads)
    result = accuracy_from_logits(logits, labels, seed=seed)
    result["settings"] = {"n_images": len(labels), "batch_size": batch_size, "num_threads": num_threads,
                          "bootstrap_resamples": 1000, "seed": seed}
    return result, logits
