"""Build one Stage 4 model: FP32 ONNX export and an INT8 version, timed (task 4.1 and M2 tooling).

Usage:  python scripts/24_build_models.py --model resnet50 [--method minmax] [--rebuild-check]
        python scripts/24_build_models.py --model efficientnet_b0 --keep-float se-all     (SE1 variants)
Needs:  the model's torchvision weights (hash-checked) and data/imagenet-1k/
Writes: models/<model>_fp32.onnx (+ .json, by scripts/01_export_model.py) and
        models/<model>_int8_percentile99.99.onnx (+ .json with build time and peak memory)

INT8 is Stage 3's chosen method applied as it is (docs/hypotheses_stage4.md): Percentile 99.99, the 512
int8_calibration images fed in groups of 128, per-channel int8 weights, per-tensor uint8 activations.
Calibration images use the model's OWN preprocessing (brokkr.export.preprocessing).

Build checks (not results): loads; finite scores; weights all per-channel; top-1 agreement with FP32
on 256 tuning images, FAIL below 20% (a broken conversion), WARNING below 90%.

--keep-float se-all | se-output (SE1, docs/hypotheses_stage4.md): the Percentile build with the model's
squeeze-and-excitation nodes (all of them, or only the output path) kept in float, chosen from the
exporter's module records (brokkr.se_float). Saved as models/<model>_int8_percentile99.99_seall.onnx or
_seoutput.onnx, with one more build check: no tensor inside the kept path is quantized.

--rebuild-check: for a model whose INT8 file already exists (MobileNetV3-Large from Stage 3), build
again into a temporary file, time it, and check the new build gives exactly the existing file's
outputs. The existing file is never replaced.

Run one model per process, so the peak memory recorded belongs to that model's build.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx

from brokkr import se_float
from brokkr.accuracy import open_image, preprocess
from brokkr.benchmark import make_session
from brokkr.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr.export import MODELS, file_info, preprocessing
from brokkr.fingerprint import machine_fingerprint
from brokkr.quantize import CALIBRATION_BATCH as BATCH
from brokkr.quantize import CALIBRATION_GROUP_BATCHES as GROUP_BATCHES
from brokkr.quantize import INT8_METHODS, INT8_SETTINGS, to_int8, weight_quantization
from brokkr.results import make_record, save_record

DATASET = "imagenet-1k-val"
# Percentile 99.99 is the 4.1 model ("int8_percentile99.99"). MinMax is the default INT8 of Stage 1,
# built for M2 with the same images and named "int8", as MobileNetV3-Large's default INT8 is.
PRECISION_NAMES = {"percentile99.99": "int8_percentile99.99", "minmax": "int8"}
FAIL_AGREEMENT, WARN_AGREEMENT = 0.20, 0.90


def peak_memory_gb() -> float:
    """Largest amount of RAM this process has used so far."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a 64-bit handle; the default int truncates it
        kernel32.K32GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters),
                                                     wintypes.DWORD)
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                                counters.cb):
            raise OSError("could not read this process's memory use")
        return counters.PeakWorkingSetSize / 1e9
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kilobytes on Linux


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODELS))
parser.add_argument("--method", choices=list(PRECISION_NAMES), default="percentile99.99")
parser.add_argument("--rebuild-check", action="store_true")
parser.add_argument("--keep-float", choices=se_float.SELECTIONS, default=None,
                    help="SE1: keep these squeeze-and-excitation nodes in float (Percentile builds only)")
parser.add_argument("--skip-symbolic-shape", action="store_true",
                    help="skip shape inference in onnxruntime's preparation step (recorded)")
args = parser.parse_args()
prep = preprocessing(args.model)
fp32_path = Path("models") / f"{args.model}_fp32.onnx"
METHOD, PRECISION = args.method, PRECISION_NAMES[args.method]
if args.keep_float and METHOD != "percentile99.99":
    sys.exit("--keep-float is only for Percentile 99.99 builds (SE1)")
if args.keep_float:
    PRECISION += "_" + args.keep_float.replace("-", "")  # int8_percentile99.99_seall / _seoutput
int8_path = Path("models") / f"{args.model}_{PRECISION}.onnx"

if int8_path.exists() and not args.rebuild_check:
    sys.exit(f"{int8_path} already exists; use --rebuild-check to time a rebuild without replacing it")

# 1. FP32 export (with its own PyTorch-vs-ONNX check), unless it already exists.
export_seconds = None
if not (fp32_path.exists() and fp32_path.with_suffix(".json").exists()):  # a half-made export is redone
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "scripts/01_export_model.py", args.model], check=True)
    export_seconds = round(time.perf_counter() - t0, 1)
export_check = json.loads(fp32_path.with_suffix(".json").read_text())["pytorch_vs_onnx"]
if not (export_check["max_abs_diff"] < 1e-4 and export_check["top1_agreement"] == 1.0):
    sys.exit(f"FAIL: {fp32_path} did not pass its PyTorch-vs-ONNX check: {export_check}")


def load_batches(files, positions):
    images = np.stack([preprocess(open_image(image), prep["resize"], prep["crop"], prep["interpolation"])
                       for image, _ in read_parquet_images(files, positions)])
    return [images[i:i + BATCH] for i in range(0, len(images), BATCH)]


def outputs(path, batches):
    session = make_session(path, num_threads=4)
    return np.concatenate([session.run(None, {"images": b})[0] for b in batches])


files = parquet_files(DATASET)
splits = make_splits(count_images(files))
calibration = load_batches(files, splits["int8_calibration"])
check = load_batches(files, splits["tuning"][:256])
reference = outputs(fp32_path, check).argmax(1)

# SE1: the squeeze-and-excitation nodes to keep in float, and the tensors that stay inside that path
# (every consumer in the export is also kept), which must not be quantized in the built model.
keep, inside, kept_float = None, [], None
if args.keep_float:
    export = onnx.load(str(fp32_path))
    keep = se_float.selected_outputs(export, args.keep_float)
    if not keep:
        sys.exit(f"FAIL: {args.model} has no {args.keep_float} nodes")
    consumers = {}
    for node in export.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    inside = [t for t in keep if consumers.get(t)
              and all(se_float.in_selection(n, args.keep_float) for n in consumers[t])]
    kept_float = {"selection": args.keep_float, "nodes": len(keep),
                  "squeeze_excitation_blocks": se_float.se_block_count(export)}
report = {}

# 2. INT8 build, timed.
with tempfile.TemporaryDirectory() as tmp:
    target = Path(tmp) / int8_path.name if int8_path.exists() else int8_path
    t0 = time.perf_counter()
    to_int8(fp32_path, target, calibration, method=METHOD, group_batches=GROUP_BATCHES,
            skip_symbolic_shape=args.skip_symbolic_shape, keep_float_outputs=keep, report=report)
    quantized_inputs = {n.input[0] for n in onnx.load(str(target)).graph.node
                        if n.op_type == "QuantizeLinear"}
    build_seconds = round(time.perf_counter() - t0, 1)
    peak_gb = round(peak_memory_gb(), 2)
    scores = outputs(target, check)
    weights = weight_quantization(target)
    same_as_existing = (bool(np.array_equal(scores, outputs(int8_path, check)))
                        if target != int8_path else None)
    same_bytes = None
    if target != int8_path:
        same_bytes = file_info(target)["sha256"] == file_info(int8_path)["sha256"]
    info = file_info(target)

agreement = float(np.mean(scores.argmax(1) == reference))
checks = {
    "loads and gives (256, 1000) scores": scores.shape == (256, 1000),
    "all scores finite": bool(np.isfinite(scores).all()),
    "all weights per-channel": weights["per_tensor"] == 0 and weights["per_channel"] > 0,
    f"top-1 agreement with FP32 >= {FAIL_AGREEMENT:.0%}": agreement >= FAIL_AGREEMENT,
}
if same_as_existing is not None:
    checks["rebuild gives exactly the existing model's outputs"] = same_as_existing
if args.keep_float:
    inside_ok = not (set(inside) & quantized_inputs)
    checks[f"no tensor inside the kept path is quantized ({len(inside)} checked)"] = inside_ok

if same_bytes is not None:
    print(f"rebuild has the same file bytes as the existing model: {same_bytes}")
print(f"{args.model}: export {export_seconds if export_seconds is not None else 'existing'} s, "
      f"INT8 build {build_seconds} s, peak memory {peak_gb} GB, {info['size_bytes'] / 1e6:.1f} MB, "
      f"agreement with FP32 {agreement:.1%} (256 tuning images)")
if agreement < WARN_AGREEMENT:
    print(f"WARNING: agreement below {WARN_AGREEMENT:.0%}")
for description, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {description}")

build = {"int8_build_seconds": build_seconds, "export_seconds": export_seconds,
         "peak_memory_gb": peak_gb, "checks": checks, "top1_agreement_with_fp32": agreement,
         "weights": weights, "skip_symbolic_shape": args.skip_symbolic_shape,
         "rebuild_same_file_bytes": same_bytes, "nodes_kept_float": report.get("nodes_kept_float", [])}
if target == int8_path:
    record = {
        "model": args.model, "precision": PRECISION, "derived_from": str(fp32_path),
        "licence": MODELS[args.model]["licence"],
        "settings": {**INT8_SETTINGS, "calibration_method": METHOD,
                     "calibration_method_detail": INT8_METHODS[METHOD]["description"],
                     "preprocessing": prep, "skip_symbolic_shape": args.skip_symbolic_shape,
                     "kept_float": kept_float,
                     "calibration": {"dataset": DATASET, "dataset_licence": DATASETS[DATASET]["licence"],
                                     "split": "int8_calibration", "n_images": len(splits["int8_calibration"]),
                                     "group_images": GROUP_BATCHES * BATCH,
                                     "excludes": "all other splits, including test"}},
        "file": {"path": str(int8_path), **info},
        "sanity_check": {"split": "tuning", "n_images": len(reference),
                         "top1_agreement_with_fp32": agreement},
        "build": build,
        "machine": machine_fingerprint(),
    }
    int8_path.with_suffix(".json").write_text(json.dumps(record, indent=2))
else:
    suffix = "_skip_symbolic_shape" if args.skip_symbolic_shape else ""
    out = Path("results/profile") / f"{args.model}_{PRECISION}_rebuild_check{suffix}.json"
    record = make_record("check", args.model, PRECISION, {
        "settings": {"split": "tuning", "n_images": len(reference), "compared_with": str(int8_path),
                     "skip_symbolic_shape": args.skip_symbolic_shape, "preprocessing": prep,
                     "note": "Rebuild check: a rebuild into a temporary file, timed and compared with "
                             "the existing model. Not a result."},
        "metrics": {"build": build},
    }, machine_fingerprint())
    save_record(record, out)
passed = all(checks.values())
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
