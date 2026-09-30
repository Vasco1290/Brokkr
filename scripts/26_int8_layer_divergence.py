"""Where does an INT8 model first drift away from FP32? A layer-by-layer diagnostic (not a result).

Usage:  python scripts/26_int8_layer_divergence.py --model mobilenet_v3_small [--images 32]
Needs:  models/<model>_fp32.onnx and models/<model>_int8_percentile99.99.onnx
Writes: results/checks/<model>_int8_layer_divergence.json (schema 2, kind "diagnostic": never a result)

Uses ONNX Runtime's own debugging tool (onnxruntime.quantization.qdq_loss_debug). Both models are
changed to also output every quantized tensor; both run on the same tuning images (the model's own
preprocessing); for each tensor the INT8 value is compared with the FP32 value as a signal-to-noise
ratio in decibels (SQNR: higher = closer; every 10 dB less means ten times more error relative to the
signal). The FP32 side is the model after ONNX Runtime's preparation step (quant_pre_process), because
that is the model the INT8 file was quantized from, so the tensor names match.

Printed, in graph order: the first tensor below 10 dB (error at least a third of the signal's size)
and the largest single drop between one tensor and the next. These two descriptions are fixed here,
before looking; they describe the failure and set nothing.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quant_pre_process
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations,
    compute_activation_error,
    create_activation_matching,
    modify_model_output_intermediate_tensors,
)

from brokkr_edge.accuracy import open_image, preprocess
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.export import MODELS, file_info, preprocessing
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.quantize import CALIBRATION_BATCH, ImageBatches
from brokkr_edge.schema import condition, make_measurement, metric, save_measurement

LOW_DB = 10.0
THREADS = 4  # the same thread count as every other Brokkr run

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--images", type=int, default=32)
args = parser.parse_args()
prep = preprocessing(args.model)
fp32_path = Path("models") / f"{args.model}_fp32.onnx"
int8_path = Path("models") / f"{args.model}_int8_percentile99.99.onnx"

files = parquet_files("imagenet-1k-val")
positions = make_splits(count_images(files))["tuning"][:args.images]
images = np.stack([preprocess(open_image(b), prep["resize"], prep["crop"], prep["interpolation"])
                   for b, _ in read_parquet_images(files, positions)])
batches = [images[i:i + CALIBRATION_BATCH] for i in range(0, len(images), CALIBRATION_BATCH)]

with tempfile.TemporaryDirectory() as tmp:
    prepared = Path(tmp) / "prepared.onnx"
    quant_pre_process(str(fp32_path), str(prepared))
    float_aug, int8_aug = Path(tmp) / "float_aug.onnx", Path(tmp) / "int8_aug.onnx"
    modify_model_output_intermediate_tensors(prepared, float_aug)
    modify_model_output_intermediate_tensors(int8_path, int8_aug)
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    float_acts = collect_activations(str(float_aug), ImageBatches(batches), options, ["CPUExecutionProvider"])
    int8_acts = collect_activations(str(int8_aug), ImageBatches(batches), options, ["CPUExecutionProvider"])

errors = compute_activation_error(create_activation_matching(int8_acts, float_acts))

# Graph order: the order of the activation QuantizeLinear nodes in the INT8 model.
graph = onnx.load(str(int8_path)).graph
stored = {i.name for i in graph.initializer}
order = [n.input[0] for n in graph.node if n.op_type == "QuantizeLinear" and n.input[0] not in stored]
rows = [{"tensor": t, "sqnr_db": round(float(errors[t]["xmodel_err"]), 2)}
        for t in order if t in errors and "xmodel_err" in errors[t]]

first_low = next((r for r in rows if r["sqnr_db"] < LOW_DB), None)
drops = [(rows[i - 1]["sqnr_db"] - rows[i]["sqnr_db"], i) for i in range(1, len(rows))]
biggest_drop, at = max(drops)
print(f"{args.model}: {len(rows)} quantized tensors compared on {args.images} tuning images")
print(f"  first tensor, input side: {rows[0]['tensor']} {rows[0]['sqnr_db']} dB; "
      f"last compared tensor: {rows[-1]['tensor']} {rows[-1]['sqnr_db']} dB")
if first_low:
    print(f"  first below {LOW_DB:.0f} dB: #{rows.index(first_low)} {first_low['tensor']} "
          f"({first_low['sqnr_db']} dB)")
else:
    print(f"  no tensor below {LOW_DB:.0f} dB")
print(f"  largest single drop: {biggest_drop:.1f} dB, from #{at - 1} {rows[at - 1]['tensor']} "
      f"({rows[at - 1]['sqnr_db']} dB) to #{at} {rows[at]['tensor']} ({rows[at]['sqnr_db']} dB)")

machine = machine_fingerprint()
spec = MODELS[args.model]
record = make_measurement(
    "diagnostic",
    {"name": args.model, "weights": str(spec["weights"]), "licence": spec["licence"]},
    "int8_percentile99.99",
    {"name": "onnxruntime", "version": ort.__version__, "execution_provider": "CPUExecutionProvider",
     "threads": THREADS},
    "laptop", machine,
    {"dataset": "imagenet-1k-val", "split": "tuning", "n_images": int(len(positions)),
     "licence": DATASETS["imagenet-1k-val"]["licence"]},
    condition(),
    {"tensors_compared": metric(len(rows)),
     "median_sqnr_db": metric(float(np.median([r["sqnr_db"] for r in rows]))),
     "first_below_low_db_index": metric(rows.index(first_low) if first_low else -1),
     "largest_drop_db": metric(round(biggest_drop, 2))},
    {"script": "scripts/26_int8_layer_divergence.py", "preprocessing": prep,
     "tool": "onnxruntime.quantization.qdq_loss_debug (SQNR, dB)", "low_db": LOW_DB,
     "note": "Diagnostic of where INT8 drifts from FP32. Not a result; sets nothing."},
    derived_from=[{"file": str(path), "sha256": file_info(path)["sha256"]}
                  for path in (fp32_path, int8_path)],
)
record["raw"] = {"tensors_in_graph_order": rows, "first_below_low_db": first_low,
                 "largest_drop": {"from": rows[at - 1], "to": rows[at]}}
out = Path("results/checks") / f"{args.model}_int8_layer_divergence.json"
print(f"Saved {save_measurement(record, out)}")
