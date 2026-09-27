"""EXPLORATORY, after the M2 verdicts: where do the largest per-tensor extra rounding errors E(t) sit?

Usage:  python scripts/36_m2_top_tensors.py
Needs:  results/m2/<model>_int8_percentile99.99_{darkness,fog}_m2.json and their .npz (scripts/35), and the
        model files (read only; no model is run)
Writes: results/checks/m2_top_tensors_<model>_<condition>.json (schema 2, kind "diagnostic", one per model
        and damaged condition: never a result, not judged, changes no verdict)

For each of the 8 M2 models, Percentile INT8, darkness (Brokkr) s5 and fog (Brokkr) s3: the 5 tensors
with the largest E(t) (M2's local extra rounding error: mean over the 128 images of clean minus damaged
SQNR), each with
- E(t) and its 95% interval (1,000 resamples of the images, seed 0, from the saved per-image SQNRs);
- its position: index among the matched tensors (#0 = the input image) and depth in % (index / last index);
  and whether it is in M2's early block (the first 10% after the input);
- the operation that produces it in the INT8 model's graph (the model M2 measured) and in the original
  FP32 export (they differ where ONNX Runtime's preparation step merged an activation into the
  operation before it, e.g. Conv + ReLU6 -> Conv, keeping the activation's output name), and the
  PyTorch module and module classes recorded by the exporter on the node of the FP32 file that produces
  the same tensor name ("pkg.torch.onnx.name_scopes" and "pkg.torch.onnx.class_hierarchy");
- its block: the deepest enclosing module whose class is one of the models' building blocks (BLOCK_CLASSES);
  outside any such block it is marked "outside a block" with its module path;
- two flags, read from the module classes and the operation, never from tensor names:
  squeeze-and-excitation = torchvision.ops.misc.SqueezeExcitation encloses the tensor's module;
  smooth activation = the producing module is SiLU, Hardswish, Hardsigmoid, Sigmoid or GELU, or the
  operation is Sigmoid, HardSigmoid, HardSwish, Gelu or Erf.
This only describes where large E(t) values sit; it draws no conclusion.
"""

import ast
import math
from pathlib import Path

import numpy as np
import onnx

from brokkr.export import MODELS
from brokkr.fingerprint import machine_fingerprint
from brokkr.m2 import EARLY_FRACTION, N_RESAMPLES, SEED
from brokkr.results import sha256_of
from brokkr.schema import load_measurement, make_measurement, metric, save_measurement

M2_MODELS = ["mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x1_0", "mnasnet1_0", "regnet_y_400mf",
             "resnet18", "resnet50", "convnext_tiny"]
PRECISION, TOP = "int8_percentile99.99", 5
SE_CLASS = "torchvision.ops.misc.SqueezeExcitation"
SMOOTH_CLASSES = {"torch.nn.modules.activation." + c for c in ("SiLU", "Hardswish", "Hardsigmoid", "Sigmoid",
                                                                "GELU")}
SMOOTH_OPS = {"Sigmoid", "HardSigmoid", "HardSwish", "Gelu", "Erf"}
BLOCK_CLASSES = {  # the building blocks of the 8 models, as torchvision names them
    "torchvision.models.mobilenetv2.InvertedResidual", "torchvision.models.efficientnet.MBConv",
    "torchvision.models.shufflenetv2.InvertedResidual", "torchvision.models.mnasnet._InvertedResidual",
    "torchvision.models.regnet.ResBottleneckBlock", "torchvision.models.resnet.BasicBlock",
    "torchvision.models.resnet.Bottleneck", "torchvision.models.convnext.CNBlock"}
RUNTIME = {"name": "numpy and onnx (no model is run)",
           "version": f"numpy {np.__version__}, onnx {onnx.__version__}", "execution_provider": "none",
           "threads": 1}


def producers(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    return {out: node for node in model.graph.node for out in node.output}


def module_info(node) -> tuple:
    """(module path, module classes) the exporter recorded on a node; the last entry is the aten op."""
    md = {p.key: p.value for p in node.metadata_props}
    scopes = ast.literal_eval(md.get("pkg.torch.onnx.name_scopes", "[]"))
    classes = ast.literal_eval(md.get("pkg.torch.onnx.class_hierarchy", "[]"))
    return scopes[:-1], classes[:-1]


machine = machine_fingerprint()
print("EXPLORATORY (after the M2 verdicts; describes where large E(t) sit, draws no conclusion).")
print("Percentile INT8; E(t) in dB (positive = the damage makes INT8 round that tensor worse); "
      "95% intervals over the 128 tuning images.")
for model in M2_MODELS:
    int8_nodes = producers(Path("models") / f"{model}_{PRECISION}.onnx")
    fp32_nodes = producers(Path("models") / f"{model}_fp32.onnx")
    for corruption in ("darkness", "fog"):
        path = Path("results/m2") / f"{model}_{PRECISION}_{corruption}_m2.json"
        record, arrays = load_measurement(path)
        tensors = [str(t) for t in arrays["tensors"]]
        diff = arrays["local_db_clean"] - arrays[f"local_db_{corruption}"]  # (images, tensors)
        e = diff.mean(axis=0)
        if not np.allclose(e, record["raw"]["E_db"]):
            raise SystemExit(f"FAIL: {path.name}: E(t) from the arrays differs from the record")
        rng = np.random.default_rng(SEED)
        n = diff.shape[0]
        boot = np.array([diff[rng.integers(0, n, n)].mean(axis=0) for _ in range(N_RESAMPLES)])
        low, high = np.percentile(boot, [2.5, 97.5], axis=0)
        early = math.ceil((len(tensors) - 1) * EARLY_FRACTION)
        order = np.argsort(-e, kind="stable")[:TOP]
        rows = []
        for i in order:
            t = tensors[i]
            op_node = int8_nodes.get(t)
            meta_node = fp32_nodes.get(t) or op_node
            scopes, classes = module_info(meta_node) if meta_node is not None else ([], [])
            enclosing = reversed(list(zip(scopes, classes, strict=True)))
            block = next((s for s, c in enclosing if c in BLOCK_CLASSES), None)
            module, module_class = (scopes[-1], classes[-1]) if scopes else ("(model input)", "")
            op = "input image" if t == "images" else (op_node.op_type if op_node is not None else "unknown")
            fp32_op = "input image" if t == "images" else (
                fp32_nodes[t].op_type if t in fp32_nodes else "not in the FP32 export")
            rows.append({
                "rank": len(rows) + 1, "tensor": t, "E_db": float(e[i]),
                "E_ci95": [float(low[i]), float(high[i])],
                "index": int(i), "depth_pct": round(100 * i / (len(tensors) - 1), 1),
                "in_early_block": bool(1 <= i <= early), "operation": op, "fp32_export_operation": fp32_op,
                "module": module,
                "module_class": module_class.split(".")[-1],
                "block": block or f"outside a block ({module})",
                "squeeze_excitation": SE_CLASS in classes,
                "smooth_activation": any(c in SMOOTH_CLASSES for c in classes[-1:]) or op in SMOOTH_OPS})
        severity = 5 if corruption == "darkness" else 3
        print(f"\n{model}, {corruption} (Brokkr) s{severity}: {len(tensors)} tensors "
              f"(#0 input, early block #1-#{early}); top {TOP} by E(t)")
        for r in rows:
            flags = ", ".join(f for f, on in (("squeeze-and-excitation", r["squeeze_excitation"]),
                                              ("smooth activation", r["smooth_activation"]),
                                              ("early block", r["in_early_block"])) if on) or "-"
            print(f"  {r['rank']}. E {r['E_db']:+.2f} dB ({r['E_ci95'][0]:+.2f} to {r['E_ci95'][1]:+.2f})  "
                  f"#{r['index']} ({r['depth_pct']:.0f}% depth)  {r['operation']} "
                  f"(FP32 export: {r['fp32_export_operation']})  "
                  f"module {r['module']} [{r['module_class']}]  block {r['block']}  flags: {flags}")

        spec = MODELS[model]
        out = make_measurement(
            "diagnostic", {"name": model, "weights": str(spec["weights"]), "licence": spec["licence"]},
            PRECISION,
            RUNTIME, "laptop", machine, record["data"], record["condition"],
            {"top_E_db": metric(rows[0]["E_db"], rows[0]["E_ci95"]),
             "top5_squeeze_excitation": metric(sum(r["squeeze_excitation"] for r in rows)),
             "top5_smooth_activation": metric(sum(r["smooth_activation"] for r in rows)),
             "top5_in_early_block": metric(sum(r["in_early_block"] for r in rows))},
            {"script": "scripts/36_m2_top_tensors.py", "exploratory": "after the M2 verdicts; not judged",
             "top": TOP, "bootstrap_resamples": N_RESAMPLES, "seed": SEED, "early_block": f"#1-#{early}",
             "definitions": {"squeeze_excitation": SE_CLASS, "smooth_classes": sorted(SMOOTH_CLASSES),
                             "smooth_ops": sorted(SMOOTH_OPS), "block_classes": sorted(BLOCK_CLASSES)}},
            None, [{"file": path.as_posix(), "arrays_sha256": record["arrays"]["sha256"]},
                   {"file": f"models/{model}_{PRECISION}.onnx",
                    "sha256": sha256_of(f"models/{model}_{PRECISION}.onnx")},
                   {"file": f"models/{model}_fp32.onnx", "sha256": sha256_of(f"models/{model}_fp32.onnx")}])
        out["raw"] = {"top_tensors": rows}
        save_measurement(out, Path("results/checks") / f"m2_top_tensors_{model}_{corruption}.json")
print("\nSaved results/checks/m2_top_tensors_*.json")
