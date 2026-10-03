"""M2: in which layers does INT8 add more rounding error under darkness and fog than on clean images?

Usage:  python scripts/35_m2_rounding_error.py
        python scripts/35_m2_rounding_error.py --dry-run --out <folder>   (a tool check, never a result:
        3 models, 16 tuning images at positions 628-643, outside M2's images; everything goes to <folder>)
Needs:  models/<model>_fp32.onnx, _int8_percentile99.99.onnx and _int8.onnx (MinMax) with build records,
        data/imagenet-1k/, results/checks/mobilenet_v3_small_int8_layer_divergence.json (scripts/26)
Writes: results/m2/<model>_<precision>_m2.npz (per-image numbers) and one schema-2 diagnostic record per
        damaged condition, results/m2/mobilenet_v3_small_m2_check.json, and the verdicts in
        results/final/m2_verdicts.json

Design and verdict rules: docs/hypotheses_stage4.md, "M2" (committed at 54c476d, before any 4.1 result)
and the dated M2 note of 27 September 2026 (written before this script ran). Measures: brokkr_edge/m2.py.
- Models: the 8 of the M2 section. Precisions: Percentile 99.99 INT8 (8 models) and default INT8
  (MinMax; only builds whose record says "usable": 7 models, EfficientNet-B0's failed).
- Images: tuning positions 500-627 (128 images, split order), each model's own preprocessing; clean,
  darkness (Brokkr) s5, fog (Brokkr) s3; damage seed = dataset position (brokkr_edge.sweep.damaged_batch).
- Tensors: each activation QuantizeLinear of the INT8 model, in graph order, matched by name to the FP32
  model after ONNX Runtime's preparation step (quant_pre_process, with the build's recorded settings).
  Values are read with ONNX Runtime's own tool (qdq_loss_debug.modify_model_output_intermediate_tensors),
  as in scripts/26, in batches of 8 images, 4 threads.
- Check first: this script's cumulative SQNR (INT8 model vs FP32 model), pooled over images as ONNX
  Runtime pools them, must reproduce scripts/26's saved MobileNetV3-Small values (32 tuning images) to
  0.01 dB at every tensor. If not, it stops before measuring anything else.
- Before judging: if any (image, tensor) has a zero-size signal or zero local error (ONNX Runtime's
  epsilon guard would set its SQNR), the verdicts are not computed and a dated note must decide.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quant_pre_process
from onnxruntime.quantization.qdq_loss_debug import modify_model_output_intermediate_tensors

from brokkr_edge import m2
from brokkr_edge.accuracy import open_image, resize_and_crop
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.export import MODELS, file_info, preprocessing
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.judge import plain
from brokkr_edge.results import make_record, save_record, sha256_of, written_atomically
from brokkr_edge.schema import (
    check_build_record,
    condition,
    condition_label,
    make_measurement,
    metric,
    save_measurement,
)
from brokkr_edge.sweep import damaged_batch, normalised

THREADS, BATCH, SAVED = 4, 8, "_ReshapedSavedOutput"
M2_MODELS = ["mobilenet_v2", "efficientnet_b0", "shufflenet_v2_x1_0", "mnasnet1_0", "regnet_y_400mf",
             "resnet18", "resnet50", "convnext_tiny"]
PRECISIONS = {"percentile": "int8_percentile99.99", "default": "int8"}
CLEAN, DARK, FOG = condition(), condition("darkness", "brokkr", 5), condition("fog", "brokkr", 3)
DAMAGED = {"darkness": DARK, "fog": FOG}
CHECK_MODEL = "mobilenet_v3_small"
CHECK_RECORD = Path("results/checks/mobilenet_v3_small_int8_layer_divergence.json")
DATASET = "imagenet-1k-val"
DRY_RUN_MODELS = ["resnet18", "efficientnet_b0", "convnext_tiny"]  # no MinMax; skip_symbolic_shape; plain

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--out", default="results/m2")
args = parser.parse_args()
if args.dry_run and args.out == "results/m2":
    sys.exit("a dry run needs its own --out folder")
OUT = Path(args.out)
VERDICT_PATH = OUT / "m2_verdicts_DRY_RUN.json" if args.dry_run else Path("results/final/m2_verdicts.json")
RUNTIME = {"name": "onnxruntime", "version": ort.__version__, "execution_provider": "CPUExecutionProvider",
           "threads": THREADS}

files = parquet_files(DATASET)
tuning = make_splits(count_images(files))["tuning"]
machine = machine_fingerprint()
options = ort.SessionOptions()
options.intra_op_num_threads = THREADS


def build(model: str, precision: str) -> dict:
    return json.loads((Path("models") / f"{model}_{precision}.json").read_text(encoding="utf-8"))


def usable(model: str, precision: str) -> bool:
    if not (Path("models") / f"{model}_{precision}.json").exists():
        return False
    status, problems = check_build_record(build(model, precision), set())
    return status == "usable" and not problems


def model_field(model: str) -> dict:
    spec = MODELS[model]
    return {"name": model, "weights": str(spec["weights"]), "licence": spec["licence"]}


def data_field(n: int) -> dict:
    return {"dataset": DATASET, "split": "tuning", "n_images": n, "licence": DATASETS[DATASET]["licence"]}


def augmented_session(path: Path, tmp: Path, name: str) -> tuple:
    """A session on a copy of the model that also outputs every float tensor (ONNX Runtime's tool)."""
    out = tmp / f"{name}_aug.onnx"
    modify_model_output_intermediate_tensors(str(path), str(out))
    session = ort.InferenceSession(str(out), options, providers=["CPUExecutionProvider"])
    return session, {o.name[:-len(SAVED)] for o in session.get_outputs() if o.name.endswith(SAVED)}


def exposed_session(path: Path, tmp: Path, name: str, names: list) -> ort.InferenceSession:
    """A session on a copy of the model that also outputs the given float tensors, added directly.

    Only a fallback for INT8 models whose graph ONNX Runtime's augmentation tool makes invalid
    (ConvNeXt-Tiny: it tries to save an int32 bias zero-point as a float tensor). It lets ONNX Runtime
    fuse differently, so the cumulative measure (reported, never judged) can differ slightly from the
    tool's; the judged local measure does not run the INT8 model (dated note of 27 September 2026)."""
    model = onnx.load(str(path))
    existing = {o.name for o in model.graph.output}
    for n in names:
        if n not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None))
    out = tmp / f"{name}_exposed.onnx"
    onnx.save(model, str(out))
    return ort.InferenceSession(str(out), options, providers=["CPUExecutionProvider"])


def run(session, names: list, images: np.ndarray) -> dict:
    """The requested saved tensors for a batch, each as (images, values)."""
    saved = {o.name for o in session.get_outputs()}
    values = session.run([n + SAVED if n + SAVED in saved else n for n in names], {"images": images})
    return {n: v.reshape(images.shape[0], -1) for n, v in zip(names, values, strict=True)}


def measure(model: str, precisions: list, positions: list, conditions: list) -> dict:
    """Per-image local and cumulative squared norms, and M2b's R, for each precision and condition."""
    skip = {build(model, p)["settings"].get("skip_symbolic_shape", False) for p in precisions}
    if len(skip) != 1:
        sys.exit(f"FAIL: {model}'s builds differ in skip_symbolic_shape; the FP32 side would be ambiguous")
    prep = preprocessing(model)
    crops = np.stack([resize_and_crop(open_image(b), prep["resize"], prep["crop"], prep["interpolation"])
                      for b, _ in read_parquet_images(files, positions)])
    found = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        prepared = tmp / "prepared.onnx"
        quant_pre_process(str(Path("models") / f"{model}_fp32.onnx"), str(prepared),
                          skip_symbolic_shape=skip.pop())
        fp32, fp32_names = augmented_session(prepared, tmp, "fp32")
        plan = {}
        for p in precisions:
            path = Path("models") / f"{model}_{p}.onnx"
            quantizers = m2.activation_quantizers(onnx.load(str(path)))
            try:
                session, int8_names = augmented_session(path, tmp, p)
                exposure = "onnxruntime qdq_loss_debug tool (as scripts/26)"
            except ort.capi.onnxruntime_pybind11_state.InvalidGraph as error:
                int8_names = {q["dequantized"] for q in quantizers if q["dequantized"]}
                session = exposed_session(path, tmp, p, sorted(int8_names))
                exposure = f"direct outputs (the tool made the graph invalid: {str(error)[:120]})"
            matched = [q for q in quantizers if q["tensor"] in fp32_names and q["dequantized"] in int8_names]
            matched_names = {q["tensor"] for q in matched}
            if not matched or matched[0]["tensor"] != "images":
                sys.exit(f"FAIL: {model} {p}: tensor #0 is not the input image")
            plan[p] = {"session": session, "q": matched, "path": path, "exposure": exposure,
                       "unmatched": [q["tensor"] for q in quantizers if q["tensor"] not in matched_names]}
        needed = sorted({q["tensor"] for info in plan.values() for q in info["q"]})
        for p, info in plan.items():
            found[p] = {"tensors": [q["tensor"] for q in info["q"]], "unmatched": info["unmatched"],
                        "int8_exposure": info["exposure"],
                        "model_sha256": file_info(info["path"])["sha256"], "conditions": {}}
        for cond in conditions:
            parts = {p: {"local": [], "cumulative": [], "R": []} for p in plan}
            for start in range(0, len(positions), BATCH):
                idx = list(range(start, min(start + BATCH, len(positions))))
                images = normalised(damaged_batch(crops, positions, idx, cond["suite"], cond["corruption"],
                                                  cond["severity"]))
                x = run(fp32, needed, images)
                for p, info in plan.items():
                    q8 = run(info["session"], [q["dequantized"] for q in info["q"]], images)
                    parts[p]["local"].append(np.array(
                        [m2.local_rounding(x[q["tensor"]], q["scale"], q["zero_point"]) for q in info["q"]]))
                    parts[p]["cumulative"].append(np.array(
                        [m2.squared_norms(x[q["tensor"]], q8[q["dequantized"]]) for q in info["q"]]))
                    q0 = info["q"][0]
                    parts[p]["R"].append(m2.input_level_ratio(images, q0["scale"], q0["zero_point"]))
            for p in plan:  # local and cumulative: (tensors, 2 = [signal², error²], images); R: (images,)
                found[p]["conditions"][condition_label(cond)] = {
                    k: np.concatenate(v, axis=-1) for k, v in parts[p].items()}
    return found


def as_db(parts: np.ndarray) -> np.ndarray:
    """(tensors, 2, images) squared norms -> (images, tensors) SQNR in dB."""
    return m2.sqnr_db(parts[:, 0], parts[:, 1]).T


def interval(ci: list, digits: int = 2) -> str:
    return f"({ci[0]:+.{digits}f} to {ci[1]:+.{digits}f})"


def outcome(part: dict) -> str:
    return "supports" if part["supports"] else ("rejects" if part["rejects"] else "neither")


# ---- 1. The check before judging: reproduce scripts/26 for MobileNetV3-Small ----
t0 = time.time()
saved = json.loads(CHECK_RECORD.read_text(encoding="utf-8"))
check_positions = tuning[:saved["data"]["n_images"]]
check = measure(CHECK_MODEL, [PRECISIONS["percentile"]], check_positions, [CLEAN])[PRECISIONS["percentile"]]
cum = check["conditions"]["clean"]["cumulative"]
ours = {t: m2.pooled_sqnr_db(cum[i, 0], cum[i, 1]) for i, t in enumerate(check["tensors"])}
rows = saved["raw"]["tensors_in_graph_order"]
diffs = [abs(ours[r["tensor"]] - r["sqnr_db"]) if r["tensor"] in ours else float("inf") for r in rows]
check_ok = len(rows) == len(ours) and max(diffs) <= 0.01
print(f"Check: cumulative SQNR pooled over {len(check_positions)} tuning images vs scripts/26's saved values "
      f"for {CHECK_MODEL}: {len(rows)} saved tensors, {len(ours)} here; "
      f"largest difference {max(diffs):.4f} dB "
      f"(tolerance 0.01): {'PASS' if check_ok else 'FAIL'}  ({time.time() - t0:.0f} s)")
check_record = make_measurement(
    "diagnostic", model_field(CHECK_MODEL), PRECISIONS["percentile"], RUNTIME, "laptop", machine,
    data_field(len(check_positions)), CLEAN,
    {"tensors": metric(len(ours)), "largest_difference_db": metric(max(diffs)),
     "pass": metric(int(check_ok))},
    {"script": "scripts/35_m2_rounding_error.py",
     "purpose": "M2's check before judging: reproduce scripts/26",
     "tolerance_db": 0.01, "batch": BATCH},
    derived_from=[{"file": CHECK_RECORD.as_posix(), "sha256": sha256_of(CHECK_RECORD)}])
check_record["raw"] = {"per_tensor": [{"tensor": r["tensor"], "saved_db": r["sqnr_db"],
                                       "here_db": ours.get(r["tensor"])} for r in rows]}
save_measurement(check_record, OUT / f"{CHECK_MODEL}_m2_check.json")
if not check_ok:
    sys.exit("STOP: the check failed; M2 is not measured or judged (a dated note decides what to do)")

# ---- 2. Measure the 8 models ----
positions = tuning[628:644] if args.dry_run else tuning[500:628]
POSITION_RANGE = "628-643 (DRY RUN)" if args.dry_run else "500-627"
if args.dry_run:
    M2_MODELS = DRY_RUN_MODELS
if not all(usable(m, PRECISIONS["percentile"]) for m in M2_MODELS):
    sys.exit("FAIL: a Percentile build of an M2 model is not usable")
precisions_of = {m: [PRECISIONS["percentile"]] + ([PRECISIONS["default"]] if usable(m, "int8") else [])
                 for m in M2_MODELS}
print(f"Default INT8 (MinMax) usable for {sum(len(v) == 2 for v in precisions_of.values())} of "
      f"{len(M2_MODELS)} models; "
      f"not usable: {[m for m, v in precisions_of.items() if len(v) == 1]}")
results, degenerate = {}, 0
for model in M2_MODELS:
    t0 = time.time()
    results[model] = measure(model, precisions_of[model], positions, [CLEAN, DARK, FOG])
    counts = []
    for p, r in results[model].items():
        for c in r["conditions"].values():
            s2, n2 = c["local"][:, 0], c["local"][:, 1]
            degenerate += int(((s2 <= m2.EPS ** 2) | (n2 <= m2.EPS ** 2)).sum())
        counts.append(f"{p} {len(r['tensors'])} tensors ({len(r['unmatched'])} unmatched: {r['unmatched']}; "
                      f"INT8 read by {r['int8_exposure'][:30]})")
    print(f"  measured {model}: {', '.join(counts)}  ({time.time() - t0:.0f} s)")
print(f"(image, tensor) pairs with a zero signal or zero local error: {degenerate}")

# ---- 3. Summaries, per-image arrays and records ----
OUT.mkdir(parents=True, exist_ok=True)
summary = {}
for model, by_precision in results.items():
    for p, r in by_precision.items():
        local = {label: as_db(c["local"]) for label, c in r["conditions"].items()}
        cumulative = {label: as_db(c["cumulative"]) for label, c in r["conditions"].items()}
        ratio = {label: c["R"] for label, c in r["conditions"].items()}
        npz = OUT / f"{model}_{p}_m2.npz"
        short = {label: label.split(" ")[0] for label in local}  # clean, darkness, fog
        with written_atomically(npz) as tmp, tmp.open("wb") as f:
            np.savez_compressed(f, positions=np.array(positions), tensors=np.array(r["tensors"]),
                                **{f"local_db_{short[k]}": v for k, v in local.items()},
                                **{f"cumulative_db_{short[k]}": v for k, v in cumulative.items()},
                                **{f"R_{short[k]}": v for k, v in ratio.items()})
        for name, cond in DAMAGED.items():
            label = condition_label(cond)
            s = m2.extra_error(local["clean"], local[label])
            extra = {f"{round(f * 100)}pct": m2.extra_error(local["clean"], local[label], f)
                     for f in m2.REPORTED_FRACTIONS}
            rc = m2.ratio_change(ratio["clean"], ratio[label])
            cum_clean, cum_damaged = cumulative["clean"].mean(axis=0), cumulative[label].mean(axis=0)
            summary[(model, p, name)] = {"s": s, "extra": extra, "R": rc, "cum_clean": cum_clean,
                                         "cum_damaged": cum_damaged}
            metrics = {"E_early_db": metric(s["E_early"], s["E_early_ci95"]),
                       "E_rest_db": metric(s["E_rest"], s["E_rest_ci95"]),
                       "E_early_minus_rest_db": metric(s["early_minus_rest"], s["early_minus_rest_ci95"]),
                       "max_E_db": metric(s["max_E"]), "R_change": metric(rc["mean"], rc["ci95"]),
                       **{f"E_early_{k}_db": metric(v["E_early"], v["E_early_ci95"])
                          for k, v in extra.items()}}
            settings = {"script": "scripts/35_m2_rounding_error.py", "compared_with": "clean (same images)",
                        "positions": f"tuning split, positions {POSITION_RANGE}", "batch": BATCH,
                        "early_fraction": m2.EARLY_FRACTION, "early_tensors": s["early_tensors"],
                        "rest_tensors": s["rest_tensors"], "unmatched_tensors": r["unmatched"],
                        "int8_exposure_for_cumulative": r["int8_exposure"],
                        "bootstrap_resamples": m2.N_RESAMPLES, "seed": m2.SEED}
            record = make_measurement(
                "diagnostic", model_field(model), p, RUNTIME, "laptop", machine, data_field(len(positions)),
                cond, metrics, settings, {"file": npz.name, "sha256": sha256_of(npz)},
                [{"file": f"models/{model}_{p}.onnx", "sha256": r["model_sha256"]}])
            record["raw"] = {"tensors": r["tensors"], "E_db": s["E"].tolist(),
                             "cumulative_mean_db_clean": cum_clean.tolist(),
                             "cumulative_mean_db_damaged": cum_damaged.tolist()}
            save_measurement(record, OUT / f"{model}_{p}_{cond['corruption']}_m2.json")

if degenerate:
    sys.exit(f"NOT JUDGED: {degenerate} (image, tensor) pairs have a zero signal or zero local error; "
             "a dated note must decide how they are handled")

# ---- 4. Verdicts ----
verdicts = {}
for short_name, p in PRECISIONS.items():
    models = [m for m in M2_MODELS if p in precisions_of[m]]
    expected = 8 if short_name == "percentile" else 7
    if len(models) != expected and not args.dry_run:
        sys.exit(f"FAIL: {len(models)} models for {p}; the M2 section counts {expected}")
    for name in DAMAGED:
        parts = {m: m2.m2a_model(summary[(m, p, name)]["s"]) for m in models}
        verdicts[f"M2a {short_name} {name}"] = {**m2.verdict(parts, m2.rejects_needed(len(models))),
                                                "parts": parts, "precision": p, "condition": name}
    if short_name == "percentile":
        for name in DAMAGED:
            parts = {m: {k: summary[(m, p, name)]["R"][k] for k in ("supports", "rejects")} for m in models}
            verdicts[f"M2b percentile {name}"] = {**m2.verdict(parts, 5), "parts": parts, "precision": p,
                                                  "condition": name}

for key, v in verdicts.items():
    print(f"\n{key}: {v['verdict']}  (supporting {v['supporting']}, {v['supports_needed']} needed; "
          f"rejecting {v['rejecting']}, {v['rejects_needed']} needed; {v['models']} models)")
    for m, part in v["parts"].items():
        d = summary[(m, v["precision"], v["condition"])]
        s = d["s"]
        if key.startswith("M2a"):
            flag = " [no extra error]" if s["no_extra_error"] else ""
            print(f"  {m:<19} E_early {s['E_early']:+.2f} dB {interval(s['E_early_ci95'])}, "
                  f"E_rest {s['E_rest']:+.2f} {interval(s['E_rest_ci95'])}, "
                  f"early-rest {s['early_minus_rest']:+.2f} {interval(s['early_minus_rest_ci95'])}, "
                  f"max E {s['max_E']:+.2f}{flag}: {outcome(part)}")
        else:
            print(f"  {m:<19} mean R change {d['R']['mean']:+.4f} {interval(d['R']['ci95'], 4)}: "
                  f"{outcome(part)}")

print("\nReported, not judged: M2b for default INT8; E_early with 5% and 20% blocks; cumulative SQNR "
      "(mean over images) at the last tensor.")
for m in M2_MODELS:
    for short_name, p in PRECISIONS.items():
        if p not in precisions_of[m]:
            continue
        for name in DAMAGED:
            d = summary[(m, p, name)]
            e5, e20 = d["extra"]["5pct"], d["extra"]["20pct"]
            print(f"  {m:<19} {short_name:<10} {name:<8} E_early 5% {e5['E_early']:+.2f} "
                  f"{interval(e5['E_early_ci95'])}, 20% {e20['E_early']:+.2f} "
                  f"{interval(e20['E_early_ci95'])}; "
                  f"R change {d['R']['mean']:+.4f} {interval(d['R']['ci95'], 4)}; cumulative "
                  f"{d['cum_clean'][-1]:.2f} clean vs {d['cum_damaged'][-1]:.2f} dB damaged")

per_model = {key: {m: {"E_early": summary[(m, v["precision"], v["condition"])]["s"]["E_early"],
                       "E_early_ci95": summary[(m, v["precision"], v["condition"])]["s"]["E_early_ci95"],
                       "E_rest": summary[(m, v["precision"], v["condition"])]["s"]["E_rest"],
                       "early_minus_rest_ci95":
                           summary[(m, v["precision"], v["condition"])]["s"]["early_minus_rest_ci95"],
                       "R_change": summary[(m, v["precision"], v["condition"])]["R"]["mean"],
                       "R_change_ci95": summary[(m, v["precision"], v["condition"])]["R"]["ci95"]}
                   for m in v["parts"]} for key, v in verdicts.items()}
verdict_record = make_record("verdicts", "m2 (8 models)", "int8_percentile99.99 and int8", {
    "settings": {"script": "scripts/35_m2_rounding_error.py", "split": "tuning", "positions": POSITION_RANGE,
                 "n_images": len(positions), "models": M2_MODELS,
                 "rules": "docs/hypotheses_stage4.md, M2 and the dated M2 note of 27 September 2026",
                 "check": {"pass": check_ok, "largest_difference_db": max(diffs)}},
    "metrics": {"verdicts": verdicts},
    "raw": {"per_model": per_model},
}, machine)
print(f"\nSaved {save_record(plain(verdict_record), VERDICT_PATH)}")
