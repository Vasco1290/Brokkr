"""Run a user's model under Brokkr's conditions and save a checked record of each run (step 3;
docs/user_models.md, plan of 10 October 2026, section a; H's decisions of 10 October 2026, 3, 11 and 13).

Two parts are run: the clean conformal-calibration part (its scores set the prediction sets later) and the
test part under every condition in the run plan (13, or 5 for a model below 32 x 32). Each build (FP32, and
INT8 when it is usable) sees the same pictures:

- each image is decoded once, resized and centre-cropped at the model's own size (user_checks.cropped, the
  code slice 1 uses), into a temporary memory-mapped file in --out (decision 11). It is deleted when the run
  ends, also after a failure or a stop; a leftover from a crash is removed when the next run starts. A
  disk-space check comes first;
- each damaged picture is made from it with the content-v1 seed of that image and damage type
  (brokkr_edge.seeds), so adding an image to the folder changes no other image's damage;
- then normalised as the user stated (user_checks.normalised) and run with the model's own input name, 32 at
  a time (1 for a model with a fixed batch of 1), with 8 threads and thread spinning off (decision 3);
- a model that outputs probabilities has them turned into logits (user_checks.as_logits) before they are
  saved, so every reliability number worked out later starts from scores whose softmax is the model's own
  probabilities.

Each run is saved as a schema-2 accuracy record with the source "user-submitted" in --out/records/, its
scores beside it, stored by each image's position in its part (canonical order; no file name, no path). A
complete record is never run again, so a stopped run carries on where it stopped.
"""

import contextlib
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr_edge.accuracy import accuracy_from_logits
from brokkr_edge.benchmark import make_session
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.results import sha256_of, written_atomically
from brokkr_edge.schema import USER_SOURCE, condition_label, make_measurement, metric, save_measurement
from brokkr_edge.seeds import content_seed, damaged_batch_seeded
from brokkr_edge.sweep import duration
from brokkr_edge.test_run import complete, keep_awake
from brokkr_edge.user_checks import as_logits, cropped, normalised
from brokkr_edge.user_settings import UserInputError

THREADS = 8  # decision 3: the study's setting, recorded on the label
DECODED = ".brokkr-decoded-{part}.tmp.npy"  # a clearly temporary name (decision 13)
SPARE_BYTES = 10**9  # free space kept beyond the decoded pictures
SEED_SCHEME = "content-v1"
INSTALL_IMAGENET_C = 'pip install -e ".[imagenet-c]" and pip install --no-deps imagecorruptions==1.1.2'


def remove_decoded(out) -> None:
    """Delete every temporary decoded-picture file in --out (a finished, stopped or crashed run's)."""
    for path in Path(out).glob(DECODED.format(part="*")):
        with contextlib.suppress(OSError):  # still open after a crash: removed when the next run starts
            path.unlink(missing_ok=True)


def _release(pictures) -> None:
    """Close a decoded-picture file now, so it can be deleted (Windows refuses to delete an open file)."""
    mapped = getattr(pictures, "_mmap", None)  # numpy's handle on the open file
    if mapped is not None:
        mapped.close()


def check_ready(plan: dict, parts: dict, out) -> None:
    """Stop before anything runs if the ImageNet-C code is missing or --out lacks room for the pictures."""
    problems = []
    if any(c["suite"] == "imagenet-c" for c in plan["conditions"]):
        try:
            import brokkr_edge.imagenet_c  # noqa: F401 (optional packages: only checked here)
        except ImportError:
            problems.append(f"the ImageNet-C conditions need the optional packages: {INSTALL_IMAGENET_C}")
    crop = plan["preprocessing"]["crop"]
    needed = sum(len(parts[p]) for p in ("conformal_calibration", "test")) * crop * crop * 3 + SPARE_BYTES
    free = shutil.disk_usage(Path(out)).free
    if free < needed:
        problems.append(f"--out needs about {needed / 1e9:.1f} GB free for the decoded pictures (1 GB of it "
                        f"spare); it has {free / 1e9:.1f} GB")
    if problems:
        raise UserInputError(problems)


def decode(entries: list, prep: dict, path: Path) -> np.ndarray:
    """Every image of a part, resized and cropped once, in a memory-mapped uint8 file (read back
    read-only)."""
    crop = prep["crop"]
    shape = (len(entries), crop, crop, 3)
    pictures = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    for i, e in enumerate(entries):
        pictures[i] = cropped(e["path"], prep)
    pictures.flush()
    del pictures
    return np.load(path, mmap_mode="r")


def damaged(pictures, entries: list, indices: list, cond: dict) -> np.ndarray:
    """uint8 pictures[indices] under the condition, each with its content-v1 seed (a copy)."""
    if cond["corruption"] == "clean":
        return np.array(pictures[indices], dtype=np.uint8)
    seeds = [content_seed(entries[i]["sha256"], cond["suite"], cond["corruption"]) for i in indices]
    return damaged_batch_seeded(pictures, indices, cond["suite"], cond["corruption"], cond["severity"], seeds)


def record_path(out, precision: str, part: str, cond: dict) -> Path:
    name = f"{precision}_{part}_{cond['suite'] or 'none'}_{cond['corruption']}_s{cond['severity']}.json"
    return Path(out) / "records" / name


def _save(path: Path, logits: np.ndarray, labels: np.ndarray, precision: str, part: str, cond: dict,
          plan: dict, runtime: dict, build_sha256: str, batch: int) -> None:
    with written_atomically(path.with_suffix(".npz")) as tmp, tmp.open("wb") as f:
        np.savez_compressed(f, logits=logits, labels=labels, positions=np.arange(len(labels)))
    acc = accuracy_from_logits(logits, labels)["metrics"]
    licences = plan["licences"]
    model = {"name": plan["name"], "weights": f"user model {plan['model_id']}",
             "licence": licences["model_weights"]}
    data = {"dataset": f"user/{plan['name']}", "split": part, "n_images": int(len(labels)),
            "licence": licences["images"]}
    metrics = {"top1": metric(acc["top1"], acc["top1_ci95"]), "top5": metric(acc["top5"], acc["top5_ci95"]),
               "top1_tied_images": metric(acc["top1_tied_images"])}
    settings = {"script": "brokkr-edge test --config", "batch_size": batch,
                "preprocessing": plan["preprocessing"],
                "seed_scheme": SEED_SCHEME, "damage_seed": "content-v1 (brokkr_edge.seeds.content_seed)",
                "outputs": plan["checks"]["outputs"]["declared"],
                "scores_saved": "logits (user_checks.as_logits)",
                "positions": "each image's place in its part, canonical order", "bootstrap_resamples": 1000,
                "seed": 0, "top1_range_over_tie_breaks": acc["top1_range_over_tie_breaks"]}
    arrays = {"file": path.with_suffix(".npz").name, "sha256": sha256_of(path.with_suffix(".npz"))}
    record = make_measurement("accuracy", model, precision, runtime, plan["device_kind"],
                              machine_fingerprint(), data, cond, metrics, settings, arrays,
                              [{"build": precision, "sha256": build_sha256}], source=USER_SOURCE)
    save_measurement(record, path)


def run_conditions(plan: dict, builds: dict, parts: dict, out, log=print, threads: int = THREADS) -> list:
    """Run every build ({precision: model file}) on the clean conformal-calibration part and on the test part
    under each of the plan's conditions; return the record paths (complete ones are kept, not run again)."""
    out = Path(out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    remove_decoded(out)  # a crashed run's leftover
    prep, batch = plan["preprocessing"], plan["runtime"]["batch"]
    declared, name = plan["checks"]["outputs"]["declared"], plan["models"]["fp32"]["input"]["name"]
    clean = [c for c in plan["conditions"] if c["corruption"] == "clean"]
    jobs = [("conformal_calibration", clean), ("test", plan["conditions"])]
    unfinished = [1 for part, conds in jobs for c in conds for p in builds
                  if not complete(record_path(out, p, part, c))]
    sessions = {}
    if unfinished:
        check_ready(plan, parts, out)
        sessions = {p: make_session(path, threads, spinning=False) for p, path in builds.items()}
    shas = {p: sha256_of(path) for p, path in builds.items()}
    runtime = {"name": "onnxruntime", "version": ort.__version__,
               "execution_provider": "CPUExecutionProvider", "threads": threads, "spinning": "off"}
    log(f"conditions: {len(plan['conditions'])}; builds {', '.join(builds)}; {threads} threads, spinning "
        f"off, batch {batch}")
    paths = []
    keep_awake(True)
    try:
        for part, conditions in jobs:
            entries = parts[part]
            labels = np.asarray([e["label"] for e in entries], dtype=np.int64)
            pictures = None
            try:
                for cond in conditions:
                    here = {p: record_path(out, p, part, cond) for p in builds}
                    paths += list(here.values())
                    todo = [p for p in builds if not complete(here[p])]
                    if not todo:
                        log(f"skip (complete): {part} {condition_label(cond)}")
                        continue
                    if pictures is None:
                        pictures = decode(entries, prep, out / DECODED.format(part=part))
                    started = time.time()
                    scores = {p: [] for p in todo}
                    for start in range(0, len(entries), batch):
                        idx = list(range(start, min(start + batch, len(entries))))
                        x = np.stack([normalised(pic, prep) for pic in damaged(pictures, entries, idx, cond)])
                        for p in todo:
                            scores[p].append(sessions[p].run(None, {name: x})[0])
                    for p in todo:
                        logits = as_logits(np.concatenate(scores[p]).astype(np.float32), declared)
                        _save(here[p], logits, labels, p, part, cond, plan, runtime, shas[p], batch)
                    took = duration(time.time() - started)
                    log(f"done: {part} {condition_label(cond)}, {len(todo)} builds, {took}")
            finally:
                _release(pictures)
    finally:
        keep_awake(False)
        remove_decoded(out)
    return paths
