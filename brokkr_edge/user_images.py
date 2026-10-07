"""A user's image folders: listing them, and splitting them into parts (docs/user_models.md, section 2).

- **Listing.** Image files (by suffix) are listed in canonical order: the sorted relative paths, with "/".
  Each image's position in that order is its damage seed (the study's rule: seed = dataset position).
  Unreadable images and exact duplicates (same SHA-256) stop the run; other files are counted, never
  silently dropped. A folder's fingerprint is the SHA-256 of its sorted (relative path, SHA-256) list, so
  records can name the folder without naming any file.
- **Parts.** INT8 calibration (512 images: from an unlabelled --calib-images folder, else from the labelled
  images; none when the user supplies the shrunk build), conformal calibration and test. Brokkr's split is
  stratified by class with fixed seeds; the user's own split has calibration/ and test/ folders. No image is
  in two parts.
- **Floors and warnings.** Below a part's floor the run stops; below its recommended size the label warns.

plan_split() works on lists only (no files), so its rules are tested on made-up lists.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from brokkr_edge.results import sha256_of
from brokkr_edge.user_settings import UserInputError

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
OWN_SPLIT_PARTS = ("calibration", "test")
INT8_CALIBRATION_IMAGES = 512  # the study's recipe, unchanged
FLOORS = {"conformal_calibration": 200, "test": 200}  # H, 6 October 2026
RECOMMENDED = {"conformal_calibration": 1000, "test": 2000}
CAPS = {"conformal_calibration": 5000, "test": 10000}  # the study's sizes
FEW_TEST_IMAGES_PER_CLASS = 20
SEEDS = {"int8_calibration": 1, "parts": 2, "caps": 3}


def _image_files(folder: Path) -> tuple:
    """(image files, number of other files) under `folder`, at any depth."""
    files = [p for p in folder.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    return images, len(files) - len(images)


def _entry(path: Path, root: Path, label, problems: list) -> dict:
    rel = path.relative_to(root).as_posix()
    try:
        with Image.open(path) as image:
            image.load()  # decodes the whole picture, so a damaged file is found now, not mid-run
    except Exception as e:  # Pillow raises many kinds of errors for a broken file
        problems.append(f"unreadable image: {rel} ({type(e).__name__})")
    return {"rel": rel, "sha256": sha256_of(path), "label": label}


def _duplicates(entries: list, what: str) -> list:
    by_hash = {}
    for e in entries:
        by_hash.setdefault(e["sha256"], []).append(e["rel"])
    return [f"{what}: identical files (one image under several names): {', '.join(names)}"
            for names in by_hash.values() if len(names) > 1]


def scan_labelled(folder, classes: list, split: str) -> dict:
    """List the labelled folder: {"images": [{"rel", "sha256", "label", "part"}...], "other_files": n}.

    split "brokkr": folder/<class>/...;
    split "own": folder/calibration/<class>/... and folder/test/<class>/...
    Raises UserInputError listing every problem.
    """
    root = Path(folder)
    if not root.is_dir():
        raise UserInputError([f"images folder not found: {folder}"])
    problems, entries, other = [], [], 0
    parts = OWN_SPLIT_PARTS if split == "own" else (None,)
    if split == "own":
        found = sorted(p.name for p in root.iterdir() if p.is_dir())
        if found != sorted(OWN_SPLIT_PARTS):
            raise UserInputError(["own split: the images folder must hold exactly the folders calibration/ "
                                  f"and test/ (found: {found})"])
    loose = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    other += sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES)
    if loose:
        problems.append(f"images outside a class folder: {', '.join(sorted(p.name for p in loose))}")
    for part in parts:
        base = root / part if part else root
        where = f"{part}/" if part else "the images folder"
        folders = sorted(p.name for p in base.iterdir() if p.is_dir())
        missing = sorted(set(classes) - set(folders))
        extra = sorted(set(folders) - set(classes))
        if missing:
            problems.append(f"{where}: no folder for the classes {missing}")
        if extra:
            problems.append(f"{where}: folders that are not in the settings file's classes: {extra}")
        if part:
            loose = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
            other += sum(1 for p in base.iterdir() if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES)
            if loose:
                names = ", ".join(sorted(p.name for p in loose))
                problems.append(f"{where}: images outside a class folder: {names}")
        for label, name in enumerate(classes):
            if not (base / name).is_dir():
                continue
            images, n_other = _image_files(base / name)
            other += n_other
            for path in images:
                entry = _entry(path, root, label, problems)
                entry["part"] = part
                entries.append(entry)
    entries.sort(key=lambda e: e["rel"])
    problems += _duplicates(entries, "labelled images")
    if problems:
        raise UserInputError(problems)
    return {"images": entries, "other_files": other}


def scan_unlabelled(folder) -> dict:
    """List an unlabelled calibration folder: {"images": [{"rel", "sha256", "label": None}...],
    "other_files"}."""
    root = Path(folder)
    if not root.is_dir():
        raise UserInputError([f"calibration images folder not found: {folder}"])
    problems = []
    images, other = _image_files(root)
    entries = sorted((_entry(p, root, None, problems) for p in images), key=lambda e: e["rel"])
    problems += _duplicates(entries, "calibration images")
    if problems:
        raise UserInputError(problems)
    return {"images": entries, "other_files": other}


def overlap(labelled: list, unlabelled: list) -> list:
    """Problems for calibration images that are also in the labelled folder (same SHA-256)."""
    in_labelled = {e["sha256"]: e["rel"] for e in labelled}
    return [f"calibration image {e['rel']} is also in the labelled folder as {in_labelled[e['sha256']]}"
            for e in unlabelled if e["sha256"] in in_labelled]


def fingerprint(entries: list) -> str:
    """SHA-256 of the sorted (relative path, SHA-256) list: names the folder without naming a file."""
    pairs = sorted([e["rel"], e["sha256"]] for e in entries)
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode("utf-8")).hexdigest()


def _by_class(indices, labels) -> dict:
    groups = {}
    for i in indices:
        groups.setdefault(labels[i], []).append(i)
    return groups


def stratified_pick(indices: list, labels: list, n: int, rng: np.random.Generator) -> list:
    """Pick n of `indices`, each class keeping its share (largest remainder), chosen at random in each class.

    Quotas: floor(n x class size / total); the images left over go one each to the classes with the largest
    remainders (ties: lower class number). Classes are visited in class order, so the result depends only on
    the inputs and the generator's seed.
    """
    groups = _by_class(sorted(indices), labels)
    total = len(indices)
    quota = {c: n * len(g) // total for c, g in groups.items()}
    left = n - sum(quota.values())
    for c in sorted(groups, key=lambda c: (-(n * len(groups[c]) % total), c))[:left]:
        quota[c] += 1
    picked = []
    for c in sorted(groups):
        members = groups[c]
        picked += [members[i] for i in rng.permutation(len(members))[: quota[c]]]
    return sorted(picked)


def plan_split(labels: list, parts: list, n_classes: int, split: str, n_calib_images: int | None,
               brokkr_builds_int8: bool,
               floors=FLOORS, recommended=RECOMMENDED, caps=CAPS, int8_n=INT8_CALIBRATION_IMAGES) -> dict:
    """Split the labelled images (given by their class labels and, in an own split, their part) into parts.

    n_calib_images: how many images the unlabelled --calib-images folder holds (None: no such folder).
    Returns {"int8_calibration": {"source", "indices"} or None, "conformal_calibration": [...], "test": [...],
    "unused": {...}, "warnings": [...]}, every index list sorted (canonical order). The "calib-images" source
    indexes the unlabelled folder; every other list indexes the labelled one. Raises UserInputError.
    """
    problems = []
    everything = list(range(len(labels)))
    if split == "own":
        calibration_pool = [i for i in everything if parts[i] == "calibration"]
        test = [i for i in everything if parts[i] == "test"]
    else:
        calibration_pool, test = everything, None

    int8 = None
    if not brokkr_builds_int8:
        if n_calib_images is not None:
            problems.append("--calib-images is given, but the shrunk build is supplied, so no INT8 build "
                            "is made "
                            "and those images would not be used; leave it out")
    elif n_calib_images is not None:
        if n_calib_images < int8_n:
            problems.append(f"--calib-images holds {n_calib_images} images; INT8 calibration needs at least "
                            f"{int8_n}")
        elif n_calib_images == int8_n:
            int8 = {"source": "calib-images", "indices": list(range(int8_n))}
        else:
            rng = np.random.default_rng(SEEDS["int8_calibration"])
            int8 = {"source": "calib-images",
                    "indices": sorted(int(i) for i in rng.choice(n_calib_images, int8_n, replace=False))}
    else:
        where = "calibration/" if split == "own" else "the labelled images"
        if len(calibration_pool) < int8_n:
            problems.append(f"INT8 calibration needs {int8_n} images and {where} hold only "
                            f"{len(calibration_pool)}; add images or give --calib-images")
        else:
            rng = np.random.default_rng(SEEDS["int8_calibration"])
            int8 = {"source": "labelled", "indices": stratified_pick(calibration_pool, labels, int8_n, rng)}
    if problems:
        raise UserInputError(problems)

    taken = set(int8["indices"]) if int8 and int8["source"] == "labelled" else set()
    rest = [i for i in calibration_pool if i not in taken]
    if split == "own":
        conformal = rest
    else:  # per class, a third (rounded down) to conformal calibration, the rest to test
        rng = np.random.default_rng(SEEDS["parts"])
        conformal, test = [], []
        for _, members in sorted(_by_class(rest, labels).items()):
            order = [members[i] for i in rng.permutation(len(members))]
            conformal += order[: len(members) // 3]
            test += order[len(members) // 3:]
        conformal, test = sorted(conformal), sorted(test)

    unused = {}
    rng = np.random.default_rng(SEEDS["caps"])
    chosen = {}
    for name, members in (("conformal_calibration", conformal), ("test", test)):
        if len(members) > caps[name]:
            chosen[name] = stratified_pick(members, labels, caps[name], rng)
            unused[name] = len(members) - caps[name]
        else:
            chosen[name] = sorted(members)

    warnings = []
    for name in ("conformal_calibration", "test"):
        present = {labels[i] for i in chosen[name]}
        absent = [c for c in range(n_classes) if c not in present]
        if absent:
            problems.append(f"{name}: no image of the classes numbered {absent}; every class needs at least "
                            "one "
                            "image in each part")
        if len(chosen[name]) < floors[name]:
            problems.append(f"{name}: {len(chosen[name])} images; the floor is {floors[name]}")
        elif len(chosen[name]) < recommended[name]:
            warnings.append({"kind": "few images", "part": name, "n_items": len(chosen[name]),
                             "recommended": recommended[name]})
    if problems:
        raise UserInputError(problems)
    counts = np.bincount([labels[i] for i in chosen["test"]], minlength=n_classes)
    few = [int(c) for c in np.flatnonzero(counts < FEW_TEST_IMAGES_PER_CLASS)]
    if few:
        warnings.append({"kind": "classes with few test images", "classes": few,
                         "below": FEW_TEST_IMAGES_PER_CLASS})
    return {"int8_calibration": int8, **chosen, "unused": unused, "warnings": warnings}
