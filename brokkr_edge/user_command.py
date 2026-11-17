"""`brokkr-edge test --config`: test a user's own model from start to label (step 3; docs/user_models.md).

The steps, in order, each built and tested on its own:
1. the checks, which write the run plan (brokkr_edge.user_plan; slice 1), and the Brokkr files the label will
   need (the seed-sensitivity record when random damage runs), so nothing runs that could not end in a label;
2. --out may hold this run, which is then carried on, but never a different one (decision 13);
3. the image parts, remade from the folders and checked against the plan's fingerprints;
4. Brokkr's INT8 build, unless the submitter supplied one (brokkr_edge.user_build);
5. the conditions, for FP32 and for the INT8 build when it is usable (brokkr_edge.user_run);
6. the label: label.json, label.md and label.html in --out (brokkr_edge.user_label).

The whole command runs with Python-level network connections blocked and counted, and ONNX Runtime telemetry
switched off (brokkr_edge.no_network; plan test 13): the model and the images never leave the machine.
"""

from pathlib import Path

import onnxruntime as ort

from brokkr_edge.no_network import no_network
from brokkr_edge.user_build import INT8_FILE, build_int8
from brokkr_edge.user_label import check_label_sources, make_label, write_label
from brokkr_edge.user_plan import check_user_inputs, rebuild_parts, same_run_problems, write_plan
from brokkr_edge.user_run import FP32, INT8_BROKKR, INT8_SUPPLIED, run_conditions
from brokkr_edge.user_settings import UserInputError, load_settings


def run_user_test(settings_path, images, calib_images, out, log=print, seed_sensitivity=None) -> dict:
    """Every step for one user model; returns {"plan", "label", "paths", "network_attempts"}.

    Raises UserInputError (the inputs cannot be used) or brokkr_edge.user_label.LabelError (the run's files
    cannot make a valid label, or a Brokkr file the label needs is missing: found before anything is
    written). seed_sensitivity: as for user_label.random_condition_sentence."""
    out = Path(out)
    with no_network() as attempts:
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
        plan = check_user_inputs(settings_path, images, calib_images, log=log)
        check_label_sources(plan, seed_sensitivity)  # seconds now, not after the whole run
        problems = same_run_problems(plan, out)
        if problems:
            raise UserInputError(problems)
        write_plan(plan, out)
        settings, base = load_settings(Path(settings_path))
        fp32 = base / settings["fp32_model"]
        parts = rebuild_parts(plan, images, calib_images)
        builds = {FP32: fp32}
        if "shrunk" in plan["models"]:
            if plan["models"]["shrunk"]["status"] == "usable":
                builds[INT8_SUPPLIED] = base / settings["shrunk_model"]["file"]
        elif build_int8(plan, fp32, parts, out, log=log)["status"] == "usable":
            builds[INT8_BROKKR] = out / INT8_FILE
        run_conditions(plan, builds, parts, out, log=log)
        label = make_label(plan, out, seed_sensitivity)
        paths = write_label(label, out)
    if attempts:  # an attempt some library caught and hid: still not allowed
        raise RuntimeError(f"network connections were attempted (and blocked): {attempts}")
    return {"plan": plan, "label": label, "paths": paths, "network_attempts": len(attempts)}
