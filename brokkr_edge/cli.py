"""The `brokkr-edge` command (the name "brokkr" belongs to another project's command).

    brokkr-edge test --model mobilenet_v3_large
    brokkr-edge test --model mobilenet_v3_large --split tuning --limit 64 --out <folder>   (a dry run)
    brokkr-edge test --config my_model.json --images <folder> [--calib-images <folder>] --out <folder>
        (a user's own model, docs/user_models.md; today it runs the checks and writes the run plan only)

Also runnable without installing the command: python -m brokkr_edge.cli test ...
"""

import argparse
import sys

from brokkr_edge import __version__
from brokkr_edge.model_list import load_model_list


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="brokkr-edge", description="Stress-test shrunk vision models.")
    parser.add_argument("--version", action="version", version=f"brokkr-edge {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    test = commands.add_parser(
        "test", help="run the clean and damaged conditions on one model and save checked records")
    which = test.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", choices=list(load_model_list()), help="one of the Brokkr study's models")
    which.add_argument("--config", help="the settings file of your own model (docs/user_models.md)")
    test.add_argument("--images", help="with --config: the folder of your labelled images")
    test.add_argument("--calib-images", help="with --config: optional folder of at least 512 unlabelled "
                      "images for INT8 calibration")
    test.add_argument("--out", default=None,
                      help="folder for the records (default with --model: results/test_runs/<model>; "
                      "required with --config)")
    test.add_argument("--split", choices=["test", "tuning"], default="test",
                      help="test = the real run (plus clean conformal_calibration); tuning = dry runs only")
    test.add_argument("--limit", type=int, default=None, help="first N images only (dry runs)")

    args = parser.parse_args(argv)
    if args.command == "test" and args.config:
        return test_own_model(parser, args)
    if args.command == "test":
        if args.images or args.calib_images:
            parser.error("--images and --calib-images go with --config")
        from brokkr_edge.test_run import run_model  # imported here so --help needs no ONNX Runtime

        paths = run_model(args.model, args.out or f"results/test_runs/{args.model}", args.split, args.limit)
        print(f"{len(paths)} records in {paths[0].parent}")
    return 0


def test_own_model(parser, args) -> int:
    """Check a user's model and images and write the run plan; exit code 2 if the inputs cannot be used."""
    if not (args.images and args.out):
        parser.error("--config needs --images and --out")
    if args.split != "test" or args.limit is not None:
        parser.error("--split and --limit go with --model")
    from brokkr_edge.user_plan import check_out_folder, check_user_inputs, write_plan
    from brokkr_edge.user_settings import UserInputError

    try:
        problems = check_out_folder(args.out)
        if problems:
            raise UserInputError(problems)
        plan = check_user_inputs(args.config, args.images, args.calib_images)
    except UserInputError as e:
        print("Stopped: your inputs cannot be used yet. Every problem found:", file=sys.stderr)
        for problem in e.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    path = write_plan(plan, args.out)
    for w in plan["warnings"]:
        print(f"warning: {w}")
    network = plan["network"]
    print(f"network: {network['what']} ({network['connection_attempts']} connection attempts)")
    print(f"Run plan: {path}")
    print("Running the damage conditions and making the label are not built yet (step 3, next slice); "
          "nothing else was run.")
    print(final_line(plan))
    return 0


def final_line(plan: dict) -> str:
    """The last line: what the checks mean for the label (docs/user_models.md, note of 7 October 2026)."""
    shrunk = plan["models"].get("shrunk")
    if shrunk and shrunk["status"] == "failed":
        a = plan["checks"]["agreement_with_fp32"]
        return (f"Checks PASS for the FP32 model and the images, but the supplied INT8 build FAILED: it "
                f"agrees with FP32 on {a['agreement']:.1%} of {a['n_items']} images (failed below "
                f"{a['failed_below']:.0%}). Downstream: FP32 numbers only; the INT8 rows will say "
                "\"INT8 build failed\" with this value, and no INT8 numbers (as for MobileNetV3-Small).")
    return "Checks PASS."


if __name__ == "__main__":
    raise SystemExit(main())
