"""The `brokkr-edge` command (the name "brokkr" belongs to another project's command).

    brokkr-edge test --model mobilenet_v3_large
    brokkr-edge test --model mobilenet_v3_large --split tuning --limit 64 --out <folder>   (a dry run)
    brokkr-edge test --config my_model.json --images <folder> [--calib-images <folder>] --out <folder>
        (a user's own model, docs/user_models.md: the checks, the INT8 build, the conditions and the label)

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
    """Test a user's model and write its label in --out. Exit code 2: the inputs cannot be used; 3: the run's
    files could not make a valid label."""
    if not (args.images and args.out):
        parser.error("--config needs --images and --out")
    if args.split != "test" or args.limit is not None:
        parser.error("--split and --limit go with --model")
    from brokkr_edge.user_command import run_user_test
    from brokkr_edge.user_label import LabelError
    from brokkr_edge.user_plan import NETWORK_WORDING, check_out_folder
    from brokkr_edge.user_settings import UserInputError

    try:
        problems = check_out_folder(args.out)
        if problems:
            raise UserInputError(problems)
        result = run_user_test(args.config, args.images, args.calib_images, args.out)
    except UserInputError as e:
        print("Stopped: your inputs cannot be used yet. Every problem found:", file=sys.stderr)
        for problem in e.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    except LabelError as e:
        print(f"Stopped: the label could not be made: {e}", file=sys.stderr)
        return 3
    for w in result["plan"]["warnings"]:
        print(f"warning: {w}")
    print(f"network: {NETWORK_WORDING} ({result['network_attempts']} connection attempts)")
    print(f"Label: {result['paths'][2]} (label.json and label.md beside it)")
    print(final_line(result["plan"], result["label"]))
    return 0


def final_line(plan: dict, label: dict | None = None) -> str:
    """The last line: what the checks mean for the label (docs/user_models.md, note of 7 October 2026), and,
    once the label is made, what it holds."""
    if label is not None:
        lab = next(b for b in label["builds"] if b["role"] == "labelled")
        tested = label["summary"]["tested_conditions"]
        if lab["status"] == "failed":
            who = "the supplied INT8 build" if "shrunk" in plan["models"] else "Brokkr's INT8 build"
            return (f"Label made (user-submitted, UNVERIFIED), but {who} FAILED "
                    f"({lab['failure']['check']}): FP32 numbers only for {tested} damaged conditions; the "
                    f"INT8 rows say \"INT8 build failed\".")
        return (f"Label made (user-submitted, UNVERIFIED): FP32 and INT8 under {tested} damaged conditions "
                f"and clean images.")
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
