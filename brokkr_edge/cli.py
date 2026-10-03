"""The `brokkr-edge` command (the name "brokkr" belongs to another project's command).

    brokkr-edge test --model mobilenet_v3_large
    brokkr-edge test --model mobilenet_v3_large --split tuning --limit 64 --out <folder>   (a dry run)

Also runnable without installing the command: python -m brokkr_edge.cli test ...
"""

import argparse

from brokkr_edge import __version__
from brokkr_edge.model_list import load_model_list


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="brokkr-edge", description="Stress-test shrunk vision models.")
    parser.add_argument("--version", action="version", version=f"brokkr-edge {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    test = commands.add_parser(
        "test", help="run the clean and damaged conditions on one model and save checked records")
    test.add_argument("--model", required=True, choices=list(load_model_list()))
    test.add_argument("--out", default=None,
                      help="folder for the records (default: results/test_runs/<model>)")
    test.add_argument("--split", choices=["test", "tuning"], default="test",
                      help="test = the real run (plus clean conformal_calibration); tuning = dry runs only")
    test.add_argument("--limit", type=int, default=None, help="first N images only (dry runs)")

    args = parser.parse_args(argv)
    if args.command == "test":
        from brokkr_edge.test_run import run_model  # imported here so --help needs no ONNX Runtime

        paths = run_model(args.model, args.out or f"results/test_runs/{args.model}", args.split, args.limit)
        print(f"{len(paths)} records in {paths[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
