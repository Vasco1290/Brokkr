"""Write the README's label example from the labels, so its numbers cannot drift from them.

Usage:  python scripts/41_readme_label_example.py [--labels labels] [--check]
Needs:  published/labels/mobilenet_v3_large/label.json (the released label, committed; the README's example
        link and screenshots show the same label) and <labels>/mobilenet_v3_small/label.json
        (scripts/39_make_labels.py)
Writes: the block between the two "label-example" markers in README.md

--check: do not write; exit 1 if the README's block differs from a fresh render of the labels, or if a
number in it is not a number of its label. scripts/40_check_labels.py runs this check too.
"""

import argparse
import sys
from pathlib import Path

from brokkr_edge.label_render import load, readme_broken_example, readme_example, unexplained_numbers

README = Path("README.md")
START = "<!-- label-example:start (written by scripts/41_readme_label_example.py; do not edit by hand) -->"
END = "<!-- label-example:end -->"

parser = argparse.ArgumentParser()
parser.add_argument("--labels", default="labels")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()

example_label = load(Path("published/labels/mobilenet_v3_large/label.json"))
broken_label = load(Path(args.labels) / "mobilenet_v3_small" / "label.json")
example, broken = readme_example(example_label), readme_broken_example(broken_label)
loose = unexplained_numbers(example_label, example) + unexplained_numbers(broken_label, broken)
if loose:
    sys.exit(f"FAIL: numbers in the example that are not in its label: {loose}")

text = README.read_text(encoding="utf-8")
if text.count(START) != 1 or text.count(END) != 1:
    sys.exit("FAIL: README.md needs exactly one pair of label-example markers")
before, rest = text.split(START)
old_block, after = rest.split(END)
new_block = f"\n{example}\n\n{broken}\n"

if args.check:
    same = old_block == new_block
    print("PASS: the README's label example equals a fresh render of the labels" if same
          else "FAIL: the README's label example is out of date; run scripts/41_readme_label_example.py")
    sys.exit(0 if same else 1)
README.write_text(before + START + new_block + END + after, encoding="utf-8", newline="\n")
print(f"wrote the label example into {README} ({len(new_block.splitlines())} lines)")
