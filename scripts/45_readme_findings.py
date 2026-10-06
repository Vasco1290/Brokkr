"""Write the README's Stage 4 findings from the records and the labels, so its numbers cannot drift.

Usage:  python scripts/45_readme_findings.py [--labels labels] [--check]
Needs:  results/final/breadth_4.1_verdicts.json (the 4.1 verdicts and summary table) and
        <labels>/*/label.json (scripts/39_make_labels.py)
Writes: the block between the two "findings" markers in README.md

The wording was approved by H on 3 October 2026 (check 4; docs/stage4_story.md holds the same text); the
headline's last clause was narrowed by H on 6 October 2026 ("we found no strong relation between clean
accuracy and which models collapsed", with the number of models).
Pre-registered verdicts (H18b, H20, H21) are stated with the conditions they judged; wider checks are
marked "exploratory, not pre-registered". Every number is read from the records or labels here.

--check: do not write; exit 1 if the README's block differs from a fresh render. scripts/40_check_labels.py
runs this check too.
"""

import argparse
import json
import statistics
import sys
import textwrap
from pathlib import Path

README = Path("README.md")
VERDICTS = Path("results/final/breadth_4.1_verdicts.json")
START = "<!-- findings:start (written by scripts/45_readme_findings.py; do not edit by hand) -->"
END = "<!-- findings:end -->"
XIAO = "Xiao et al., 2023, arXiv:2304.03968"
RECTI = "Yaghoubi Araghi et al., 2026, 4-bit, arXiv:2607.18540"


def pts(x: float) -> str:
    return f"{100 * x:.1f}"


def join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def findings(v: dict, labels: dict) -> list:
    """The findings bullets, in order, as plain sentences."""
    verdicts, settings = v["metrics"]["verdicts"], v["settings"]
    judged = settings["judged_models"]
    name = {m: lab["model"]["display_name"] for m, lab in labels.items()}
    h18b, h20, h21 = verdicts["H18b"], verdicts["H20"], verdicts["H21"]
    two = [m for m, item in h21["items"].items() if item["holds"]]
    if two != [m for m, item in h20["items"].items() if item["holds"]]:
        sys.exit("FAIL: H20 and H21 hold for different models; the approved wording does not fit")

    first = next(iter(labels.values()))
    rule = first["envelope"]["rule"]
    tested = first["summary"]["tested_conditions"]
    sanity = settings["sanity_check"]
    broken = [name[m] for m in settings["left_out"]]
    overview = (
        f"{len(sanity)} torchvision models, {tested + 1} test conditions (clean and {tested} damaged). All "
        f"{len(sanity)} full-precision models passed the sanity check against their published accuracy. "
        f"{len(judged)} INT8 builds were usable; the one for {join(broken)} broke."
    )
    headline = (
        f"Consistent with prior work on quantized models ({XIAO}; {RECTI}), we found that shrunk models can "
        f"pass a normal accuracy check and still collapse in dark or low-contrast images, and in our small "
        f"pre-registered test ({len(judged)} models), we found no strong relation between clean accuracy and "
        f"which models collapsed. In wider exploratory checks, which models "
        f"were hit varied with the condition, and fog hit some models too."
    )
    prereg = (
        f"Pre-registered (Stage 4, {len(judged)} models, {settings['n_images']:,} test images; verdicts "
        f"unchanged): the two predictions about collapse were judged at two conditions only, darkness "
        f"(Brokkr) s5 (H21) and contrast (ImageNet-C) s3 (H20). In each, {h21['holding']} of {h21['judged']} "
        f"models lost much more to shrinking under the damage than on clean images "
        f"({join(name[m] for m in two)}); we had predicted at least {h21['needed']}, so both are FAIL. Clean "
        f"accuracy and how much shrinking cost under damage showed no strong relation (H18b: PASS, "
        f"{h18b['holding']} of {h18b['judged']} judged conditions, {h18b['needed']} needed; a weak test with "
        f"{len(judged)} models)."
    )

    def measurement(model, metric, cid, build=None):
        lab = labels[model]
        return next(
            m
            for m in lab["measurements"]
            if m["metric"] == metric
            and m["condition_id"] == cid
            and m["build_id"] == (build or lab["label_id"])
        )

    ex = max(two, key=lambda m: measurement(m, "shrinking_cost", "clean")["value"])
    example = (
        f"For example, {name[ex]}'s shrunk build scores "
        f"{100 * measurement(ex, 'top1', 'clean')['value']:.2f}% "
        f"on clean images, {pts(-measurement(ex, 'shrinking_cost', 'clean')['value'])} points below its "
        f"full-precision build, but {pts(-measurement(ex, 'shrinking_cost', 'brokkr/darkness/5')['value'])} "
        f"points below it under darkness (Brokkr) s5."
    )

    usable = {m: lab for m, lab in labels.items() if m in judged}
    with_any = [m for m, lab in usable.items() if lab["summary"]["large_shrinking_cost"]["count"] > 0]
    by_condition = {}
    for m, lab in usable.items():
        for cid in lab["summary"]["large_shrinking_cost"]["conditions"]:
            by_condition.setdefault(cid, []).append(m)
    cond_label = {c["condition_id"]: c["label"] for c in first["conditions"]}
    widest = max(by_condition, key=lambda c: len(by_condition[c]))
    beyond = sorted({m for ms in by_condition.values() for m in ms if m not in two}, key=lambda m: name[m])
    explore = (
        f"Exploratory, not pre-registered: across all {tested} damaged conditions on the labels, "
        f"{len(with_any)} of the {len(usable)} usable shrunk builds have at least one condition with a large "
        f"shrinking cost (the whole interval more than "
        f"{pts(-rule['large_shrinking_cost_below'])} points below "
        f"the full-precision build). Besides {join(name[m] for m in two)}, this includes "
        f"{join(name[m] for m in beyond)}. The condition hitting the most models is {cond_label[widest]} "
        f"({len(by_condition[widest])} models)."
    )

    # Noise and blur against darkness, contrast and fog: judged models, near-floor cells left out (as in 4.1).
    table = [
        r
        for r in v["raw"]["table"]
        if r["condition"] != "clean" and r["model"] in judged and not r["near_floor"]
    ]
    cid_of = {c["label"]: c["condition_id"] for c in first["conditions"]}

    def group(cond):
        return "nb" if ("noise" in cond or "blur" in cond) else "dcf"

    def flagged(r):
        lab = labels[r["model"]]
        row = next(
            x
            for x in lab["envelope"]["rows"]
            if x["build_id"] == lab["label_id"] and x["condition_id"] == cid_of[r["condition"]]
        )
        return "large shrinking cost" in row["shrinking_cost_flags"]

    worst = {
        g: min(r["extra_gap"]["value"] for r in table if group(r["condition"]) == g) for g in ("nb", "dcf")
    }
    pairs = {g: sum(1 for r in table if group(r["condition"]) == g and flagged(r)) for g in ("nb", "dcf")}
    totals = {g: sum(1 for r in table if group(r["condition"]) == g) for g in ("nb", "dcf")}
    medians = {}
    for r in table:
        medians.setdefault(r["condition"], []).append(r["extra_gap"]["value"])
    medians = {c: statistics.median(x) for c, x in medians.items()}
    cut = rule["large_shrinking_cost_below"]
    past = [c for c, m in medians.items() if m < cut]
    typical = f"The median build's extra loss was under {pts(-cut)} points in every condition" + (
        f" except {join(f'{c} ({pts(-medians[c])} points)' for c in past)}." if past else "."
    )
    noise_blur = (
        f"Exploratory, not pre-registered: large shrinking costs appeared in "
        f"{pairs['dcf']} of {totals['dcf']} "
        f"usable build-condition pairs under darkness, fog and low contrast, against {pairs['nb']} of "
        f"{totals['nb']} under noise and blur (worst extra gap {pts(-worst['dcf'])} vs {pts(-worst['nb'])} "
        f"points; near-floor cells excluded). {typical} Our conditions did not include impulse noise, which "
        f"Xiao et al. found hit quantized models most."
    )
    why = (
        "The tests of *why* it happens have not found the cause: each was inconclusive "
        "or rejected our guess. "
        "The question is parked while the labels and the website ship."
    )
    return [overview, headline, prereg, example, explore, noise_blur, why]


def render(bullets: list) -> str:
    return "\n".join(
        textwrap.fill(b, width=104, initial_indent="- ", subsequent_indent="  ", break_on_hyphens=False)
        for b in bullets
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="labels")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    v = json.loads(VERDICTS.read_text(encoding="utf-8"))
    labels = {
        p.parent.name: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(args.labels).glob("*/label.json"))
    }
    block = f"\n{render(findings(v, labels))}\n"
    text = README.read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        sys.exit("FAIL: README.md needs exactly one pair of findings markers")
    before, rest = text.split(START)
    old, after = rest.split(END)
    if args.check:
        same = old == block
        print(
            "PASS: the README's findings equal a fresh render of the records and labels"
            if same
            else "FAIL: the README's findings are out of date; run scripts/45_readme_findings.py"
        )
        sys.exit(0 if same else 1)
    README.write_text(before + START + block + END + after, encoding="utf-8", newline="\n")
    print(f"wrote the findings into {README} ({len(block.splitlines()) - 1} lines)")


if __name__ == "__main__":
    main()
