"""The website's committed sources other than the labels and figures (docs/website_v0_plan.md, section 14).

- findings(): the Stage 4 findings, read word for word from the generated block between the "findings"
  markers in README.md. scripts/45_readme_findings.py writes that block from the 4.1 records and the
  labels, and its --check (run by scripts/40_check_labels.py) fails if the block differs from a fresh
  render, so the site quotes checked text without needing the records (which are not committed).
- roadmap(): ROADMAP.md's stages and Platform plan steps with their status marks, and the "Later
  versions" list, with every date taken out (H, 3 October 2026: no dates on the website).
- CITATIONS: the related work cited on the site, as verified on each source's own page on 3 October 2026
  (scratch/related_work.md; not committed yet). Bibliographic facts only, no claims about the papers.
"""

import re

from site_labels import ROOT

README = ROOT / "README.md"
ROADMAP = ROOT / "ROADMAP.md"
FINDINGS_START, FINDINGS_END = "<!-- findings:start", "<!-- findings:end -->"
# Each findings paragraph, known by its first words (scripts/45_readme_findings.py writes them in this order).
FINDINGS = {
    "overview": " torchvision models,",
    "headline": "Consistent with prior work on quantized models",
    "prereg": "Pre-registered (",
    "example": "For example,",
    "explore": "Exploratory, not pre-registered: across all",
    "noise_blur": "Exploratory, not pre-registered: large shrinking costs",
    "why": "The tests of *why*",
}


def findings_block(text: str | None = None) -> str:
    text = README.read_text(encoding="utf-8") if text is None else text
    return text.split(FINDINGS_START)[1].split("-->", 1)[1].split(FINDINGS_END)[0]


def findings(text: str | None = None) -> dict:
    """{key: paragraph} for every findings bullet, unwrapped to one line each."""
    bullets = [" ".join(b.split()) for b in re.split(r"\n- ", "\n" + findings_block(text).strip())[1:]]
    out = {}
    for key, start in FINDINGS.items():
        found = [b for b in bullets if (start in b[:40] if key == "overview" else b.startswith(start))]
        if len(found) != 1:
            raise SystemExit(
                f"FAIL: README findings: expected one paragraph starting {start!r}, found {len(found)}"
            )
        out[key] = found[0]
    return out


# ---- the roadmap ----

STATUS = {"[x]": "done", "[~]": "in progress", "[ ]": "not started"}
HEADING = re.compile(r"^(#{2,3}) (.+?) `(\[[x~ ]\])`")


def clean_title(title: str) -> str:
    """'Stage 1 — Core measurement' stays; '1. Labels (P1)' -> 'Labels'; backticks dropped."""
    title = re.sub(r"\s*\(P\d\)", "", title).replace("`", "")
    return re.sub(r"^\d+b?\.\s*", "", title)


def undated(text: str) -> str:
    """A 'Later versions' item without its bracketed notes about dates, people, STATUS.md or plan codes."""
    text = re.sub(r"\s*\([^()]*(?:20\d\d|STATUS|by H|P\d)[^()]*\)", "", text).replace("`", "")
    return text.strip()


def roadmap(text: str | None = None) -> dict:
    """{'built', 'now', 'next', 'later', 'later_versions'}: lists of (step, title) or plain items."""
    text = ROADMAP.read_text(encoding="utf-8") if text is None else text
    lines = text.splitlines()
    stages, steps, later_versions = [], [], []
    section = None
    for i, line in enumerate(lines):
        if line.startswith("## "):
            section = (
                "platform" if line.startswith("## Platform plan") else "stages" if "Stage" in line else None
            )
        m = HEADING.match(line)
        if m and section == "stages" and m.group(1) == "##":
            stages.append((None, m.group(2), STATUS[m.group(3)]))
        elif m and section == "platform" and m.group(1) == "###":
            number = re.match(r"(\d+b?)\.", m.group(2)).group(1)
            steps.append((number, clean_title(m.group(2)), STATUS[m.group(3)]))
        elif section == "platform" and line.startswith("### Later versions"):
            for item in lines[i + 1 :]:
                if item.startswith("#"):
                    break
                if item.startswith("- "):
                    later_versions.append(undated(item[2:]))
    not_started = [s for s in steps if s[2] == "not started"]
    return {
        "built": [s for s in stages + steps if s[2] == "done"],
        "now": [s for s in stages + steps if s[2] == "in progress"],
        "next": not_started[:1],
        "later": not_started[1:],
        "later_versions": later_versions,
    }


# ---- related work ----

CITATIONS = [  # (key, authors, year, title, where, link)
    (
        "xiao",
        "Yisong Xiao, Tianyuan Zhang, Shunchang Liu, Haotong Qin",
        "2023",
        "Benchmarking the Robustness of Quantized Models",
        "arXiv:2304.03968; CVPR 2023 workshop",
        "https://arxiv.org/abs/2304.03968",
    ),
    (
        "recti",
        "Hamidreza Yaghoubi Araghi, Parastoo Pilevar, Ming C. Lin",
        "2026",
        "Recti-Q: Feature-Space Rectification for Out-of-Distribution-Robust Quantized Perception in Edge "
        "Robotics",
        "arXiv:2607.18540",
        "https://arxiv.org/abs/2607.18540",
    ),
    (
        "kasa",
        "Kevin Kasa, Graham W. Taylor",
        "2023",
        "Empirically Validating Conformal Prediction on Modern Vision Architectures Under Distribution Shift "
        "and Long-tailed Data",
        "arXiv:2307.01088",
        "https://arxiv.org/abs/2307.01088",
    ),
    (
        "hendrycks",
        "Dan Hendrycks, Thomas Dietterich",
        "2019",
        "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
        "arXiv:1903.12261; ICLR 2019",
        "https://arxiv.org/abs/1903.12261",
    ),
    (
        "michaelis",
        "Claudio Michaelis, Benjamin Mitzkus, Robert Geirhos, Evgenia Rusak, Oliver Bringmann, "
        "Alexander S. Ecker, Matthias Bethge, Wieland Brendel",
        "2019",
        "Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming",
        "arXiv:1907.07484",
        "https://arxiv.org/abs/1907.07484",
    ),
]
FURTHER = [  # verified too, not cited in the text: listed by title only
    (
        "karimov",
        "Toghrul Karimov, Hassan Imani, Allan Kazakov",
        "2025",
        "Quantization Robustness to Input Degradations for Object Detection",
        "arXiv:2508.19600",
        "https://arxiv.org/abs/2508.19600",
    ),
    (
        "kasa2024",
        "Kevin Kasa, Zhiyu Zhang, Heng Yang, Graham W. Taylor",
        "2024",
        "Adapting Prediction Sets to Distribution Shifts Without Labels",
        "arXiv:2406.01416; UAI 2025",
        "https://arxiv.org/abs/2406.01416",
    ),
    (
        "mitchell",
        "Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben "
        "Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, Timnit Gebru",
        "2019",
        "Model Cards for Model Reporting",
        "arXiv:1810.03993; FAT* 2019",
        "https://arxiv.org/abs/1810.03993",
    ),
]
