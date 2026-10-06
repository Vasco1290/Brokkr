# Pre-public checklist (Platform plan step 2), run on 6 October 2026

Asked by H on 6 October 2026 (round 2, slice C). Every item below lists what was checked, the command or
file that shows it, and the result, as run on branch `step-2-site`. Numbers here were printed by commands
run that day; nothing was estimated. Where an item needs a decision from H, it says so; nothing on the
"for H" lists has been changed.

Summary: **all items pass**, with three things for H to decide: the overclaim list (section 2), the real
name on five GitHub-made commits (section 3), and the publishing plan (`docs/publishing_plan.md`).

## 1. Licences and commercial use

| What | Licence (where recorded) | Commercial use |
|---|---|---|
| Brokkr's code | Apache-2.0: `LICENSE` (Apache License 2.0 text), `pyproject.toml` (`license = Apache-2.0`) | allowed |
| Released labels (`published/labels/`) | CC BY 4.0: each label's `licences.label_data`, its HTML and Markdown, the README, every site page's footer | allowed, with credit |
| Fonts (`web/fonts/`) | SIL Open Font License 1.1: an `OFL.txt` in each folder; `web/fonts/README.md` records the source (google/fonts, `ofl/`), the download date and that the files are unmodified (IBM Plex's Reserved Font Name "Plex" respected: no subsetting). The two Plex `OFL.txt` files are stored with LF line endings by `.gitattributes`; their text is unchanged | allowed |
| Model code (torchvision) | BSD-3-Clause: each label's `licences.model_code`, from the build records | allowed |
| Model weights (torchvision, ImageNet-1k) | as torchvision states it: "no separate licence for the weights, and ImageNet's terms of access are for non-commercial research. Check before redistributing." (`brokkr_edge/model_list.json`, every label, every model page) | **not cleared**: every label says the weights carry ImageNet's non-commercial terms of access |
| ImageNet-1k validation images | ImageNet Terms of Access, non-commercial research and educational use only (`brokkr_edge/datasets.py`; Methods page; README) | **research only** (tagged `"use": "research only"` in `brokkr_edge/datasets.py` on 6 October 2026) |
| ImageNetV2 (Stage 3 only; not on the labels or site) | as the sources state it (`brokkr_edge/datasets.py`); images keep their Flickr licences | **research only** (tagged the same day) |
| imagecorruptions 1.1.2 (vendored) | Apache-2.0: `brokkr_edge/third_party/imagecorruptions/LICENSE` (copied from the wheel), the one-line change in `numpy2_fix.diff`, recorded in `CHANGES.md` | allowed |
| Its dependencies, as installed (package metadata, read 6 October 2026) | numpy 2.2.6 BSD; scikit-image 0.25.2 BSD; scipy 1.15.3 BSD; Pillow 12.3.0 MIT-CMU; opencv-python-headless 4.14.0.94 Apache-2.0 | allowed |
| Runtime and tools (same source) | onnxruntime 1.23.2 MIT; onnx 1.23.0 Apache-2.0; onnxconverter-common 1.16.0 MIT; torch 2.14.0+cpu Apache-2.0 (with LLVM exception); torchvision 0.29.0+cpu BSD; pyarrow 25.0.1 Apache-2.0; psutil 7.2.2 BSD-3-Clause; py-cpuinfo 9.0.0 MIT; huggingface_hub 1.32.0 Apache-2.0 | allowed |

Checks: `tests/test_datasets.py::test_every_dataset_has_a_licence_and_a_use_tag`; the label validator
refuses a label with an empty licence field; `scripts/40_check_labels.py` checks each label's licences
against its build records. No image from ImageNet, ImageNet-C or ImageNetV2 is in the repository or on the
site: the six tracked pictures are the banner, logo, social preview, favicon and the two label-summary
screenshots in `docs/assets/`, and no commit on any branch ever touched `results/` or `data/` (`git log --all
-- results/ data/ '*.JPEG' '*.jpeg'`: 0 commits). Labels hold aggregate numbers only.

## 2. Overclaiming re-read of every page

Every page of the built site was read (its visible text, hover notes and alt text: 1,125 distinct
sentences once numbers were masked, from `scripts/49_build_site.py`'s build of `00dfdd4`). The README was
searched for strong words (guarantee, first, best, proves, always, never, robust, reliable, honest, edge
device, real-world) and each hit read in context. **Sentences that claim more than their evidence, for H
(not changed):**

1. **Glossary, "Pinned"** (hover note on every model page, and the Methods page; it is label renderer text,
   so it is also in every `label.html`): "the timing was only allowed to run on the named CPU cores, so the
   operating system could not move it to slower ones." The pin was set but **not read back** (said
   elsewhere on the same pages), so "could not move it" is not shown. Suggested: "the timing was asked to
   run only on the named CPU cores (the pin was set but not read back)". Changing it changes every
   `label.html`: it would need a dated entry in `docs/label_changes.json` and a re-release.
2. **Why labels?, "The 'I'm not sure' signal"** (claims register `fig-uncertainty`): "...while their
   average size changes little." True for FP32 (2.04 to 2.97 classes under damage, 2.28 clean); for INT8
   the set size falls to 0.99 under contrast (ImageNet-C) s5 (2.63 clean, coverage 12.6%) and rises to 3.58
   under fog (ImageNet-C) s5. Suggested: "...while their average size barely grows (for INT8 under contrast
   (ImageNet-C) s5 it even shrinks)."
3. **Landing page and Why labels? headline** (README findings, approved wording of 3 October 2026): "in our
   small pre-registered test, clean accuracy didn't predict which." The test (H18b, passed) found no strong
   relation with 9 models, a weak test; "didn't predict" reads as a shown absence. Suggested: "in our small
   pre-registered test, we found no strong relation between clean accuracy and which models collapsed."
4. **Methods, honesty rules**: "Every number on a label and on this site comes from a checked result
   file." The site also shows numbers that are not results (the meter's scale, dates and commits in the
   threshold history, version numbers), each on a page's reviewed allow-list. Suggested: "Every measured
   number...".
5. **Methods, "Why these values, without looking at the data:"** The reasons given do not depend on the
   results, but they were written on 4 October 2026, after the results existed. Suggested: "Reasons for
   these values that do not depend on the results:".
6. **Methods, honesty rules**: "Nothing is estimated: ..." The intervals are bootstrap estimates; the rule
   means no value is guessed. Suggested: "Nothing is guessed: ...".
7. **Tagline** (landing page, README): "Shrink AI models for small hardware, and find out honestly what you
   lost." Only one laptop CPU has been measured; the sentence below it on both pages says so ("one laptop
   CPU, simulated damage"). Low risk; listed for completeness.

Read and judged not to overclaim: "Hurt by shrinking ... the shrinking caused the failure" (same images,
only the build differs); "Not harmful in our tests ... not a guarantee"; every speed sentence (always
"laptop latency", never an edge device; INT8 slower said plainly); "marks what was not tested" (every label
says "Not tested: every damage type and severity not listed above"); the ImageNet-C sentences (corrected on
5 October 2026; `brokkr_edge.wording` finds no "official" claim anywhere on the site, in the labels, the
README or the public docs).

## 3. Git-history scan (all branches and tags)

- **Refs:** 15 (local `main`, `stage-4`, `step-2`, `step-2-site`; their four `origin/` copies and
  `origin/HEAD`; tags `report-1`, `stage-1-done` to `stage-4-done`, `stage3-final-run`), after `git fetch`.
  **199 commits** (`git rev-list --all`).
- **Author and committer emails:** every commit uses `143497911+Vasco1290@users.noreply.github.com`, except
  that GitHub itself is the committer of five commits (`noreply@github.com`). No other email.
- **Names:** 194 commits are by "Vasco1290". **Five commits carry the author name "Harshit Rawal"** (H's
  GitHub display name, with the noreply email): `b373a21` "Initial commit" and `216610e` (24 September
  2026), and the pull-request merges `0422a8c` (#4), `f48afa7` (#5), `1448e98` (#6), all made in GitHub's web
  interface. They are on `main`; rewriting them would rewrite `main`'s history. **For H:** keep (the same
  name is on the GitHub profile) or rewrite.
- **Contents** (`git log --all -p`, 10,571,193 bytes, every diff and message): no `C:\Users\<name>`,
  `C:/Users/<name>` or `/c/Users/<name>` path for this machine (`Users/harsh`: 0 hits); the only machine-path
  strings are the made-up examples in tests (`C:/Users/someone/...`, `/home/someone/...`), the checks that
  look for such paths, the checklist text, and commit `de75df0`'s message, which names the problem it fixed
  without the path ("recorded as C:/Users/..."); this machine's host name: 0 hits; the real name: 0 hits; no email address other than the noreply
  address (the only `@...` matches are Python decorators). The 11 matches of the word "harsh" are the
  label term "harsh conditions" and variables named after it.
- Label sources are repository paths (checked by the label validator, which refuses an absolute path).

## 4. The claims register is complete

`docs/claims.json` has 12 entries: the five figures, the landing finding (`site-finding`), and one per
paragraph of the Why labels? page (`why-example`, `why-prereg`, `why-explore`, `why-noise-blur`,
`fig-uncertainty`, `why-label-design`, `imagenet-c-method`). Each has its claim, where it appears, scope,
pre-registered or exploratory status, prior work, and the commands that make and check it. Checks:
`scripts/47_figures.py --check` (every figure has a complete entry); the site check "claims register" fails
if a findings paragraph quoted on any page has no entry with its findings key, if a Why labels? paragraph
names no entry, if an entry misses a field or names a missing script, or if it cites an arXiv number that is
not in `docs/related_work.md`. Both PASS.

## 5. Every external link resolves

12 distinct external links on the site, in the README, ROADMAP.md and `docs/*.md`: **12 of 12 answered 200**
(HEAD, then GET where HEAD was refused): the eight arXiv abstract pages cited, the GitHub repository,
`download.pytorch.org/whl/cpu` and the ImageNet-1k dataset page on Hugging Face. Internal links and anchors
are checked by the site build ("internal links resolve").

## 6. The site builds from a fresh clone

`git clone --branch step-2-site https://github.com/Vasco1290/Brokkr.git` at `00dfdd4` (212 tracked files; no
`results/`, `models/`, `labels/` or `data/`), with `PYTHONPATH` set to the clone so its own `brokkr_edge` is
used: `scripts/49_build_site.py` printed PASS for all 11 site checks and wrote 16 pages and 60 files. The
same build in the working folder gave **60 of 60 files byte-identical** (two font licence files first
differed only in line endings: the working copies still had Windows line endings from the download, while
git stores LF; the working copies were refreshed from git). The site, label, wording, figure and theme tests
(84) also passed inside the clone.

## Decisions of 6 October 2026 (H, round 3), and what was done

Added after H's review; the sections above are left as they were run.

- **Section 2, the seven sentences: all narrowed** to what the evidence shows (H: "never strengthen").
  Before and after:
  1. Headline (README findings block, landing page, Why labels?, claim `site-finding`): "...and in our
     small pre-registered test, clean accuracy didn't predict which." -> "...and in our small
     pre-registered test (9 models), we found no strong relation between clean accuracy and which models
     collapsed." The number of models is generated by `scripts/45_readme_findings.py`.
  2. Uncertainty (Why labels?, claim `fig-uncertainty`): "...while their average size changes little." ->
     "...while the average set size doesn't grow to compensate; under strong low contrast (contrast
     (ImageNet-C) s5) the INT8 sets even get smaller." No number is typed. `tests/test_site.py` checks on
     the label that the INT8 sets are smaller there than on clean images.
  3. Glossary "Pinned" (every `label.html` and `label.md`, model pages, Methods): "the timing was only
     allowed to run on the named CPU cores, so the operating system could not move it to slower ones." ->
     "the timing was set to run only on the named cores; this was not read back to confirm." Listed in
     `docs/label_changes.json`; the labels were remade and released again.
  4. Methods, honesty rule: "Every number on a label and on this site comes from a checked result file;
     the site is not built if a page shows a number that is not in a label, a checked figure or the
     checked findings." -> "Every measured number on a label and on this site comes from a checked result
     file; the site is not built if a page shows a number that is not in a label, a checked figure, the
     checked findings or the page's reviewed list of other numbers (such as scales, dates and versions)."
     The same narrowing was applied to the landing page ("Every number comes from a checked result file"
     -> "Every measured number ...") and the README's rule ("Every number comes from running the code" ->
     "Every measured number ...").
  5. Methods: "Why these values, without looking at the data:" -> "Reasons for these values that do not
     depend on the results:".
  6. Methods, honesty rule: "Nothing is estimated: ..." -> "Nothing is guessed: ...".
  7. Tagline (landing page, README, design prototype): "Shrink AI models for small hardware, and find out
     honestly what you lost." -> "Shrink AI models for small hardware, and measure what you lost." The
     sentence under it on both pages still says "one laptop CPU".
- **Section 3, the five commits with H's real name: kept** (H). No history rewrite.
- **Publishing plan: approved** (H). A 404 page was added first; see `docs/publishing_plan.md`.
