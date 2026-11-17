# Fix mode (`shrink --auto`): plan

**Draft written 17 November 2026; H's answers of the same day are in section 11.** Nothing here is measured
yet. It revises the "Brokkr Fix Agent" feature brief after checking it against the repository. Its home in
the roadmap is "Later versions", item `shrink --auto` (P5 in `ROADMAP.md`), extended with diagnosis, a
diary and a baseline. **It starts after Platform plan step 4 (`pip install`) ships** (H, 17 November 2026).
The numbers marked **(approved)** were approved by H on 17 November 2026 (section 12); they go into a
hypotheses note committed before any measurement. Numbers quoted from earlier work name their source.

## 1. What fix mode is, in plain words

Brokkr today shrinks every model with one recipe: INT8 with Percentile 99.99 calibration, chosen in Stage 3
on one model (MobileNetV3-Large). Fix mode would, for one model:

1. **Diagnose:** which damage conditions hurt the shrunk build, whether that harm comes from shrinking or
   was already there at full size (FP32), and which layers lose the most precision when shrunk.
2. **Try known recipes** from a fixed menu, one at a time, on tuning images only, in an order set by fixed
   rules that read the diagnosis.
3. **Keep the best build** that stays inside the limits, or say "no fix found".
4. **Write a second label** beside the plain INT8 label (which stays, unchanged), saying what was changed
   and what it cost, plus a **diary**: every try, its result and the rule that chose it.

It is an automated, logged search over recipes Brokkr already knows. The "agent" is a set of fixed rules,
not a language model: nothing in Brokkr uses one, and fixed rules give the same choices on every run, as
the reproducibility rules require.

**What it cannot do.** Shrinking recipes can at most bring INT8 back to FP32's level. Where FP32 itself
fails, the part of the failure FP32 already has is not fixable by shrinking, and the label says so.
Coverage above FP32's level comes only from bigger prediction sets (section 4, S1).

**A correction to the brief.** The brief says the uncertainty signal "does not survive INT8". The labels
show it mostly does not survive the damage, at full size too: for MobileNetV3-Large, FP32 coverage is
below 80% in 8 of 12 damaged conditions (`published/labels/mobilenet_v3_large/label.json`).

## 2. What is already known

| Recipe | MobileNetV3-Large (Stage 3) | The other models |
|---|---|---|
| Per-channel vs per-tensor weights | Per-channel fixed for every build, by design (`docs/hypotheses_stage3.md`) | Same |
| Calibration method: MinMax, Entropy, Percentile 99.99 / 99.999 | Compared (3.2); Percentile 99.99 chosen; clean gap 15.4 -> 2.0 points (H10) | **Not compared**: all use Percentile 99.99 |
| Damaged images in INT8 calibration | H12 FAIL: −0.67 points at darkness s5 (−1.41 to +0.06) | **Not tried** |
| Last layer's output unrounded | H11 FAIL (within noise) | **Not tried** |
| Squeeze-and-excitation blocks in float | SE1: did not recover half of the loss | SE1 FAIL for EfficientNet-B0 too (`docs/hypotheses_stage4.md`) |
| Prediction sets re-tuned on damaged images ("robust conformal") | H15 PASS: +12.8 points coverage; clean sets 2.28 -> 7.72 classes | **Not tried** |
| Each build's own conformal threshold | Already done: INT8 has its own (clean coverage 90.22%) | Already done |

Where fix mode could matter, from the labels and Stage 4:
- **MobileNetV3-Small's INT8 build is broken:** it matched FP32's first answer on 2.73% of 256 tuning
  images (README). Why is a parked question.
- **Large shrinking costs under damage:** 6 of the 9 usable INT8 builds have at least one condition where
  INT8 is far below FP32 (README, exploratory). EfficientNet-B0's shrinking cost at darkness (Brokkr) s5 is
  −43.72 points against −4.78 on clean images, an extra gap of −38.94 points (its label,
  `published/labels/efficientnet_b0/label.json`; corrected on 17 November 2026: this line first called −43.7 the
  extra gap).
- **Coverage lost to shrinking, held-out conditions** (computed on 17 November 2026 from the 9 usable
  labels; the 8 ImageNet-C conditions, leaving out cells where FP32 top-1 is below 10%): 58 cells; INT8
  coverage minus FP32 coverage ranges from −47.92 points (MobileNetV3-Large, contrast s5: FP32 60.55%,
  INT8 12.63%) to +3.53, median −1.06. 36 of the 58 are already within 2 points (point estimates).
- **Clean accuracy:** EfficientNet-B0 −4.78, MobileNetV3-Large −1.98, ConvNeXt-Tiny −1.62 and ResNet-50
  −1.14 points below FP32 on clean images (the labels' `shrinking_cost`).

H decided on 3 October 2026 not to research damage-aware calibration further (prior work and H12 were
negative) and parked it as "a cited recipe in `shrink --auto`" (STATUS.md). The menu below keeps it on
those terms.

## 3. Design rules (from the brief, kept)

- **A layer on top.** `brokkr-edge test`, the INT8 build and the labels stay as they are and remain the
  default. Fix mode calls the existing code (`brokkr_edge/quantize.py`, `brokkr_edge/shift/`,
  `brokkr_edge/label_build.py`); it never changes their behaviour. `scripts/43` must still pass 28 of 28.
- **The plain INT8 result is always shown.** A repaired build is a second result, never a replacement.
- **No hidden failures.** If nothing works, the output is the normal label plus "no fix found", with the
  diary.
- **Results are JSON first.** The diary, the repaired label and the summary table are made from records.

## 4. The menu (v1)

Shrinking recipes (change the INT8 file):
- **R1 Calibration method:** Percentile 99.99 (today's default), Percentile 99.999, Entropy, MinMax.
- **R2 Calibration images:** clean only (default), or clean plus damaged images from the tuning conditions
  only (as Stage 3's 3.3). Labelled "a cited recipe; negative in Brokkr's H12 and in prior work".
- **R3 Layers kept in float:** none (default); the last layer's output (as 3.4); the squeeze-and-excitation
  blocks (as SE1; only models that have them); or the k most fragile layers from the diagnosis, for
  k = 1, 2, 4 or 8 **(approved)**.

Dropped from the brief: **per-tensor weights.** Every build already uses per-channel weights, and Stage 3
recorded why per-tensor would be worse (depthwise layers with very different ranges per channel).

A separate option, not a shrinking recipe, **included in v1** (H, 17 November 2026):
- **S1 Prediction sets re-tuned on damaged images** (robust conformal, as 3.6), from tuning conditions only,
  applied to the plain build and to the chosen build. It changes set size, not the model, so it is shown
  as its own row with its set size, never mixed into the shrinking verdict. Limit, stated on the label:
  conformal prediction promises nothing for damage it was not tuned on.

Out of scope: quantization-aware training (no GPU) and FP16 builds (not part of this menu). A new recipe
joins the menu only by a dated note, before it is tried.

## 5. How fix mode is tested: which images and conditions each part may see

The aim is better models, so the test must show whether a fix also helps on damage the search never saw;
a fix that only helps on the damage it was tuned on is not a fix. Four layers, cheapest first:

- **Images.** The search uses only the `tuning` split (5,000 images) and `int8_calibration` (512); S1 tunes
  on `conformal_calibration` (5,000). The `test` split (10,000) is used once, for the final candidates, as
  in Stage 3.
- **Search conditions:** Brokkr's own damage (fog, defocus blur, motion blur, noise, darkness), on tuning
  images.
- **Leave one kind of damage out (the Stage 3 method)** for the two options that learn from damaged images,
  R2 and S1: each is built five times, each time without one kind of damage, and judged only on the kind it
  did not see. So every number for R2 or S1 comes from damage that option never learned from. Cost: five
  builds (R2) or five thresholds (S1) per try instead of one.
- **Held-out conditions for the final report:** the 8 ImageNet-C conditions on the labels (fog s3/s5,
  contrast s3/s5, defocus blur s3/s5, Gaussian noise s3/s5), on test images. Limit: fog, blur and noise
  exist in both groups (different code, same kind of damage); only contrast is a kind of damage the search
  never sees. The report says this.
- **Real new photos:** the chosen builds are also measured on ImageNetV2 (10,000 photos collected the same
  way years later; already used in Stage 3, task 3.9; licence recorded in `brokkr_edge/datasets.py`;
  evaluation only). This is the one check that is not simulated damage.
- **Multi-fidelity:** the search runs on a fixed, seeded subset of 1,000 tuning images **(approved)**. Only
  the final 1–2 candidates per model run on all 10,000 test images, the 13 conditions and ImageNetV2.

## 6. What "fixed" means (approved by H, 17 November 2026)

The brief's targets cannot be met as written: coverage of at least 88% is out of reach where FP32 itself is
far below it; set size had no limit; and 4 of 9 usable builds already miss "top-1 within 1 point of FP32"
on clean images. Instead, the target is to close the part of the gap that shrinking caused.

- **Judged conditions:** the 8 held-out conditions, except near-floor cells (FP32 top-1 below 10%, the
  rule already used in Stage 4). FP32 failing a condition does not remove it: if FP32 covers 60% and INT8
  13%, shrinking caused the 47 points between them, and a fix can recover them.
- **A condition's gap is closed** unless the paired 95% interval of INT8 coverage minus FP32 coverage (same
  images) lies wholly below **−2 points (approved)**. This is the "whole interval past a line" style of
  the label's existing judgements.
- **Fixed:** every judged condition closed. **Improved:** more conditions closed than the plain INT8 build,
  none newly open. **No fix found:** neither. All three are reported with the counts.
- **Limits a kept build must meet:**
  - Average set size at most **1.25 times FP32's (approved)** in each judged condition. Plain INT8 is above
    this in 11 of the 58 held-out cells today (7 of them EfficientNet-B0), so the limit forbids buying
    coverage with bigger sets beyond what most plain builds already show.
  - Clean top-1 not worse than the plain INT8 build by more than **0.5 points (approved)**. Reason: Stage
    3's INT8 build-to-build calibration luck was 0.28 points top-1 (MobileNetV3-Large); 0.5 is above that
    noise but small.
  - File size at most **1.3 times the plain INT8 build** (from the brief).
  - Speed: a float layer can make the build slower. Laptop latency of each chosen build is measured by H's
    long run (the existing method), or the label says "not measured".
- **Reported apart:** the share of each condition's failure that FP32 already has ("FP32 also fails; not
  fixable by shrinking"), as on today's labels.

## 7. The search, the stop rules and the diary

- **Rules, in order (draft):** broken build (agreement below the existing 20% floor) -> R3 first; large
  shrinking cost in darkness- or fog-type search conditions -> R1 first; otherwise R1, then R3, then R2.
  Each try changes one setting from the current best build.
- **Keep or revert:** keep a try only if it closes more search conditions than the current best (by the
  section 6 rule, on the 1,000-image subset) without crossing a limit; otherwise revert. With 1,000
  images, one coverage near 80% has a 95% interval of about ±2.5 points (binomial formula, not measured);
  paired differences are narrower. Wide intervals mean fewer keeps: the search can miss a real fix, but is
  less likely to keep a lucky one.
- **Stop:** the target is met, the budget is used, or a full pass of the menu keeps nothing.
- **Budget: 20 tries per model (approved)**, a try being one build plus its evaluation (an R2 try counts as
  one even though it needs five builds). The time of one try is not measured yet; milestone 3 measures it,
  and H confirms the budget then.
- **Diary:** one JSON record per model: each try, its settings, its subset numbers with intervals, keep or
  revert, and the rule that chose it. Rendered as a short table.

## 8. Baseline: a plain grid search (a product check)

**A product check, not a research question** (H, 17 November 2026): it shows whether the rules choose as
well as trying everything. It makes no research claim and needs no pre-registration; if its result raises
a research question, that question goes to STATUS.md's parked list.

- The grid tries **every** combination of R1, R2 and R3: 4 x 2 x 7 = 56 per model with
  squeeze-and-excitation blocks, 4 x 2 x 6 = 48 without. Its best build is the reference answer.
- Compared: whether the rules reach the grid's best (or one as good by the section 6 rule), how many
  tries each used, and the final size, accuracy, coverage and set size. If the grid does better, the
  report says so.

## 9. Outputs

- The command **`brokkr-edge shrink --auto`** (H, 17 November 2026), beside `brokkr-edge test`.
- Per model: the repaired build's label (a label-format addition, written in `docs/label_schema.md` before
  code: which plain build it comes from, what changed, the cost) and the diary.
- A summary table across the 9 usable models plus MobileNetV3-Small: rules vs grid.
- The website shows the repaired label next to the original, only after the results exist (rule 5).
- Every report states the limits: simulated damage (plus ImageNetV2), ImageNet classification, torchvision
  models, one laptop CPU.

## 10. Milestones (each ends with a check that prints PASS or FAIL)

1. **Diagnosis as reusable code:** the per-layer tool of `scripts/26` moved into `brokkr_edge`, plus the
   per-condition and FP32-vs-INT8 breakdown read from existing records. *Check:* it reproduces `scripts/26`'s
   saved values for MobileNetV3-Small (32 images) to 0.01 dB; `scripts/43` 28 of 28.
2. **The menu as build options** on `quantize.py`'s existing settings. *Check:* the default options give
   the existing INT8 files bit for bit; each option has a test.
3. **The cheap evaluation subset, the time of one try, and the hypotheses note** (section 6 and 7 numbers
   as approved by H). *Check:* the note is committed before any search run; `scripts/22` PASS.
4. **The rules, stop conditions and diary.** *Check:* tests with made-up scores, including the
   "no fix found" case.
5. **The grid baseline.** *Check:* the same try accounting as the rules, tested.
6. **Final test-split and ImageNetV2 runs** for the chosen candidates (H runs; long). *Check:* `scripts/22`,
   `scripts/40`.
7. **Labels, site, write-up.** *Check:* `scripts/40` PASS; site build PASS.

## 11. H's answers (17 November 2026)

1. **When:** after Platform plan step 4 ships.
2. **The numbers:** proposed in sections 4–7, reviewed once more by H and approved the same day
   (section 12).
3. **Testing:** the proposed split, plus the Stage 3 leave-one-out method where it makes the test stricter
   (R2 and S1), plus ImageNetV2 for the chosen builds (section 5). H's aim: models that get better, with
   any testing that shows it honestly.
4. **Command name:** `brokkr-edge shrink --auto`.
5. **Rules vs grid:** a product check first (section 8).
6. **S1 (robust conformal):** included in v1.

## 12. The approved numbers (H, 17 November 2026)

Approved by H on 17 November 2026. The budget is confirmed again once milestone 3 has measured the time
of one try. In one place:

| Setting | Approved | Where |
|---|---|---|
| Gap line for "closed" | paired interval not wholly below −2 points | section 6 |
| Conditions that must be closed for "fixed" | every judged held-out condition | section 6 |
| Set-size limit | 1.25 x FP32's, per judged condition | section 6 |
| Clean top-1 limit | at most 0.5 points below plain INT8 | section 6 |
| File-size limit | 1.3 x plain INT8 (from the brief) | section 6 |
| Fragile layers kept in float | k = 1, 2, 4 or 8 | section 4 |
| Search subset | 1,000 tuning images, fixed and seeded | section 5 |
| Budget | 20 tries per model; confirmed after milestone 3 measures one try | section 7 |
