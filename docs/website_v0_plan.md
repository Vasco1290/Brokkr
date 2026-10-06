# Website v0 plan (Platform plan step 2): draft

*Drafted 3 October 2026 in `scratch/`, committed here on 3 October 2026 at H's request. A plan only: no
code. The points marked **(decide)** were decided by H on 3 October 2026 (section 9, which wins where it
differs from the sections above). No number in sections 1-10 is a result; the only numbers there are
settings (page widths) and facts about the label files read that day. Section 11's numbers were printed
from the records and labels by the check-4 script; on the site they come from the labels and records
through the number test, never typed.*

Step 2's "done when" (ROADMAP): the site builds from label files alone, and a test fails if any number
on it is not in a `label.json`; filter and compare work on the 10 labels; a user-submitted test label
shows "unverified"; every checklist item is ticked with its date; the README is current; the
repository is public and the site is live.

---

## 1. Pages

| Page | Address | What it shows | Made from |
|---|---|---|---|
| Home / catalog | `/` | One row per label: model, labelled build, summary counts (fine / borderline / harmful / not tested), clean top-1 of both builds, size, laptop latency (4 threads); filters | every `label.json` |
| Model page | `/models/<model>/` | The model's facts (publisher, weights, licence, input) and a list of its labels (today one per model) | that model's `label.json` files |
| Label page | `/models/<model>/<build>/` | The full label, as `label_render.render_html` makes it today, inside the site layout | one `label.json` |
| Compare | `/compare/` | 2–4 labels side by side: summary groups, clean top-1, size, latency, and the damage table for the conditions the user picks | the chosen `label.json` files |
| Why labels? | `/why-labels/` | Why accuracy on clean images is not enough; examples drawn **live** from the labels (see 1a) | fixed text + `label.json` |
| Methods | `/methods/` | Splits and image counts, conditions (Brokkr vs ImageNet-C, named separately), the harm definition and its threshold history as recorded in `docs/hypotheses_stage4.md`, the "12 conditions" sentence, the latency method (pinning, threads, sessions, the "unstable" rule), what "not tested" means | fixed text + values read from `label.json` (thresholds, counts) |
| Roadmap | `/roadmap/` | What exists now and what is planned, instead of greyed-out menu items (see 6) | fixed text **(decide: source)** |
| Licences | `/licences/` | Code Apache-2.0, labels CC BY 4.0, each model's and dataset's licence as recorded in the labels, ImageNet terms, ImageNetV2 citation | `label.json` licence fields + fixed text |
| 404 | `/404.html` | Plain "not found" with a link home | fixed text |

### 1a. "Why labels?" and the findings wording
- Every example on this page is pulled from a label at build time (e.g. "the label for X shows ...
  under Y"), never typed. This is where H's fix-list item 4 lands: any statement like "only N models
  collapse" must name its measure (e.g. shrinking cost, INT8 minus FP32, paired, on which conditions)
  and be generated from the labels, so it cannot drift from them.
- The wording "all failures are phone-optimised designs" is in no tracked file (searched 3 October
  2026); if it is wanted here, it must be checked against the label numbers first (fix 4).

### 1b. Speed wording on every page (from H's decision of 3 October 2026)
- Absolute times only (p50 / p95 / p99 per build). No "speed-up" column and no ratio in v0.
- If a relative line is shown, it is computed by the label generator and stored in `label.json`,
  and it says plainly "INT8 is slower than FP32 on this laptop CPU" where it is (a relative comparison
  on one machine only), never a negative speed-up. **(decide)** whether to show no relative line when
  either row is unstable (proposed: yes, no relative line then).
- Unstable rows: "Speed varied a lot between repeat runs (spread X%, from the record); treat as
  rough", plus a glossary entry (H's decision; the exact text goes in the next session's dated note).
- Every speed number says "laptop latency (Intel Core i5-1235U, Windows 11, batch 1, N threads on
  performance cores)", as the 29 September method note fixes. The methods page also says what the
  records show about those cores (see the read-only check of 3 October: logical CPUs 0–3 belong to
  two physical cores, two hardware threads each) **(decide: wording)**.

---

## 2. How pages are generated from `label.json`

- **One script** (proposed `scripts/45_build_site.py`) reads `labels/*/label.json` and nothing else
  (no results files, no model files). Output: a folder of static HTML, CSS and images (proposed
  `site_v0/`, gitignored).
- **Order inside the script:**
  1. Load every `label.json`; run the schema validator (`brokkr_edge/label_schema.py`); stop on any
     problem. A label whose `schema_version` the site does not know is refused, not guessed.
  2. Make the slim catalog file (`catalog.json`) from the full labels: generated, never edited, not a
     second source (label_schema note of 30 September, point 5).
  3. Run each page type (see 5) over the labels; each returns `{address: html}`.
  4. Write the files; copy `favicon.png` and the logo from `docs/assets/`.
  5. Run the number test and the page checks (sections 3 and 4) on the written files; exit non-zero
     on any failure.
- **Rendering reuses `brokkr_edge/label_render.py`** (its `fmt()` for every number, its glossary
  lines, its hover notes). The site adds a shared layout (header, nav, footer) around it.
- **JavaScript only for filter and compare** (CLAUDE.md: "JavaScript only when search needs it"). The
  data comes from `catalog.json`; JS only shows and hides rows and picks columns, it computes no
  number. Without JS the catalog still shows the full table.
- **Nothing is fetched from other sites** (no fonts, analytics or CDN scripts).

---

## 3. The test that fails on any number not in a label

Built on what `scripts/40_check_labels.py` already does for the Markdown and HTML labels.
- For every written page: strip the HTML tags but keep text in `title` attributes (hover notes) and
  `alt` text; find every number token (integers, decimals, percentages, negatives, ranges).
- Allowed set for a page = every number in the `label.json` files that page was built from, formatted
  with the same `fmt()`, plus a short, fixed allow-list of non-measurement numbers, each with its
  reason: schema version, Brokkr version, commit hashes, dates, severities inside condition names,
  the CI level as written in `label.json`. The allow-list lives in the test file and is reviewed by H.
- The test fails, naming the page and the number, if any number is outside its allowed set.
- Further tests, same file:
  - a made-up `user-submitted` label (tests only) shows "unverified" on every page it appears on
    (catalog row, model page, label page, compare);
  - every page states the label schema version and the Brokkr version it was built from (see 6);
  - no page contains a speed ratio or the words "speed-up"/"faster"/"slower" except in the stored
    relative line of 1b;
  - a coverage value never appears without its mean set size on the same row;
  - the site build refuses a label with an unknown `schema_version`.

---

## 4. GitHub Pages setup

- **Where the built site lives (decide).** The labels are gitignored today (hard rule 4: generated
  results are not committed), so GitHub's own CI cannot build the site: it has no labels. Options:
  - **(a) proposed:** build locally from a clean commit and push the built folder to a separate
    `gh-pages` branch (ROADMAP 1.7 already planned "gh-pages branch"). `main` never holds generated
    files; the footer of every page names the commit it was built from.
  - (b) commit the published `label.json` files to `main` (they are the CC BY 4.0 product). This
    changes rule 4 for labels, and step 4's label-submission pull requests will put user labels in
    the repository anyway. Worth deciding once, before step 4.
- Settings: Pages source = `gh-pages` branch, root; add `.nojekyll`; every link relative, so the
  site works under the project address (`/<repo>/`); a `404.html`.
- Pages on a private repository depends on the GitHub plan; the site goes live after H makes the
  repository public (step 2's order), so no setting is needed before then.
- A pass/fail script after publishing: fetch the live home page and one label page and check they
  equal the local build (byte for byte).

---

## 5. Skeleton: room to grow without rewriting

### 5a. Label schema items: does version 1 have them, and what would adding them cost?

Read from `docs/label_schema.md` and `labels/convnext_tiny/label.json` on 3 October 2026.

| Item | In schema v1? | What adding / changing it would cost |
|---|---|---|
| **Device** | **Yes.** `hardware[]` (`hardware_id`, `kind` from a list: laptop, raspberry-pi-5, cloud-arm, ...; CPU, OS, architecture, cores, features); every measurement and speed row points to a `hardware_id`; speed rows also name `hardware_kind`, with `hardware_id: null` for a device not measured yet. | None for v0. A new device kind is a new list entry (no version change). |
| **Runtime** | **Partly.** `hardware[].runtime` (name, version, execution provider, threads). It sits *inside* the hardware entry, one per device, and today records the accuracy runs' thread count; the speed rows keep their own thread counts in `settings`. Two runtimes on one device (e.g. a second ONNX Runtime version, or another runtime) have no clean place. | Small if optional: add a `runtime` block to each speed row and measurement (optional field, no version change), filled from the records, checked by `scripts/40`. Moving runtime out of `hardware` would be a rename, so schema version 2: not proposed for v0. |
| **Evidence level** | **No.** Pieces exist (`source.kind`, `verified`, `how_made`, each dataset's `n_items`), but no single field says "full study on the 10,000-image test split" vs "fast mode" vs "user images, labelled" vs "user images, unlabelled (agreement only)". | Small to medium: an optional `evidence` field with a fixed list of levels, defined in a dated note before code (the levels depend on steps 3, 4b and fast mode). Optional, so no version change; making it required later would be version 2. The site can show it from v0 if added before labels are regenerated. |
| **Task type** | **Yes.** `model.task` ("classification") and `model.modality` ("image" or "signal"). Metric names are open strings, so detection metrics later fit. | None now. A new task needs its own metrics and renderer sections, not a schema change. |
| **Source (study / user-submitted)** | **Yes, with different words.** `source.kind` is `official` or `user-submitted`; `verified` is true only for official. CLAUDE.md rule 7 says "community-submitted"; ROADMAP step 2 says user labels show "unverified". | None if only the display word changes (e.g. show "Brokkr study" for `official`) **(decide the words)**. Renaming the stored value would be schema version 2: not proposed. |

### 5b. Page addresses
- Now: `/models/<model>/` and `/models/<model>/<build>/`. Later, without moving anything:
  `/devices/<device-kind>/`, `/packs/<pack>/` (e.g. EEG/EMG, step 7), `/community/`.
- Slugs are made from label fields by one tested function (lowercase, safe characters only; `/`, `@`
  and `#` in IDs are not URL-safe). **(decide)** the build slug: readable `<precision>-<recipe>`
  (proposed), with the file hash shown on the page; or include the first 12 hash characters (unique
  even when a model is rebuilt, but less readable). A test fails if two labels map to one address.
- GitHub Pages has no server redirects, so addresses should be fixed before going public.

### 5c. Adding page types without rewriting existing pages
- Page types are a list in the generator; each is one function `labels -> {address: html}`. The nav
  is built from that list, so a new type (devices, packs, community results) is one new function.
- Test: adding a made-up page type (tests only) leaves every existing page's content unchanged
  except the shared nav.

### 5d. A roadmap page instead of greyed-out menu items
- The nav shows only pages that exist. Planned things appear on `/roadmap/` in plain words ("planned,
  not built yet"), never as disabled links.
- **(decide)** its source: fixed text in the generator, or generated from ROADMAP.md's Platform plan
  headings and status marks; and whether to show target dates (they have moved twice).

### 5e. Versions on every page
- Footer of every page: "Built by brokkr-edge <version> at commit <hash> from label schema version
  <n>". Today's `label.html` states the brokkr-edge version and commit but not the schema version
  (checked 3 October 2026): a one-line renderer change plus the test in section 3.
- Pages built from several labels (catalog, compare) also say when labels were made by different
  versions.

---

## 6. Phone and dark-mode checks

- Today's `label.html` already has the viewport tag and a `prefers-color-scheme` dark block (checked
  3 October 2026); the site layout keeps both.
- Wide tables scroll inside their own box (`overflow-x: auto`), never the whole page; the summary
  and key numbers come before the tables.
- Colour is never the only signal: every state has its word and mark, as on the labels today.
- Automated (in the page checks): every page has the viewport tag and a dark-mode block; no page loads
  anything from another site.
- Manual, in the browser pane, with screenshots kept in `scratch/`: catalog, one label page, compare
  and methods at phone width (375 px) and desktop width, each in light and dark; filter and compare
  used by touch-size taps; text contrast checked in both themes.

---

## 7. Pre-public checklist (each item ticked with its date)

From ROADMAP (Stage 1 "before making the repository public", and step 2):
- [ ] Publish the site to GitHub Pages (after the repository is public).
- [ ] Re-read README, ROADMAP, docs and the site for overclaiming: laptop latency never called edge
      speed; "Raspberry Pi 5" only for a measured Pi; "improved coverage in our tests", never
      "guaranteed"; ImageNet-C numbers not called directly comparable.
- [ ] Licence and commercial-use check of every model and dataset; each dataset tagged "research only"
      or "commercial use allowed" in `brokkr_edge/datasets.py`; torchvision weights' licence text as
      recorded ("torchvision states no separate licence for the weights ...").
- [ ] README current (it still says "Stages 1–2 of 7") and the licences stated: Apache-2.0 (code,
      `LICENSE`), CC BY 4.0 (labels).

ImageNet terms:
- [ ] No ImageNet, ImageNet-C or ImageNetV2 image, crop, cache or sample sheet is in the repository,
      its history, or the site (`results/samples/` stays local); labels hold only aggregate numbers.
- [ ] The site's licences page states the ImageNet terms as recorded and cites ImageNetV2 (Recht et
      al., ICML 2019).

Git history and personal details:
- [ ] Search every commit on every branch and tag (`git log --all -p`, plus author/committer fields)
      for full machine paths (`C:\Users\...`, `/c/Users/...`), H's username, real name and personal
      email, and the machine's host name. ROADMAP records that the personal email was already
      rewritten to the noreply address; this re-checks it, and checks paths and usernames, which were
      not part of that fix.
- [ ] The same search over everything that will be published: `label.json` files (`sources[].file`
      should be repository-relative paths), the built site, `catalog.json`.
- [ ] No tracked tool or editor folders with local settings or paths (e.g. `.claude/`, `.vscode/`).
- [ ] No secrets or tokens in history.

Other:
- [ ] Technical Report 1's "Related work" placeholder ("To be written by H"): **(decide)** publish
      as is, or hold the report back until written.
- [ ] `docs/HANDOFF.md` (out of date since 1 October): **(decide)** update, mark superseded, or remove.
- [ ] Social preview uploaded by H in GitHub settings.

---

## 8. Decisions for H, collected

1. Where the built site lives: `gh-pages` branch built locally (proposed) or labels committed to
   `main` (section 4).
2. Build slug in addresses: readable (proposed) or with hash characters (5b).
3. Display words for `official` and `user-submitted` ("Brokkr study"? "community-submitted"?) (5a).
4. Optional `evidence` field: add before the labels are regenerated, or after v0 (5a).
5. Relative speed line: shown at all, and hidden when a row is unstable (1b).
6. Methods-page wording for "4 threads on performance cores", given the record shows two physical
   cores with two hardware threads each (1b).
7. Roadmap page source and whether it shows dates (5d).
8. Report 1 with its "Related work" placeholder, and `docs/HANDOFF.md` (7).

---

## 9. H's decisions on section 8 (3 October 2026)

Recorded as given; where a decision changes a proposal above, this section wins.

1. **Where the site and released labels live: a hybrid.** A committed folder `published/labels/`
   holds released labels only, added at release points; working labels stay gitignored in `labels/`.
   This changes hard rule 4 for that folder only, recorded in a dated note. For v0 the site is built
   locally and pushed to `gh-pages` (as proposed in section 4); building on GitHub comes later.
2. **Readable addresses**, e.g. `/models/mobilenet-v3-large/int8-percentile-99.99/`. A generator test
   fails if two builds produce the same address. The build checksum is shown on the page, not in the
   address.
3. **Display words:** `official` → "Brokkr study"; `user-submitted` → "User-submitted".
   "Community-submitted" is reserved for results others send to the site later.
4. **Evidence field:** after v0 (settled; also H's decision d of 3 October).
5. **Relative speed line: shown, never hidden.** Wording that works both ways: "INT8 takes N× the time
   of FP32", with "(slower)" added when INT8 is slower; N generated from the records. When either row
   is unstable, the line is still shown, with the unstable flag and the plain-words note. (Replaces the
   proposal in 1b to hide it.) *Detail to fix in the dated note: which statistic N uses (proposed:
   the p50 ratio, per thread count), and that N is stored in `label.json` so the number test covers it.*
6. **Cores wording:** settled by H's decision c of 3 October ("Pinned to the laptop's 2 performance
   cores (4 hardware threads, as reported by Windows). The 4-thread setting therefore runs on 2
   physical cores."; the pin was not read back, stated as a limitation).
7. **Roadmap page generated from ROADMAP.md:** phases and Now / Next / Later plus what is built; no
   dates on the website.
8. **Report 1:** replace the "Related work" placeholder ("To be written by H") with "Related work: not
   yet written; planned before the next report." **`docs/HANDOFF.md`:** read-only summary first; H
   decides after.

## 10. Added 3 October 2026 (H): a claims register on the pre-public checklist

- [ ] **Claims register:** every public claim (README, site, "Why labels?") lists its scope (models,
      conditions, pre-registered or exploratory), its prior-work citation (`scratch/related_work.md`,
      later a tracked file), and the command that backs it; the checker verifies each claim against
      the records. No claim of being "first".


## 11. "Why labels?" page text (approved by H, 3 October 2026)

Generated from the records by the check-4 script (`scratch/check4_draft.md`); when the site is built,
these numbers come from the labels and records through the number test, never typed.

> **A normal accuracy check is not enough.** Consistent with prior work on quantized models (Xiao et al., 2023, arXiv:2304.03968; Yaghoubi Araghi et al., 2026, 4-bit, arXiv:2607.18540), we found that shrunk models can pass a normal accuracy check and still collapse in dark or low-contrast images, and in our small pre-registered test, clean accuracy didn't predict which. In wider exploratory checks, which models were hit varied with the condition, and fog hit some models too.
>
> For example, MobileNetV3-Large's shrunk build scores 73.60% on clean images, 2.0 points below its full-precision build, but 13.6 points below it under darkness (Brokkr) s5.
>
> Pre-registered (Stage 4, 9 models, 10,000 test images; verdicts unchanged): the two predictions about collapse were judged at two conditions only, darkness (Brokkr) s5 (H21) and contrast (ImageNet-C) s3 (H20). In each, 2 of 9 models lost much more to shrinking under the damage than on clean images (MobileNetV3-Large and EfficientNet-B0); we had predicted at least 5, so both are FAIL. Clean accuracy and how much shrinking cost under damage showed no strong relation (H18b: PASS, 8 of 11 judged conditions, 8 needed; a weak test with 9 models).
>
> Exploratory, not pre-registered: across all 12 damaged conditions on the labels, 6 of the 9 usable shrunk builds have at least one condition with a large shrinking cost (the whole interval more than 5.0 points below the full-precision build). Besides MobileNetV3-Large and EfficientNet-B0, this includes ConvNeXt-Tiny, MobileNetV2, RegNetY-400MF and ResNet-50. The condition hitting the most models is contrast (ImageNet-C) s5 (6 models).
>
> Exploratory, not pre-registered: large shrinking costs appeared in 19 of 51 usable build-condition pairs under darkness, fog and low contrast, against 3 of 43 under noise and blur (worst extra gap 38.9 vs 8.8 points; near-floor cells excluded). The median build's extra loss was under 5.0 points in every condition except contrast (ImageNet-C) s5 (14.0 points). Our conditions did not include impulse noise, which Xiao et al. found hit quantized models most.
>
> That is why every Brokkr label shows each tested condition separately, says whether the full-precision
> model also fails there, and marks what was not tested. The ImageNet-C conditions were generated with the imagecorruptions package v1.1.2 (Michaelis et al., 2019, arXiv:1907.07484; an extension of the ImageNet-C code of Hendrycks & Dietterich, 2019, arXiv:1903.12261), with a one-line fix so that fog runs on NumPy 2, tested pixel-identical to the unmodified package's fog. They are not directly comparable to the released ImageNet-C files.

## 12. Design decisions (H, 4 October 2026; after reviewing the prototypes)

Recorded as given. Where this section differs from sections 1–9, it wins. The prototypes in
`web/prototypes/` (built by `web/prototypes/build_prototypes.py` from `published/labels/`) are the
approved **visual** reference: theme, type, colours, the meter and the controls. Their **page layout**
predates the site map below (the prototype landing page still has the full results table and all
thirteen condition keys; the prototype model page lists condition names in its key facts). For layout,
this section is the reference.

### 12a. Theme: "the forge's test bench"
- Brokkr forges the part (shrinks the model), then measures it on the bench (the label).
- **Shell** (landing, navigation, "Why labels?"): dark-first forge, charcoal and stone with cream text and
  an ember accent. **Evidence** (model pages, tables, methods, compare): datasheet-crisp panels inside the
  shell.
- **Exactly three signature elements:**
  1. the pixel-cooling hero: a static SVG of squares cooling from ember to stone in discrete steps (a
     quantization metaphor);
  2. the analog bench meter on the landing page: needle, a Nixie-style readout (styled monospace text,
     never images) and a shaded interval band. Two zones only, small and large, split at the label's
     large-shrinking-cost line (the one threshold the rules define). The verdict comes from the whole
     interval and the label's own flag, never from the needle. Values come from published labels only and
     are checked like the figures. The needle overshoots slightly and settles; with
     `prefers-reduced-motion` it jumps;
  3. tactile controls for choosing conditions: raised = pressable, sunken = selected, always a visible
     border, a clear pressed state and a visible focus ring.
- Neumorphic styling only on controls, never on content cards, tables or text. Glow only on the Nixie
  digits. Gauges only for single headline values, always beside the exact number and its interval;
  comparisons across models stay as bars and tables. No severity knob in v0 (only severities 3 and 5
  were tested).

### 12b. Themes and colour tokens
- Every colour is a named CSS token, generated from `web/themes.py`; adding a theme is one new entry
  there, with no page changes. A theme changes only the shell: backgrounds, panels, accent, glow and
  control styling.
- **Fixed in every theme** (they depend only on light or dark mode): the data colours (INT8 blue, FP32
  grey, as in the figures), the verdict and warning colours (always with an icon), and the datasheet
  panels behind tables and figures. The dark datasheet panel is a **neutral grey with no hue, `#181818`**
  (light: white); the figures' dark versions use the same surface, so a figure looks the same under every
  theme. `tests/test_web_themes.py` keeps the panels and data colours identical to the figures'.
- **v0 ships Ember forge** (default for dark-mode visitors) **and Light** (default for light-mode visitors,
  datasheet style). Brass foundry, Runic frost, Deep mine and Anvil steel follow in the week after launch.
- Automated rules for every theme (`check()` in `web/themes.py`, run by the builder and by the test): text
  at least 4.5:1, marks and controls at least 3:1; a theme's accent at least 15 (OKLab distance x100) from
  the INT8 blue (written for Runic frost); no theme may override a fixed colour (so Deep mine's warnings
  always use the fixed warning colour and icon, never its green glow).
- A theme picker in the navigation; the choice is kept in the visitor's browser only (`localStorage`
  inside try/catch; the site works without storage); no tracking; the default follows the system
  light/dark setting.

### 12c. Typography
- **Big Shoulders Display** (display), **IBM Plex Sans** (body), **IBM Plex Mono** (every number and ID,
  and the Nixie readout). All SIL Open Font License 1.1, self-hosted in `web/fonts/` with their licence
  files; nothing is loaded from other sites. The files are unmodified copies from Google's font
  repository: IBM Plex has the Reserved Font Name "Plex", so it is not subset or converted under that name.

### 12d. Build IDs and addresses
- **Address:** `/models/<model>/<build>/`, with `<model>` the model's name with `_` turned into `-`
  (`mobilenet-v3-large`) and `<build>` the precision and recipe split into words (`int8-percentile-99.99`;
  FP32: `fp32`).
- **Build ID:** `BRK-<model code>-<precision>-<recipe code>`, generated from the address, never typed.
  Model code: the display name split at spaces, hyphens and capitals; a plain word gives its first letter,
  a part with digits or in capitals is kept whole without dots (MobileNetV3-Large → `MNV3L`, ConvNeXt-Tiny
  → `CNXT`, ShuffleNetV2 x1.0 → `SNV2X10`). Recipe code: the method's first letter and its digits
  (`P9999`); FP32 builds have none. The generator fails if two builds get the same ID or address.

### 12e. Site map: one job per page
- **Landing** (under about three phone screens): the hero; the meter with **four conditions only** (clean,
  darkness (Brokkr) s5, fog (ImageNet-C) s3, contrast (ImageNet-C) s5) and a "see all 12 conditions" link
  (the count generated from the labels); the hero figure with a two-sentence finding (generated from the
  records and labels, checked like the README findings); buttons to Catalog and "Why labels?". No results
  table on the landing page.
- **Catalog** `/models/`: one card per model: name, build ID, clean shrinking cost with its interval,
  large-shrinking-cost count, link to the label.
- **Model page**: key facts show **counts only** (the condition names move into collapsed `<details>`).
  Order: key facts → envelope groups → condition details → speed → uncertainty → licences and provenance
  → raw label JSON, with a short in-page menu at the top.
- **Compare**: the grid figure and a sortable table. **"Why labels?"**, **Methods** and **Roadmap** as in
  sections 1, 1a and 9 (point 7).
- **Navigation:** Catalog · Compare · Why labels? · Methods · Roadmap · GitHub · Theme.

### 12f. Progressive disclosure
- Details are collapsed by default with `<details>` (no JavaScript needed).
- On phones, condition details are **one card per condition**, never a wide table.
- Usable with JavaScript off (only the optional sorting, the theme picker and the condition switches need
  it), keyboard accessible, no external scripts, fonts or trackers.

### 12g. The site builder
- **Refuses to build from a dirty working tree**, like the measurement scripts; every page's footer names
  the commit it was built from, the brokkr-edge version and the label schema version.
- Every number a reader sees is checked against the labels (and, for the figures and the finding, the
  records), as the prototype builder and `scripts/47_figures.py` already do.

## 13. Slice A as built (4 October 2026; for H's review)

Slice A (asked by H on 4 October 2026): the generator, the landing page, the catalog and the ten model
pages, with their checks. Compare, "Why labels?", Methods and Roadmap come in slice B. Built by
`scripts/49_build_site.py` (`web/site_labels.py`, `site_meter.py`, `site_pages.py`, `site_build.py`,
`site_checks.py`; tests in `tests/test_site.py`); the prototypes now share its label lookups, meter and
hero picture. Choices made while building, each open to H's change:
- **Addresses.** Each model page is at `/models/<model>/<build>/` (12d). There is no `/models/<model>/` page
  yet (one label per model today); breadcrumbs link to the catalog.
- **Navigation and buttons** are built from the page types, so slice A shows only Catalog, GitHub and the
  theme picker, and the landing page has only the Catalog button; "Why labels?" appears in both when its
  page type is added (5c, 5d).
- **Meter keys** sit in one row inside the readout panel (the suite is named in the readout and in the
  table shown without JavaScript), to keep the landing page short on phones. Measured at 343 px wide
  (headless Chrome, device emulation): 2,530 px tall, about 3.1 screens of an 812 px tall phone (12e asks
  for about three).
- **The finding** is generated from the hero figure's values (H21, in `docs/figures/figures.json`) and the
  MobileNetV3-Large label, in the approved wording of section 11, with the label's two-decimal shrinking
  costs to match the meter beside it. Claims register entry: `site-finding`. The hero figure's
  `appears_in` now lists the landing page.
- **Model pages.** Key facts show counts only (condition lists collapsed), plus clean top-1 and file size.
  Condition details are one table on wide screens and one card per condition below 760 px (CSS only, the
  same markup). Speed and licence lines reuse the label's own wording (`label_render`); a slower INT8 also
  gets a callout above the speed table. A failed build opens with a "Do not use this INT8 build" panel
  before any number, and its catalog card is outlined in the warning colour.
- **No-JavaScript check** (automated): no element is hidden until a script runs, and every number the
  script can show is also on the page without it (the meter's four readings are in a plain table that
  shows only without JavaScript).

## 14. Slice A fixes and slice B (4 October 2026)

H approved slice A on 4 October 2026 with five fixes (accepted: the landing page at 3.1 phone screens), then
asked for slice B. Built on `step-2-site`; choices made while building, each open to H's change:
- **Fixes.** (1) The landing finding is the approved headline of section 11, quoted word for word from the
  README's findings block; the H21 detail is on "Why labels?". (2) The hero figure's column is headed
  "Much worse in the dark?" (each phone row says the same), and its caption names "Pre-registered test H21"
  and the overall verdict (regenerated by `scripts/47_figures.py`; the README picked it up). (3) The meter's
  large zone is a muted red (light `#c0726a`, dark `#a8564f`); a new theme rule fails if it is as loud as
  the needle, and the contrast rules still pass. (4) On phones each condition is a collapsed `<details>`
  card whose summary line is condition · INT8 state · shrinking cost (with "⚠ large" when flagged); wide
  screens keep the table. (5) A failed build's "Where it holds up" is one line ("All 12 conditions: INT8
  build failed") that expands to the groups; its FP32 results stay visible.
- **Where the findings text comes from.** The full findings (headline, the pre-registered paragraph with
  H18b, H20 and H21, the two exploratory paragraphs) are computed from the 4.1 records, which are not
  committed. The site therefore quotes them word for word from the generated block between the "findings"
  markers in README.md (`web/site_sources.py`); `scripts/45_readme_findings.py --check`, run by
  `scripts/40_check_labels.py`, fails if that block differs from a fresh render of the records and labels.
  A page that quotes it may show its numbers. No new committed results file was needed.
- **Compare.** A table of every label, sortable by any column with JavaScript (rows stay in name order
  without it), the grid figure in a box that scrolls sideways on phones, and the speed figure.
- **Why labels?** The approved text of section 11 (the order of section 11; the README's sentences), the
  hero, grid and uncertainty figures, the uncertainty figure's sentence from the claims register
  (`fig-uncertainty`), and the related work: the five sources cited on the page and the three others
  verified on 3 October 2026, as bibliographic entries only (typed in `web/site_sources.py` from
  `scratch/related_work.md`, which is not committed).
- **Methods.** Images and their uses, the test conditions, the imagecorruptions sentence, the envelope
  rules (the label's own wording), the threshold history as recorded in `docs/hypotheses_stage4.md`
  (`a957652`, `237effa`, `621cc43`, with their data-independent reasons), "With 12 conditions, an
  occasional result may cross a line by chance", how speed was measured (the label's notes, with the cores
  wording and the pin limitation), the honesty rules and the label's limits.
- **Roadmap.** Generated from ROADMAP.md's status marks: Built (marked done), Now (in progress, which
  includes Stage 4 as ROADMAP.md marks it), Next (the next platform step), Later (the other platform
  steps, then "Later versions" without bracketed dates, people or plan codes). A check fails on any month
  name or year on the page.
- **Navigation** now shows Catalog · Compare · Why labels? · Methods · Roadmap · GitHub, and the landing
  page has both buttons, all from the page list. On phones the links sit on their own row that scrolls
  sideways.
- **Number check tightened:** a number with thousands separators ("10,000") is one number, so a changed
  "10,500" is caught (before, its two parts were each allowed). Identifiers with digits on a page
  (commits, citations, step names) are listed per page, each with its reason.

## 15. Round 1 of the next session: H's fixes 0–4 (5 October 2026; for H's review)

H's fix list of 4 October 2026, done in this order on `step-2-site`; choices made while building, each open
to H's change:
- **0. Findings check.** Every findings paragraph a page quotes sits in a marked element
  (`<span class="quote" data-findings="<key>">`). A new site check reads README.md on its own (not through
  `web/site_sources.py`, which the pages use) and fails if a marked paragraph is not exactly one paragraph of
  the generated findings block. Only line breaks and repeated spaces are read as one space (neither Markdown
  nor HTML shows them); every other character must match. It also fails if a page that quotes the findings
  shows no marked paragraph, or a page that does not quote them shows one. Tests: one changed character,
  one changed digit, one missing letter, a changed README and a missing marker are each caught.
- **1. ImageNet-C wording.** The labels' limit and the label glossary no longer say "official" (dated note
  of 5 October 2026 in `docs/label_schema.md`). The ten labels were remade at the clean commit `0b02393`,
  passed `scripts/40_check_labels.py`, and were released again. `scripts/48_compare_labels.py` against the
  4 October release found one difference per label beyond the commit and date: `limits/1`, the ImageNet-C
  sentence. `brokkr_edge.wording` holds the check. It is run by `scripts/40` (labels and README), the site
  build (every page and raw label) and `tests/test_wording.py`. A sentence ends at ". ", a line break or an
  HTML tag, so "Official label" in one list item and "ImageNet-C" in another are not one sentence. The same
  wording was also corrected in two code notes and the vendored package's `CHANGES.md`.
  Left for slice C's overclaiming re-read (not changed here): ROADMAP.md's 4.1 scope says "the official
  corruption code", and `docs/hypotheses_stage4.md` (pre-registered, never edited above its outcomes) says
  the same.
- **2. Roadmap.** Stage 4 is marked done, with a note naming what stays open and where it moved. Every
  stage, Platform plan step and "Later versions" item in ROADMAP.md ends with a hidden mark,
  `<!-- public -->` or `<!-- internal -->`. A public mark may carry a plain name
  (`<!-- public: A minimal EEG/EMG pack -->`); without one, the name is the heading without its number. The
  site shows public items only, in file order, with no "Step N" labels. The build stops on an unmarked item,
  or on a public name holding a stage, step or task number or a plan code; the roadmap page check also fails
  on such a number. Marked internal: Stages 5–7 (their content is in "Later versions"), 4.1's summary tables
  script, Report 1's related work, the Stage 3 cache rebuild and the parked research questions. Stage 4's
  public name is "The study across ten models", because its "recommendations" were not built.
- **3. Compare on phones.** The model column stays fixed while the other columns scroll under it, with a
  thin line at its edge. Below 760 px, rows are one line each (intervals and sub-lines inline, smaller text
  and padding) and headings wrap. A "scroll →" hint shows above the table on screens under 960 px.
- **4. Model pages.** The speed timing note reads "Timed on 3 October 2026, 07:55–08:01 UTC", to the minute,
  with the exact times from `label.json` on hover. A window that crosses midnight names both dates. The
  number check allows the window only as `web/site_labels.readable_window` writes it from that label's own
  times. The conformal thresholds moved out of the coverage table into a collapsed "Calibration thresholds
  (conformal)" section, with their calibration image counts. The model page's Limits heading has an anchor
  (`#limits`).

## 16. Round 2: slice C (6 October 2026; for H's review)

H approved round 1 on 6 October 2026, including the public/internal choices, and asked for round 2. Choices
made while building, each open to H's change:
- **Claims register.** Every paragraph of the Why labels? page names its entry in `docs/claims.json`
  (`data-claim`), and every quoted findings paragraph has an entry with its findings key. The new entries
  are `why-example`, `why-prereg`, `why-explore`, `why-noise-blur`, `why-label-design` and
  `imagenet-c-method`; `site-finding` and `fig-uncertainty` already existed. As the register's rule says,
  no result number is typed in an entry: the numbers stay in the checked paragraphs. A site check fails on
  a missing entry or field, a script that does not exist, or an arXiv number that is not in
  `docs/related_work.md`.
- **Related work.** `scratch/related_work.md` is now `docs/related_work.md` (unchanged apart from its
  opening paragraph, the word "advisor" and one `Site:` line per entry). The site reads its citations from
  it (`web/site_sources.related_work()`); the typed list in `web/site_sources.py` is gone. The Why labels?
  page's visible text is unchanged (compared with the build of `0d1117f`; no other page changed).
- **Wording check, docs.** `brokkr_edge.wording.check_docs` reads the README and the public docs. It skips
  the text above the correction note of 6 October 2026 in `docs/hypotheses_stage4.md`, and the note's own
  section, and reports that the note exists. Four docs that record the fix or this check are listed as
  "not read", with their reasons. Every Markdown file git tracks must be in one of the two lists.
  - The sentence splitter changed: a single line break inside a paragraph no longer ends a sentence, so a
    claim wrapped across two lines is caught. Blank lines, headings, list items and HTML tags still end
    one.
- **Pre-public checklist:** `docs/prepublic_checklist.md`, with its evidence and the list for H.
  **Publishing plan:** `docs/publishing_plan.md` (plan only).
- **Datasets** carry a use tag, "research only" or "commercial use allowed" (`brokkr_edge/datasets.py`):
  both are "research only".

## 17. Round 3: narrowed wording and the 404 page (6 October 2026; for H's review)

H approved round 2 and decided: narrow all seven listed sentences (never strengthen), keep the five commits
with the real author name, approve the publishing plan with a 404 page added. Choices made while building,
each open to H's change:
- **Wording.** The before/after of all seven is in `docs/prepublic_checklist.md` (decisions of 6 October).
  - The same "every number" narrowing was applied where the sentence recurs: the landing page and the
    README's rules.
  - The tagline keeps "for small hardware" (the project's aim) and drops "find out honestly".
  - The uncertainty sentence types no number (the register's rule); a test checks its "even get smaller"
    against the label.
  - `docs/stage4_story.md` holds the narrowed headline too, with a dated note of what it said before.
- **"Pinned".** The labels were remade at `625e3de` and released again. `scripts/48_compare_labels.py`
  against the release of 5 October: PASS, each label with the one listed change.
- **404 page.** It reuses the landing page's pixel-cooling picture and buttons: the title "Not found", one
  line "This page doesn't exist.", and buttons Home and Catalog. Because GitHub serves it at any missing
  address, it alone carries `<base href="/Brokkr/">`, derived from the repository name; the published
  address is `SITE_ADDRESS` in `web/site_build.py`. A twelfth site check covers it, and the link check
  skips the base tag.
