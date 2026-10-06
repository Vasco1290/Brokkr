# Website v0 design prototypes

The approved **visual** reference for website v0 (H, 4 October 2026): the Ember forge and Light themes,
the fonts, the pixel-cooling hero, the bench meter and the tactile condition keys. Not the site generator.

```bash
.venv/Scripts/python.exe web/prototypes/build_prototypes.py --open
```

This builds two pages into `web/prototypes/out/` (gitignored) from `published/labels/` and opens the
first. Add `?theme=ember` or `?theme=light` to a page address to see a theme. Before writing, the builder
checks that every number comes from the labels, that the meter's needle and band match their label values,
that the themes pass the rules in `web/themes.py`, and that build IDs are unique.

**Layout:** these pages predate H's site map. The landing page here still has the full results table and
all thirteen condition keys, and the model page lists condition names in its key facts. The site map, the
disclosure rules and the builder rules are in `docs/website_v0_plan.md`, section 12, which wins.
