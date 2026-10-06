# Publishing plan: the website on GitHub Pages (plan only)

*Written 6 October 2026 at H's request (round 2, slice C, item d); approved by H the same day (round 3),
with a 404 page added. **A plan only: nothing here has been done.** No `gh-pages` branch exists, nothing has been pushed to one, and GitHub Pages is not switched on.
Every step waits for H's go-ahead (STATUS.md: "Do not publish until H gives the go-ahead").*

## What gets published, and where

- **Address:** `https://vasco1290.github.io/Brokkr/`. This is GitHub's address for a project site: the
  account name in lower case, then the repository name, whose capital B matters.
- **Content:** exactly the folder `site_v0/` that `python scripts/49_build_site.py` writes: 17 pages and 61
  files (the counts printed by the build on 6 October 2026, round 3). It holds the pages, `404.html`,
  `assets/`, `fonts/`, `figures/`, the ten raw labels (`label.json`, `label.html`, CC BY 4.0),
  `favicon.png` and `.nojekyll`. `.nojekyll`
  tells GitHub to serve the files as they are, without running its Jekyll site builder.
- **Where the files live:** a separate branch `gh-pages` that holds only the built site (H's decision of
  3 October 2026, `docs/website_v0_plan.md` section 9, point 1). It shares no history with `main`, so
  `main` never contains generated files (hard rule 4). Building on GitHub is a later step.
- **The not-found page.** GitHub Pages shows `404.html` for any address under the site that does not
  exist, at the address that was asked for. So that page alone carries `<base href="/Brokkr/">`
  (`SITE_BASE` in `web/site_build.py`, the path of the address above), and its links to the landing page
  and the catalog, and its stylesheets, resolve from the site's root at any depth. A site check fails if
  the page is missing, lacks that base or either link, or if any other page has a base. Opened from the
  file system or the local preview server, the 404 page is unstyled (its base points at `/Brokkr/`); it is
  meant only for GitHub Pages.
- **Already true of the build** (checked by the site build on every run): every internal link is
  relative, so the site works under `/Brokkr/`; nothing is loaded from another site; every page names the
  commit it was built from; the build refuses a dirty working tree.

## Steps

**0. Before anything (H).** Step 2's work is reviewed and merged: a pull request `step-2-site` -> `main`
with a merge commit, as for PRs #1-#6. H opens it on GitHub: `gh` is not installed here.

**1. Build from a clean, tagged commit (Claude or H).**

```
git checkout main
git pull
git status                      # must print "nothing to commit, working tree clean"
git tag site-v0                 # the commit the published site comes from
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/40_check_labels.py     # needs the local records; about 11 minutes
.venv/Scripts/python.exe scripts/49_build_site.py       # every site check must print PASS
```

**2. Put the built folder on a new `gh-pages` branch, without touching `main`'s folder.** A git
*worktree* is a second working folder of the same repository, so the orphan branch is made outside the
project folder:

```
git worktree add --detach ../brokkr-gh-pages
cd ../brokkr-gh-pages
git checkout --orphan gh-pages        # a branch with no history
git rm -rf --quiet .                  # start empty
cp -r ../Brokkr/site_v0/. .           # the built site, including .nojekyll
git add -A
git commit -m "Website v0, built from <commit> (tag site-v0)"
git push origin gh-pages              # H's go-ahead needed for this push
cd ../Brokkr
git worktree remove ../brokkr-gh-pages
```

The commit uses the same noreply identity as every other commit. Check before pushing: `git -C
../brokkr-gh-pages ls-files | wc -l` prints the file count the build printed, and no file outside `site_v0/`
is included.

**3. Switch Pages on (H, in GitHub).** Repository Settings -> Pages -> Build and deployment -> Source:
"Deploy from a branch"; Branch: `gh-pages`, folder `/ (root)`; Save. The repository is public, so Pages
needs no paid plan. HTTPS is on by default for `github.io` addresses. No custom domain.

**4. Wait for GitHub's "pages build and deployment" run** (repository's Actions tab) to finish, then open
`https://vasco1290.github.io/Brokkr/`.

**5. Check the live site against the local build.** A small script, to be written when publishing
(proposed: `scripts/50_check_live_site.py`; not built yet), fetches every file of the build from the
live address and compares it byte for byte with `site_v0/`, printing PASS or FAIL (plan section 4). Then
check by hand: landing, catalog, one model page, Compare, Methods and Roadmap on a phone and a desktop, in
both themes, and the theme picker; and an address that does not exist, at the root and a few folders deep
(it must show the 404 page, styled, with working links home and to the catalog).

**6. Record it.** ROADMAP.md: tick "Publish the results page to GitHub Pages" and step 2's "the site is live"
with the date; README: add the site's address under Status; STATUS.md: the commit and tag published.

## Updating the site later

Rebuild from a new clean, tagged commit (step 1), then in a worktree of `gh-pages` replace every file with
the new build and add a **new commit** (no force-push), so the branch keeps a record of each release.
Then repeat steps 4 and 5.

## Taking it down

Settings -> Pages -> "Unpublish site" stops serving it at once. To go back one release, revert the last
`gh-pages` commit and push.

## Decided by H (6 October 2026, round 3)

- The plan is approved; a simple on-brand 404 page was added (above).
- The overclaim list was fixed before publishing (`docs/prepublic_checklist.md`, decisions of 6 October).
- The five commits with H's real author name are kept; no history rewrite.
- Publishing itself (steps 1 to 6) waits for H's go-ahead.
