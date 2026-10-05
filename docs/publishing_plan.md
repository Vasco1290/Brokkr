# Publishing plan: the website on GitHub Pages (plan only)

*Written 6 October 2026 at H's request (round 2, slice C, item d). **A plan only: nothing here has been
done.** No `gh-pages` branch exists, nothing has been pushed to one, and GitHub Pages is not switched on.
Every step waits for H's go-ahead (STATUS.md: "Do not publish until H gives the go-ahead").*

## What gets published, and where

- **Address:** `https://vasco1290.github.io/Brokkr/`. This is GitHub's address for a project site: the
  account name in lower case, then the repository name, whose capital B matters.
- **Content:** exactly the folder `site_v0/` that `python scripts/49_build_site.py` writes: 16 pages and 60
  files (5.5 MB on 6 October 2026, built at `00dfdd4`). It holds the pages, `assets/`, `fonts/`, `figures/`,
  the ten raw labels (`label.json`, `label.html`, CC BY 4.0), `favicon.png` and `.nojekyll`. `.nojekyll`
  tells GitHub to serve the files as they are, without running its Jekyll site builder.
- **Where the files live:** a separate branch `gh-pages` that holds only the built site (H's decision of
  3 October 2026, `docs/website_v0_plan.md` section 9, point 1). It shares no history with `main`, so
  `main` never contains generated files (hard rule 4). Building on GitHub is a later step.
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
../brokkr-gh-pages ls-files | wc -l` prints 60, and no file outside `site_v0/` is included.

**3. Switch Pages on (H, in GitHub).** Repository Settings -> Pages -> Build and deployment -> Source:
"Deploy from a branch"; Branch: `gh-pages`, folder `/ (root)`; Save. The repository is public, so Pages
needs no paid plan. HTTPS is on by default for `github.io` addresses. No custom domain.

**4. Wait for GitHub's "pages build and deployment" run** (repository's Actions tab) to finish, then open
`https://vasco1290.github.io/Brokkr/`.

**5. Check the live site against the local build.** A small script, to be written when publishing
(proposed: `scripts/50_check_live_site.py`; not built yet), fetches every one of the 60 files from the
live address and compares it byte for byte with `site_v0/`, printing PASS or FAIL (plan section 4). Then
check by hand: landing, catalog, one model page, Compare, Methods and Roadmap on a phone and a desktop, in
both themes, and the theme picker.

**6. Record it.** ROADMAP.md: tick "Publish the results page to GitHub Pages" and step 2's "the site is live"
with the date; README: add the site's address under Status; STATUS.md: the commit and tag published.

## Updating the site later

Rebuild from a new clean, tagged commit (step 1), then in a worktree of `gh-pages` replace every file with
the new build and add a **new commit** (no force-push), so the branch keeps a record of each release.
Then repeat steps 4 and 5.

## Taking it down

Settings -> Pages -> "Unpublish site" stops serving it at once. To go back one release, revert the last
`gh-pages` commit and push.

## Open points for H

- **No `404.html`** yet (`docs/website_v0_plan.md` section 1 listed one): GitHub then shows its own
  "page not found". Optional before publishing.
- **The overclaim list** in `docs/prepublic_checklist.md` (section 2): any wording H changes before
  publishing needs a rebuild (and, for label text, a dated entry in `docs/label_changes.json` and a
  re-release of the labels).
- **The five commits with the real author name** (`docs/prepublic_checklist.md`, section 3): keep or
  rewrite before the site points people to the repository.
