---
name: quokka-checkout-pr-and-default
description: "Check out a Quokka pull request and the default branch (development) side by side on throwaway branches, so the two can be built and compared, and clean up afterwards."
---

# Check Out a PR and the Default Branch

Check out a PR and the default branch side by side, so that the two can be built and compared.

**Terms:** `<short-name>` (Step 2 of `quokka-review-pr`); `<pr-number>` (**Terms** in `quokka-review-pr`).

The **default branch** is `development`, the branch every PR merges into. The other skills refer to it by that name, and point here for its definition.

The throwaway branches are named after a `<short-name>` for the work at hand. `quokka-review-pr`, for example, derives it from the PR title.

## Set up (once per session)

```bash
# This must name a repository. If it fails, run `gh repo set-default quokka-astro/quokka` once.
gh repo set-default --view
```

## Start from a clean tree

Checking out other branches needs a clean working tree:

```bash
git status --porcelain
```

If this prints anything, stop and ask the user either to commit their changes or to work in a separate worktree (`git worktree add ../review-<short-name>`). Never stash their work.

Check out both branches as throwaway branches named after `<short-name>`, not after a commit hash, so that reviews running at the same time do not collide.

## Check out the default branch

```bash
git fetch origin development
git checkout -B dev-<short-name> origin/development
git submodule update --init --recursive
```

## Check out the PR

`gh pr checkout` also works for PRs opened from forks, and `--force` resets a branch left over from an earlier round:

```bash
gh pr checkout <pr-number> --branch pr-<short-name> --force
git submodule update --init --recursive
```

After either checkout, configure and build with `quokka-build-problem`. `quokka config` has to run again after every checkout or submodule update.

## Clean up

Delete the throwaway branches when you are done:

```bash
git branch -D dev-<short-name> pr-<short-name>
```
