---
name: quokka-checkout-pr-and-default
description: "Check out a Quokka pull request and the default branch (development) side by side on throwaway branches, so the two can be built and compared, and clean up afterwards."
---

# Check Out a PR and the Default Branch

Uses a `<slug>` for the work at hand, for example derived by `quokka-review-pr` from the PR title.

## Setup (once per session)

```bash
# Must name a repo. If it errors, run `gh repo set-default quokka-astro/quokka` once.
gh repo set-default --view
```

## Isolation

Checking out other refs needs a clean tree:

```bash
git status --porcelain
```

Non-empty → stop and ask the user to commit, or to work in a separate worktree (`git worktree add ../review-<slug>`). Never stash their work.

Build both refs on throwaway branches keyed to `<slug>`, not a commit hash, so concurrent reviews don't collide.

## Check out the default branch

```bash
git fetch origin development
git checkout -B dev-<slug> origin/development
git submodule update --init --recursive
```

## Check out the PR

`gh pr checkout` also resolves PRs opened from forks; `--force` resets a branch left by a prior round:

```bash
gh pr checkout NNNN --branch pr-<slug> --force
git submodule update --init --recursive
```

After either checkout, configure and build with `quokka-build-problem`; `quokka config` must run after every checkout or submodule update.

## Clean up

Delete the throwaway branches when done:

```bash
git branch -D dev-<slug> pr-<slug>
```
