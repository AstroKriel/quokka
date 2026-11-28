---
name: quokka-debug-build-failure
description: "Narrow down why a Quokka problem fails to configure or build, and whether the failure comes from the PR or was already on the default branch."
---

# Debug a Build Failure

Find why a problem fails to configure or build, and whether the PR caused it.

**Before this:** the failing build, from `quokka-build-problem`.
**Terms:** default branch (`quokka-checkout-pr-and-default`); pre-existing (**Terms** in `quokka-review-pr`); preset (**Presets** in `quokka-build-problem`).

## Narrow it down

1. **Read the first error, not the last.** Compiler errors cascade, so the first one in the log is usually the cause.
2. **Check whether the default branch builds** the same problem with the same preset. If it fails there too, the failure is not the PR's: report it as pre-existing.
3. **Reconfigure from scratch** with `quokka config ... --delete` after any checkout or submodule update. A stale build directory is the most common cause of errors that make no sense.
4. **Check the submodules** with `git submodule status`. A `+` or `-` in front of a path means it is not at the commit the branch expects. Fix it with `git submodule update --init --recursive`.
5. **For a failure only on the GPU,** where the CPU preset builds and the GPU preset does not, look first at code inside `ParallelFor` and other device lambdas. The usual cause is capturing a host variable or pointer. See **GPU Lambda Safety** in `CLAUDE.md`, and "Common bugs" in `docs/markdown/debugging.md`.

## Report

Report the command, the commit, the preset and machine, and the first error message, pasted exactly as it appears. Say whether the default branch builds.
