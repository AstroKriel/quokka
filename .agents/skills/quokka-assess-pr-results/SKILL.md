---
name: quokka-assess-pr-results
description: "Assess the output of a Quokka run on a pull request against the default branch: which numbers matter, which messages to ignore, and when a marginal failure is the PR's fault."
---

# Assess PR Results

First run the same scenario on the PR and on the default branch (defined in `quokka-checkout-pr-and-default`; see `quokka-checkout-pr-and-default`, `quokka-build-problem` and `quokka-run-problem`).

## What to weigh

Weigh convergence rates, error norms, `SUCCESS` and `FAIL` lines, and the exact abort messages.

Ignore AMReX performance advisories such as "grid blocking factor (N) is too small for reasonable performance". They are about how the grid is laid out, not about correctness. Mention them only when the PR is about grid decomposition.

## Marginal failures

An error norm within about 2× of its tolerance is not automatically the PR's fault, so compare it against the default branch:

- If both branches give nearly the same norm close to the tolerance, the failure is tolerance noise that was already there. Note it as pre-existing (defined under **Terms** in `quokka-review-pr`).
- If the norm jumps by an order of magnitude from the default branch to the PR, it is a regression, even if it still passes.
