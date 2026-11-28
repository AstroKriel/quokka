---
name: quokka-assess-pr-results
description: "Assess the output of a Quokka run on a pull request against the default branch: which numbers matter, which messages to ignore, and when a marginal failure is the PR's fault."
---

# Assess PR Results

Run the same scenario on the PR and on the default branch first (`quokka-checkout-pr-and-default`, `quokka-build-problem`, `quokka-run-problem`).

## Reading output

Ignore AMReX performance advisories: "grid blocking factor (N) is too small for reasonable performance" is a grid-layout preference, not correctness. Mention them only when the PR is about grid decomposition.

Weigh: convergence rates, error norms, SUCCESS/FAIL lines, actual abort messages.

**Marginal failures.** An error norm within ~2× of its tolerance is not automatically the PR's fault. Compare against the base branch: near-identical norms both hovering at the threshold are tolerance noise, a pre-existing note. A norm that jumps an order of magnitude between base and PR is a regression even if it still passes.
