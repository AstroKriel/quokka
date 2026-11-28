---
name: quokka-check-pr-readiness
description: "Check that a Quokka pull request is ready for review: out of draft, CI passed, PR description complete, GPU tests triggered, review threads resolved, and new tests run by CI. Run it before asking for review, and as the first step of quokka-review-pr."
---

# Check PR Readiness

Check that a PR has what a maintainer needs before reviewing starts. This is the author's side of the PR; whether the change is correct is for the review.

**Terms:** maintainer, readiness, `<pr-number>` (defined in `.agents/skills/glossary.md`).

## Run the checks

```bash
python3 .agents/skills/quokka-check-pr-readiness/scripts/check_pr_readiness.py \
    <pr-number>
```

It prints one row per check and exits 0 when every check passes. The last column says what the author does when a check fails.

| Check | Passes when | If it fails, the author |
|---|---|---|
| State | the PR is open and out of draft | marks the PR ready for review |
| CI | every check on the latest commit passed or was skipped, and none is still running | fixes the failing checks named in the detail, or waits for the running ones to finish |
| Description | the template placeholder is gone, and every checklist item is ticked or struck through as not applicable | completes the PR description, and ticks or strikes through each checklist item |
| GPU tests | a `/azp run` comment exists, or the author struck the GPU checklist item through as not applicable | comments the `/azp run` command that the PR template gives for the change |
| Review threads | none is unresolved | addresses each open review thread, and resolves it |
| New tests run in CI | each new or changed test is built and run by a CI job (checked by reading, below) | registers each test for a dimension that a CI job builds |

## Check by reading: new tests run in CI

One check needs reading, not a script. A new or changed test only protects the code if a CI job builds and runs it:

1. List the tests the PR adds or changes: new problems in `src/problems/`, new inputs in `inputs/`, and changes to `regression/quokka-tests.ini`.
2. For each, find its `add_test` and the dimensions it is registered for.
3. Check that at least one CI job builds that dimension and runs the test.

A test registered only for a dimension no CI job builds never runs, and counts as missing.

## Report

Report a table with one row per check, giving its result and detail. Then, for each check that fails, write one instruction to the author, taken from the last column of the table above and made specific to the PR: name the failing check, the open thread, or the checklist item. When every check passes, the PR is ready for review.
