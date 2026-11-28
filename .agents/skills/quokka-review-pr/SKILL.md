---
name: quokka-review-pr
description: "Review a Quokka pull request: build and run the code to test each claim it makes, then post a short verdict comment. Analysis only; does not implement fixes."
---

# Quokka PR Review

Analyse and report. Never change source.

## What this review is for

This review is the baseline a maintainer reads before reviewing the PR themselves. It settles what can be settled by building, running and reading, with evidence for each finding, and states what it leaves to the maintainer. It never approves a PR: only a person does.

Three words are used in one sense each:

| Word | Meaning |
|---|---|
| Claim | what the PR says it does |
| Evidence | what a build, a run, or a reading of the code shows |
| Finding | a problem that the evidence supports |

## The argument

A target, plus an optional instruction:

- `pr 2158`, or a bare `2158`: a pull request
- text after the target: a specific instruction, such as `pr 2158. suggest a fix for the nvcc error`

With no argument, ask which target. For an issue, use `quokka-review-issue`.

An instruction sets the scope: it decides what you build, what you run, and what the report covers, and the comment must answer it directly, in the **Verdict** line when it is the whole point of the review. It never waives the rule in Step 3 that a fix is only confirmed after watching the bug fail, and a request to fix something is answered with a suggested fix in the comment, never an edit.

## Comments

Post every PR and issue comment with `--body-file`, never inline `--body "…"`: backticks and `$` in a review comment get command-substituted out of an inline body. End every comment with the fixed line `🤖 AI-assisted review`.

**Exception:** `/azp run …` CI triggers post raw with no footer; Azure Pipelines matches on exact command text.

## Setup (once per session)

```bash
# <review-dir>: record it; substitute literally, never as a shell variable
echo "$(git rev-parse --path-format=absolute --git-common-dir)/quokka-review"
```

## Steps

| Step | Skills used |
|---|---|
| 1. Check readiness | `quokka-check-pr-readiness` |
| 2. List the claims | none; see `references/choosing-test-problems.md` |
| 3. Test each claim | `quokka-checkout-pr-and-default`, `quokka-build-problem`, `quokka-run-problem`, `quokka-assess-pr-results`, `quokka-assess-pr-performance` |
| 4. Check unaffected behaviour | the same as Step 3 |
| 5. Validate the findings | none |
| 6. Write the report and comment | none |
| 7. Post the comment | none |

When a build fails, use `quokka-debug-build-failure`; when a run crashes or gives a wrong result, use `quokka-debug-run-failure`. For one review from start to finish, see `references/example-review.md`.

### Step 1: Check readiness

Run `quokka-check-pr-readiness`. If a GPU test is missing and you can trigger it, post the `/azp run` command yourself (see **Comments**) rather than list it as a finding.

If the PR is not ready for any other reason, stop: post its readiness table as the comment, with the verdict `not ready for review` and one line per missing item, and do not review further. Readiness is the author's side and correctness is the review's; a draft or an incomplete PR gets the short note, not a full review.

### Step 2: List the claims

```bash
gh pr view NNNN
gh pr diff NNNN
```

Derive a kebab-case `<slug>` from the title, such as `fix-amr-regrid` from "Fix AMR regrid bug".

List every claim the PR makes, from the PR description, linked issues and the diff. Where the description is vague, infer the claim from the diff and note the gap. Each kind of claim sets the evidence Step 3 produces:

| Claim | Evidence needed |
|---|---|
| Fixes a bug, or makes a run robust | the failure on the default branch, then the same scenario passing on the PR |
| Adds a feature | each promised behaviour tested on the PR: new parameters and TOML keys, outputs and effects, stated limits and error conditions |
| Improves accuracy, or an order of convergence | a convergence test at three or more resolutions, on both branches |
| Improves performance | run time and memory on both branches, on the same machine and inputs |

A claim with no evidence the review can produce, and none in the PR description, is a finding. When unsure whether a PR fixes a bug or adds a feature, treat it as a bug fix; that is stricter. For which problem to run for each kind of claim, see `references/choosing-test-problems.md`.

Scan the diff for `AMREX_GPU_DEVICE`, `amrex::ParallelFor`, or changes to `simulation.hpp` / `QuokkaSimulation.hpp`; those make the GPU build mandatory. A passing CPU build does not clear a PR that touches GPU kernels.

### Step 3: Test each claim

Check out the PR and the default branch with `quokka-checkout-pr-and-default`, build with `quokka-build-problem` (read it before the first `quokka config`), and run with `quokka-run-problem`. Record each run as a row of the evidence table (Step 6).

**A bug fix.** Never claim a PR fixes something without first watching it fail on the default branch.

1. On the default branch, run with the parameters that expose the bug, and record exactly what you observe: file count, error message, incorrect value. If the bug does not reproduce, stop and tell the user why: already fixed upstream, wrong parameters, platform difference.
2. On the PR, run the identical scenario. Confirm the wrong behaviour is gone, the run completes cleanly, and the output matches the claim.

**A feature.** Test each promised behaviour on the PR, the usual case first and then the stated limits. Mark each one verified, failed, or not testable.

**An accuracy claim.** Run the convergence test on both branches, and compare error norms and convergence rates with `quokka-assess-pr-results`.

**A performance claim,** or a change to a hot loop: compare the branches with `quokka-assess-pr-performance`.

Build on the GPU as well when Step 2 flagged device code.

### Step 4: Check unaffected behaviour

Run one scenario the PR should not change, such as an explicit configuration, a different code path, or an adjacent feature, and one boundary the PR description does not cover. Confirm with `quokka-assess-pr-results` that the results match the default branch.

### Step 5: Validate the findings

Before anything is written up, check each finding again, on its own:

1. Re-read the code at its `file:line` on the PR head, not from memory of the diff.
2. Confirm the evidence behind it: a run, a build or a number recorded in the evidence table, or a quoted line of code.
3. Keep it only if both hold. A finding that cannot be confirmed is dropped, not softened.

Where the tool allows it, give each finding to a fresh agent with only the PR description, the finding and its evidence, so that the check does not inherit the reasoning that produced it.

Do not report:

- issues that were already on the default branch, beyond one row marked `no` in the last column;
- anything a linter or formatter would catch, or that CI already reports;
- style, naming or taste, unless it breaks a rule written in `AGENTS.md` or `CLAUDE.md`;
- AMReX performance advisories, unless the PR is about grid decomposition.

### Step 6: Write the report and comment

Two files, two audiences. The **report** is the full record, as detailed as the work warrants. The **comment** is what a maintainer reads first, to decide what to do. Never post the report as the comment.

One rule governs the comment: **the author does not need convincing, they need locating.**

#### Structure levels

Each part of a review takes the lowest structure level that holds it, and moves up a level when it no longer fits. Higher levels add structure, not words, so the meaningful information stays easy to find as a review grows.

| Level | Form | Used for |
|---|---|---|
| 1 | One line of prose | The verdict, what was checked, and what is left for the maintainer |
| 3 | Table | Findings and evidence: every row has the same fields |
| 4 | Subsection | A finding whose explanation does not fit in its row, such as a root cause or a derivation; in the report only |

A table needs real columns: each one has a name and holds the same kind of value in every row. If a column could only be called "Notes", the content is a bullet list, not a table.

#### The report

```bash
ls <review-dir>/ 2>/dev/null | grep "^prNNNN-" || echo "none"
mkdir -p <review-dir>/prNNNN-<slug>
```

Write inside an existing `prNNNN-*` folder if there is one. Filename `<folder>-REVIEW.md`; follow-up rounds append `-v2`, `-v3`.

The report holds, in this order:

1. **Verdict, what was checked, and what is left for the maintainer:** one line each, as in the comment.
2. **Claims:** a table with one row per claim from Step 2, and whether the evidence supports it.
3. **Evidence:** a table with one row per run or check.

   | Step | Command | Commit | Preset and machine | Result | Output |
   |---|---|---|---|---|---|

4. **Findings:** a table with one row per finding, the same as in the comment.
5. **One subsection per finding that needs explaining:** the root cause with `file:line` for a bug fix, or the match against the claim for a feature. Derivations, parameter sweeps and raw numbers go here.
6. **Left for the maintainer:** a bullet list of what the review did not settle, such as whether the physics is right, design choices, and claims it could not test.
7. **Pre-existing issues:** a bullet list.

#### The comment

A separate file, `<folder>-COMMENT.md`:

```markdown
**Verdict:** <ready for maintainer review | N blocking | cannot verify X>
**Checked:** <preset and problems built, tests run> at `<short-sha>`
**Left for the maintainer:** <what the review did not settle, in one line>

| Finding | Where | Evidence | Blocks merge |
|---|---|---|---|
| <what is wrong> | `file.cpp:NN` | <trigger and result, with the number> | yes / no |

<details><summary>Evidence</summary>

| Step | Command | Commit | Result |
|---|---|---|---|
| … | … | … | … |

</details>

🤖 AI-assisted review
```

For example, one finding:

| Finding | Where | Evidence | Blocks merge |
|---|---|---|---|
| `num_periods` is not validated | 7 problem files | `num_periods=0` or `=-1` runs zero steps and reports PASS at error norm 6.37e-18 | yes |

1. **Verdict first.** Line 1 is the conclusion a maintainer acts on. The verdict is one of: `ready for maintainer review` (no blocking findings), `N blocking`, `cannot verify X`, or `not ready for review` (from Step 1). Never write "ship it" or approve: only a person approves.
2. **One row per finding,** at most 5, worst first, with the rows that block merging before the rest. Drop any finding whose fix would not change what the author does. A pre-existing issue gets one row with `no` in the last column, or none.
3. **Numbers, not adjectives.** `error norm 1.33e-09 vs tol 3e-03`, not "well within tolerance". Paste the real abort message.
4. **A row holds the finding and its number, not the explanation.** An explanation that does not fit goes into the report as a subsection.
5. **No preamble, no recap, no closer.** Start with the verdict, end with the table.
6. **No findings is a complete answer.** Replace the table with `No findings.`

Before posting, check: reading **only** the Verdict line, does a maintainer know what to do next? If not, rewrite that line.

**Follow-up rounds (v2, v3, …):** re-read the code at every carried-over `file:line` and drop findings whose code is gone. A round where everything was fixed is three lines: verdict, what you re-ran, what still stands.

### Step 7: Post the comment

```bash
gh pr comment NNNN --body-file <review-dir>/<folder>/<folder>-COMMENT.md
```

Then a 1–2 sentence summary in chat.
