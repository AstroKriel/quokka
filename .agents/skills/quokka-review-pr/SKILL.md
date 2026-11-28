---
name: quokka-review-pr
description: "Review a Quokka pull request — build and run the code to verify the claim, then post a short verdict comment. Analysis only; does not implement fixes."
---

# Quokka PR Review

Analyse and report. Never change source.

## The argument

A target, plus an optional instruction:

- `pr 2158`, or a bare `2158` — a pull request
- text after the target — a specific instruction: `pr 2158. suggest a fix for the nvcc error`

With no argument, ask which target. For an issue, use `quokka-review-issue`.

An instruction sets the scope: it decides what you build, what you run, and what the report covers, and the comment must answer it directly — in the **Verdict** line when it is the whole point of the review. It never waives the Iron Law, and a request to fix something is answered with a suggested fix in the comment, never an edit.

## Comments

Post every PR and issue comment with `--body-file`, never inline `--body "…"` — backticks and `$` in a review comment get command-substituted out of an inline body. End every comment with the fixed line `🤖 AI-assisted review`.

**Exception:** `/azp run …` CI triggers post raw with no footer; Azure Pipelines matches on exact command text.

## Setup (once per session)

```bash
# <review-dir> — record it; substitute literally, never as a shell variable
echo "$(git rev-parse --path-format=absolute --git-common-dir)/quokka-review"
```

## Skills this review uses

| Step | Skill |
|---|---|
| Check out the PR and the default branch on throwaway branches, and clean up after | `quokka-checkout-pr-and-default` |
| Build a problem with the right preset | `quokka-build-problem` |
| Run a built problem | `quokka-run-problem` |
| Read the output against the default branch | `quokka-assess-pr-results` |

## Reviewing a Pull Request

### Step 1 — Read and classify

```bash
gh pr view NNNN
gh pr diff NNNN
```

Derive a kebab-case `<slug>` from the title — `fix-amr-regrid` from "Fix AMR regrid bug".

| Type | Signal | Cycle |
|---|---|---|
| Bug fix | "fix", "closes issue #N", describes wrong existing behavior | RED → GREEN → EDGE |
| Feature | "add", "implement", "new", describes a new capability | SPEC → VERIFY → EDGE |

When in doubt use the **bug fix** cycle; it is stricter.

Scan the diff for `AMREX_GPU_DEVICE`, `amrex::ParallelFor`, or changes to `simulation.hpp` / `QuokkaSimulation.hpp` — those make the GPU build mandatory.

**Before the first `quokka config`, read `quokka-build-problem`.** It sets up the environment, picks the preset, and holds the exact build commands.

---

## Bug Fix PRs: RED → GREEN → EDGE

```
NO CLAIM THAT A PR "FIXES" SOMETHING WITHOUT FIRST WATCHING IT FAIL ON THE BASE BRANCH
```

### Step 2 — RED

Build `origin/development` as `dev-<slug>`, then run with the parameters that expose the bug.

Document exactly what you observe: file count, error message, incorrect value. If the bug does not reproduce, stop and tell the user why — already fixed upstream, wrong parameters, platform difference.

### Step 3 — GREEN

Build the PR as `pr-<slug>` and run the identical scenario. Confirm the wrong behavior is gone, the run completes clean, and the output matches the PR's claim.

Add the GPU build if Step 1 flagged device code. A passing CPU build does not clear a PR that touches GPU kernels.

### Step 4 — EDGE

Run one scenario the PR should not affect — an explicit configuration, a different code path, an adjacent feature. Confirm it records the same value it records on the base branch.

Then go to **Reporting**.

---

## Feature PRs: SPEC → VERIFY → EDGE

### Step 2 — SPEC

Before building, write the contract from the PR description, linked issues, and diff:

- new inputs: parameters, TOML keys, function arguments
- outputs and effects
- stated boundary and error conditions

Infer from the diff where the description is vague, and note the gaps.

### Step 3 — VERIFY

Build the PR as `pr-<slug>`. Work the checklist — golden path first, then the stated boundaries. Mark every SPEC item verified, failed, or not-testable before writing the report.

Add the GPU build on the same rule as the bug-fix cycle.

### Step 4 — EDGE

Test one boundary the description does not cover, and run one existing adjacent test for regression.

Then go to **Reporting**.

---

## Reporting

Two files, two audiences. The **report** is the full record, as detailed as the work warrants. The **comment** is what a maintainer reads first, to decide what to do. Never post the report as the comment.

One rule governs the comment: **the author does not need convincing, they need locating.**

### Structure levels

Each part of a review takes the lowest structure level that holds it, and moves up a level when it no longer fits. Higher levels add structure, not words, so the meaningful information stays easy to find as a review grows.

| Level | Form | Used for |
|---|---|---|
| 1 | One line of prose | The verdict, and what was checked |
| 3 | Table | Findings and evidence: every row has the same fields |
| 4 | Subsection | A finding whose explanation does not fit in its row, such as a root cause or a derivation; in the report only |

A table needs real columns: each one has a name and holds the same kind of value in every row. If a column could only be called "Notes", the content is a bullet list, not a table.

### 1. Report

```bash
ls <review-dir>/ 2>/dev/null | grep "^prNNNN-" || echo "none"
mkdir -p <review-dir>/prNNNN-<slug>
```

Write inside an existing `prNNNN-*` folder if there is one. Filename `<folder>-REVIEW.md`; follow-up rounds append `-v2`, `-v3`.

The report holds, in this order:

1. **Verdict and what was checked:** one line each, as in the comment.
2. **Evidence:** a table with one row per phase.

   | Phase | Command | Commit | Preset and machine | Result | Output |
   |---|---|---|---|---|---|

3. **Findings:** a table with one row per finding, the same as in the comment.
4. **One subsection per finding that needs explaining:** the root cause with `file:line` for a bug fix, or the match against the design for a feature. Derivations, parameter sweeps and raw numbers go here.
5. **Pre-existing issues:** a bullet list.

### 2. Comment

A separate file, `<folder>-COMMENT.md`:

```markdown
**Verdict:** <ship it | N blocking | cannot verify X>
**Checked:** <preset and problems built, tests run> at `<short-sha>`

| Finding | Where | Evidence | Blocks merge |
|---|---|---|---|
| <what is wrong> | `file.cpp:NN` | <trigger and result, with the number> | yes / no |

<details><summary>Evidence</summary>

| Phase | Command | Commit | Result |
|---|---|---|---|
| … | … | … | … |

</details>

🤖 AI-assisted review
```

For example, one finding:

| Finding | Where | Evidence | Blocks merge |
|---|---|---|---|
| `num_periods` is not validated | 7 problem files | `num_periods=0` or `=-1` runs zero steps and reports PASS at error norm 6.37e-18 | yes |

1. **Verdict first.** Line 1 is the conclusion a maintainer acts on.
2. **One row per finding,** at most 5, worst first, with the rows that block merging before the rest. Drop any finding whose fix would not change what the author does. A pre-existing issue gets one row with `no` in the last column, or none.
3. **Numbers, not adjectives.** `error norm 1.33e-09 vs tol 3e-03`, not "well within tolerance". Paste the real abort message.
4. **A row holds the claim and its number, not the explanation.** An explanation that does not fit goes into the report as a subsection.
5. **No preamble, no recap, no closer.** Start with the verdict, end with the table.
6. **No findings is a complete answer.** Replace the table with `No findings.`

Before posting, check: reading **only** the Verdict line, does a maintainer know whether to merge? If not, rewrite that line.

**Follow-up rounds (v2, v3, …):** re-read the code at every carried-over `file:line` and drop findings whose code is gone. A round where everything was fixed is three lines — verdict, what you re-ran, what still stands.

### 3. Post

```bash
gh pr comment NNNN --body-file <review-dir>/<folder>/<folder>-COMMENT.md
```

Then a 1–2 sentence summary in chat.
