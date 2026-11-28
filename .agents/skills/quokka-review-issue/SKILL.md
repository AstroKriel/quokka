---
name: quokka-review-issue
description: "Review a Quokka GitHub issue or audit finding: check it against the current code, reproduce if needed, then post a short verdict comment. Analysis only; does not implement fixes."
---

# Quokka Issue Review

This skill analyses and reports. It never changes the source code.

## The argument

A target, plus an optional instruction:

- `issue 2110`: an issue
- text after the target: a specific instruction

With no argument, ask which target. For a pull request, use `quokka-review-pr`.

## Comments, setup and reporting

Follow **Comments**, **Set up**, and **Step 6: Write the report and comment** in `quokka-review-pr`: use the same `<review-dir>`, the same two files, and the same comment rules.

## Reviewing a GitHub Issue

```bash
gh issue view NNNN
```

Derive a `<slug>` from the title. Inspect the affected source files and check git log for related commits and PRs. If the root cause is clear from code and history, that is enough; otherwise build `origin/development` as `dev-<slug>` (`quokka-checkout-pr-and-default`, then `quokka-build-problem`) and reproduce.

For audit issues (labels `agent/audit` / `agent/codex`), verify each claimed finding against the **current** code, and state per finding: still valid, or invalid and why.

Write the same two files as for a PR. The report goes to `<review-dir>/issueNNNN-<slug>/issueNNNN-<slug>.md`, and gives the root cause, where the fix belongs, any latent concerns, and whether the issue is resolved or still open. The comment goes to `issueNNNN-<slug>-COMMENT.md`, following all the comment rules in `quokka-review-pr`:

```markdown
**Verdict:** <root cause in one line | still open | already fixed by #NNNN>
**Fix location:** `file.cpp:NN` <what changes>

**Also worth knowing**
- <one line each, at most 3>

🤖 AI-assisted review
```

For audit issues, one line per claimed finding: `**Finding N:** valid / invalid because X`.

```bash
gh issue comment NNNN --body-file <review-dir>/issueNNNN-<slug>/issueNNNN-<slug>-COMMENT.md
```

Then provide a summary of one or two sentences in the chat.
