---
name: quokka-review-issue
description: "Review a Quokka GitHub issue or audit finding — check it against the current code, reproduce if needed, then post a short verdict comment. Analysis only; does not implement fixes."
---

# Quokka Issue Review

Analyse and report. Never change source.

## The argument

A target, plus an optional instruction:

- `issue 2110` — an issue
- text after the target — a specific instruction

With no argument, ask which target. For a pull request, use `quokka-review-pr`.

## Comments, setup and reporting

Follow **Comments**, **Setup** and **Reporting** in `quokka-review-pr`: the same `<review-dir>`, the same two files, and the same comment rules.

## Reviewing a GitHub Issue

```bash
gh issue view NNNN
```

Derive a `<slug>` from the title. Inspect the affected source files and check git log for related commits and PRs. If the root cause is clear from code and history, that is enough; otherwise build `origin/development` as `dev-<slug>` (`quokka-checkout-pr-and-default`, then `quokka-build-problem`) and reproduce.

For audit issues — labels `agent/audit` / `agent/codex` — verify each claimed finding against the **current** code, and state per finding: still valid, or invalid and why.

Same two files. Report to `<review-dir>/issueNNNN-<slug>/issueNNNN-<slug>.md`: root cause, fix location(s), latent concerns, and whether the issue is resolved or still open. Then `issueNNNN-<slug>-COMMENT.md`, under all the comment rules in `quokka-review-pr`:

```markdown
**Verdict:** <root cause in one line | still open | already fixed by #NNNN>
**Fix location:** `file.cpp:NN` <what changes>

**Also worth knowing**
- <one line each, at most 3>

🤖 AI-assisted review
```

For audit issues, one line per claimed finding: `**Finding N** — valid / invalid because X`.

```bash
gh issue comment NNNN --body-file <review-dir>/issueNNNN-<slug>/issueNNNN-<slug>-COMMENT.md
```

Then a 1–2 sentence summary in chat.
