---
name: quokka-review-issue
description: "Review a Quokka GitHub issue or audit finding: check it against the current code, reproduce if needed, then post a short verdict comment. Analysis only; does not implement fixes."
---

# Quokka Issue Review

Check an issue, or an audit finding, against the current code, and post a short verdict. This skill analyses and reports; it never changes the source code.

**Before this:** follow **Comments** and **Set up** in `quokka-review-pr`. This review uses the same `<review-dir>`, the same two files, and the same comment rules as **Step 6** there.
**Terms** (defined in `.agents/skills/glossary.md`): claim, default branch, finding, `<file>:<line>`, `<issue-number>`, `<pr-number>`, `<review-dir>`, `<short-name>`.

## The argument

A target, plus an optional instruction:

- `issue 2110`: an issue, whose number is written `<issue-number>` below
- text after the target: a specific instruction

With no argument, ask which target. For a pull request, use `quokka-review-pr`.

## Review the issue

```bash
gh issue view <issue-number>
```

Derive a `<short-name>` from the title. Then:

1. Read the affected source files, and check the git log for related commits and PRs.
2. If the code and its history make the root cause clear, that is enough.
3. Otherwise, reproduce the issue on the default branch: check it out with `quokka-checkout-pr-and-default`, and build it with `quokka-build-problem`.

For audit issues (labels `agent/audit` and `agent/codex`), check each finding the audit claims against the **current** code. For each one, state whether it is still valid, or why it is not.

## Write the report and comment

The report goes to `<review-dir>/issue<issue-number>-<short-name>/issue<issue-number>-<short-name>.md`. It gives:

- the root cause;
- where the fix belongs;
- any latent concerns;
- whether the issue is resolved or still open.

The comment goes to `issue<issue-number>-<short-name>-COMMENT.md`, and follows the comment rules in `quokka-review-pr`:

```markdown
**Verdict:** <root cause in one line | still open | already fixed by #<pr-number>>
**Fix location:** `<file>:<line>` <what changes>

**Also worth knowing**
- <one line each, at most 3>

🤖 AI-assisted review
```

For an audit issue, write one line per finding instead: `**Finding N:** valid / invalid because X`.

## Post the comment

```bash
gh issue comment <issue-number> \
    --body-file <review-dir>/issue<issue-number>-<short-name>/issue<issue-number>-<short-name>-COMMENT.md
```

Then provide a summary of one or two sentences in the chat.
