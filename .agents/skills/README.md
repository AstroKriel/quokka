# Quokka Agent Skills

Skills for AI coding agents working on Quokka, also written to be read and followed by people. Each skill is a folder with a `SKILL.md`; Codex reads them here, and `.claude/skills/` links to each one for Claude Code.

Names follow `quokka-<verb>-<subject>`. Skills that are variants of the same task share the verb and differ in the specific case at the end, for example `quokka-review-pr` and `quokka-review-issue`.

| Skill | Use it to |
|---|---|
| `quokka-review-pr` | review a pull request: build and run it against the default branch, then post a short verdict |
| `quokka-review-issue` | check an issue or audit finding against the current code |
| `quokka-check-pr-readiness` | check that a PR has what a maintainer needs before reviewing: out of draft, CI passed, description complete, GPU tests triggered, threads resolved, new tests run by CI |
| `quokka-checkout-pr-and-default` | check out a PR and the default branch side by side on throwaway branches, and clean up after |
| `quokka-build-problem` | build one problem with the right preset, on CPU or GPU |
| `quokka-run-problem` | run a built problem with its inputs and parameter overrides |
| `quokka-assess-pr-results` | assess the output of a PR against the default branch: what to weigh, what to ignore, marginal failures |
| `quokka-assess-pr-performance` | assess the run time and memory of a PR against the default branch, on the same machine and inputs |
| `quokka-debug-build-failure` | narrow down why a problem fails to configure or build |
| `quokka-debug-run-failure` | narrow down why a problem crashes, aborts or gives a wrong result at run time |

## Terms

Each term is defined in one place, and every skill that uses it points there.

| Term | Defined in |
|---|---|
| Claim, evidence, finding, maintainer, pre-existing | **Terms** in `quokka-review-pr` |
| Structure level, evidence table | Step 6 of `quokka-review-pr` |
| Default branch | `quokka-checkout-pr-and-default` |
| Readiness | `quokka-check-pr-readiness` |
| Hot path | `quokka-assess-pr-performance` |
| Preset | **Presets** in `quokka-build-problem` |
| `<repo-root>`, `<Problem>`, `<preset>`, `<num-dims>`, `<suffix>` | **Placeholders** in `quokka-build-problem` |
| `<short-name>` | Step 2 of `quokka-review-pr` |
| `<pr-number>`, `<file>:<line>` | **Terms** in `quokka-review-pr` |
| `<issue-number>` | **The argument** in `quokka-review-issue` |
| `<review-dir>` | **Set up** in `quokka-review-pr` |

## Adding a skill

A new skill gets a row here, a folder in `.agents/skills/`, and a link in `.claude/skills/` (`ln -s ../../.agents/skills/<skill> .claude/skills/<skill>`).
