# Glossary

Every term the Quokka skills use in a specific sense is defined here, once. Each skill lists the terms it uses in its header, and points here.

## Terms

| Term | Meaning |
|---|---|
| Claim | what a PR says it does: fixes a bug, adds a feature, improves accuracy, or improves performance |
| Default branch | `development`, the branch every PR merges into |
| Evidence | what a build, a run, or a reading of the code shows |
| Evidence table | the table in a review report with one row per run or check; its columns are given in Step 6 of `quokka-review-pr` |
| Finding | a problem that the evidence supports |
| Hot path | the code that runs on every time step; how to tell whether a change is on it is set out in **The hot path** in `quokka-assess-pr-performance` |
| Maintainer | the person who reviews a PR after the agent's review, and who alone can approve it |
| Pre-existing | already present on the default branch, so not caused by the PR |
| Preset | a named build configuration of the `quokka` script, setting the number of dimensions, Release or Debug, and the GPU backend or compiler; the presets are listed in **Presets** in `quokka-build-problem` |
| Readiness | whether a PR has what a maintainer needs before reviewing starts: the author's side of the PR, not whether the change is correct; checked by `quokka-check-pr-readiness` |
| Structure level | the form a part of a review takes, from a line of prose to a table to a subsection; the levels a review uses are given in Step 6 of `quokka-review-pr` |

## Placeholders

| Placeholder | Stands for |
|---|---|
| `<file>:<line>` | where a finding is in the code: a path from the repository root and a line number, such as `src/hydro/HLLC.hpp:120` |
| `<folder>` | the folder of one review, `pr<pr-number>-<short-name>` or `issue<issue-number>-<short-name>`, inside `<review-dir>` |
| `<issue-number>` | the number of the issue under review, such as `2110` |
| `<num-dims>` | the number of dimensions of a preset: `1`, `2` or `3` |
| `<pr-number>` | the number of the PR under review, such as `2468` |
| `<preset>` | a preset name, such as `1d` or `3d-cuda` |
| `<Problem>` | a problem, which is the name of its folder in `src/problems/`, such as `HydroShocktube` |
| `<repo-root>` | the root of the repository, from `git rev-parse --show-toplevel` |
| `<review-dir>` | the folder that holds all review reports, set up in `quokka-review-pr` |
| `<short-name>` | a few lowercase words joined by hyphens, derived from a PR or issue title, such as `fix-amr-regrid`; it names throwaway branches and review folders |
| `<suffix>` | the GPU backend part of a preset: `cuda` or `hip` |
