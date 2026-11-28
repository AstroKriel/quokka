---
name: quokka-run-problem
description: "Run a built Quokka problem with its inputs and any parameter overrides, without leaving artifacts in the source tree."
---

# Run a Quokka Problem

Run a built problem, with its inputs and any parameter overrides.

**Before this:** build the problem with `quokka-build-problem`.
**Terms:** `<repo-root>`, `<preset>` and `<Problem>` (**Placeholders** in `quokka-build-problem`); default branch (`quokka-checkout-pr-and-default`).

## Run the binary directly

Run from `<repo-root>/tests`. Git ignores that folder, so the output files do not clutter the source tree.

The working directory is then not the repository root, so provide both the binary and the input file as absolute paths:

```bash
cd <repo-root>/tests
<repo-root>/build/<preset>/src/problems/<Problem>/<Problem> \
    <repo-root>/inputs/<Problem>.toml <param>=<value>
```

Each `<param>=<value>` after the input file overrides one parameter in it. `quokka run` does the same thing for you.

To compare the output against the default branch, use `quokka-assess-pr-results`.
