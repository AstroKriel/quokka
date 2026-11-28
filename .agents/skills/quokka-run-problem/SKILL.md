---
name: quokka-run-problem
description: "Run a built Quokka problem with its inputs and any parameter overrides, without leaving artifacts in the source tree."
---

# Run a Quokka Problem

Build the problem first with `quokka-build-problem`, which also defines the placeholders used here: `<repo-root>`, `<preset>` and `<Problem>`.

## Run the binary directly

Run from `<repo-root>/tests`. That folder is ignored by git, so the output files do not clutter the source tree. Provide both the binary and the input file as absolute paths, since the working directory is not the repository root:

```bash
cd <repo-root>/tests
<repo-root>/build/<preset>/src/problems/<Problem>/<Problem> \
    <repo-root>/inputs/<Problem>.toml <param>=<value>
```

Each `<param>=<value>` after the input file overrides one parameter in it. `quokka run` does the same thing for you.

To compare the output against the default branch, use `quokka-assess-pr-results`.
