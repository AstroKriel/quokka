---
name: quokka-run-problem
description: "Run a built Quokka problem with its inputs and any parameter overrides, without leaving artifacts in the source tree."
---

# Run a Quokka Problem

Build it first with `quokka-build-problem`.

## Running a binary directly

Run from `<repo-root>/tests`, which is gitignored, so artifacts do not dirty the tree. Both paths absolute:

```bash
cd <repo-root>/tests
<repo-root>/build/<preset>/src/problems/<Target>/<Target> \
    <repo-root>/inputs/<Target>.toml <param>=<value>
```

`quokka run` already does this.

To read the output against the default branch, use `quokka-assess-pr-results`.
