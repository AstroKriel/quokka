---
name: quokka-build-problem
description: "Build one Quokka problem with the right preset, on CPU or GPU: pick the preset, detect the GPU toolchain and device, then configure and build."
---

# Build a Quokka Problem

## Setup (once per session, from the repo root)

Install `quokka` and confirm it is not stale — see **Build & Test Commands** in `CLAUDE.md`.

```bash
# Repo root — record it; substitute literally below, never as a shell variable
git rev-parse --show-toplevel
```

Every `quokka` command below carries `--root <repo-root>` and `--source default`. Keep both.

## Choosing the preset

- **Lowest dimensionality the problem allows.** Move to `2d` / `3d` only when it demands it.
- **macOS** — append `-apple` (`1d-apple`, `2d-apple`, `3d-apple`).
- **GPU-touching change** — build with the GPU preset whenever a toolchain exists. Toolchain but no device: the compile alone is the check, runtime is CI's job.
- **Slow CPU test and a GPU device present** — run on the GPU preset.

Detect toolchain and device separately: the toolchain decides whether you can compile, the device whether you can run. `nvcc` may be off the interactive PATH.

```bash
# toolchain → GPU preset suffix
if [ -x /usr/local/cuda/bin/nvcc ] || command -v nvcc &>/dev/null; then
    echo "CUDA → suffix: -cuda"
elif command -v hipcc &>/dev/null || [ -d /opt/rocm ]; then
    echo "ROCm → suffix: -hip"
else
    echo "No GPU toolchain → skip GPU build steps"
fi

# device → can you RUN on the GPU, or only compile?
nvidia-smi -L 2>/dev/null || rocm-smi --showid 2>/dev/null || ls /dev/kfd 2>/dev/null \
    || echo "No GPU device → compile-only check"
```

## Configure and build

On a checked-out ref (`quokka-checkout-pr-and-default`) — `quokka config` must run after every checkout or submodule update:

```bash
quokka config -d <preset> --delete --source default --root <repo-root> -DQUOKKA_PYTHON=OFF
quokka build -d <preset> <Target> --source default --root <repo-root>
```

For the GPU build, the same two commands with `-d <Nd>-<suffix>`.

To run what was built, use `quokka-run-problem`.
