---
name: quokka-build-problem
description: "Build one Quokka problem with the right preset, on CPU or GPU: pick the preset, detect the GPU toolchain and device, then configure and build."
---

# Build a Quokka Problem

Build one problem with the right preset, on CPU or GPU.

**Before this:** check out the branch to build, for example with `quokka-checkout-pr-and-default`.

## Set up (once per session, from the repository root)

Install `quokka` and confirm that it is not stale, as described under **Build & Test Commands** in `CLAUDE.md`. Then record the repository root:

```bash
# Record the output and substitute it literally below, not as a shell variable.
git rev-parse --show-toplevel
```

Every `quokka` command below takes `--root <repo-root>` and `--source default`. Keep both.

## Placeholders

These placeholders are used here, in `quokka-run-problem`, and in the skills that call them:

| Placeholder | Stands for |
|---|---|
| `<repo-root>` | the repository root, recorded above |
| `<Problem>` | the problem to build, which is the name of its folder in `src/problems/`, such as `HydroShocktube` |
| `<preset>` | the build preset (defined in **Presets** below), such as `1d` or `3d-cuda` |
| `<num-dims>` | the number of dimensions of a preset: `1`, `2` or `3` |
| `<suffix>` | the GPU backend of a preset, `cuda` or `hip`, found in **Choose the preset** below |

## Presets

A preset is a named build configuration of the `quokka` script. One name sets the number of dimensions, whether the build is Release or Debug, and the GPU backend or compiler.

Each preset builds into its own folder, `<repo-root>/build/<preset>`. `quokka --help` lists them all, and with no `-d` the script uses `1d`.

| Presets | Build |
|---|---|
| `1d`, `2d`, `3d` | Release, CPU |
| `1d-debug`, `2d-debug`, `3d-debug` | Debug, CPU |
| `1d-cuda`, `2d-cuda`, `3d-cuda` | Release, with the CUDA GPU backend |
| `1d-hip`, `2d-hip`, `3d-hip` | Release, with the HIP GPU backend |
| `1d-apple`, `2d-apple`, `3d-apple` | Release, with Apple clang |

## Choose the preset

- **Use the lowest dimension the problem allows.** Move to `2d` or `3d` only when the problem needs it.
- **On macOS,** append `-apple` to the preset: `1d-apple`, `2d-apple`, `3d-apple`.
- **For a change that touches GPU code,** also build with the GPU preset whenever a GPU toolchain is installed. On a machine with a toolchain but no GPU, the build alone is the check; running on a GPU is left to CI.
- **For a slow CPU test on a machine with a GPU,** run it with the GPU preset.

Check for the toolchain and the device separately: the toolchain decides whether you can compile for the GPU, and the device decides whether you can run on it. `nvcc` may be installed but missing from the interactive `PATH`.

```bash
# Toolchain: which GPU preset suffix to use
if [ -x /usr/local/cuda/bin/nvcc ] || command -v nvcc &>/dev/null; then
    echo "CUDA → suffix: -cuda"
elif command -v hipcc &>/dev/null || [ -d /opt/rocm ]; then
    echo "ROCm → suffix: -hip"
else
    echo "No GPU toolchain → skip GPU build steps"
fi

# Device: can you run on the GPU, or only compile?
nvidia-smi -L 2>/dev/null || rocm-smi --showid 2>/dev/null || ls /dev/kfd 2>/dev/null \
    || echo "No GPU device → compile-only check"
```

## Configure and build

Run `quokka config` again after every checkout or submodule update, then build:

```bash
quokka config -d <preset> --delete --source default --root <repo-root> -DQUOKKA_PYTHON=OFF
quokka build -d <preset> <Problem> --source default --root <repo-root>
```

For the GPU build, run the same two commands with `-d <num-dims>d-<suffix>`, for example `-d 3d-cuda`.

To run what was built, use `quokka-run-problem`.
