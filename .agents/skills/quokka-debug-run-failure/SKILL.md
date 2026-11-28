---
name: quokka-debug-run-failure
description: "Narrow down why a Quokka problem crashes, aborts or gives a wrong result at run time, and whether the PR caused it."
---

# Debug a Run Failure

The full guide, including GPU debuggers and sanitizers, is `docs/markdown/debugging.md`. This is the order to work through it in during a review.

## Narrow it down

1. **Check whether the default branch fails the same way,** with the same inputs (`quokka-checkout-pr-and-default`, `quokka-build-problem`, `quokka-run-problem`). If it does, the failure is not the PR's: report it as pre-existing.
2. **Make it small:** the lowest dimension, the smallest grid and the fewest steps that still show the failure, on one MPI rank.
3. **Move to the CPU** if it failed on a GPU. A failure that also happens on the CPU is far easier to find.
4. **Rebuild in Debug** (`-DCMAKE_BUILD_TYPE=Debug`) and rerun. An out-of-bounds access on an `amrex::Array4` then stops with the array and the indices.
5. **Try AddressSanitizer** (`-DENABLE_ASAN=ON`) if Debug is too slow or finds nothing; set the environment variables given in `docs/markdown/debugging.md`.
6. **GPU only:** if it fails on the GPU and not on the CPU, rerun with `CUDA_LAUNCH_BLOCKING=1` (NVIDIA) or `HIP_LAUNCH_BLOCKING=1` (AMD) to rule out asynchronous kernels, then follow "How to actually debug on GPUs" in `docs/markdown/debugging.md`.

## Report

Report the smallest inputs that fail, the command, the commit, the preset and machine, and the abort message, or the wrong value next to the expected one. Say whether the default branch fails the same way.
