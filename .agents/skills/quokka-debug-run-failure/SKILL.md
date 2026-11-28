---
name: quokka-debug-run-failure
description: "Narrow down why a Quokka problem crashes, aborts or gives a wrong result at run time, and whether the PR caused it."
---

# Debug a Run Failure

Find why a problem crashes, aborts or gives a wrong result at run time, and whether the PR caused it. The full guide, including GPU debuggers and sanitizers, is `docs/markdown/debugging.md`; this skill gives the order to work through it during a review.

**Before this:** the failing run, from `quokka-run-problem`.
**Terms:** default branch, pre-existing, preset, `<num-dims>` (defined in `.agents/skills/glossary.md`).

## Narrow it down

1. **Check whether the default branch fails the same way.** Run it with the same inputs. If it fails, the failure is not the PR's: report it as pre-existing.
2. **Make it small.** Find the lowest dimension, the smallest grid and the fewest steps that still show the failure, on one MPI rank.
3. **Move to the CPU.** If it failed on a GPU, try the CPU: a failure that also happens there is far easier to find.
4. **Rebuild with the Debug preset.** Use `<num-dims>d-debug` for the same dimension, and rerun. An out-of-bounds access on an `amrex::Array4` then stops with the array and the indices.
5. **Try AddressSanitizer.** Use it if the Debug build is too slow or finds nothing. Configure a Release preset with `-DENABLE_ASAN=ON` added to `quokka config`, and set the environment variables given in `docs/markdown/debugging.md`.
6. **Rule out asynchronous kernels for a failure only on the GPU.** Rerun with `CUDA_LAUNCH_BLOCKING=1` (NVIDIA) or `HIP_LAUNCH_BLOCKING=1` (AMD). Then follow "How to actually debug on GPUs" in `docs/markdown/debugging.md`.

## Report

Report the smallest inputs that fail, the command, the commit, the preset and machine, and the abort message, or the wrong value next to the expected one. Say whether the default branch fails the same way.
