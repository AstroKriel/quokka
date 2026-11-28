# Example Review: an Accuracy Claim

An illustration of one review from start to finish, for a PR that changes the hydro reconstruction and claims better accuracy. It is not a real review: values are shown as `<placeholders>`.

## Step 1: Check readiness

`quokka-check-pr-readiness` reports every check passed, except GPU tests: no `/azp run` yet, and the PR changes shared hydro code. The agent posts `/azp run` itself and continues.

## Step 2: List the claims

| Claim | Evidence needed |
|---|---|
| Lower error on smooth flow | a convergence test at three or more resolutions, on both branches |
| A new parameter `hydro.<option>`, off by default | the default gives the same result as the default branch; the option changes it |

The diff touches code inside `amrex::ParallelFor`, so the GPU build is needed too.

## Step 3: Test each claim

Both branches are checked out (`quokka-checkout-pr-and-default`), and `HydroWaveConvergence` is built in 1D (`quokka-build-problem`, chosen from `choosing-test-problems.md`) and run on both (`quokka-run-problem`). `quokka-assess-pr-results` compares the error norms and convergence rates. The PR is also built with the GPU preset; the machine has no GPU, so the build is the check.

## Step 4: Check unaffected behaviour

`HydroShocktube` with the new option off, on both branches: the results match. With the option on, the shock is captured without new oscillations.

## Step 5: Validate the findings

One finding survives: the PR description claims fourth order, and the measured rate is `<rate>`. A second, about a missing default in the parameter documentation, is dropped: the default is documented in `docs/markdown/parameters.md`.

## Step 6: The comment

```markdown
**Verdict:** 1 blocking
**Checked:** 1d preset, HydroWaveConvergence at 32 to 256 cells and HydroShocktube on both branches; 3d-cuda build of the PR, at `<short-sha>`
**Left for the maintainer:** whether this reconstruction is the right default for the solver

| Finding | Where | Evidence | Blocks merge |
|---|---|---|---|
| Claimed fourth-order convergence is not reached | PR description | measured rate `<rate>` between 128 and 256 cells; default branch `<rate>` | yes |

<details><summary>Evidence</summary>

| Step | Command | Commit | Result |
|---|---|---|---|
| 3 | HydroWaveConvergence, default branch | `<sha>` | L1 errors `<e32>`, `<e64>`, `<e128>`, `<e256>` |
| 3 | HydroWaveConvergence, PR | `<sha>` | L1 errors `<e32>`, `<e64>`, `<e128>`, `<e256>` |
| 3 | 3d-cuda build of the PR | `<sha>` | builds |
| 4 | HydroShocktube, option off, both branches | `<sha>` | identical |

</details>

🤖 AI-assisted review
```

The report holds the same rows in full, plus a subsection with the convergence table, and the list of what is left for the maintainer.
