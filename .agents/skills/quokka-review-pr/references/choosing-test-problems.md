# Choosing Test Problems

Which Quokka problems to run for each kind of claim, and for the unaffected case. Each problem is a folder in `src/problems/` with its inputs in `inputs/`; read its test file for what it checks and its pass criterion before relying on it. Pick the lowest dimension that exercises the change (`quokka-build-problem`).

## Accuracy and convergence

For a claim of better accuracy or a higher order of convergence: a smooth problem with a known solution, run at three or more resolutions on both branches.

| Physics | Problems |
|---|---|
| Hydrodynamics | `HydroWaveConvergence` (a Richardson convergence sweep, also run in CI by `hydro-wave-convergence.yml`), `Advection` |
| MHD | `MHDAlfvenWaveLinearConvergence`, `MHDAlfvenWaveCircularConvergence`, `MHDEntropyWaveConvergence`, `MHDFastWaveConvergence`, `MHDSlowWaveConvergence` |
| Thermal conduction | `ThermalConductionAnisoGaussian`, `ThermalConductionSpitzerGaussian` |

Several convergence problems choose between a convergence sweep and a single run from their inputs (for example `setup.run_convergence` and `setup.run_sim`); check which one the input file selects.

## Shocks and robustness

For a bug fix or robustness claim about shocks, strong gradients or extreme states, and as the unaffected case for a change aimed at smooth flow.

| Physics | Problems |
|---|---|
| Hydrodynamics | `HydroShocktube`, `HydroShuOsher`, `HydroLeblanc`, `HydroHighMach`, `HydroContact`, `HydroQuirk`, `HydroBlast3D` |
| MHD | `MHDBrioWuShockTube`, `MHDRyuJones2aShockTube`, `IsothermalMHDShockTube`, `OrszagTang`, `MHDBlast`, `MHDQuirk`, `FieldLoop` |
| Radiation hydrodynamics | `RadhydroShock`, `RadMarshak`, `RadSuOlson`, `RadTube`, `RadStreaming` |

## The unaffected case

Step 4 runs one scenario the PR should not change. Choose it from the other table: a shock problem for a change aimed at smooth flow, a convergence problem for a change aimed at shocks, or a problem from another module when the change is meant to be local to one.

## The test suites

- **CI tests:** the problems registered with `add_test` in their `CMakeLists.txt`, run with `ctest -R <name>`. Check the dimensions each test is registered for: a test registered only for a dimension no CI job builds never runs.
- **The nightly regression suite:** `regression/quokka-tests.ini`, which compares output against stored benchmarks. A change that alters results on purpose needs its benchmarks regenerated, which only the suite maintainer can do; say so in the review.
