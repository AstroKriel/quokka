---
name: quokka-assess-pr-performance
description: "Assess the run time and memory of a Quokka pull request against the default branch, on the same machine and inputs, when the PR claims or risks a change in performance."
---

# Assess PR Performance

Compare the run time and memory of the PR with the default branch. Use this when the PR claims to change performance, or when it changes code on the hot path (defined below). For why GPU kernels behave the way they do, see `docs/markdown/performance.md`.

**Before this:** check out both branches with `quokka-checkout-pr-and-default`.
**Terms** (defined in `.agents/skills/glossary.md`): claim, default branch, hot path, preset.

## The hot path

The hot path is the code that runs on every time step. That is anything reached from the time-step loop: `AMRSimulation::timeStepWithSubcycling` calls `QuokkaSimulation::advanceSingleTimestepAtLevel` on each level, and `computeTimestepAtLevel` sets each step's size. To tell whether a change is on the hot path, follow the calls in the diff back to one of these functions.

Code that runs once per run, such as initial conditions, reading inputs, and writing plotfiles and checkpoints, is not on the hot path, unless the problem runs it every step. Within the hot path, loops over every cell or particle cost the most, such as an `amrex::ParallelFor`, because their cost grows with the size of the grid.

When unsure how much a change matters, measure it: the TinyProfiler tables at the end of a run (see **What to read** below) show the time spent in each `BL_PROFILE` region.

## Set up a fair comparison

Build the default branch and the PR with the same preset, using `quokka-build-problem`. Then run both:

- on the same machine, one after the other, with nothing else running on it;
- with the same inputs, a problem large enough that a run takes at least tens of seconds;
- with output turned down, so writing files does not dominate the time;
- at least three times each, since a single run can be off by several percent.

Run the binary directly (`quokka-run-problem`), not with `quokka run`: the wrapper passes `tiny_profiler.enabled=0`, which turns off the per-region timing.

## What to read

| Number | Where it comes from | What it says |
|---|---|---|
| Figure of merit | the line `Performance figure-of-merit: <x> μs/zone-update` at the end of the run | time per cell update; lower is faster |
| Zone-updates per level | the lines after it | whether the two runs did the same work; if they differ, the comparison is not like for like |
| Time per region | the TinyProfiler tables at the end, when it is enabled | which part of the code changed speed |
| Memory | the memory lines in the TinyProfiler output | whether the PR uses more device or host memory |

## Report

Report a table with one row per branch and run, followed by the median of each branch:

| Branch | Commit | Run | Figure of merit (μs/zone-update) | Total time (s) |
|---|---|---|---|---|

A difference smaller than the spread between runs of the same branch is not a change. When the zone-updates differ between the branches, say so and do not compare the figures of merit.
