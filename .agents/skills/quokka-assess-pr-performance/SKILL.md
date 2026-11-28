---
name: quokka-assess-pr-performance
description: "Assess the run time and memory of a Quokka pull request against the default branch, on the same machine and inputs, when the PR claims or risks a change in performance."
---

# Assess PR Performance

Use this when the PR claims to change performance, or touches code on the hot path (GPU kernels, `ParallelFor` loops, the hydro or radiation update). For why GPU kernels behave the way they do, see `docs/markdown/performance.md`.

## Set up a fair comparison

Build the default branch and the PR with the same preset (`quokka-checkout-pr-and-default`, `quokka-build-problem`), and run both:

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

A table with one row per branch and run, then the median of each branch:

| Branch | Commit | Run | Figure of merit (μs/zone-update) | Total time (s) |
|---|---|---|---|---|

A difference smaller than the spread between runs of the same branch is not a change. When the zone-updates differ between the branches, say so and do not compare the figures of merit.
