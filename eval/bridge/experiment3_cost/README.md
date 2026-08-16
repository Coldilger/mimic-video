# Experiment 3 — Cost per decision (mimic-video)

**Status: run (smoke, n=36 replans per condition). Full-scale run not yet done.**

## What this tests

Context for interpreting Experiments 1 and 2: how much inference-time
compute does each model actually spend on its world-model computation, and
at what latency does that translate?

## Method

`timing_wrapper.py`'s `add_timing` wraps `VAMInference.step` (mimic has no
separate "predict a chunk"-style method — the world-model + action-decoder
forward pass is inlined directly in `step()`), gated on `action_buffer is
None` so only calls that trigger a real replan get timed, not every 5Hz
control tick that just pops a cached action. Wrapping the base
`VAMInference` class once is sufficient — neither `VAMAblatedInference` nor
`VAMShuffledInference` (Experiment 1's variants) override `step()`, so both
resolve to the same wrapped method via normal inheritance.

## Results (smoke, 3 real closed-loop episodes per condition, Carrot task, 2026-08-16)

| variant | n (replans) | median | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| baseline (self-imagined) | 36 | 10249.7 ms | 10463.8 ms | 10965.8 ms | — | 35850.1 ms |
| ablated (zero crossattn) | 36 | **132.3 ms** | 340.4 ms | 1133.0 ms | — | 35729.8 ms |
| shuffled (wrong-episode real) | 36 | 10385.7 ms | 10787.4 ms | 11167.2 ms | — | 38048.3 ms |

**Ablated is ~77x faster than baseline at the median** (132ms vs 10250ms) —
confirms `ZeroWorldModelPipeline` genuinely skips the real 35-step
video-diffusion generation after its one-time shape-probing call, not just
discarding its output. **Shuffled costs essentially the same as baseline**
(10386ms vs 10250ms median) — unlike F1-VLA's shuffle (which reuses the
same cheap `oracle_indices` substitution that only skips a *sampling* step,
not the expensive forward passes), mimic's shuffle still runs the full real
video-diffusion backbone on every replan — it's just fed a different
episode's real frame instead of the current one. The max latencies
(~36-38s) are the one-time warm-up cost on each condition's first replan
(CUDA graph construction), not representative of steady-state cost.

This puts mimic-video's own world-model computation at a **much larger
share of total inference cost** than F1-VLA's — F1's own ablated variant
was only ~26% faster than its baseline (per F1-VLA's
`experiment3_cost/README.md`), vs mimic's ~99% latency reduction when
ablated. Read together with Experiment 1's closed-loop result (ablation
collapses mimic's success rate to 0% on every task, a much larger effect
than F1's degradation-but-not-collapse), the two experiments tell a
consistent story: for mimic-video, the world-model computation is not just
expensive, it's carrying most of both the compute budget *and* the
causal weight for successful action selection.

## Not yet done

- [ ] Scale up from the 3-episode smoke sample to a larger/multi-task
      latency measurement, matching F1-VLA's n=45 across 3 episodes.
- [ ] Measure LDA-1B's Experiment 3 row (blocked on a working closed-loop
      baseline — see `../../RESULTS.md`).
