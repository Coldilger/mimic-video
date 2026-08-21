# Experiment 3 — Cost per decision (mimic-video)

**Status: done. Recomputed 2026-08-21 at the model's correct operating point
(`--vam-stop-video-denoising-step=1`, not `=23`) — median latency 1060ms,
not 10227ms, closing most of the "40-77x more expensive" gap to ~4-5x. See
"Update 2026-08-21" below for the canonical number; the `stop=23` numbers
throughout this doc are superseded but kept for provenance. Original
full-scale measurement: n=48 replans/condition, 4 episodes, Carrot task,
2026-08-19, confirming the smoke-scale numbers below.**

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

## Results (full scale, 4 real closed-loop episodes per condition, Carrot task, 2026-08-19)

| variant | n (replans) | median | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| baseline (self-imagined) | 48 | 10226.9 ms | 10416.5 ms | 10750.0 ms | 10074.7 ms | 35424.1 ms |
| ablated (zero crossattn) | 48 | **133.1 ms** | 344.5 ms | 901.0 ms | 132.6 ms | 36342.6 ms |
| shuffled (wrong-episode real) | 48 | 10287.7 ms | 10506.5 ms | 10840.0 ms | 10154.0 ms | 36285.9 ms |

Matches the smoke-scale numbers (below) closely — the extra episode mainly
tightens the estimate, doesn't change it. First attempt at this run (jobs
631081/631083) hit `CPU time limit exceeded (core dumped)`: the partition's
default soft `RLIMIT_CPU` is 600s, which baseline/shuffled's ~10s/replan cost
exceeds on 4 episodes even though wall-clock stayed well under the SBATCH
`--time` limit (ablated stayed under it, being ~77x cheaper). Fixed with
`ulimit -t unlimited` at the top of `timing_full.slurm` (hard limit is
unlimited) and reran clean.

### Results (smoke, 3 real closed-loop episodes per condition, Carrot task, 2026-08-16)

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

## Why is mimic ~40-77x slower than F1-VLA/LDA-1B? (2026-08-19)

**Not parameter count.** Exact counts from each checkpoint's own header
(safetensors header for F1, `map_location='meta'` load for the others — no
weight data materialized):

| model | params | latency (median) |
|---|---|---|
| LDA-1B | **7.21B** (bf16) | 254.7 ms |
| F1-VLA | 4.19B (bf16) | 215.7 ms |
| mimic-video | **2.46B** (1.96B video backbone + 0.50B action decoder, bf16) | 10226.9 ms |

LDA-1B has ~3x mimic's parameter count and is ~40x faster. The relationship
is inverted, not correlated — ruling out model size as the driver.

**It's the number of full-video denoising steps, confirmed directly, not
just plausible from reading the code.** mimic is "latent-only": the action
decoder never sees a decoded pixel frame, only reads a hidden state from an
intermediate transformer layer (`world2action_model.py`'s `get_crossattn_emb`,
`xattn_layer_idx`, matching the checkpoint filename's `layer20`) via
`return_only_hidden_states_up_to`/`return_decoded_video=False` — so being
latent-only does skip the VAE pixel-decode step and running the transformer's
remaining depth past that layer. What it does *not* skip is the iterative
diffusion sampling loop itself: `num_sampling_step=35` (`stop_after_step=23`
in our eval config), and at **every one of those steps**, the transformer
processes the *entire* video latent's token grid — `crossattn_emb.shape` is
`(B, T*H*W, D)` with **T·H·W = 19200 tokens** (confirmed in
`experiment4_probing/README.md`'s own feature-shape probe), not a single
frame's worth. Compare F1-VLA's VAR foresight: 10 scales,
`v_patch_nums=(1,2,3,4,5,6,8,10,13,16)`, summing to only **680 tokens total
across all 10 steps combined** (per F1-VLA's `modeling_f1.py`/`wm/vqvae.py`) —
mimic processes ~28x more tokens *per single step* than F1 processes in its
*entire* foresight generation, on top of running ~2.3x more steps (23 vs 10).

Direct empirical test (not just consistent-with-the-code): ran the same
baseline condition at `stop=12` (halfway), 1 episode, job 631132:

| stop-steps | median latency |
|---|---|
| 0 (ablated) | 133.1 ms |
| 12 | 5660.2 ms |
| 23 (baseline) | 10226.9 ms |

Linear prediction for 12 steps from the 0/23 endpoints: 5399.4 ms. Measured:
5660.2 ms — 4.8% off, well within run-to-run noise. Slope is near-constant
across the range (460.6 ms/step on 0→12, 415.2 ms/step on 12→23). This is a
test that could have falsified "it's the step count" (e.g. if cost were
dominated by a fixed setup/first-step cost, latency at stop=12 would land
much closer to the stop=23 value than to a linear midpoint) and didn't.

**Conclusion:** the ~77x gap is specific to Cosmos-Predict2's video-diffusion
mechanism — an iterative multi-step sampling process over a full-video-length
token grid — not to raw model size, and not eliminated by mimic's own
latent-only tap point (that saves the pixel-decode cost, not the sampling-loop
cost). F1-VLA's VAR and LDA-1B's absence of any foresight-sampling loop are
both architecturally cheap for the same underlying reason: far fewer
token-processing steps, not fewer parameters.

**Caveat on the table above, noticed 2026-08-21:** the `0 (ablated)` row
isn't actually a `stop=0` run of the real denoising loop — it's
`VAMAblatedInference`/`ZeroWorldModelPipeline`, a different code path that
skips generation entirely (Experiment 1's ablation mechanism), reused here
as a stand-in for "zero steps." The `12`/`23` pair is the only genuinely
apples-to-apples comparison in that table (both real runs of the actual
denoising loop at different `stop_after_step` values); the implied
"0→23 is linear and validates the step-count story" reading leans on a
comparison across two different mechanisms. The qualitative conclusion
(cost scales with step count) still holds — see the `stop=1` measurement
directly below, which is a real run of the actual loop and lands close to
what the `12→23`-only slope alone would predict (~1097ms) — but this
should have been flagged before now.

## Update 2026-08-21 — real cost at the correct operating point (`stop=1`)

Every number above used `stop=23`, since established to be the wrong
operating point for this model — see `../experiment1_ablation/README.md`'s
"Recompute at the correct operating point" for the full story. Re-measured
at `stop=1` (job 632645, same method as "full scale" above, n=48 replans,
4 episodes, Carrot task):

| variant | n | median | p95 | mean | min | max |
|---|---|---|---|---|---|---|
| baseline, `stop=1` | 48 | **1060.0 ms** | 1349.8 ms | 1615.5 ms | 1054.2 ms | 26830.4 ms |
| baseline, `stop=23` (above) | 48 | 10226.9 ms | 10416.5 ms | 10750.0 ms | 10074.7 ms | 35424.1 ms |

**Median drops ~9.6x** (1060ms vs 10227ms). The min (1054.2ms) sits almost
exactly at the median, meaning the large majority of calls are tightly
clustered near 1.05-1.06s — consistent with the `12→23`-slope-only
prediction above (~1097ms), not the (methodologically weaker) `0→23`
prediction. The one `max=26830.4ms` outlier is the one-time CUDA-graph
warmup cost this doc's own smoke-scale table already documented ("max
latencies ~36-38s are the one-time warm-up cost on each condition's first
replan... not representative of steady-state cost") — same mechanism,
same order of magnitude, still one-time.

**Revised cross-model comparison** (was: mimic 40-77x pricier than F1/LDA):

| model | latency, median | vs. F1-VLA (215.7ms) | vs. LDA-1B (254.7ms) |
|---|---|---|---|
| F1-VLA | 215.7 ms | 1x | — |
| LDA-1B | 254.7 ms | — | 1x |
| mimic-video, `stop=23` (superseded) | 10226.9 ms | 47.4x | 40.2x |
| mimic-video, `stop=1` (correct) | **1060.0 ms** | **4.9x** | **4.2x** |

Still the most expensive of the three at its actual operating point — the
qualitative ranking (mimic > LDA > F1, or mimic > F1 > LDA depending on
exact numbers) doesn't flip — but the magnitude of the gap was
substantially overstated by measuring the wrong `stop` value. **A debugging
note, since it briefly cost real time:** a real `ulimit`/CPU-time-limit bug
initially made the `stop=1` closed-loop recompute look like it might take
hours per task; an unreliable indirect timing proxy (gaps between log
timestamps unrelated to this instrumented measurement) then suggested a
~39s/replan recurring cost, which looked like a second, separate bug. Both
were red herrings — this table, from the same purpose-built timing wrapper
used for every other number in this document, is the number that should be
cited.

## Not yet done

- [x] Scale up from the 3-episode smoke sample to a larger/multi-task
      latency measurement, matching F1-VLA's n=45 across 3 episodes (n=48,
      2026-08-19 — see "full scale" above).
- [x] LDA-1B's Experiment 3 row — measured 2026-08-19, via its own confirmed-
      working RoboCasa checkpoint rather than Bridge (still 0% closed-loop).
      See `LDA-1B/eval/bridge/experiment3_cost/README.md`.
- [x] Explain the ~40-77x latency gap — done 2026-08-19, see above.
