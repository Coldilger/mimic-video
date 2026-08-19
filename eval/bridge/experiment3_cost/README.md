# Experiment 3 — Cost per decision (mimic-video)

**Status: done. Full scale (n=48 replans/condition, 4 episodes, Carrot task, 2026-08-19), confirming the smoke-scale numbers below.**

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

## Not yet done

- [x] Scale up from the 3-episode smoke sample to a larger/multi-task
      latency measurement, matching F1-VLA's n=45 across 3 episodes (n=48,
      2026-08-19 — see "full scale" above).
- [x] LDA-1B's Experiment 3 row — measured 2026-08-19, via its own confirmed-
      working RoboCasa checkpoint rather than Bridge (still 0% closed-loop).
      See `LDA-1B/eval/bridge/experiment3_cost/README.md`.
- [x] Explain the ~40-77x latency gap — done 2026-08-19, see above.
