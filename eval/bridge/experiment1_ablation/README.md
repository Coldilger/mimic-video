# Experiment 1 — Ablation of the world-model signal (mimic-video)

**Status: both variants implemented and run offline (n=120 each) and
variant 1 (zero out) run closed-loop (all 4 tasks × 3 seeds). Variant 2
(shuffle) closed-loop in progress.**

## What this tests

One hypothesis, two opposite interventions, applied per-model depending on
whether the model's world-model computation normally runs at inference:

- Where it **normally runs** (F1, mimic-video): turn it **off**, and see if
  action prediction gets worse.
- Where it **normally doesn't run** (LDA-1B): turn it **on**, and see if
  action prediction gets better.

This is the complement to Experiment 2 (oracle injection): E2 asks "does a
*perfect* future help", E1 asks "does the *actual, currently-computed*
future matter at all, in either direction."

## mimic-video's mechanism: suppress the cross-attention embedding

mimic's action decoder (`world2action_pipeline`) never looks at a
finished/rendered image — it reads `crossattn_emb`, hidden activations from
one specific layer of the video-diffusion backbone (Cosmos-Predict2), passed
in as a plain tensor argument
(`model/cosmos_predict2/pipelines/video2world2action.py:60-67`). Unlike F1's
KV-cache-based connection (see `../experiment1_ablation` in the F1-VLA repo),
there is no existing "leave this slot empty" code path to exploit here,
since `crossattn_emb` is a required positional argument, not an optional
cache entry that the model already knows how to skip.

**Variant 1 (implemented): force `crossattn_emb` to zeros.**
`zero_world_model_pipeline.py`'s `ZeroWorldModelPipeline` is a new,
standalone class with the same `__call__` signature as the real
`Video2World2ActionPipeline` — a drop-in replacement for `VAMInference.model`
(`vam_ablated_inference.py`'s `VAMAblatedInference`). Nothing under
`model/cosmos_predict2/pipelines/` is touched. To genuinely skip the
(expensive) video-diffusion computation rather than just discarding its
output: `crossattn_emb`'s shape and the paired context-timestep value are
data-independent (fixed by the backbone's architecture, not by the input
video's content), so both are computed for real exactly once (first call),
cached, and reused as zeros/constants on every later call — only the very
first sample in an eval run pays the cost of running the real video
backbone.

**Variant 2 (implemented): shuffle the world model's input image across
episodes.** Unlike F1's KV-cache mechanism, mimic's video-diffusion backbone
has exactly one input that determines `crossattn_emb`: the image history fed
to `video2world_pipeline` (`VAMInference.step`'s `image` argument ->
`_add_image_to_history` -> `input_vid`; proprioceptive state is built
separately from `ee_pose_proprio`/`gripper_proprio` and never touches
`image`). So unlike variant 1, this needs no new pipeline class at all --
`shuffle_offline_probe.py` calls the real, unmodified `VAMInference` with a
different episode's real frame standing in for the current one, while state
and task description stay matched to the real, current sample. Same
shift-by-`samples_per_episode` pairing scheme as F1's
`kv_shuffle_offline_probe.py`, on the same (episode, t) samples as variant 1.

## Results — offline probe (both variants)

Both `ablation_offline_probe.py` (variant 1) and `shuffle_offline_probe.py`
(variant 2): for a sampled (episode, t), predict the action chunk on the same
real logged Bridge moment under each condition, compared against the real
logged next-step action. Same protocol/metric as
`experiment2_oracle/oracle_offline_probe.py` (position L1 against the next
observation.state, gripper L1), for direct comparability across all four
conditions. Run 2026-08-16, n=120 each (24 episodes × 5 samples/episode,
same (episode, t) samples for both):

| metric | baseline (self-imagined) | ablated (zero crossattn) | shuffled (wrong-episode real) | no-motion baseline |
|---|---|---|---|---|
| position L1 | 0.0098 (sd 0.0080) | 0.0100 (sd 0.0078) | 0.0099 (sd 0.0081) | 0.0096 |
| gripper L1 | 0.3327 | 0.3145 | 0.3581 | — |

**Interpretation:** unlike F1-VLA, where ablation clearly hurt offline L1
and shuffling clearly recovered it (ablated 0.0272 vs baseline 0.0176 vs
shuffled 0.0157 vs oracle 0.0157, i.e. shuffled ≈ oracle ≫ ablated — see the
F1-VLA repo's `experiment1_ablation/README.md`), mimic-video shows
**all three conditions landing in the same narrow band**, indistinguishable
from each other and from the no-motion floor (0.0096). Zeroing the signal,
feeding it real-but-wrong-episode content, or leaving it as the model's own
self-imagined future — none of it moves this metric. Position L1 for every
condition sits almost exactly at the no-motion baseline — the single-step
position-prediction task itself is dominated by small consecutive-frame
motion in Bridge, which limits how much this metric alone can distinguish
any of these conditions. Gripper L1 is noisy across all three (ablated
slightly better, shuffled slightly worse than baseline, within what a
small-n smoke test already showed is an unstable reading at this sample
size) — no consistent ordering.

This does **not** yet answer whether the world-model computation is causally
load-bearing for mimic-video — per the same reasoning that drove F1's
closed-loop follow-up (memorization confound: this model was fine-tuned on
the full Bridge dataset, no held-out split), the real test is closed-loop
SimplerEnv success rate, not offline L1 against logged actions. Unlike F1,
where the offline metric alone already told a clear story before closed-loop
even ran, mimic's offline numbers are uninformative either way here — the
closed-loop result isn't just a robustness check for mimic, it's the first
real signal.

## Results — closed-loop (variant 1, zero out)

Real SimplerEnv rollouts, matching F1-VLA/LDA-1B's exact protocol (4 tasks ×
3 seeds × 24 episodes, `eval_ablated.slurm`), 2026-08-16/17:

| task | baseline | ablated (zero crossattn) |
|---|---|---|
| Put Carrot on Plate | 41.7% | **0.0%** |
| Put Spoon on Towel | 45.8% | **0.0%** |
| Stack Green Cube | 16.7% | **0.0%** |
| Put Eggplant in Basket | 95.8% | **0.0%** |
| **average** | **50.0%** | **0.0%** |

**0 of 12 runs (all 4 tasks × 3 seeds) succeeded at all.** Every job
completed cleanly (`EVAL_EXIT=0`, 24 episodes each, no crashes) — this is a
real result, not a harness failure.

**This is a much larger effect than F1-VLA's ablation**, which hurt every
task but stayed well above zero (48.6% → 34.7% average, still succeeding on
a real fraction of episodes — see F1-VLA's `experiment1_ablation/README.md`).
mimic-video's world-model signal, once removed, doesn't just degrade
performance — closed-loop success collapses entirely on all four tasks,
including Eggplant, which baseline solved 95.8% of the time. Combined with
the offline probe finding no signal at all (`ablated` ≈ `baseline` ≈
`shuffled` on single-step L1, above) — offline L1 completely failed to
predict this. The world-model computation is clearly causally load-bearing
for mimic-video's closed-loop behavior, more so than for F1-VLA, even
though nothing in the offline metric hinted at it.

## Not yet done

- [ ] Run variant 2 (shuffle) in closed-loop SimplerEnv — in progress,
      first attempt (`--time=01:00:00`) timed out on every job since
      shuffled runs the real, expensive video-diffusion backbone every
      replan (same ~10s/decision as baseline, per Experiment 3 — unlike
      ablated's ~130ms, which finished comfortably inside the same limit).
      Resubmitted with `--time=04:00:00`.
