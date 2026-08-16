# Experiment 1 — Ablation of the world-model signal (mimic-video)

**Status: variant 1 (zero out) and variant 2 (shuffle) both implemented and
run offline (n=120 each). Closed-loop SimplerEnv runs not yet done.**

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

## Not yet done

- [ ] Run variant 1 in real closed-loop SimplerEnv (baseline vs ablated
      success rate) — the memorization-robust test, matching F1-VLA's
      Experiment 1 protocol.
- [ ] Run variant 2 in closed-loop SimplerEnv.
