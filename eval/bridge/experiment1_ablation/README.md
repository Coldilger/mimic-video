# Experiment 1 — Ablation of the world-model signal (mimic-video)

**Status: variant 1 (zero out) implemented and run offline (n=120). Variant
2 (shuffle across episodes) and closed-loop SimplerEnv runs not yet done.**

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

**Variant 2 (not yet implemented): shuffle `crossattn_emb` across
episodes** — same idea as F1's variant 2, pairing each sample with a
different episode's real activations instead of zeros, to separate "no
signal present" from "wrong but real-shaped signal present."

## Results — offline probe (variant 1, zero out)

`ablation_offline_probe.py`: for a sampled (episode, t), predicts the action
chunk twice on the same real logged Bridge moment — baseline (`VAMInference`,
self-imagined future via the real video backbone) and ablated
(`VAMAblatedInference`, `crossattn_emb` forced to zeros) — compared against
the real logged next-step action. Same protocol/metric as
`experiment2_oracle/oracle_offline_probe.py` (position L1 against the next
observation.state, gripper L1), for direct comparability. Run 2026-08-16,
n=120 (24 episodes × 5 samples/episode, `eval/bridge/experiment1_ablation/ablation_offline_probe.slurm`):

| metric | baseline (self-imagined) | ablated (zero crossattn) | no-motion baseline |
|---|---|---|---|
| position L1 | 0.0098 (sd 0.0080) | 0.0100 (sd 0.0078) | 0.0096 |
| gripper L1 | 0.3327 | 0.3145 | — |

**Interpretation:** unlike F1-VLA, where ablation clearly hurt offline L1
(0.0272 ablated vs 0.0176 baseline vs 0.0946 zero-action baseline — see the
F1-VLA repo's `experiment1_ablation/README.md`), mimic-video shows
essentially **no offline degradation** from zeroing the world-model signal.
Position L1 for both baseline and ablated sit almost exactly at the
no-motion baseline (0.0096) — the single-step position-prediction task
itself is dominated by small consecutive-frame motion in Bridge, which
limits how much this metric alone can distinguish the two conditions.
Gripper L1 is noisy and, if anything, slightly favors ablated (likely within
sampling noise at n=120 — an early n=4 smoke test showed the opposite gap by
a wide margin, underscoring how unstable this specific metric is at small n).

This does **not** yet answer whether the world-model computation is causally
load-bearing for mimic-video — per the same reasoning that drove F1's
closed-loop follow-up (memorization confound: this model was fine-tuned on
the full Bridge dataset, no held-out split), the real test is closed-loop
SimplerEnv success rate, not offline L1 against logged actions. Not yet run.

## Not yet done

- [ ] Implement variant 2 (shuffle across episodes) offline probe.
- [ ] Run variant 1 in real closed-loop SimplerEnv (baseline vs ablated
      success rate) — the memorization-robust test, matching F1-VLA's
      Experiment 1 protocol.
- [ ] Run variant 2 in closed-loop SimplerEnv once implemented.
