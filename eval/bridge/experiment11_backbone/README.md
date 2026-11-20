# Experiment 11 — Action-only comparison (mimic-video)

**Question (advisor's point 5).** Train a comparable model on actions only:
does learning to forecast help the policy? Same point as F1-VLA's
`eval/bridge/experiment11_nogenloss` (Bridge finetune with forecasting-loss
weight 0) and LDA-1B's `eval/robocasa/experiment11` (`only_policy`). It extends
the authors' own "No-Gen" ablation in F1-VLA (Table 4) and the co-training
comparison of LDA-1B (Fig. 10) to mimic-video.

## How the design maps onto mimic-video

mimic-video learns in two separate stages (`MODEL.md`):

1. **Video model finetuning.** The Cosmos-Predict2 2B video model gets a LoRA
   (rank 256) trained to predict Bridge videos (70k iterations, batch 64). This
   is the only stage in which the model learns to forecast robot futures.
2. **Action decoder training.** With the video model frozen, a decoder is
   trained on its layer-20 activations with the action loss only (14,112
   iterations, batch 256, lr 1e-4).

The forecasting objective therefore lives entirely in stage 1. The authors
released a second, matched pair in which stage 1 is skipped:

| | video model | action decoder |
|---|---|---|
| deployed | `v2w_bridge_lora_rank256_..._iter_000070043_fused` (Bridge forecasting finetune) | `w2a_bridge_v2w_bridge_lora_..._layer20_bsz256_iter_000014112` |
| **no Bridge forecasting** | `v2w_pretrained_cosmos` (generic Cosmos-Predict2, never trained on Bridge videos) | `w2a_bridge_v2w_pretrained_cosmos_..._layer20_bsz256_iter_000014112` |

Same decoder architecture, layer, learning rate, batch size and iteration
count; the two decoder files differ (they are trained on different
activations). The second pair is the mimic-video analogue of F1-VLA's
"finetune without the forecasting loss": in both cases the model starts from
weights that were pretrained with a forecasting objective (F1-VLA's
pretraining; Cosmos' web-video pretraining) and the robot-domain stage
optimises actions only.

**What this does not cover.** In F1-VLA the trunk still adapts to Bridge
through the action loss; here the video model stays frozen at its generic
weights. A closer match would train the video model's LoRA end to end through
the decoder with the action loss only (not implemented: mimic-video's
training code always freezes the video model). Whether that is worth doing
depends on the result below.

## Protocol

`../experiment1_ablation/closed_loop/closed_loop.slurm` with
`CONDITION=baseline BACKBONE=pretrained`: identical simulator settings,
operating point (stop = 1 of 35 denoising steps, 5 of 15 actions executed),
4 tasks × 24 episodes; save tag `exp1_baseline_pretrainedbackbone_stop1_seed0`.
mimic-video is deterministic (fixed seed 0), so 96 episodes, paired episode by
episode with `exp1_baseline_stop1_seed0` (`../paired_stats.py`). Jobs
781201–781204 (`../experiment1_ablation/closed_loop/jobs_exp11_pretrained.txt`).

## Results

Pending.
