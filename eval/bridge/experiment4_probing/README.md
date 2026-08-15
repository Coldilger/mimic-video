# Experiment 4 — Representation probing (mimic-video)

**Status: planned, not yet run.**

## What this tests

The direct test of the "training-time representational effect" side of the
central research question: does mimic-video's frozen video backbone already
encode useful future information in its representations, independent of
what the inference-time denoising/foresight computation does with it? If a
small probe can recover future end-effector pose from a single frozen
hidden state, that's evidence the *training* process already built the
representation.

## Method

- Freeze the backbone, take one hidden state at the same relative position
  in all three models (candidate for mimic: the same intermediate
  Cosmos-Predict2 layer the action decoder itself reads from, per
  `experiment2_oracle/ORACLE_EXPERIMENT.md` — not yet confirmed as the right
  comparable point against F1/LDA's own extraction points).
- Train a small probe head (same size/architecture for all three models) on
  top of the frozen features.
- Probe target: **future end-effector pose** — deliberately not what the
  model is already trained to predict.
- Use the same Bridge checkpoint as Experiments 1–3 (mimic-video's
  checkpoint is the authors' own release, not a project finetune — see the
  caveat already noted in `experiment2_oracle/ORACLE_EXPERIMENT.md`).
- Cost: forward pass only, train just the small probe head — hours, no
  retraining of the backbone itself.

## Not yet done

- [ ] Confirm the extraction point (which layer / `τv` value) is the right
      comparable choice against F1 and LDA-1B's own probing points.
- [ ] Implement frozen-feature extraction + probe head training script.
- [ ] Run on real logged Bridge trajectories, report probe accuracy against
      future end-effector pose.
