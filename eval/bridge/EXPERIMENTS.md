# Experiments — index

## Central research question

Does the world-model computation performed at inference time carry causal
weight for action selection — or is the benefit entirely a training-time
representational effect?

Each experiment below attacks this question from a different angle, across
all three models under comparison (F1-VLA, mimic-video, LDA-1B). This repo
is mimic-video's fork; the same four experiments also live in the F1-VLA and
LDA-1B forks, each with a model-specific implementation.

## Experiments

- [`experiment1_ablation/`](experiment1_ablation/) — **Ablation of the
  world-model signal.** One hypothesis, two opposite interventions:
  suppress the foresight signal where the model normally uses it (F1,
  mimic-video), or turn it on where the model normally doesn't (LDA-1B).
  **Done** — both variants, offline and closed-loop: zeroing *or*
  shuffling the signal collapses closed-loop success to 0% on every task
  (vs. F1-VLA, where shuffling recovers to near-baseline) — see
  `experiment1_ablation/README.md`.
- [`experiment2_oracle/`](experiment2_oracle/) — **Oracle injection.**
  Replace the predicted future with the ground-truth future, encoded
  through each model's own pipeline. Tests whether a *perfect* forecast
  would even be used if the model had one. Extends the case study in
  mimic-video's own paper (arXiv:2512.15692, Section III / Fig. 2) — this
  model's own authors already ran a version of this. **Offline probe (L1)
  done; closed-loop success rate — the actual metric that case study
  reports — is still outstanding**, see
  `experiment2_oracle/ORACLE_EXPERIMENT.md`'s "Not yet done" section.
- [`experiment3_cost/`](experiment3_cost/) — **Cost per decision.**
  Characterizes how much inference-time compute each model actually spends
  on its world-model computation, and at what latency/success-rate
  trade-off. Context for interpreting Experiments 1 and 2. **Done** — full
  scale (n=48/condition) measured 2026-08-19, see
  `experiment3_cost/README.md`.
- [`experiment4_probing/`](experiment4_probing/) — **Representation
  probing.** Freezes each model's backbone and trains a small probe head to
  predict future end-effector pose from a single frozen hidden state. Tests
  the "training-time representational effect" side of the research question
  directly.
