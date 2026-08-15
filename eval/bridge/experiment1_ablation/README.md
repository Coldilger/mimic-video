# Experiment 1 — Ablation of the world-model signal (mimic-video)

**Status: planned, not yet run — implementation mechanism not yet
identified.**

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

Per `experiment2_oracle/ORACLE_EXPERIMENT.md`, mimic's action decoder never
looks at a finished/rendered image (real or generated) — it reads hidden
activations from a specific layer of the video backbone (Cosmos-Predict2),
attached to the decoder as a cross-attention conditioning signal. Suppressing
this signal means either:

1. **Zero/mask the cross-attention embedding** the decoder receives (module
   removed, changes what shape/content of conditioning the decoder sees —
   same "shape the model never saw in training" confound as F1's variant 1).
2. **Shuffle it across episodes** (same shape, wrong episode's content —
   the clean version, same idea as F1's variant 2, no shape confound).

**Not yet done:** identify the exact module/hook where this cross-attention
conditioning attaches (candidate: `model/cosmos_predict2/models/text2image_dit.py`
has cross-attention code, but that appears to be text conditioning, not
necessarily the video-backbone-to-action-decoder connection used here — needs
verification against how `VAMInference`/`video_action_model.py` actually
wires the two together before either variant can be implemented).

## Not yet done

- [ ] Identify the exact cross-attention hook this needs to suppress.
- [ ] Implement variant 1 (zero/mask) and variant 2 (shuffle across
      episodes).
- [ ] Run both offline against real logged Bridge moments (same style as
      `experiment2_oracle/oracle_offline_probe.py`), compare predicted-action
      L1 error against the unmodified baseline.
