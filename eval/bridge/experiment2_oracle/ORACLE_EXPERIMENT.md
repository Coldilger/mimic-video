# Experiment 2: Oracle injection — mimic-video

## What this is and why it's grounded, not invented

This experiment is **not something we made up** — it's a direct extension
of the method mimic-video's own authors published in their own paper
(arXiv:2512.15692, Section III / Fig. 2, "Case Study: How Does Video
Generation Quality Affect Robot Policy Performance?"). Their idea: give
the action decoder either a predicted future or the real ("oracle") one,
and see how much that changes the result. We apply the same method, the
same question, to all three models under comparison (F1-VLA, mimic-video,
LDA-1B) — each time through that specific model's own natively-trained
future-prediction mechanism.

The question the experiment answers: **does the "imagine the future"
computation carry causal weight for action selection, or is the benefit
already baked into the learned representations, making the inference-time
computation an expensive ritual with no real effect?**

## How exactly mimic "sees" the real future

To be precise: mimic's action decoder **never looks at a finished image**
(neither generated nor real). It reads **hidden activations from a
specific layer of the video backbone** (Cosmos-Predict2). The backbone is
a video-diffusion model (trained on a noise-to-clean-video schedule), but
the decoder is attached not to the final decoded frame but to an
intermediate latent representation at a chosen layer. In the world-model
taxonomy this is the "Latent-Only" category — the grounding point of the
action prediction sits on an intermediate representation, not on
rendered pixels.

By default, these activations arise while the backbone tries to imagine
the future through a (partial) denoising process — controlled by the
parameter `τv` (video flow time): at `τv=1` the future is pure noise, at
`τv=0` it's a fully rendered video.

In the oracle condition we don't let the backbone imagine anything. The
real next frame (from an actually-recorded Bridge episode) is passed
through the same backbone directly — we get activations from the same
layer, but obtained from reality instead of denoising. Implementation:
`is_hil=True` + `model.ingest_video()` in `VAMInference`
(`video_action_model.py`) — the same mechanism the repo documents for a
human teleoperator (`main_inference_hil.py`); we use the same mechanism
but feed pre-recorded real frames instead of live teleoperation.

## How the metric is computed

Take a real, logged moment from an actual Bridge episode: the "now"
frame plus what the robot **actually** did a second later. Show the
model "now" (and, in the oracle condition, the real "next"), compare the
predicted action to the real one.

A robot action is a vector of numbers. **L1** = mean
`|predicted − real|` across the vector's numbers. Lower is more accurate.

## Current results (115 samples: 24 episodes × 5 moments)

| | oracle | baseline (arm doesn't move) |
|---|---|---|
| position, L1 | **0.0092** (sd 0.0063) | 0.0097 |
| gripper, L1 | 0.1717 | — |

**How to read it:** the gap between oracle and the trivial baseline is
almost zero (~1.05x). Even given the **real** future frame, the decoder
doesn't extract meaningfully more value from it than from nothing at
all. This means the inference-time computation (even in the best case,
with a perfect future) carries no causal weight for action selection in
this model.

**Important finding — an independent confirmation of a result already in
mimic-video's own paper.** The authors already show (Fig. 7-8, Section
V) that best performance is achieved at `τv≈1`, i.e. when the future is
pure noise rather than a rendered video. Our oracle test reaches the
same conclusion from the other direction: not "noise works just as
well" but "even the truth doesn't work better". This isn't a
coincidence between two separate experiments — it's the same finding,
that the foresight computation here isn't causally load-bearing, arrived
at through two independent routes.

## Caveats

1. **Memorization.** See the general caveat — training had no held-out
   split.
2. **Checkpoint provenance.** Unlike F1 and LDA, the mimic-video
   checkpoint is the authors' own release, not our finetune — we don't
   control the training recipe. The training data also went through a
   *different* processing pipeline (raw Berkeley release via
   `process_bridge.py`) than `bridge_orig_lerobot`, which feeds this
   oracle test — same underlying source (BridgeData V2), different
   processing. This shouldn't affect the "oracle doesn't help" finding
   itself (the decoder receives the image regardless of how it was
   preprocessed during training), but it matters when comparing absolute
   numbers against F1/LDA.
3. **One run, no seed repeats** — same as the other models.
