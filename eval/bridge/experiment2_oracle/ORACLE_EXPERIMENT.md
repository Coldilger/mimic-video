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

## Live oracle probe (closed-loop, randomized) — preliminary, unresolved anomaly

The offline probe above replays `bridge_orig_lerobot` — the exact dataset
this checkpoint was finetuned on, with no held-out split (caveat 1 below).
This probe reuses the identical oracle mechanism (`VAMInference`'s
`is_hil=True` + `ingest_video()`, unmodified, on a second model instance)
but sources it from a live, randomized SimplerEnv-Bridge rollout instead:
object placement is randomized per episode by SimplerEnv itself (the same
mechanism Experiment 1's own closed-loop eval already uses), so verbatim
recall of a specific trajectory is impossible here.

**Design.** A normal closed-loop rollout runs with the REAL (non-oracle)
`VAMInference` driving the robot — pure side computation, behavior
unaffected in principle. Every control tick, once a 12-frame future window
is available, a second oracle-only instance is queried and its predicted
ABSOLUTE position (composing the predicted delta with the current pose, the
same way `VAMInference.step()` does internally) is compared against the
REAL achieved position one step later — not against the driving policy's
own action, since (as for F1-VLA's copy of this probe) comparing against a
policy that only succeeds part of the time isn't a meaningful reference;
restricted to episodes SimplerEnv scored successful.
Implementation: `main_inference_live_oracle.py`.

### Results (job 631443, task PutCarrotOnPlateInScene-v0, 24 episodes)

Average success: **20.8%** (5/24).

| | all samples (n=96) | successful episodes only (n=20) |
|---|---|---|
| oracle L1 (predicted position vs. real next position) | 0.00293 (sd 0.00169) | 0.00314 (sd 0.00185) |
| "arm doesn't move" baseline | 0.00339 | 0.00382 |

**This run is flagged, not adopted as a result yet — one number here does
not match what's already established.** 20.8% is well below the 41.7%
baseline already measured on this exact task and episode range
(`eval_baseline.slurm`, see `../../RESULTS.md`). The pipeline itself ran
clean (`EVAL_EXIT=0`, no errors), so this isn't a crash — but a ~2x success
drop on the identical task/episode range is not yet explained. Leading
candidate: running two full `VAMInference` instances simultaneously
(driving + oracle, ~2 full video-diffusion backbones resident on one GPU)
may be perturbing the driving instance through resource contention, even
though the wrapper design (side-channel query, no coupling into the
driving instance's own state or return value) should leave it unaffected in
principle. **Not yet isolated.**

Given this, the oracle-vs-baseline gap above (0.00314 vs 0.00382 — much
closer together than F1-VLA's equivalent live-probe gap) should not yet be
read as a finding. Two different things could be true independently: the
success-rate anomaly could be a real perturbation from the side-computation
(in which case the episode set being averaged over here isn't the same
distribution as the 41.7%-baseline runs, and these L1 numbers are measuring
something slightly different than intended), or it could be ordinary
between-run variance on a task this repo's own docs already flag as noisy
at n=24 (see the Eggplant anomaly in `../experiment1_ablation/README.md`).

**Note on the metric's construction:** unlike F1-VLA's copy of this probe
(action vs. the driving policy's own action), this one compares a predicted
absolute position against a real physical outcome — the two models' gaps
are not on the same scale and should not be compared to each other directly.

## Not yet done

- [ ] **Resolve the success-rate anomaly** before trusting the table above:
  re-run the same 24-episode Carrot range through the plain (non-oracle-
  wrapped) `eval_baseline.slurm` path to check whether 41.7% reproduces on
  this exact venv/checkpoint/day, or whether it's also lower — this isolates
  whether the oracle side-computation is perturbing the driving model.
- [ ] **Closed-loop success-rate evaluation with the oracle actually
  driving the robot** (not just a side-channel query). This repo's own paper
  (Section III/Fig. 2) reports **closed-loop success rate**, not offline/live
  single-step L1 — everything above (both probes) is a cheaper proxy for the
  causal question, not a replication of the paper's own reported metric.
  Getting a genuine closed-loop oracle number is harder than it looks: once
  the model's own action diverges from the logged trajectory, there is no
  pre-recorded "real future" left to inject at the next step. The source
  paper (and this repo's own `main_inference_hil.py` / `eval_hil.sh`,
  "human-in-the-loop evaluation (oracle study)") handles this via live human
  teleoperation — expensive per episode, and not yet run for any of the
  three models.
- [ ] Seed repeats once the anomaly above is resolved.

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
