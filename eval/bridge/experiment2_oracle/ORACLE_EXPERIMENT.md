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

A robot action is a vector of numbers. **L1** = mean `|predicted − real|`
across the vector's numbers. Lower is more accurate. The comparison is
between the oracle-conditioned prediction (real next frame via
`is_hil=True` + `ingest_video()`) and a reference, on the same real
decision points.

An earlier version of this probe replayed `bridge_orig_lerobot` — the exact
dataset this checkpoint was finetuned on, with no held-out split —
comparing the oracle's prediction against the logged expert action. That
number is not reported here: with no held-out split, a null result there
can't be cleanly attributed to "the mechanism genuinely doesn't benefit
from foresight" versus "the model already memorized this trajectory well
enough that neither condition needed the signal" — so it isn't decisive
either way. Kept for the record, not cited, in `OFFLINE_PROBE_BACKLOG.md`.
The live probe below replaces it. (The independent confirmation against
this model's own published finding — Fig. 7-8, Section V, best performance
at `τv≈1` — is reproduced by the live probe's own result below, so nothing
is lost by not citing the offline number.)

## Live oracle probe (closed-loop, randomized)

This probe reuses the identical oracle mechanism (`VAMInference`'s
`is_hil=True` + `ingest_video()`, unmodified, on a second model instance)
but sources it from a live, randomized SimplerEnv-Bridge rollout: object
placement is randomized per episode by SimplerEnv itself (the same
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

**Sanity-checked against the driving model's own success rate — 20.8% is
correct, not an anomaly.** This number initially looked suspicious against
the 41.7% figure quoted elsewhere in this repo for the same task, and a
follow-up run (job 631931) re-ran the identical 24-episode range through
the plain, non-oracle-wrapped `eval_baseline.slurm` to check whether the
oracle side-computation (a second `VAMInference` instance resident on the
same GPU) was perturbing the driving model. It reproduced **20.8% exactly**
— confirming the side-computation has no effect. What actually happened:
**41.7% was the wrong number to compare against.** It comes from an older
July sweep over `--vam-stop-video-denoising-step` (best-of-sweep per task),
not from the fixed `stop=23` this probe and every Experiment 1 condition
actually use. The correct, matched-settings baseline for Carrot at
`stop=23` is **20.8%** — already established independently on 2026-08-17
(see `../experiment1_ablation/README.md`'s "Baseline sanity re-run" /
Eggplant-anomaly section) and now reproduced twice more (job 631443 with
the oracle instance attached, job 631931 without it). Three independent
runs, same number, with and without the thing under suspicion — this is
resolved, not open.

**Note on the metric's construction:** unlike F1-VLA's copy of this probe
(action vs. the driving policy's own action), this one compares a predicted
absolute position against a real physical outcome — the two models' gaps
are not on the same scale and should not be compared to each other directly.

### Update 2026-08-21 — rerun at the corrected operating point (`stop=1`)

Everything above ran at `--vam-stop-video-denoising-step=23`, since
established to be the wrong operating point for this model — see
`../experiment1_ablation/README.md`'s "Recompute at the correct operating
point" section for the full story (a real `ulimit` bug and a misleading
indirect timing measurement both had to be worked through before landing
on this). Rerun at `stop=1` (job 632644 and 632660, two independent
launches, same 24-episode Carrot range):

| | all samples (n=96) | successful episodes only (n=36) |
|---|---|---|
| oracle L1 (job 632644) | 0.00486 | 0.00430 |
| oracle L1 (job 632660) | 0.00490 | 0.00449 |
| "arm doesn't move" baseline | 0.00319 | 0.00402 |

Both independent runs agree closely with each other (0.00486 vs 0.00490
overall — good internal consistency). Closed-loop success on this run:
37.5%, matching `experiment1_ablation/README.md`'s own `stop=1` baseline
number exactly, as expected (same config).

**This flips the sign of the earlier reading.** At `stop=23`, oracle L1
(0.00314, successful episodes) was slightly *better* than the trivial
baseline (0.00382). At `stop=1`, oracle L1 (~0.0044, averaged across the
two runs) is slightly *worse* than the trivial baseline (0.00402) — the
same qualitative pattern found for oracle-vs-baseline at `stop=23`'s
*offline* probe, interestingly, though that number is retired to backlog
and not being resurrected as evidence here. Flagged, not over-interpreted:
both L1 values are small in absolute terms and close to each other; n=36
successful-episode samples is not a large sample; and this is exactly the
kind of near-baseline comparison the success-episode filter was built to
make trustworthy in the first place, not a settled contradiction of the
`stop=23` reading. Worth an independent repeat before treating "oracle is
now worse than trivial at the correct operating point" as a real finding.

### Update 2026-08-22 — unified construction: action vs. real policy's own action

Every result above compares oracle's predicted *position* against the real
*achieved* position one step later — a genuinely different construction
from F1-VLA's and LDA-1B's own Exp2 (predicted *action* vs. the real
*policy's own action*, restricted to successful episodes), despite an
earlier commit message claiming parity with F1-VLA's copy of this script.
`main_inference_live_oracle.py` now also computes the unified comparison:
oracle's raw predicted delta action against the delta action the real
(self-imagining) driving model actually took at that same historical tick
— not composed into a position at all. Both constructions are computed and
reported side by side now; the position-based one is kept for continuity
with everything above, not because it's wrong.

Job 634004 (`live_oracle_stop1.slurm`, `stop=1`, PutCarrotOnPlate, 24
episodes, closed-loop success 37.5% — matches other `stop=1` runs exactly,
as expected, same config):

| | all samples (n=96) | successful episodes only (n=36) |
|---|---|---|
| oracle **action** L1 (unified w/ F1-VLA/LDA-1B) | 0.03080 (sd 0.04063) | 0.05141 (sd 0.05236) |
| zero-action baseline | 0.28982 | 0.26718 |
| oracle **position** L1 (original construction) | 0.00477 (sd 0.00379) | 0.00466 (sd 0.00462) |
| "arm doesn't move" baseline | 0.00319 | 0.00402 |

**Oracle's action stays close to what the real driving model's own
imagination already produces** — about 9x closer than the zero-action
baseline on all samples, ~5x on successful episodes only. Same qualitative
shape as F1-VLA's own result (oracle ≈8x closer to the real policy's
action than a zero-action guess). Read in isolation this would be exactly
as ambiguous as F1's own number was before Exp1 resolved it (see that
model's own doc): either mimic's self-imagined future is already close to
correct (explanation (a)), or the action-decoder doesn't read out forecast
content in a fine-grained way (explanation (b)).

**Exp1 already rules out (b) for mimic — decisively, in the opposite
direction from F1.** `experiment1_ablation/README.md`'s own result: both
ablated *and* shuffled collapse to exactly 0.0% on every task. A wrong
frame is exactly as fatal as no frame at all. If the action-decoder didn't
read out forecast content (explanation (b)), a wrong-episode frame should
land somewhere in between "no signal" and "correct signal" in its effect —
instead it's indistinguishable from having no signal, the strongest
possible evidence that content is read out and used. That leaves (a) as
the only explanation consistent with both experiments: **mimic-video's own
video-diffusion forecast is already close enough to correct that the
oracle's perfect version barely changes the decision.** Unlike F1-VLA
(structural dependency, content-blind) or LDA-1B (open, see that model's
own doc), mimic-video's foresight computation is doing real, accurate,
load-bearing work — the opposite verdict from F1-VLA, reached by the same
two-experiment logic.

**Caveats.** Single run, n=36 successful-episode samples for the filtered
comparison — same small-sample caveat as everything else in this document;
worth an independent repeat. sd is large relative to the mean for the
action-L1 numbers (0.04-0.05 sd on a 0.03-0.05 mean) — the *direction* of
the effect (oracle much closer to self-imagined than to zero) is clear,
the precise ratio less so.

## Not pursued (a deliberate scope decision, not an oversight)

- **Closed-loop success-rate evaluation with the oracle actually
  driving the robot** (not just a side-channel query). This repo's own paper
  (Section III/Fig. 2) reports **closed-loop success rate**, not single-step
  L1 — the live probe above is a cheaper proxy for the causal question, not
  a replication of the paper's own reported metric.
  Getting a genuine closed-loop oracle number is harder than it looks: once
  the model's own action diverges from the logged trajectory, there is no
  pre-recorded "real future" left to inject at the next step. The source
  paper (and this repo's own `main_inference_hil.py` / `eval_hil.sh`,
  "human-in-the-loop evaluation (oracle study)") handles this via a real
  live human teleoperator per episode, not a script. **Not being run for
  any of the three models in this thesis** — the infrastructure and time
  cost don't fit the scope here, not a TODO waiting on availability. Read
  the L1 numbers above as evidence about the causal question this probe was
  built for, not as a stand-in for this number, permanently — not "for now."

## Not yet done

- [ ] Seed repeats — see caveat 3 below (note: mimic's episode outcomes are
  fully deterministic given task+episode-range, per
  `../experiment1_ablation/README.md`'s "not three seeds" finding, so a
  repeat here means a different task or episode range, not a different
  seed value).

## Caveats

1. **Distributional overfitting.** Object placement is randomized per
   episode, so verbatim memorization of this exact trajectory is
   impossible — but overfitting to the task *family* is a separate, softer
   question this probe doesn't rule out. Not the same failure mode as the
   offline probe this one replaced (see `OFFLINE_PROBE_BACKLOG.md`).
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
