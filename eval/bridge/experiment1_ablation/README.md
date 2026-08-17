# Experiment 1 — Ablation of the world-model signal (mimic-video)

**Status: complete.** Both variants run offline (n=120 each) and
closed-loop (all 4 tasks × 24 episodes each).

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

Real SimplerEnv rollouts, on the same 4 tasks × 24 episodes as F1-VLA/LDA-1B
(`eval_ablated.slurm`), each condition launched 3× per task, 2026-08-16/17.
See "Method note: the three runs per task are not three seeds" below before
reading the per-run counts.

| task | baseline | ablated (zero crossattn) |
|---|---|---|
| Put Carrot on Plate | 41.7% | **0.0%** |
| Put Spoon on Towel | 45.8% | **0.0%** |
| Stack Green Cube | 16.7% | **0.0%** |
| Put Eggplant in Basket | 95.8% | **0.0%** |
| **average** | **50.0%** | **0.0%** |

**0 of 24 episodes succeeded on any of the 4 tasks.** Every job completed
cleanly (`EVAL_EXIT=0`, 24 episodes each, no crashes) — this is a real
result, not a harness failure.

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

## Results — closed-loop (variant 2, shuffle)

Same protocol, `eval_shuffled.slurm` (first attempt at `--time=01:00:00`
timed out on every job, since shuffled runs the real, expensive
video-diffusion backbone every replan — same ~10s/decision as baseline per
Experiment 3, unlike ablated's ~130ms; resubmitted at `--time=04:00:00`),
2026-08-17:

| task | baseline | ablated (zero) | shuffled (wrong-episode real) |
|---|---|---|---|
| Put Carrot on Plate | 41.7% | 0.0% | **0.0%** |
| Put Spoon on Towel | 45.8% | 0.0% | **0.0%** |
| Stack Green Cube | 16.7% | 0.0% | **0.0%** |
| Put Eggplant in Basket | 95.8% | 0.0% | **0.0%** |
| **average** | **50.0%** | **0.0%** | **0.0%** |

**Shuffled collapses just as completely as ablated — 0 of 24 episodes
succeeded on any task.** All 24 episodes per job completed cleanly on every job (no
timeouts, no crashes) — a real result.

**This is the sharpest contrast with F1-VLA in the whole experiment.** For
F1, shuffled (real image, wrong episode) recovers success almost exactly to
baseline on every task, both offline and closed-loop — the model apparently
only needs *some* real visual grounding in that slot, not the right one.
For mimic-video, shuffled is statistically indistinguishable from ablated:
a real image from a different episode helps exactly as much as zeros do,
i.e. not at all. Combined with the offline probe (all three conditions
identical there too), the picture for mimic-video is unusually clean and
severe: the action decoder isn't reacting to "is there a real-shaped signal
present" the way F1's does — it needs the *correct, current-episode*
`crossattn_emb` specifically, and any substitute, real or not, is as good as
no signal at all. Where F1-VLA's world-model dependency is about the *act*
of conditioning on real visual information, mimic-video's is about the
*content* being right — a genuinely different failure mode between the two
architectures, not just a difference in effect size.

## Method note: the three runs per task are not three seeds

Each condition above was launched three times per task, and earlier versions
of this document described that as "3 seeds". It isn't, and the distinction
matters for how much statistical weight these numbers carry.

`SimplerEnv`'s evaluator has no seed argument at all — for *any* model. The
recorded `Namespace(...)` of two supposedly-different-seed runs differs only
in `additional_env_save_tags` (the output directory name); nothing about the
environment or the policy is reseeded. Object layouts come from
`--obj-variation-mode episode`, which is a deterministic function of the
episode index, not of a seed.

For mimic-video specifically, the pipeline is then fully deterministic:
`model/cosmos_predict2/pipelines/video2world2action.py` takes `seed: int = 0`
as a hardcoded default and passes it to `generate_noise`
(`pipelines/base.py:147`, `torch.Generator(device).manual_seed(seed)`), so the
video-diffusion backbone always denoises from the same starting noise. Three
launches of the same task therefore produce byte-identical rollouts.

Verified empirically by the baseline sanity re-run (2026-08-17,
`eval_baseline.slurm`): PutCarrotOnPlate scored `0.20833333333333334` on both
"seeds" — matching to 17 decimal places, with an identical per-episode
success pattern. StackGreenCube likewise scored `0.041666666666666664` twice.

**Consequence:** the effective sample size is 24 episodes per task, not 72.
The three launches are exact repeats and add no variance information.

This does not change any conclusion here — 0 successes out of 24 distinct
episodes is still a complete collapse, on every task, against a baseline
that succeeds on a real fraction of those same episodes (see below). What it
changes is the precision of the claim: the effect is measured across 24
episodes per task, not replicated across three independent runs.

**This is a mimic-video-specific property, not a general SimplerEnv one.**
F1-VLA's repeat launches *do* differ from each other (Carrot 41.7% / 25.0% /
37.5%, Eggplant 62.5% / 58.3% / 87.5%; see `slurm/logs/seeded-*` in the
F1-VLA repo), because F1's policy does not pin its sampling RNG. So F1's
three runs are genuine samples of run-to-run variance even though they, too,
were never passed an explicit seed. mimic-video could be given the same
property by plumbing a varying seed into the pipeline call; that has not been
done, so mimic's numbers here are single deterministic measurements.

## Baseline sanity re-run (2026-08-17)

The 0.0% results above were produced after this session's `model/.venv`
rebuild, so the unmodified model was re-run through the identical protocol
(`eval_baseline.slurm`, real `VAMInference`, no Experiment 1 subclass) to
confirm they aren't an artifact of the rebuilt environment:

| task | baseline, this re-run (`stop=23`) | baseline, July sweep (table above) | ablated | shuffled |
|---|---|---|---|---|
| Put Carrot on Plate | 20.8% | 41.7% | 0.0% | 0.0% |
| Put Spoon on Towel | 12.5% | 45.8% | 0.0% | 0.0% |
| Stack Green Cube | 4.2% | 16.7% | 0.0% | 0.0% |
| Put Eggplant in Basket | 8.3% | 95.8% | 0.0% | 0.0% |
| **average** | **11.5%** | **50.0%** | **0.0%** | **0.0%** |

The unmodified model succeeds on a real fraction of episodes on every task
measured, through the same venv, partition, and launcher that produced the
0.0% ablated/shuffled results — so those zeros are a real effect of the
intervention, not a broken harness.

**Why these differ from the baseline column in the tables above.** Those
figures (41.7% / 45.8% / 16.7% / 95.8%) come from the original July sweep
(`../eval.sh`), which ranged `--vam-stop-video-denoising-step` over
`stop_steps=(0 … 35)`; the recorded numbers are from that sweep rather than
from the fixed `stop=23` used by every Experiment 1 condition. The
apples-to-apples comparison is therefore this re-run's column against
ablated/shuffled — both at `stop=23` — and it tells the same story, with the
same sign and the same collapse to zero. The July run's own logs are no
longer on disk (scratch's 30-day purge), so the exact per-`stop` provenance
of those four numbers could not be re-derived directly.

**Read the ablation's magnitude off the matched column, not the July one.**
At the settings every Experiment 1 condition actually used, baseline averages
**11.5%**, not 50.0%. The intervention still drives every task to exactly
zero, so the qualitative finding — mimic-video's world-model signal is
causally load-bearing, and a wrong-episode substitute is worth no more than
zeros — is unaffected. But the drop is 11.5 → 0.0 points, not 50 → 0, and the
interpretation paragraphs above were written against the larger figure.

**Eggplant is an outlier in this discrepancy and is not fully explained.**
Carrot/Spoon/Stack land at 0.25-0.50× their July figures, which a
best-of-sweep selection effect covers comfortably. Eggplant lands at 0.09×
(8.3% vs 95.8%) — an order of magnitude, not a factor of two to four. Either
that task is unusually sensitive to `--vam-stop-video-denoising-step`, or
something else differs that has not been identified; with the July logs gone,
this was not resolved. Statements elsewhere in this document that lean on
Eggplant's 95.8% (e.g. "including Eggplant, which baseline solved 95.8% of
the time") describe the July sweep's figure, which did not reproduce at
`stop=23`.
