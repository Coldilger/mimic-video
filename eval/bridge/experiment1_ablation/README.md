# Experiment 1 — Ablation of the world-model signal (mimic-video)

**Status: complete.** Both variants run closed-loop (all 4 tasks × 24
episodes each) — the decisive result. (An offline probe also ran first;
retired to `OFFLINE_PROBE_BACKLOG.md` — no held-out split, not a citable
result, see below.)

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

## Offline probe (both variants) — retired to backlog

Both variants were first checked with an offline probe (predicted action
vs. logged action on real `bridge_orig_lerobot` moments) before the real
closed-loop runs below. Not reported here: no held-out split, so any result
is confounded with memorization — same reasoning that makes closed-loop
success rate this experiment's actual metric. Kept for the record, not
cited, in `OFFLINE_PROBE_BACKLOG.md`. Unlike F1-VLA (where the retired
offline probe at least showed *some* pattern), mimic's version found no
signal in any direction — which, as the closed-loop results below show, was
itself uninformative about what closed-loop success rate would do.

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
including Eggplant, which baseline solved 95.8% of the time. The
world-model computation is clearly causally load-bearing for mimic-video's
closed-loop behavior, more so than for F1-VLA.

## Results — closed-loop (variant 2, shuffle)

Same protocol, `eval_shuffled.slurm` (first attempt at `--time=01:00:00`
timed out on every job, since shuffled runs the real, expensive
video-diffusion backbone every replan — same ~10s/decision as baseline per
Experiment 3, unlike ablated's ~130ms; resubmitted at `--time=04:00:00`),
2026-08-17. Implementation: `main_inference_shuffled.py` ->
`VAMShuffledInference`, which overrides `_add_image_to_history` (the only
entry point `VAMInference.step()` uses to push an observed frame) to
discard the real frame and push one drawn from a 64-frame pool instead.

**Precisely what "wrong episode" means here (verified against the code,
2026-08-21, same check run against F1-VLA's copy of this mechanism — see
that repo's own README): the pool (`_load_frame_pool`) is built once via
`rng.choice(meta.total_episodes, size=64, replace=False)`
(`np.random.default_rng(seed=123)`) over every episode in the entire
`bridge_orig_lerobot` dataset, no task filtering — while evaluating Carrot,
the substituted frame can come from Spoon, Stack, or Eggplant (a different
scene, and for Eggplant a different camera rig) just as easily as from a
different Carrot episode. A fresh index is drawn from this pool on every
control step (`self._shuffle_rng.integers(...)`, seed 123), not fixed once
per episode.**

| task | baseline | ablated (zero) | shuffled (wrong episode/task real) |
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
F1, shuffled (real image, wrong episode, and — per the pool mechanism above
— often a wrong task/camera rig too) recovers success almost exactly to
baseline on every task, closed-loop — the model apparently only needs
*some* real visual grounding in that slot, not the right one.
For mimic-video, shuffled is statistically indistinguishable from ablated:
a real image from a different episode (or a different task entirely) helps exactly as much as zeros do,
i.e. not at all. The picture for mimic-video is unusually clean and
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

**Decision: not pursuing further.** The July logs that could have settled
this are gone (30-day `/scratch` purge), and the qualitative finding this
discrepancy sits next to (ablation drives every task to exactly zero) is
unaffected by which baseline figure is correct. Documenting as an open,
unresolved measurement anomaly rather than spending more compute chasing it.

## Not yet done

- [ ] **Confound not yet separated:** same as F1-VLA's copy of this
      experiment — the shuffle pool draws from the entire
      `bridge_orig_lerobot` dataset with no task filtering (verified
      2026-08-21), so "wrong episode, same task" and "wrong task entirely"
      are mixed into one condition. Given mimic already collapses to 0%
      under this mixed condition, a same-task-only variant could only
      sharpen the finding (rule out "maybe a same-task wrong episode would
      have partially worked"), not weaken it — lower priority than F1's
      version of this gap, where the direction of a possible correction
      actually matters.
