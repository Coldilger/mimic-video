# Experiment 9 — How deep does mimic-video's forecast go, and does the decoder follow it? (open loop, 2026-11-12)

Advisor point 4 (vary the number of generation steps; success vs. total
latency), mimic-video part. Motivation: at the deployed operating point
(`stop=1`) foresight and perception share the same context tokens and cannot
be ablated separately (`../experiment1_ablation/README.md`, "Can the future
be removed while keeping perception?"). This experiment varies the step `k`
after which the decoder reads the video model's layer-20 activations, as in
the mimic-video paper's own analysis of the video flow time τv, and adds a
test that leaves perception untouched: the video model's random seed.

Key fact from the training code (`model/cosmos_predict2/models/world2action_model.py`,
`get_crossattn_emb`): the decoder was trained on activations of the **real**
future video (56 frames at 5 Hz) noised to a random sigma. At small sigma it
expects the real future in the future tokens; at inference it gets the
video model's own prediction.

## Setup

`depth_open_loop.py` (job 755694, H100, 38 min; log `depth-ol-755694.out`,
per-clip data `depth_open_loop_8ep.json`). 16 real Bridge clips (8 episodes ×
2 time points, own task string): 5 observed frames + the 56 real future frames
as in training (past the episode end the last frame repeats). Fixed robot
state, decoder seed 0. `k` ∈ {0, 1, 2, 4, 8, 16, 23, 30} of 35 steps.

- **Self-check:** the test re-implements the sampling loop to read several
  steps per run; at `k=1` its action equals the production call exactly
  (max |diff| = 0).
- **Forecast quality:** the video model's clean-video estimate at step `k`
  (latent space), error to the real future, divided by the error of a
  "nothing moves" future (last observed frame repeated). < 1: closer to the
  truth than a static scene.
- **Seed test:** video seeds 0–3. Same frames, same instruction, so
  perception is identical; only the imagined future differs. *Follow
  future*: seed 0's context with the future tokens (latent frames 2–15) of
  seed s; 0 = the action stays with seed 0, 1 = it moves to seed s's action.
  *Follow observed* (control): the same with the observed tokens swapped.
- **Oracle:** the context built as in training from the real future noised to
  the same sigma; action change vs. the model's own forecast.
- Actions are compared between conditions only, never to the dataset's
  actions.

## Results (mean over 16 clips)

| k | sigma | forecast error vs. static | future spread across seeds | action spread across seeds | follow future | follow observed (control) | oracle vs. own forecast | translation | rotation error | latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 80.0 | 0.69 | 0.09 | 3% | 0.77 | 0.23 | 14% | 0.070 | 0.007 | 0.82 |
| **1 (deployed)** | 68.3 | 0.69 | 0.11 | 4% | 0.79 | 0.22 | 16% | 0.069 | 0.007 | 1.36 |
| 2 | 58.1 | 0.70 | 0.15 | 6% | 0.82 | 0.20 | 16% | 0.068 | 0.007 | 1.89 |
| 4 | 41.6 | 0.74 | 0.31 | 14% | 0.87 | 0.13 | 21% | 0.068 | 0.008 | 2.95 |
| 8 | 20.2 | 0.86 | 0.50 | 19% | 0.85 | 0.17 | 27% | 0.066 | 0.008 | 5.09 |
| 16 | 3.65 | 0.95 | 0.59 | 32% | 0.84 | 0.18 | 31% | 0.068 | 0.011 | 9.30 |
| 23 | 0.52 | 0.97 | 0.61 | 32% | 0.80 | 0.21 | 31% | 0.068 | 0.010 | 13.03 |
| 30 | 0.04 | 0.99 | 0.63 | 21% | 0.76 | 0.26 | 19% | 0.049 | 0.023 | 16.71 |

No NaN. "Follow future" exceeds "follow observed" in 16/16 clips at every `k`
checked (1, 4, 16, 23). Latency: production call without CUDA graphs, one
clip, H100 NVL (Experiment 3 measured 1.06 s at `k=1` in the real eval loop).

## What this means

1. **The decoder follows the imagined future.** With perception held fixed,
   swapping in another seed's future tokens moves the action ~80% of the way
   to that seed's action; swapping the observed tokens moves it ~20%. This
   holds at every depth. So the decoder does read the future part of its
   context, not only a view of the present.
2. **With depth, the imagined future becomes specific and the action depends
   on it.** Different seeds' futures differ by 9% at `k=0` and ~60% from
   `k=16` on; the action differs by 3–4% at `k≤1` and ~32% at `k=16–23`.
   At the deployed `k=1` the future is still close to a seed-independent
   average, so the action barely depends on the seed.
3. **Deeper forecasts are not closer to the truth.** The error to the real
   future grows from 0.69 to 0.99 of a static scene's error. Caveat: a
   squared-error metric favours a blurry average over a sharp guess, so this
   says the specific futures are not *on average* closer to the truth, not
   that they are useless. Also, Bridge episodes are short: the clips have
   7–41 real future frames (of 56), the rest repeats the episode's last frame
   (the same clamping as in training). That part of the "real future" *is*
   static, which favours the static baseline; the near-future ratio (first 3
   future latent frames, where 15/16 clips still have real frames) shows the
   same trend (0.63 → 0.94).
4. **Given the real future, the decoder acts differently** (14–31% change),
   about as much as between two seeds at the same depth: to the decoder the
   real future looks like one more plausible future.
5. **Cost:** latency grows ~0.5 s per step, from 1.4 s at `k=1` to 13 s at
   `k=23`. At `k=30` the decoder starts leaving its normal range (rotation
   error ×3, less motion).

**Hypothesis for the closed loop (not yet tested):** the earlier unpaired
result (success 47.9% at `k=1`, 11.5% at `k=23`) fits a decoder that follows
its forecast: deeper forecasts commit to one specific future, the action
follows it, and when that future is wrong the episode fails. Next: paired
closed-loop success vs. `k` on the 96 Experiment 1 episodes, with latency.
