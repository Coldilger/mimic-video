# Experiment 3 — Cost per decision (mimic-video)

**Status: planned, not yet run.**

## What this tests

Context for interpreting Experiments 1 and 2: how much inference-time
compute does each model actually spend on its world-model computation, and
at what latency / success-rate trade-off?

| Category | Model | World-model compute @ inference | Latency / control step (median · p95) | Success rate, SimplerEnv-Bridge (95% CI) |
|---|---|---|---|---|
| 1 | F1-VLA | VAR foresight loop, re-run every control step | TBD | TBD |
| 2 | **mimic-video** | **One** video-backbone forward pass per action chunk (amortised over the chunk) | TBD | TBD |
| 3 | LDA-1B | None beyond the shared MM-DiT — the visual-forecasting head is a training-time co-objective, unused at inference | TBD | TBD |

mimic-video sits in the middle structurally: cheaper than F1 (which re-runs
its foresight loop every control step) because the video-backbone forward
pass is amortized over a whole action chunk, but not free like LDA-1B
(which does no extra world-model compute at inference at all).

## Not yet done

- [ ] Instrument the eval wrapper (`main_inference_hil.py` /
      `VAMInference`) to record per-step wall-clock latency, split into
      video-backbone forward pass vs action-decoder compute vs I/O.
- [ ] Run on real hardware across a real SimplerEnv-Bridge eval sweep,
      report median/p95.
- [ ] Cross-reference against the already-measured success rate for the
      same checkpoint/seeds.
