# Offline oracle probe — backlog, not a cited result

Moved out of `ORACLE_EXPERIMENT.md` (2026-08-20). This probe replays
`bridge_orig_lerobot` — the exact dataset this checkpoint was finetuned on,
with no held-out split — so any gap it shows is confounded with
memorization and cannot be trusted as evidence about the causal question
this experiment asks. Established as a general rule for this thesis, not
specific to mimic-video: offline/replay-based L1 probes are not a decisive
or even citable metric anywhere, precisely because of this confound (same
reasoning that already applies to Experiment 1 — see
`feedback_exp1_success_rate_metric` memory). The live oracle probe in
`ORACLE_EXPERIMENT.md` (randomized SimplerEnv-Bridge rollouts, not the
training dataset) is what replaced this, specifically because it does not
have this problem.

Kept here only for provenance/traceability — not linked from the main
doc's narrative, not to be cited in the thesis write-up or presentation.

## What it measured

Take a real, logged moment from an actual Bridge episode: the "now" frame
plus what the robot actually did a second later. Show the model "now" (and,
in the oracle condition, the real "next" frame via `is_hil=True` +
`ingest_video()`), compare the predicted action's L1 distance to the real
logged action.

## Results (115 samples: 24 episodes × 5 moments)

| | oracle | baseline (arm doesn't move) |
|---|---|---|
| position, L1 | 0.0092 (sd 0.0063) | 0.0097 |
| gripper, L1 | 0.1717 | — |

Oracle and the trivial "arm doesn't move" baseline came out almost
identical here — but since this replays the training set with no held-out
split, that null result cannot be cleanly attributed to "the mechanism
genuinely doesn't benefit from foresight" versus "the model already
memorized this trajectory well enough that neither condition needed the
signal." Do not read anything into this table beyond "the probe ran and
produced a number" — the live probe is the one that answers the actual
question.
