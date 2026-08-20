# Offline ablation probe — backlog, not a cited result

Moved out of `README.md` (2026-08-20). Compares a predicted action against
the logged action from `bridge_orig_lerobot` — the exact dataset this
checkpoint was finetuned on, with no held-out split — so any result is
confounded with memorization and isn't decisive evidence either way. This
experiment's real, citable result is the closed-loop success-rate work in
`README.md`, which doesn't have this problem. Same rule as
`experiment2_oracle/OFFLINE_PROBE_BACKLOG.md` and F1-VLA's copy of this
file: offline/replay-based L1 probes are not a decisive or citable metric
anywhere in this thesis — this holds even though (as it turned out here)
the offline probe found no signal at all in either direction; a null result
from a memorization-confounded probe is just as uninformative as a
positive one.

Kept here only for provenance. Not linked from the main doc's narrative,
not to be cited in the thesis write-up or presentation.

## Results (both variants, n=120: 24 episodes × 5 samples/episode, run 2026-08-16)

| metric | baseline (self-imagined) | ablated (zero crossattn) | shuffled (wrong-episode real) | no-motion baseline |
|---|---|---|---|---|
| position L1 | 0.0098 (sd 0.0080) | 0.0100 (sd 0.0078) | 0.0099 (sd 0.0081) | 0.0096 |
| gripper L1 | 0.3327 | 0.3145 | 0.3581 | — |

All three conditions landed in the same narrow band, indistinguishable from
each other and from the no-motion floor. This turned out to say nothing
about the real, closed-loop result (`README.md`'s own closed-loop tables):
all three conditions collapsing offline did not predict that ablated and
shuffled would collapse all the way to 0% success while baseline stayed
above 0% — the closed-loop test is what actually carries evidential weight
here, not this probe.
