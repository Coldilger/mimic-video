# Experiment 2 — Oracle injection (mimic-video)

**Status: done.** See [`ORACLE_EXPERIMENT.md`](ORACLE_EXPERIMENT.md) for the
full write-up (mechanism, metric, results, caveats).

Short version: instead of letting the Cosmos-Predict2 video backbone imagine
the future through denoising, feed it the real next frame directly
(`is_hil=True` + `model.ingest_video()`, the same mechanism the repo
documents for live human teleoperation, repurposed here for offline replay
of real frames) and compare predicted-action L1 error against the trivial
"arm doesn't move" baseline. Result: the gap between oracle and the trivial
baseline is almost zero — even given the real future, the decoder doesn't
extract meaningfully more value from it. This independently confirms, from
the opposite direction, a finding already in mimic-video's own paper (best
performance at `τv≈1`, i.e. pure noise rather than a rendered video).

Files: `oracle_offline_probe.py` (offline probe, no simulator),
`oracle_offline_probe.slurm` (launcher), `oracle-offline-*.out` (past run
logs).
