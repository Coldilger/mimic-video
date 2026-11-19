# Experiment 8 — Instruction–foresight conflict (mimic-video)

**Question (advisor's point 7).** Same observation, different instruction:
does the action follow the instruction or the imagined future? Same design
and instructions as F1-VLA's `eval/bridge/experiment8_conflict` (cue
conflict, Geirhos et al. 2019; interchange interventions, Geiger et al.
2021).

## How the design maps onto mimic-video

mimic-video has **no direct route** from the instruction to the action: the
action decoder reads only the robot state and the video model's layer-20
activations (`world2action_model.py`: `self.pipe(state, crossattn_emb,
sigma)`); the instruction is the video model's prompt. The closest analogue
of F1's "understanding block vs. forecast" split is the two parts of the
decoder's context at the deployed operating point (k = 1 of 35 denoising
steps, 19,200 tokens):

| part | tokens | content |
|---|---|---|
| observed-frame tokens | first 2,400 (2 of 16 latent frames) | the 5 input frames |
| future-frame tokens | other 16,800 (14 latent frames) | the imagined future |

Both parts are computed under the prompt, so neither is free of the
instruction; the swap separates "what the model sees" from "what it
imagines", not language from vision. Cells: AA (= production), AA2 (video
seed 1: noise floor), BB, AB = observed(A) + future(B), BA = observed(B) +
future(A); video and decoder seeds fixed at 0 as in production.

`follow(AB) = d(AB,AA) / (d(AB,AA) + d(AB,BB))`: 0 = the action stays with
the prompt of the observed-frame tokens, 1 = it goes with the prompt of the
future tokens.

## Files

| file | role |
|---|---|
| `vam_conflict_inference.py` | `ConflictPipeline` (drop-in for the production pipeline), `VAMConflictInference` |
| `main_inference_conflict.py`, `conflict.slurm` | entry point; `MODE=probe` or `MODE=cl OBS=A/B FUT=A/B` |
| `analyze_conflict.py` | probe and closed-loop tables, `conflict_summary.json` |
| `jobs_smoke.txt`, `records_smoke/` | checks below |

## Checks (2026-11-19, Carrot, episodes 0–2)

- The closed-loop cell obs A / fut A and the probe (driven by AA) reproduce
  the Experiment 1 baseline episodes exactly (outcomes and progress flags).
- Token split read from the model: 19,200 tokens, 2,400 observed.
- First probe numbers (3 episodes, 36 decisions, not a result): follow
  0.82–0.90 for all four alternative instructions, i.e. the action goes
  with the future tokens' prompt — the opposite of F1-VLA (0.01–0.02).

## Results

Full runs submitted 2026-11-19 (probe, 4 tasks × 24 episodes; closed loop
obs A/fut B, obs B/fut A, obs B/fut B with the mirrored instruction; the
obs A/fut A cell is Experiment 1's `exp1_baseline_stop1_seed0`).
