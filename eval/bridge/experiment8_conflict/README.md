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

## Results (2026-11-20)

16 jobs, all `EVAL_EXIT=0`. Tables from `analyze_conflict.py`.

### Single decisions (1,440 decisions on states of the deployed rollout, 4 tasks × 24 episodes)

Noise floor (video seed 1 vs 0, same prompt): the future tokens change by
105% (relative L2) and the observed tokens by 12%, the action by
0.036–0.091. Prompt A vs. B (mean over the four tasks; per task and per
alternative in `analyze_conflict.py`):

| B | context change observed / future | d(BB, AA) | d(AB, AA) | d(BA, BB) | follow(AB) | follow(BA) |
|---|---|---|---|---|---|---|
| mirror (both objects present) | 0.08–0.17 / 0.25–0.37 | 0.17–0.27 | 0.17–0.27 | 0.17–0.27 | 0.86–0.92 | 0.86–0.93 |
| other three tasks' instructions | 0.07–0.11 / 0.20–0.30 | 0.07–0.23 | 0.06–0.23 | 0.06–0.23 | 0.79–0.93 | 0.80–0.94 |

Swapping in the future tokens of the other prompt moves the action almost
all the way to where the other prompt sends it (follow 0.79–0.93 in all 16
pairs, both directions); the observed-frame tokens carry the rest.

### Closed loop (mirrored instruction, 96 episodes per cell)

| observed tokens from | future tokens from | success (task A) | moved A's object | moved the other object | grasped A's object |
|---|---|---|---|---|---|
| A | A (= deployed, Exp 1 baseline) | 46/96 | 64/96 | 0/96 | 61/96 |
| A | B | 0/96 | 0/96 | 20/96 | 0/96 |
| B | A | 48/96 | 69/96 | 0/96 | 68/96 |
| B | B | 0/96 | 0/96 | 18/96 | 0/96 |

Paired: obs A/fut A → obs A/fut B −47.9 pp [−56.2, −39.6], p < 0.0001;
obs B/fut B → obs B/fut A +50.0 pp [+42.7, +57.3], p < 0.0001. Per task
(success / moved A's object / moved the other, of 24): obs A/fut B Carrot
0/0/0, Eggplant 0/0/0, Spoon 0/0/4, Stack 0/0/16; obs B/fut B Spoon 0/0/3,
Stack 0/0/15; obs B/fut A Carrot 9/11/0, Eggplant 24/24/0, Spoon 13/15/0,
Stack 2/19/0.

**What this shows.**
1. **mimic-video's action follows the instruction carried by the imagined
   future.** With the future tokens of the mirrored prompt the robot never
   completes A and, on Stack, moves the other cube in 16/24 episodes —
   the same as when the whole context comes from B (15/24). With the
   future of A and the observed-frame tokens of B it completes A as often
   as the deployed policy (48 vs 46/96).
2. This is the opposite of F1-VLA, where a forecast made under another
   instruction moves the action no more than noise (follow 0.01–0.02,
   closed loop unchanged).
3. **Caveat (same as Experiment 1).** At the deployed operating point the
   future tokens are the decoder's main view of the scene as well as of
   the future; the swap moves "where the video model thinks the episode is
   going", which includes its reading of the instruction. It shows that
   the instruction reaches the action through the future-frame part of the
   context, not that a *predicted future* is what the decoder uses.
4. **Splice check (2026-11-20, from the 1,440 probe records).** The
   split is by token position; each run (A, B) is computed by the video
   model on its own and only the decoder sees the spliced sequence. The
   spliced actions coincide with the unspliced action of the future
   tokens' prompt: median d(AB, BB) = 0.008 and d(BA, AA) = 0.008, against
   d(BB, AA) = 0.118 and the seed noise floor d(AA2, AA) = 0.032. An
   incoherent ("mushed") context would land far from both. Together with
   the closed-loop B/A cell (48/96, as deployed) this rules out the splice
   itself as the cause of the 0/96. It matches Experiment 1
   (`../experiment1_ablation/README.md`, "Can the future be removed while
   keeping perception?"): the decoder puts 14% of its attention on the
   observed-frame tokens, and dropping them changes the action by 5%.
5. Under the mirrored instruction alone (B/B) the robot fails A everywhere
   but only reaches the other object on Stack and Spoon; on Carrot and
   Eggplant the other object is the plate / basket, which it never moves.
