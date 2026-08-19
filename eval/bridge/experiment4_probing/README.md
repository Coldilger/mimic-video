# Experiment 4 — Representation probing (mimic-video)

**Status: implemented and run; the measurement does not work as designed. No
conclusion about mimic-video's representation is available yet.** The pipeline
end-to-end is in place and validated; what fails is the probe's ability to
recover anything from mean+std-pooled features at 40 episodes. See
"Why the current design cannot answer the question" below before reading any
number here as a result.

## What this tests

The direct test of the "training-time representational effect" side of the
central research question: does mimic-video's world-model representation
already encode useful future information, independent of what the
inference-time denoising computation does with it?

## Design

**Extraction point: `crossattn_emb`** — the same tensor Experiment 1 ablates
(`../experiment1_ablation/zero_world_model_pipeline.py`), and the only channel
by which the video-diffusion backbone reaches the action decoder. Probing and
ablation therefore address the same place, which makes the two experiments
directly comparable. Captured by wrapping `video2world_pipeline.generate_video`
on the live instance; nothing under `model/` is modified.

**Probe target: the delta end-effector pose K=5 steps ahead**, not the absolute
pose. Experiment 1's offline probe showed absolute single-step position sits at
the no-motion floor (every condition 0.0098–0.0100 against a 0.0096 floor),
i.e. it measures "the arm barely moves" rather than representation quality.

**Checkpoint comparison: `pretrained_cosmos_bridge` vs
`finetuned_cosmos_bridge`.** This is the memorization-robustness control:
both models were finetuned on all of Bridge with no held-out split, so a
single absolute probe score cannot be separated from memorization. The two
released checkpoints differ *exactly* in the video backbone — generic Cosmos
vs. Bridge-LoRA-finetuned — while both ship the same Bridge-trained action
decoder, which this experiment never probes. So the comparison isolates what
Bridge finetuning built into the probed representation.

Verified as a controlled comparison at full scale: targets identical,
episode ids identical, features different (cosine similarity 0.93, feature
norms 104.9 finetuned vs 177.9 pretrained). The norm gap is why features are
standardized on train-split statistics — without it a pure scale difference
would masquerade as an information difference.

**Probe: ridge regression with cross-validated alpha**, dual form (the n-by-n
solve, since d=4096 >> n). An MLP head is computed alongside as a capacity
check but is not the measurement — see below.

**Split: by episode, never by sample.** Consecutive frames are near-duplicates,
so a per-sample split would put near-copies of training frames in validation.

## Results

Run 2026-08-18, n=400 (40 episodes × 10 samples), 30 train / 10 val episodes.

| variant | ridge L1 | MLP L1 | best constant | no-motion |
|---|---|---|---|---|
| finetuned | 0.05292 | 0.05699 | 0.05326 | 0.05266 |
| pretrained | 0.05379 | 0.06275 | 0.05326 | 0.05266 |

Neither variant beats the no-motion reference. The finetuned model edges past
the best-constant predictor by 0.6% and the pretrained one does not, so the
ordering is in the expected direction — **but see below: this number comes
from an instrument that fails its own control, and should not be quoted.**

The ridge probe beats the MLP for both variants, confirming the MLP was
overfitting: 4096 input dimensions against 300 training samples gives a
first layer alone over a million parameters. A regularized linear probe is
the right tool for "is this information linearly accessible", and its
capacity is controlled rather than assumed.

## Why the current design cannot answer the question

Two controls were run to separate "the representation lacks future
information" from "the measurement cannot see it".

**Control 1 — episode identity (`positive_control.py`).** Can the features
identify which episode a sample came from? **100% accuracy, 40× chance, both
variants.** The features are far from empty.

This looked like a pass, but it is a weak control: episode identity is a
*global* scene property, exactly what mean+std pooling over 19200 tokens
preserves. Future arm motion is spatially localized, exactly what pooling
would destroy. Passing control 1 rules out "pooling destroyed everything",
not "pooling destroyed the localized part".

**Control 2 — current pose (`compare_targets.py`).** The sharp version: probe
the same features, same standardization, same episode split, for the *current*
end-effector pose. It is localized like the future target, but unambiguously
present in the input frame. Current pose was recovered without re-running the
model, by replaying the extraction's RNG draws (`recover_current_pose.py`,
verified: all 400 samples matched the saved episode order).

| target | probe L1 | best constant | improvement |
|---|---|---|---|
| current pose (finetuned) | 0.31156 | 0.08867 | **−251%** |
| current pose (pretrained) | 0.30779 | 0.08867 | **−247%** |
| future delta (finetuned) | 0.05292 | 0.05326 | +0.6% |
| future delta (pretrained) | 0.05379 | 0.05326 | −1.0% |

**The probe is ~3.5× worse than a constant predictor on current pose, for both
variants.** That is not a weak signal, it is active failure on a task the
features must support.

**Mechanism, and why control 1's perfect score was a warning rather than good
news.** The features encode episode identity so strongly that scene identity
is the dominant direction of variance. With an episode-level split and only 30
training episodes, ridge fits "this scene → that pose" and then extrapolates
catastrophically onto unseen scenes. The future-delta target escapes the worst
of this only because deltas are far less scene-specific than absolute poses —
which is also why its score sits inertly at the constant-predictor level rather
than going negative.

**Consequence: the 0.6% finetuned-over-constant figure is not a finding.** It
is produced by the same instrument that fails control 2, and nothing about
mimic-video's representation — including the finetuned-vs-pretrained ordering
— can be concluded from this run.

## What would have to change

- **Far more episodes.** 40 is not enough when features are scene-dominated;
  the probe needs enough distinct scenes to learn a scene-invariant mapping.
  This is the cheapest lever and probably the decisive one.
- **Less aggressive pooling.** Mean+std over 19200 tokens discards spatial
  structure, which is where manipulator state lives. Storing every token is
  infeasible (~157MB/sample), but pooling over a coarse spatial grid rather
  than globally, or projecting tokens down before pooling, would keep some of
  it.

Both are extraction-side changes, so the probe code and the controls above can
be reused unchanged.

**Update (2026-08-18): the spatial-grid pooling fix (v2 extraction, see
`extract_features_v2.py`) is done and confirmed a controlled comparison
(smoke test: identical targets across variants, different real features), but
before its own controls finished, F1-VLA's copy of this experiment ran the
identical probe+controls on features that were NEVER pooled this way at
all — F1's real world-model signal is only 30 tokens, kept individually, no
spatial averaging anywhere — and got the **same catastrophic current-pose
failure** (−254%/−254% vs. mimic's −251%/−247%). That rules out "pooling
destroyed the localized signal" as the sole explanation here too: whatever's
wrong is shared across two structurally different extraction designs, which
points at episode/scene diversity (the other item above), not pooling
granularity, as the dominant lever. See F1-VLA's
`eval/bridge/experiment4_probing/README.md`, "Cross-model finding" section,
for the full argument. mimic's own v2-pooled features should still be run
through the same two controls once extraction finishes, for a same-model
before/after comparison — but the prior going in is now that pooling alone
won't fix it.

## Files

- `extract_features.py` / `extract.slurm` — frozen-feature extraction, either
  checkpoint variant
- `download_pretrained.slurm` — fetches `pretrained_cosmos_bridge`
- `train_probe.py` / `train_probes.slurm` — ridge (primary) and MLP (capacity
  check) probes, with no-motion and best-constant references
- `positive_control.py` / `positive_control.slurm` — control 1, episode identity
- `recover_current_pose.py` / `recover_pose.slurm` — replays extraction RNG to
  recover current pose without the model
- `compare_targets.py` / `compare_targets.slurm` — control 2, current vs future

## Update (2026-08-18): the pooling fix alone doesn't help, confirmed directly

`extract_features_v2.py` (spatial-grid pooling, see its docstring) was run at
the same scale as the original (40 episodes x 10 samples) specifically to
isolate pooling from episode count. Result: current-pose gain −244.2%
(finetuned) / −246.9% (pretrained) -- statistically indistinguishable from
the original mean+std-pooled design's −251.4%/−247.1%.

This is now confirmed by three independent methods pointing the same
direction: F1-VLA's extraction (no spatial pooling at all, 30 tokens kept
individually) fails identically; LEACE-based erasure of the scene-identity
confound (`experiment5_erasure/`) leaves pose exactly as unrecoverable;
and now re-pooling with spatial structure preserved, at the same episode
count, changes nothing. Pooling design is not the bottleneck. A
substantially-larger-episode-count re-extraction is in progress to test the
remaining candidate (data scarcity) directly.

## Update (2026-08-18): 6.25x more episodes helps in degree, not in kind

Re-extracted with 250 episodes x 2 samples (v2, spatial-grid pooling) vs.
the original 40 x 10 -- 6.25x more distinct episodes, similar total sample
count (498 vs 400, two episodes too short to sample were skipped).

| | 40 episodes (30 train / 10 val) | 250 episodes (~187 train / ~62 val) |
|---|---|---|
| current-pose gain | −244.2% / −246.9% | **−110.6% / −110.7%** |
| future-pose gain | −18.6% / −9.3% | **+0.8% / +0.4%** |
| episode-ID accuracy | 100% (chance 2.5%, 40x) | 58.9% (chance 0.4%, **147x**) |

Same story as F1-VLA's independent re-run (see that repo's README): real
improvement in *degree* -- current-pose failure roughly halves -- but not in
*kind* -- still nowhere near recoverable, and future-pose still shows no
real signal. Episode-identity accuracy drops in absolute terms (harder
249-way problem vs. 30-way) but is if anything *more* extreme relative to
chance (147x vs 40x): scene identity remains trivially easy to recover at
this scale; pose does not. Two structurally different models, two very
different episode-count multipliers (6.25x here, 25x for F1), same
qualitative result -- this is now a well-triangulated finding, not a
one-model artifact.

## Is current pose nonlinearly recoverable? (2026-08-19, cross-check from LDA-1B)

While probing LDA-1B, an MLP recovered its current pose (+60.2% over
constant) where ridge had failed badly (−229%) — raising the question of
whether mimic's own current-pose failure is also a linear-probe artifact.
Tested directly
(`LDA-1B/eval/bridge/experiment4_probing/mlp_current_pose.py`, same
`ProbeHead` architecture this repo's own `train_probe.py` already uses for
the future target, run against this repo's own
`features_finetuned_v2_more.npz` / `current_pose_more.npz`): **MLP
current-pose val L1 0.172 vs. constant 0.160 — a −7.7% gain, i.e. still
worse than guessing the mean.** Unlike LDA, nonlinearity is not the
explanation for mimic's current-pose failure — `crossattn_emb` genuinely
does not encode pose here, under either probe family.

## Not yet done

- [ ] Determine whether an order-of-magnitude-larger episode count closes
      the remaining gap, or whether the failure plateaus -- current data
      can't distinguish "needs much more data" from "extraction caps
      recoverability regardless of data volume."
- [x] Extend to LDA-1B — done 2026-08-19, see
      `LDA-1B/eval/bridge/experiment4_probing/README.md`. Notably different
      result: LDA's shared `vl_embs` backbone *does* encode both current and
      future pose, nonlinearly (unlike F1-VLA and mimic-video's own
      world-model-specific representations, neither of which recovers pose
      under any probe family tried).
