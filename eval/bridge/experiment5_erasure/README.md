# Experiment 5 — Concept erasure (mimic-video)

**Status: Level A run and decisive. Level B (causal downstream re-injection)
not started, scoped pending Level A's outcome.**

## Motivation

Experiment 4's decisive control found the probe ~3.5x WORSE than a constant
predictor on current end-effector pose, while an episode-identity probe
scored near-perfect (see the corrected number below — the original "100%,
40x chance" figure had a bug, fixed as part of this experiment). Read
together, this is consistent with the representation being dominated by
which-scene-is-this, with a probe fit on ~30 training episodes extrapolating
catastrophically to unseen scenes regardless of pooling design (confirmed
across two structurally different extraction designs, see
`experiment4_probing/README.md`'s "Cross-model finding").

Rather than re-extracting with more episodes (expensive, ~1.5h/run, unclear
payoff given Bridge episodes cluster into a handful of scene families) this
tests the hypothesis directly and cheaply: **surgically remove the
scene-identity direction from the already-extracted features, then check
whether pose becomes recoverable in what's left.** If it does, scene
dominance was masking real pose information. If it doesn't, the problem is
elsewhere (extraction/pooling or genuine data scarcity), and no amount of
"more episodes" alone would have been guaranteed to fix it either.

## Method: LEACE, not INLP

Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman, "LEACE:
Perfect linear concept erasure in closed form" (NeurIPS 2023,
arXiv:2306.03819). Chosen over the more commonly-cited INLP (Ravfogel et
al. 2020) for three concrete reasons relevant here, not just novelty:

- **Continuous/vector targets are native, not a retrofit.** INLP's machinery
  and guarantees are built around linear *classifiers*; LEACE is derived
  directly from the cross-covariance `Cov(X, Z)`, well-defined for any real
  vector `Z`. Level B's actual target (9-D pose) needs this; Level A's
  target (episode identity) is used here as a one-hot vector for the same
  reason, so both stages share one method.
- **Provably minimal distortion.** Among all affine maps that make `Z`
  linearly unrecoverable from the result, LEACE's closed-form choice is the
  one minimizing `E[||r(x)-x||^2]`. This matters for Level B's planned
  comparison against Experiment 1's full-signal ablation: if erasure itself
  over-damaged the representation, a resulting behavior change wouldn't be
  attributable to removing *just* the pose-relevant content.
- **Closed-form.** No iterative train/remove/retrain loop (unlike INLP), so
  the same code runs across multiple models/checkpoints without a separate
  training cycle each time -- practical given this needs to run on 2+
  checkpoints across at least 2 models.

**Not used: O-LEACE (the oracle variant).** The paper documents that fitting
a per-point erasure directly from that point's own true label can
paradoxically leave *more* nonlinearly-recoverable concept information than
proper train-fit/held-out-apply erasure -- the opposite of the goal. `fit_leace`
is only ever called on the train split; `erase()` is then applied unchanged
to held-out data. Same train/val discipline already used for feature
standardization and ridge alpha selection elsewhere in this experiment, not
a new one introduced here.

**Implementation notes** (`leace.py`): whitening uses ridge-regularized
eigenvalues of `Cov(X)`, not a raw pseudo-inverse -- with `d` in the
thousands and `n` in the hundreds, `Cov(X)` has rank <= n-1, and
un-regularized whitening would divide by near-zero eigenvalues and amplify
noise directions. Validated on synthetic data with a KNOWN injected concept
direction before ever touching real features (`python leace.py`):

```
R^2 before erasure: 0.6694   (d=200 < n=300 regime)
R^2 after erasure:  -0.0052  (held-out, never seen by fit)
SELF_TEST_PASSED

R^2 before (d=3072 >> n=300, matching our real regime): 0.3787
R^2 after erasure:  0.0026
relative distortion: 0.0619
PASS
```

## A bug found and fixed along the way

Building this surfaced a real leak in `experiment4_probing/positive_control.py`:
its alpha was selected by directly maximizing accuracy **on the held-out val
set** -- val used for both hyperparameter selection and evaluation. This
inflated the originally-reported "100%, 40x chance" episode-identity result
for both mimic-video and F1-VLA. Fixed to select alpha via k-fold CV within
the train split only (matching `train_probe.py`/`compare_targets.py`'s
already-correct methodology), and re-run -- see
`experiment4_probing/positive_control.py`'s updated docstring and the
corrected numbers there. The qualitative finding survives (episode identity
is still far above chance), the specific magnitude needed correcting.

A related design subtlety surfaced while building Level A's own
episode-identity validation: it must stay within the SAME set of episodes
the eraser was fit to distinguish. Checking a 30-episode eraser's effect
against a *different* 40-episode/different-split classification task tests
a higher-rank problem (up to 39 separating directions vs. the 29 the eraser
targets) that a working eraser isn't obligated to solve -- this produced a
misleadingly bad "erasure failed" reading before being caught by checking
against fresh samples of the *same* 30 episodes instead.

## Results

Run 2026-08-18, features from `experiment4_probing/features_{variant}.npz`
(the original mean+std-pooled extraction), episode-level split (30 train /
10 val episodes), n=400.

| variant | episode-ID before | episode-ID after erasure | current-pose gain before | current-pose gain after |
|---|---|---|---|---|
| finetuned | 100.0% (chance 3.3%) | **2.7%** | −251.4% | **−250.0%** |
| pretrained | 97.3% (chance 3.3%) | **4.0%** | −247.1% | **−249.9%** |

Erasure worked as designed -- episode identity collapses from near-perfect
to indistinguishable from chance, on fresh held-out samples of the exact
episodes it was fit to separate, for both checkpoints. **Current pose does
not become recoverable.** The gain over a constant predictor is essentially
unchanged (within noise) before and after erasure.

## Verdict: scene dominance is refuted as the (sole) explanation

The hypothesis motivating this experiment -- "pose information is present
but masked by the scene-identity confound" -- does not hold. Once the
confound is genuinely and completely removed (verified independently, not
assumed), pose recoverability doesn't improve at all. Whatever limits
current-pose recovery is not simply "the probe was distracted by a stronger,
irrelevant signal."

This narrows, rather than closes, the remaining explanations from
`experiment4_probing/README.md`:
- extraction/pooling is still too lossy (plausible for the original
  mean+std-over-19200-tokens design; less clear for F1-VLA's
  individually-kept 30 tokens -- see that repo's Experiment 5 results)
- genuine data scarcity: not enough distinct episodes/samples for a linear
  probe to find a real but weak pose signal at all, confound or not
- the representation genuinely does not encode pose linearly at this
  extraction point

## Level B: not started

The causal downstream test -- erase the pose-relevant direction specifically
(once one can be reliably identified), re-inject the erased representation
into the frozen action decoder at the same hook point as Experiment 1
(`crossattn_emb`), and compare the resulting action sensitivity/success rate
against Experiment 1's full-signal ablation -- was scoped to run only if
Level A suggested a real, recoverable pose direction worth erasing causally.
Given Level A's negative result (pose isn't reliably identifiable as a
direction in the first place, confound removed or not), building Level B on
top of a noisy, poorly-identified erasure target would not be interpretable
right now. Revisit if a future extraction change makes pose linearly
recoverable at all.

## Files

- `leace.py` -- LEACE fit/erase implementation + synthetic self-test
- `scene_erasure_diagnostic.py` / `scene_erasure.slurm` -- Level A

## Not yet done

- [ ] Re-run against `experiment4_probing/features_*_v2.npz` (spatial-grid
      pooled re-extraction) once that finishes, for a same-model
      before/after-pooling-fix comparison.
- [ ] Decide on Level B based on whether any future extraction change makes
      pose linearly identifiable.
