"""Experiment 8 (instruction-foresight conflict, advisor's point 7) for
mimic-video.

Extends the cue-conflict design (Geirhos et al., 2019) and interchange
interventions (Geiger et al., 2021) as in F1-VLA's
eval/bridge/experiment8_conflict. mimic-video has no second route for the
instruction: the action decoder reads only the robot state and the video
model's layer-20 activations (world2action_model.py: self.pipe(state,
crossattn_emb, sigma)); the instruction enters as the video model's prompt.
The closest analogue of F1's "understanding block vs. forecast" split is
therefore the two parts of the decoder's context:

    observed-frame tokens  (first 2 of 16 latent frames: the 5 input frames)
    future-frame tokens    (the other 14 latent frames: the imagined future)

Both parts are computed by the video model under the prompt, so neither is
free of the instruction; the split separates "what the model sees" from
"what it imagines", not language from vision.

Cells (A = the scene's instruction, B = the alternative):

    AA   observed(A) + future(A)                       = production call
    AA2  as AA, video seed 1 instead of 0              noise floor
    BB   observed(B) + future(B)
    AB   observed(A) + future(B)                       forecast from B
    BA   observed(B) + future(A)                       forecast from A

The video model and decoder seeds are fixed (0) as in production, so every
difference comes from the prompt or the token swap.

Modes (env CONFLICT_MODE):
  probe  the robot is driven by AA (= the deployed policy); every decision
         also decodes all cells and appends one record to
         records/<TASK>/episode_NNN.npz
  cl     the robot is driven by the cell CONFLICT_OBS (A|B) / CONFLICT_FUT (A|B)

Nothing under model/ or SimplerEnv/ is modified.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")

from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402

CARROT = "put carrot on plate"
SPOON = "put the spoon on the towel"
STACK = "stack the green block on the yellow block"
EGGPLANT = "put eggplant into yellow basket"
# same alternatives as F1-VLA's experiment8_conflict: "mirror" swaps the
# scene's two objects; the others are the other tasks' instructions
INSTRUCTIONS = {
    "PutCarrotOnPlateInScene-v0": [("mirror", "put plate on carrot"), ("spoon", SPOON), ("stack", STACK), ("eggplant", EGGPLANT)],
    "PutSpoonOnTableClothInScene-v0": [("mirror", "put the towel on the spoon"), ("carrot", CARROT), ("stack", STACK), ("eggplant", EGGPLANT)],
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0": [("mirror", "stack the yellow block on the green block"), ("carrot", CARROT), ("spoon", SPOON), ("eggplant", EGGPLANT)],
    "PutEggplantInBasketScene-v0": [("mirror", "put yellow basket into eggplant"), ("carrot", CARROT), ("spoon", SPOON), ("stack", STACK)],
}


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """mean over the chunk's steps of the L2 distance (10-dim action rows)"""
    return float(np.linalg.norm(a - b, axis=-1).mean())


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-8))


class ConflictPipeline(nn.Module):
    """Drop-in for Video2World2ActionPipeline (same __call__ signature)."""

    def __init__(self, v2w, w2a, task: str, mode: str, obs_src: str = "A", fut_src: str = "A", alt: str = "mirror"):
        super().__init__()
        self.video2world_pipeline, self.world2action_pipeline = v2w, w2a
        self.mode, self.obs_src, self.fut_src = mode, obs_src, fut_src
        self.alts = INSTRUCTIONS[task] if mode == "probe" else [(alt, dict(INSTRUCTIONS[task])[alt])]
        self.records: list[dict] = []
        self._logged = 0

    def _ctx(self, input_vid, prompt, seed, num_sampling_step, stop_after_step, ucg):
        T = input_vid.shape[2]
        emb, sigma = self.video2world_pipeline.generate_video(
            vid_input=input_vid, is_video_embedding=False, num_latent_conditional_frames=1 if T == 1 else 2,
            prompt=prompt, prompt_embedding=None, negative_prompt="", guidance=0.0,
            num_sampling_step=num_sampling_step, seed=seed, use_cuda_graphs=ucg,
            return_context_at_step=stop_after_step,
            hidden_state_layer_idx=self.world2action_pipeline.config.xattn_layer_idx,
        )
        n_lat = emb.shape[1] if emb.dim() == 5 else self.video2world_pipeline.config.state_t
        emb = emb.reshape(emb.shape[0], -1, emb.shape[-1])
        n_obs = emb.shape[1] // n_lat * (1 if T == 1 else 2)
        return emb, sigma, n_obs

    def _dec(self, state, emb, sigma, ucg):
        return self.world2action_pipeline(state_B_HO_O=state, crossattn_emb=emb,
                                          context_timesteps_B_1=sigma.unsqueeze(1), seed=0, use_cuda_graphs=ucg)

    @staticmethod
    def _mix(obs_from, fut_from, n_obs):
        return torch.cat([obs_from[:, :n_obs], fut_from[:, n_obs:]], 1)

    @torch.no_grad()
    def __call__(self, input_vid, state_B_HO_O, prompt, prompt_embedding=None, num_sampling_step=35,
                 stop_after_step=None, seed=0, use_cuda_graphs=False):
        args = (num_sampling_step, stop_after_step, use_cuda_graphs)
        if self.mode == "cl":
            text = {"A": prompt, "B": self.alts[0][1]}
            eo, sig, n_obs = self._ctx(input_vid, text[self.obs_src], 0, *args)
            if self.fut_src == self.obs_src:
                emb = eo
            else:
                ef, _, _ = self._ctx(input_vid, text[self.fut_src], 0, *args)
                emb = self._mix(eo, ef, n_obs)
            if self._logged < 2:
                print(f"[conflict-cl] obs={self.obs_src} fut={self.fut_src} B={text['B']!r} tokens={emb.shape[1]} n_obs={n_obs}", flush=True)
                self._logged += 1
            return self._dec(state_B_HO_O, emb, sig, use_cuda_graphs)

        # probe: drive with AA (the production call), decode every cell
        eA, sig, n_obs = self._ctx(input_vid, prompt, 0, *args)
        eA2, sig2, _ = self._ctx(input_vid, prompt, 1, *args)
        out = self._dec(state_B_HO_O, eA, sig, use_cuda_graphs)
        aA = out[0].float().cpu().numpy()
        aA2 = self._dec(state_B_HO_O, eA2, sig2, use_cuda_graphs)[0].float().cpu().numpy()
        rec = {"n_tokens": int(eA.shape[1]), "n_obs": int(n_obs), "sigma": float(sig.flatten()[0]),
               "action_AA": aA, "action_AA2": aA2, "d_AA2_AA": _dist(aA2, aA),
               "seed_ctx_obs": _rel(eA2[:, :n_obs], eA[:, :n_obs]), "seed_ctx_fut": _rel(eA2[:, n_obs:], eA[:, n_obs:]),
               "alts": {}}
        for name, text in self.alts:
            eB, sigB, _ = self._ctx(input_vid, text, 0, *args)
            bb = self._dec(state_B_HO_O, eB, sigB, use_cuda_graphs)[0].float().cpu().numpy()
            ab = self._dec(state_B_HO_O, self._mix(eA, eB, n_obs), sig, use_cuda_graphs)[0].float().cpu().numpy()
            ba = self._dec(state_B_HO_O, self._mix(eB, eA, n_obs), sig, use_cuda_graphs)[0].float().cpu().numpy()
            d_ab_aa, d_ab_bb = _dist(ab, aA), _dist(ab, bb)
            d_ba_bb, d_ba_aa = _dist(ba, bb), _dist(ba, aA)
            rec["alts"][name] = {
                "instruction": text, "action_BB": bb, "action_AB": ab, "action_BA": ba,
                "ctx_obs_AB": _rel(eB[:, :n_obs], eA[:, :n_obs]), "ctx_fut_AB": _rel(eB[:, n_obs:], eA[:, n_obs:]),
                "d_BB_AA": _dist(bb, aA), "d_AB_AA": d_ab_aa, "d_BA_BB": d_ba_bb,
                # 0 = the action stays with the observed-frame tokens' prompt,
                # 1 = it goes with the future tokens' prompt
                "follow_AB": d_ab_aa / (d_ab_aa + d_ab_bb + 1e-12),
                "follow_BA": d_ba_bb / (d_ba_bb + d_ba_aa + 1e-12),
            }
        self.records.append(rec)
        return out


class VAMConflictInference(VAMInference):
    def __init__(self, *args, **kwargs):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        task = os.environ["TASK"]
        mode = os.environ.get("CONFLICT_MODE", "probe")
        self.model = ConflictPipeline(
            self.model.video2world_pipeline, self.model.world2action_pipeline, task, mode,
            os.environ.get("CONFLICT_OBS", "A"), os.environ.get("CONFLICT_FUT", "A"),
            os.environ.get("CONFLICT_ALT", "mirror"),
        ).cuda()
        self._out = pathlib.Path(os.environ["CONFLICT_OUT"]) / task if mode == "probe" else None
        if self._out is not None:
            self._out.mkdir(parents=True, exist_ok=True)
        self._episode = -1

    def _flush(self):
        if self._out is not None and self._episode >= 0 and self.model.records:
            np.savez_compressed(self._out / f"episode_{self._episode:03d}.npz",
                                records=np.array(self.model.records, dtype=object))
        self.model.records = []

    def reset(self, task_description):
        # called once per episode by maniskill2_evaluator
        self._flush()
        self._episode += 1
        super().reset(task_description)

    def close(self):
        self._flush()
