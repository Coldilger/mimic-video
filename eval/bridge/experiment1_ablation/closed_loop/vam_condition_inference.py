"""Closed-loop Experiment 1 conditions added after the ablation sanity check
(ablation_sanity/): the controls that separate "world-model information was
removed" from "the decoder was pushed off its activation distribution".

- mean:     crossattn_emb replaced by the MEAN real context of the SAME task,
            pooled over that task's 24 baseline rollouts in the sanity check
            (task-conditional mean: keeps "which task" information, removes
            everything specific to the current state -- matching the
            diagnostic's own mean condition). Same mechanics
            as ZeroWorldModelPipeline (backbone runs once for sigma, then is
            skipped) -- only the constant differs: an on-distribution average
            instead of zeros. Standard mean-ablation (Heimersheim & Nanda 2024).
- coherent: the world model sees a temporally COHERENT clip from one other
            Bridge episode (same 64-episode pool and per-episode choice as the
            sanity check), frame t at control step t -- wrong content, intact
            temporal structure. Contrast with Exp1's per-step shuffle, which
            mixes up to five unrelated episodes into each 5-frame clip.
- stale:    the REAL context of this episode's first replan, reused for every
            later replan -- the right episode and scene, but a forecast made
            at the start and never updated (bridges to Experiment 7,
            horizon-shifted foresight).
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # zero_world_model_pipeline
sys.path.insert(0, str(HERE.parent / "ablation_sanity"))  # _FrameSource
sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")

from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402

from vam_sanity_inference import _FrameSource  # noqa: E402
from zero_world_model_pipeline import ZeroWorldModelPipeline  # noqa: E402

POOL_GLOB = HERE.parent / "ablation_sanity" / "records"


def load_mean_context() -> torch.Tensor:
    f = POOL_GLOB / os.environ["TASK"] / "real_context_pool.pt"
    assert f.exists(), f"{f} missing -- run the sanity check for this task first"
    pool = torch.load(f)
    print(f"mean context for {os.environ['TASK']}: {pool['count']} replans")
    return pool["sum"] / pool["count"]


class MeanWorldModelPipeline(ZeroWorldModelPipeline):
    def __init__(self, video2world_pipeline, world2action_pipeline, mean_ctx: torch.Tensor) -> None:
        super().__init__(video2world_pipeline, world2action_pipeline)
        self._mean_ctx = mean_ctx

    def __call__(self, *args, **kwargs):
        first = self._cached_zero_crossattn is None
        out = super().__call__(*args, **kwargs)  # first call learns shape/sigma
        if first:
            assert self._mean_ctx.shape == self._cached_zero_crossattn.shape, (
                self._mean_ctx.shape, self._cached_zero_crossattn.shape)
            # from now on the "zero" cache holds the mean; the very first
            # decision also gets redone with it so no step sees zeros
            self._cached_zero_crossattn = self._mean_ctx.to(self._cached_zero_crossattn)
            out = super().__call__(*args, **kwargs)
        return out


class VAMMeanInference(VAMInference):
    def __init__(self, *args, **kwargs):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        self.model = MeanWorldModelPipeline(
            self.model.video2world_pipeline, self.model.world2action_pipeline, load_mean_context()
        ).cuda()


class VAMCoherentShuffledInference(VAMInference):
    def __init__(self, *args, **kwargs):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        self._frames = _FrameSource(self._process_image, self._image_horizon)
        self._episode_counter = -1
        self._t = 0

    def reset(self, task_description):
        super().reset(task_description)
        self._episode_counter += 1
        self._frames.start_episode(self._episode_counter)
        self._t = 0

    def _add_image_to_history(self, image) -> None:
        # `image` (the real frame) is ignored on purpose
        ep = self._frames.episodes[self._frames._coherent_ep]
        super()._add_image_to_history(self._process_image(ep[min(self._t, len(ep) - 1)]))
        self._t += 1


class StaleWorldModelPipeline(ZeroWorldModelPipeline):
    """Real context computed once per episode (first replan after reset), then
    reused; reset_episode() clears it."""

    def reset_episode(self) -> None:
        self._cached_zero_crossattn = None

    def __call__(self, input_vid, state_B_HO_O, prompt, prompt_embedding=None, num_sampling_step=35,
                 stop_after_step=None, seed=0, use_cuda_graphs=False):
        if self._cached_zero_crossattn is None:
            T = input_vid.shape[2]
            emb, sigma = self.video2world_pipeline.generate_video(
                vid_input=input_vid, is_video_embedding=False,
                num_latent_conditional_frames=1 if T == 1 else 2, prompt=prompt,
                prompt_embedding=prompt_embedding, negative_prompt="", guidance=0.0,
                num_sampling_step=num_sampling_step, seed=seed, use_cuda_graphs=use_cuda_graphs,
                return_context_at_step=stop_after_step,
                hidden_state_layer_idx=self.world2action_pipeline.config.xattn_layer_idx,
            )
            self._cached_zero_crossattn = emb.reshape(emb.shape[0], -1, emb.shape[-1])  # holds the stale REAL context
            self._cached_context_timesteps = sigma.unsqueeze(1)
        B = state_B_HO_O.shape[0]
        return self.world2action_pipeline(
            state_B_HO_O=state_B_HO_O,
            crossattn_emb=self._cached_zero_crossattn.expand(B, -1, -1),
            context_timesteps_B_1=self._cached_context_timesteps.expand(B, -1),
            seed=seed, use_cuda_graphs=use_cuda_graphs,
        )


class VAMStaleInference(VAMInference):
    def __init__(self, *args, **kwargs):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        self.model = StaleWorldModelPipeline(
            self.model.video2world_pipeline, self.model.world2action_pipeline
        ).cuda()

    def reset(self, task_description):
        super().reset(task_description)
        self.model.reset_episode()
