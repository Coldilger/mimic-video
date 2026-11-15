"""Experiment 9 closed-loop conditions beyond "baseline at depth k".

seedavg: at every decision the video model is run with N different seeds
    (VIDEO_SEEDS, e.g. "0:1:2:3" -- colons, since sbatch --export splits on commas); the decoder (its own seed fixed at 0, as in
    the open-loop seed test) produces one action chunk per imagined future,
    and the robot executes their mean. Positions and gripper are averaged
    directly; the 6D rotation rows are averaged and re-orthonormalised by
    VAMInference._pose_from_lowdim. With one seed in VIDEO_SEEDS this is the
    baseline policy run with that video seed (seed 0 == production).

    Purpose: separate two explanations of the success drop at deep k
    (README, "Closed loop", point 4). If the drop comes from committing to
    one specific wrong future, averaging over futures should recover
    success; if it comes from the decoder never having seen a model-made
    low-noise future in training, every seed's context is equally unfamiliar
    and averaging should not help. Caveat: a mean of actions that follow
    different plans need not be a valid plan, so "no recovery" is weaker
    evidence than "recovery".

Nothing under model/ or SimplerEnv/ is modified.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")

from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402


class SeedAveragePipeline(nn.Module):
    """Drop-in for Video2World2ActionPipeline (same __call__ signature)."""

    def __init__(self, video2world_pipeline, world2action_pipeline, video_seeds: list[int]) -> None:
        super().__init__()
        self.video2world_pipeline = video2world_pipeline
        self.world2action_pipeline = world2action_pipeline
        self.video_seeds = video_seeds
        self._n_calls = 0

    @torch.no_grad()
    def __call__(self, input_vid, state_B_HO_O, prompt, prompt_embedding=None, num_sampling_step=35,
                 stop_after_step=None, seed=0, use_cuda_graphs=False):
        T = input_vid.shape[2]
        assert T in {1, 5}
        actions = []
        for s in self.video_seeds:
            emb, sigma = self.video2world_pipeline.generate_video(
                vid_input=input_vid, is_video_embedding=False,
                num_latent_conditional_frames=1 if T == 1 else 2, prompt=prompt,
                prompt_embedding=prompt_embedding, negative_prompt="", guidance=0.0,
                num_sampling_step=num_sampling_step, seed=s, use_cuda_graphs=use_cuda_graphs,
                return_context_at_step=stop_after_step,
                hidden_state_layer_idx=self.world2action_pipeline.config.xattn_layer_idx,
            )
            actions.append(self.world2action_pipeline(
                state_B_HO_O=state_B_HO_O,
                crossattn_emb=emb.reshape(emb.shape[0], -1, emb.shape[-1]),
                context_timesteps_B_1=sigma.unsqueeze(1),
                seed=0, use_cuda_graphs=use_cuda_graphs,
            ).float())
        stack = torch.stack(actions)  # (N, B, H, 10)
        if self._n_calls < 3:  # log the first decisions: chunk shape and how far the seeds disagree
            spread = (stack - stack.mean(0)).norm(dim=-1).mean().item() if len(actions) > 1 else 0.0
            print(f"[seedavg] seeds={self.video_seeds} chunk={tuple(stack.shape[1:])} "
                  f"mean |a_s - mean| = {spread:.4f}, mean |a| = {stack.norm(dim=-1).mean().item():.4f}")
        self._n_calls += 1
        return stack.mean(0)


class VAMSeedAverageInference(VAMInference):
    def __init__(self, *args, **kwargs):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        seeds = [int(s) for s in os.environ["VIDEO_SEEDS"].replace(":", ",").split(",")]
        self.model = SeedAveragePipeline(
            self.model.video2world_pipeline, self.model.world2action_pipeline, seeds
        ).cuda()
