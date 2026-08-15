"""Experiment 1, variant 1 for mimic-video: "remove the module".

Does NOT modify anything under model/cosmos_predict2/pipelines/ -- those are
shared, working code (main eval, Experiment 2's oracle mechanism). This is a
new, standalone pipeline class with the SAME __call__ signature as the real
Video2World2ActionPipeline (model/cosmos_predict2/pipelines/video2world2action.py),
so it's a drop-in replacement for VAMInference's self.model.

Mechanism: mimic's action decoder (world2action_pipeline) never looks at a
finished image -- it reads `crossattn_emb`, hidden activations from one
specific layer of the video-diffusion backbone (Cosmos-Predict2), passed in
as a plain tensor argument (video2world2action.py:60-67). Unlike F1's
KV-cache-based connection, there is no existing "skip this slot" behavior to
exploit here, since crossattn_emb is a required argument, not an optional
cache entry. So this ablation constructs it explicitly as zeros, matching
the "changed: tokens absent -> nothing there" row already documented in
../README.md's confound table for this variant.

To genuinely skip the (expensive) video-diffusion computation rather than
just discarding its output: crossattn_emb's shape and the paired
context_timesteps value are DATA-INDEPENDENT (fixed by `stop_after_step` and
the backbone's architecture, not by the input video's content) -- so both
are computed for real exactly ONCE (the first call), cached, and reused as
plain zeros/constants for every later call. Only the first sample in a
given eval run pays the cost of running the real video backbone; every
other sample genuinely never runs it.
"""

from __future__ import annotations

import torch
from torch import nn


class ZeroWorldModelPipeline(nn.Module):
    def __init__(self, video2world_pipeline, world2action_pipeline) -> None:
        super().__init__()
        self.video2world_pipeline = video2world_pipeline
        self.world2action_pipeline = world2action_pipeline
        self._cached_zero_crossattn: torch.Tensor | None = None
        self._cached_context_timesteps: torch.Tensor | None = None

    @torch.no_grad()
    def __call__(
        self,
        input_vid: torch.Tensor,
        state_B_HO_O: torch.Tensor,
        prompt: str,
        prompt_embedding: torch.Tensor | None = None,
        num_sampling_step: int = 35,
        stop_after_step: int | None = None,
        seed: int = 0,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor:
        if self._cached_zero_crossattn is None:
            # First call only: run the real backbone once, purely to learn
            # the (data-independent) shape/context-timestep it produces,
            # then discard the content. Every later call skips this
            # entirely.
            T = input_vid.shape[2]
            assert T in {1, 5}
            crossattn_emb, video_sigma = self.video2world_pipeline.generate_video(
                vid_input=input_vid,
                is_video_embedding=False,
                num_latent_conditional_frames=1 if T == 1 else 2,
                prompt=prompt,
                prompt_embedding=prompt_embedding,
                negative_prompt="",
                guidance=0.0,
                num_sampling_step=num_sampling_step,
                seed=seed,
                use_cuda_graphs=use_cuda_graphs,
                return_context_at_step=stop_after_step,
                hidden_state_layer_idx=self.world2action_pipeline.config.xattn_layer_idx,
            )
            hidden_state_shape = crossattn_emb.shape
            crossattn_emb = crossattn_emb.reshape(hidden_state_shape[0], -1, hidden_state_shape[-1])
            self._cached_zero_crossattn = torch.zeros_like(crossattn_emb)
            self._cached_context_timesteps = video_sigma.unsqueeze(1)

        B = state_B_HO_O.shape[0]
        crossattn_emb = self._cached_zero_crossattn.expand(B, -1, -1)
        context_timesteps_B_1 = self._cached_context_timesteps.expand(B, -1)

        return self.world2action_pipeline(
            state_B_HO_O=state_B_HO_O,
            crossattn_emb=crossattn_emb,
            context_timesteps_B_1=context_timesteps_B_1,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
