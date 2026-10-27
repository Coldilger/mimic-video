"""Ablation sanity check for Experiment 1 (advisor review, 2026-10-27).

Drop-in replacement for VAMInference.model (same __call__ signature as
Video2World2ActionPipeline). On every replan it:

1. computes the REAL crossattn_emb exactly as the baseline pipeline does and
   returns the real action, so the robot drives a normal baseline rollout;
2. re-runs the action decoder on the SAME state and seed under several
   context conditions, so every condition is compared on identical inputs;
3. records, per condition: raw and post-ctx_norm statistics of the context,
   the cross-attention entropy of every decoder block, and the decoded
   action chunk.

Question it answers: when Experiment 1's zero/shuffle ablations give 0%, is
that because the decoder lost world-model *information*, or because it was
pushed off its activation distribution (Heimersheim & Nanda 2024)? Mean-
ablation and a temporally coherent shuffle are the controls that separate
the two. Nothing under model/ or SimplerEnv/ is modified.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

CONDITIONS = ("real", "zero", "mean", "noise", "stale", "shuffle_perstep", "shuffle_coherent")


class _CrossAttnEntropyHook:
    """Recomputes cross-attention weights from each block's own q/k to get
    their entropy (the fused attention kernel does not expose weights)."""

    def __init__(self, blocks):
        self.entropies: list[float] = []
        self.enabled = False
        self._handles = [b.cross_attn.register_forward_pre_hook(self._hook, with_kwargs=True) for b in blocks]

    def _hook(self, module, args, kwargs):
        if not self.enabled:
            return
        x = args[0] if len(args) > 0 else kwargs["x"]
        context = args[1] if len(args) > 1 else kwargs.get("context")
        q, k, _ = module.compute_qkv(x, context)  # [B, S, H, D], [B, N, H, D]
        logits = torch.einsum("bshd,bnhd->bhsn", q.float(), k.float()) / math.sqrt(module.head_dim)
        p = logits.softmax(dim=-1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1)  # [B, H, S]
        self.entropies.append(float(ent.mean() / math.log(k.shape[1])))  # 1.0 == uniform

    def remove(self):
        for h in self._handles:
            h.remove()


def _ctx_stats(raw: torch.Tensor, ctx_norm: nn.Module) -> dict:
    """raw: [1, N, D]. Stats before and after the decoder's own ctx_norm."""
    r = raw.float()
    normed = ctx_norm(raw).float()[0]  # [N, D]
    idx = torch.linspace(0, normed.shape[0] - 1, min(64, normed.shape[0])).long()
    sub = torch.nn.functional.normalize(normed[idx], dim=-1)
    cos = sub @ sub.T
    off_diag = cos[~torch.eye(len(idx), dtype=torch.bool, device=cos.device)]
    return {
        "raw_mean": float(r.mean()),
        "raw_std": float(r.std()),
        "raw_token_norm": float(r[0].norm(dim=-1).mean()),
        "raw_nan": int(torch.isnan(r).sum()),
        "normed_token_std": float(normed.std(dim=-1).mean()),
        "normed_token_norm": float(normed.norm(dim=-1).mean()),
        "normed_mean_pairwise_cos": float(off_diag.mean()),
    }


class SanityPipeline(nn.Module):
    def __init__(self, base_pipeline, frame_source) -> None:
        super().__init__()
        self.video2world_pipeline = base_pipeline.video2world_pipeline
        self.world2action_pipeline = base_pipeline.world2action_pipeline
        self.frame_source = frame_source  # provides shuffled clips, see vam_sanity_inference.py
        dit = self.world2action_pipeline.dit
        self._ctx_norm = dit.ctx_norm
        self._hook = _CrossAttnEntropyHook(dit.blocks)
        self._noise_gen = torch.Generator(device="cuda").manual_seed(0)
        # mean-ablation pool: sum of real contexts from PREVIOUS episodes only
        self._mean_sum: torch.Tensor | None = None
        self._mean_count = 0
        self._episode_sum: torch.Tensor | None = None
        self._episode_count = 0
        self._stale: torch.Tensor | None = None
        self.records: list[dict] = []

    def real_context_pool(self) -> tuple[torch.Tensor | None, int]:
        """Sum and count of real contexts over ALL finished + current episodes."""
        total, n = self._mean_sum, self._mean_count
        if self._episode_sum is not None:
            total = self._episode_sum if total is None else total + self._episode_sum
            n += self._episode_count
        return total, n

    # --- episode bookkeeping, called by VAMSanityInference.reset() ---
    def start_episode(self) -> None:
        if self._episode_sum is not None:
            self._mean_sum = self._episode_sum if self._mean_sum is None else self._mean_sum + self._episode_sum
            self._mean_count += self._episode_count
        self._episode_sum, self._episode_count = None, 0
        self._stale = None

    def _context(self, input_vid, prompt, prompt_embedding, num_sampling_step, stop_after_step, seed, use_cuda_graphs):
        T = input_vid.shape[2]
        assert T in {1, 5}
        emb, sigma = self.video2world_pipeline.generate_video(
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
        return emb.reshape(emb.shape[0], -1, emb.shape[-1]), sigma.unsqueeze(1)

    def _decode(self, state, emb, sigma, seed, use_cuda_graphs, record_attention):
        self._hook.entropies = []
        self._hook.enabled = record_attention
        out = self.world2action_pipeline(
            state_B_HO_O=state,
            crossattn_emb=emb,
            context_timesteps_B_1=sigma,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
        self._hook.enabled = False
        return out, list(self._hook.entropies)

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
        gen_args = (prompt, prompt_embedding, num_sampling_step, stop_after_step, seed, use_cuda_graphs)
        real_emb, sigma = self._context(input_vid, *gen_args)
        # The action that drives the robot: identical to the baseline pipeline's call.
        driving_action, _ = self._decode(state_B_HO_O, real_emb, sigma, seed, use_cuda_graphs, False)

        if self._stale is None:
            self._stale = real_emb.clone()
        contexts = {
            "real": real_emb,
            "zero": torch.zeros_like(real_emb),
            "noise": real_emb.float().mean(-1, keepdim=True)
            + real_emb.float().std(-1, keepdim=True)
            * torch.randn(real_emb.shape, generator=self._noise_gen, device=real_emb.device),
            "stale": self._stale,
        }
        contexts["noise"] = contexts["noise"].to(real_emb.dtype)
        if self._mean_count > 0:
            contexts["mean"] = (self._mean_sum / self._mean_count).to(real_emb.dtype)
        perstep_vid = self.frame_source.perstep_clip(input_vid)
        coherent_vid = self.frame_source.coherent_clip(input_vid)
        contexts["shuffle_perstep"], _ = self._context(perstep_vid, *gen_args)
        contexts["shuffle_coherent"], _ = self._context(coherent_vid, *gen_args)

        rec = {
            "global_step": self.frame_source.global_step,
            "sigma": float(sigma.float().mean()),
            "driving_action": driving_action[0].float().cpu().numpy(),
            "conditions": {},
        }
        real_action = None
        for name in CONDITIONS:
            if name not in contexts:
                continue
            emb = contexts[name]
            # Hooks need eager execution, so diagnostics never use CUDA graphs.
            action, ent = self._decode(state_B_HO_O, emb, sigma, seed, False, True)
            a = action[0].float().cpu().numpy()
            if name == "real":
                real_action = a
            rec["conditions"][name] = {
                "ctx": _ctx_stats(emb, self._ctx_norm),
                "attn_entropy_per_block": ent,
                "action": a,
                "action_nan": int(np.isnan(a).sum()),
            }
        for c in rec["conditions"].values():
            c["l2_to_real"] = float(np.linalg.norm(c["action"] - real_action))
        self.records.append(rec)

        f = real_emb.float()
        self._episode_sum = f if self._episode_sum is None else self._episode_sum + f
        self._episode_count += 1
        return driving_action
