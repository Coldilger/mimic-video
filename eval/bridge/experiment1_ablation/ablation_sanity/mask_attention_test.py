"""Empirical mask / attention test for the action decoder's cross-attention
(advisor review, 2026-10-28). The sanity check (sanity_pipeline.py) relied on
two things it had only established by reading the code:

  (a) the cross-attention has no mask: every context token can be attended
      (world2action_dit.py builds DotProductAttention with
      attn_mask_type="no_mask");
  (b) the fused kernel computes plain softmax(q k^T / sqrt(d)) v, so the
      attention entropy the sanity check recomputes from q and k is the
      attention the kernel actually uses.

This script tests both on the real model, block by block, for the real and
the zero context:

  1. kernel vs manual: the module's own output is compared with a manual,
     mask-free softmax attention computed from the same q, k, v
     (Attention.compute_qkv) -> confirms (b), and that no hidden mask exists;
  2. per-token perturbation: one context token at a time (first, quartiles,
     last) is replaced by noise and the REAL module is re-run. A masked
     position cannot change the output, so a non-zero change for every probed
     token confirms (a) on the kernel itself;
  3. test of the test: the same perturbation through a manual attention with
     one position deliberately masked must give exactly zero change there.

Input: 5 consecutive real Bridge frames (same pool as the sanity check), a
fixed plausible robot state. No simulator needed.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch
from einops import rearrange

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, str(HERE))

from simpler_env.policies.vam.video_action_model import VAMInference, load_video2world2action_pipeline  # noqa: E402
from vam_sanity_inference import _FrameSource  # noqa: E402

CK = "/home/3295540/mimic-video/model/checkpoints"
EXP = "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
VIDEO = f"{CK}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION = f"{CK}/action_decoder/{EXP}_iter_000014112.pt"
STATS = pathlib.Path(f"{CK}/dataset_statistics/bridge.json")


def manual_attention(module, x, context, mask_pos: int | None = None):
    q, k, v = module.compute_qkv(x, context)  # [B, S, H, D], [B, N, H, D]
    logits = torch.einsum("bshd,bnhd->bhsn", q.float(), k.float()) / math.sqrt(module.head_dim)
    if mask_pos is not None:
        logits[..., mask_pos] = float("-inf")
    w = logits.softmax(-1)
    o = torch.einsum("bhsn,bnhd->bshd", w, v.float()).to(v.dtype)
    return module.output_proj(rearrange(o, "b s h d -> b s (h d)")), w


def rel(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    pipe = load_video2world2action_pipeline(EXP, VIDEO, ACTION, STATS, is_hil=False)
    v2w, w2a = pipe.video2world_pipeline, pipe.world2action_pipeline
    dit = w2a.dit

    frames = _FrameSource(lambda f: VAMInference._process_image(None, f), horizon=5)
    ep = frames.episodes[0]
    clip = np.concatenate([VAMInference._process_image(None, ep[i]) for i in range(5)], axis=1)
    input_vid = torch.from_numpy(clip[None]).cuda().bfloat16()
    # plausible proprio (pos, rot6 = identity, gripper open); normalized inside the pipeline
    state = torch.tensor([[[0.3, 0.0, 0.1, 1, 0, 0, 0, 1, 0, 1.0]]]).cuda().bfloat16()

    emb, sigma = v2w.generate_video(
        vid_input=input_vid, is_video_embedding=False, num_latent_conditional_frames=2,
        prompt="put the carrot on the plate", prompt_embedding=None, negative_prompt="",
        guidance=0.0, num_sampling_step=35, seed=0, use_cuda_graphs=False,
        return_context_at_step=1, hidden_state_layer_idx=w2a.config.xattn_layer_idx,
    )
    real = emb.reshape(emb.shape[0], -1, emb.shape[-1])
    contexts = {"real": real, "zero": torch.zeros_like(real)}

    report = {"n_context_tokens": int(real.shape[1]), "n_blocks": len(dit.blocks), "conditions": {}}
    for name, ctx in contexts.items():
        captured = {}

        def make_hook(i):
            def hook(module, args, kwargs, output):
                if i not in captured:  # first denoising step only
                    x = args[0]
                    c = args[1] if len(args) > 1 else kwargs["context"]
                    captured[i] = (module, x.clone(), c.clone(), output.clone())
            return hook

        handles = [b.cross_attn.register_forward_hook(make_hook(i), with_kwargs=True) for i, b in enumerate(dit.blocks)]
        w2a(state_B_HO_O=state, crossattn_emb=ctx, context_timesteps_B_1=sigma.unsqueeze(1), seed=0, use_cuda_graphs=False)
        for h in handles:
            h.remove()

        blocks = []
        for i in sorted(captured):
            module, x, c, out = captured[i]
            manual, w = manual_attention(module, x, c)
            n = c.shape[1]
            probe = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
            base = module(x, c)
            pert = {}
            for j in probe:
                c2 = c.clone()
                c2[:, j] = torch.randn_like(c2[:, j]) * c.float().std().clamp_min(1.0).to(c.dtype)
                pert[j] = rel(module(x, c2), base)
            # test of the test: position probe[2] masked in a manual attention
            mpos = probe[2]
            c3 = c.clone()
            c3[:, mpos] = torch.randn_like(c3[:, mpos]) * 5
            masked_a, _ = manual_attention(module, x, c, mask_pos=mpos)
            masked_b, _ = manual_attention(module, x, c3, mask_pos=mpos)
            blocks.append({
                "block": i,
                "kernel_vs_manual_rel_err": rel(out, manual),
                "min_attention_weight": float(w.min()),
                "max_attention_weight": float(w.max()),
                "perturbation_rel_change": pert,
                "masked_control_rel_change": rel(masked_b, masked_a),
            })
        report["conditions"][name] = blocks
        errs = [b["kernel_vs_manual_rel_err"] for b in blocks]
        mins = [min(b["perturbation_rel_change"].values()) for b in blocks]
        ctrl = [b["masked_control_rel_change"] for b in blocks]
        wmin = [b["min_attention_weight"] for b in blocks]
        print(f"[{name}] kernel vs manual rel err: max {max(errs):.2e} | "
              f"smallest per-token perturbation effect: {min(mins):.2e} | "
              f"min attention weight: {min(wmin):.2e} | masked control: max {max(ctrl):.2e}", flush=True)

    out = HERE / "mask_attention_test.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
