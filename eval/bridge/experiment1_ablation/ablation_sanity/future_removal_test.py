"""Does mimic-video's action decoder use the FUTURE part of its context, or
only the observed-frame part? (advisor review, 2026-11-02)

crossattn_emb is layer-20 activations of the video DiT, shape
[B, T=16, H=30, W=40, D] flattened T-major to [B, 19200, D]: latent frames
0-1 (tokens 0-2399) encode the 5 observed frames, frames 2-15 (tokens
2400-19199) are the future, which at stop=1 (sigma ~68 of 80) is still almost
pure, seed-fixed noise. Exp1's zero/shuffle ablations replaced all of it, i.e.
the decoder's only view of the scene (it has no image or text input of its
own), so their 0% cannot separate "uses the forecast" from "uses perception".

Conditions, decoded on the same state and seed:
  full            19200 tokens, as in the policy
  obs_only        the video DiT run with state_t=2: only the 2 observed latent
                  frames exist, no future is generated, so nothing mixes into
                  them (2400 tokens)
  cur_from_full   the first 2400 tokens of `full` (future dropped AFTER mixing)
  fut_from_full   the last 16800 tokens of `full` (observed part dropped)
  obs_only_pad    obs_only + the future tokens' mean over the OTHER clips
                  (19200 tokens: controls for the token-count change)
  full_fut_mean   full's own observed tokens + the other clips' mean future

The decoder's cross-attention has no positional encoding (RoPE is applied
only in self-attention, world2action_dit.py compute_qkv), so dropping tokens
shrinks the attended set but shifts no positions.

Also measured: how much the observed tokens change when no future positions
exist (mixing), and the decoder's attention mass on observed vs future tokens.
Inputs: real Bridge clips (5 consecutive frames) with their own task string,
a fixed plausible robot state (as in mask_attention_test.py). No simulator.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, str(HERE))

from simpler_env.policies.vam.video_action_model import VAMInference, load_video2world2action_pipeline  # noqa: E402
from vam_sanity_inference import _fast_path_is_file, _to_uint8_hwc  # noqa: E402

CK = "/home/3295540/mimic-video/model/checkpoints"
EXP = "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
VIDEO = f"{CK}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION = f"{CK}/action_decoder/{EXP}_iter_000014112.pt"
STATS = pathlib.Path(f"{CK}/dataset_statistics/bridge.json")
DATA = "/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"
CAM = "observation.images.image_0"
N_EPISODES = 8
N_OBS = 2400  # 2 observed latent frames x 30 x 40 tokens


def load_clips():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    rng = np.random.default_rng(7)
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=DATA)
        ids = rng.choice(meta.total_episodes, size=N_EPISODES, replace=False).tolist()
        ds = LeRobotDataset(repo_id="bridge_orig_lerobot", root=DATA, episodes=ids, video_backend="pyav")
    clips = []
    for pos in range(len(ids)):
        lo, hi = int(ds.episode_data_index["from"][pos]), int(ds.episode_data_index["to"][pos])
        for t in sorted({lo + 4, (lo + hi) // 2}):
            if t >= hi:
                continue
            frames = [_to_uint8_hwc(ds[i][CAM]) for i in range(t - 4, t + 1)]
            clip = np.concatenate([VAMInference._process_image(None, f) for f in frames], axis=1)
            clips.append({"episode": int(ids[pos]), "t": t - lo, "task": ds[t]["task"],
                          "vid": torch.from_numpy(clip[None]).cuda().bfloat16()})
    return clips


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    pipe = load_video2world2action_pipeline(EXP, VIDEO, ACTION, STATS, is_hil=False)
    v2w, w2a = pipe.video2world_pipeline, pipe.world2action_pipeline
    dit = w2a.dit
    state = torch.tensor([[[0.3, 0.0, 0.1, 1, 0, 0, 0, 1, 0, 1.0]]]).cuda().bfloat16()
    full_state_t = v2w.config.state_t

    orig_encode = v2w.encode

    def context(vid, task, state_t):
        # encode() always zero-pads a 5-frame clip to 16 latent frames
        # (video2world.py encode); with state_t=2 keep only the 2 real ones.
        v2w.config.state_t = state_t
        if state_t != full_state_t:
            v2w.encode = lambda s: orig_encode(s)[:, :, :state_t]
        try:
            emb, sigma = v2w.generate_video(
                vid_input=vid, is_video_embedding=False, num_latent_conditional_frames=2,
                prompt=task, prompt_embedding=None, negative_prompt="", guidance=0.0,
                num_sampling_step=35, seed=0, use_cuda_graphs=False,
                return_context_at_step=1, hidden_state_layer_idx=w2a.config.xattn_layer_idx,
            )
        finally:
            v2w.config.state_t = full_state_t
            v2w.encode = orig_encode
        return emb, sigma

    # attention mass of action queries on observed (first N_OBS) vs future keys
    attn = {"on": False, "obs_mass": []}

    def hook(module, args, kwargs):
        if not attn["on"]:
            return
        x = args[0] if args else kwargs["x"]
        c = args[1] if len(args) > 1 else kwargs["context"]
        q, k, _ = module.compute_qkv(x, c)
        w = (torch.einsum("bshd,bnhd->bhsn", q.float(), k.float()) / math.sqrt(module.head_dim)).softmax(-1)
        attn["obs_mass"].append(float(w[..., :N_OBS].sum(-1).mean()))

    handles = [b.cross_attn.register_forward_pre_hook(hook, with_kwargs=True) for b in dit.blocks]

    def decode(ctx, sigma, record=False):
        attn["on"], attn["obs_mass"] = record, []
        a = w2a(state_B_HO_O=state, crossattn_emb=ctx, context_timesteps_B_1=sigma.unsqueeze(1),
                seed=0, use_cuda_graphs=False)
        attn["on"] = False
        return a[0].float().cpu().numpy(), list(attn["obs_mass"])

    clips = load_clips()
    print(f"clips: {len(clips)}", flush=True)
    for c in clips:
        e_full, sigma = context(c["vid"], c["task"], full_state_t)
        e_obs, sigma_obs = context(c["vid"], c["task"], 2)
        c["shape_full"], c["shape_obs"] = list(e_full.shape), list(e_obs.shape)
        assert e_full.shape[1] == full_state_t and e_obs.shape[1] == 2, (e_full.shape, e_obs.shape)
        c["full"] = e_full.reshape(1, -1, e_full.shape[-1])
        c["obs"] = e_obs.reshape(1, -1, e_obs.shape[-1])
        c["sigma"], c["sigma_obs"] = sigma, sigma_obs
    fut_sum = sum(c["full"][:, N_OBS:].float() for c in clips)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))

    def act_stats(a):
        rot = (np.abs(np.linalg.norm(a[:, 3:6], axis=-1) - 1).mean()
               + np.abs(np.linalg.norm(a[:, 6:9], axis=-1) - 1).mean()) / 2
        return {"translation": float(np.linalg.norm(a[:, :3], axis=-1).mean()),
                "rot6_norm_err": float(rot), "gripper": float(a[:, 9].mean()), "nan": int(np.isnan(a).sum())}

    results = []
    for c in clips:
        full, obs, sigma = c["full"], c["obs"], c["sigma"]
        other_fut = ((fut_sum - full[:, N_OBS:].float()) / (len(clips) - 1)).to(full.dtype)
        cur_f, cur_o = full[0, :N_OBS].float(), obs[0].float()
        mixing = {
            "rel_l2_obs_tokens": float((cur_o - cur_f).norm() / cur_f.norm()),
            "cos_obs_tokens": float(torch.nn.functional.cosine_similarity(cur_o, cur_f, dim=-1).mean()),
            "sigma_full": float(sigma.float().mean()), "sigma_obs_only": float(c["sigma_obs"].float().mean()),
        }
        conds = {
            "full": (full, True),
            "full_repeat": (full, False),
            "obs_only": (obs, True),
            "cur_from_full": (full[:, :N_OBS], True),
            "fut_from_full": (full[:, N_OBS:], False),
            "obs_only_pad": (torch.cat([obs, other_fut], 1), True),
            "full_fut_mean": (torch.cat([full[:, :N_OBS], other_fut], 1), True),
        }
        rec = {"episode": c["episode"], "t": c["t"], "task": c["task"], "shape_full": c["shape_full"],
               "shape_obs": c["shape_obs"], "mixing": mixing, "conditions": {}}
        ref = None
        for name, (ctx, rec_attn) in conds.items():
            a, mass = decode(ctx, sigma, rec_attn and ctx.shape[1] > N_OBS)
            if name == "full":
                ref = a
            rec["conditions"][name] = {"n_tokens": int(ctx.shape[1]), "rel_l2_to_full": rel(a, ref),
                                       "obs_attn_mass": float(np.mean(mass)) if mass else None,
                                       "action": a.tolist(), **act_stats(a)}
        results.append(rec)
        print(f"ep {c['episode']} t={c['t']} '{c['task'][:40]}': mixing rel {mixing['rel_l2_obs_tokens']:.3f} | "
              + " | ".join(f"{k} {v['rel_l2_to_full']:.3f}" for k, v in rec["conditions"].items()), flush=True)
    for h in handles:
        h.remove()

    print("\nshapes: full", results[0]["shape_full"], "obs_only", results[0]["shape_obs"])
    m = np.array([r["mixing"]["rel_l2_obs_tokens"] for r in results])
    cs = np.array([r["mixing"]["cos_obs_tokens"] for r in results])
    print(f"observed tokens, full vs obs_only: rel L2 {m.mean():.3f} ± {m.std():.3f}, cosine {cs.mean():.3f}")
    print(f"sigma full {results[0]['mixing']['sigma_full']:.3f}, obs_only {results[0]['mixing']['sigma_obs_only']:.3f}")
    print("| condition | tokens | rel L2 to full | obs attn mass | translation | rot6 err | gripper | NaN |")
    print("|---|---|---|---|---|---|---|---|")
    for name in results[0]["conditions"]:
        rows = [r["conditions"][name] for r in results]
        f = lambda k: np.array([x[k] for x in rows if x[k] is not None], dtype=float)
        mass = f("obs_attn_mass")
        ms = f"{mass.mean():.3f}" if mass.size else "-"
        print(f"| {name} | {rows[0]['n_tokens']} | {f('rel_l2_to_full').mean():.3f} ± {f('rel_l2_to_full').std():.3f} "
              f"| {ms} | {f('translation').mean():.4f} | {f('rot6_norm_err').mean():.4f} | {f('gripper').mean():.3f} "
              f"| {int(f('nan').sum())} |")
    out = HERE / "future_removal_test.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
