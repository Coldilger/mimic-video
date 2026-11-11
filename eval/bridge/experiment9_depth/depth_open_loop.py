"""Experiment 9 (advisor point 4), open-loop part: how far into its denoising
does mimic-video's video model have to go before the decoder reads an actual
forecast, and does the decoder follow that forecast?

The policy reads the video model's layer-20 activations after step k of 35
denoising steps (`stop=k`; deployed: k=1, sigma 68 of 80). The decoder was
trained on activations of the REAL future video noised to a random sigma
(world2action_model.py get_crossattn_emb), so at small sigma it expects a
real future in the future tokens; at inference it gets the model's own
prediction.

Per real Bridge clip (5 observed frames + the 56 real future frames at 5 Hz,
as in training, policy_io/bridge.yaml), and per k in K:

1. forecast quality: the video model's own clean-video estimate at step k
   (x0_pred, latent space) vs the real future's latent, next to the error of
   a "nothing moves" future (the last observed frame repeated). ratio < 1:
   the forecast is closer to the real future than a static scene is.
2. seed test: video seeds 0-3, decoder seed fixed at 0. Same frames and
   instruction, so perception is identical; only the imagined future
   differs. Recorded: spread of the imagined futures and of the actions
   across seeds; and a patch test: seed-0 context with the future tokens
   (latent frames 2-15) of seed s. follow = d(patch, a0) / (d(patch, a0) +
   d(patch, a_s)): 0 = the action stays with seed 0, 1 = it follows the
   donor's future. The reverse patch (observed tokens of s) is the control.
3. oracle at the same sigma: the context built as in training from the real
   future video noised to sigma_k (video2world2action_gtvid.py), 2 noise
   seeds; action change vs the model's own forecast at the same k.
4. latency of the production call per k (one clip, median of 3).

Actions are compared between conditions only, never to the dataset's actions
(no held-out split; offline action error is memorization-confounded).
Fixed plausible robot state as in the Experiment 1 sanity tests.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch
from einops import rearrange

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/experiment1_ablation/ablation_sanity")

from imaginaire.utils import misc  # noqa: E402
from simpler_env.policies.vam.video_action_model import VAMInference, load_video2world2action_pipeline  # noqa: E402
from vam_sanity_inference import _fast_path_is_file, _to_uint8_hwc  # noqa: E402

CK = "/home/3295540/mimic-video/model/checkpoints"
EXP = "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
VIDEO = f"{CK}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION = f"{CK}/action_decoder/{EXP}_iter_000014112.pt"
STATS = pathlib.Path(f"{CK}/dataset_statistics/bridge.json")
DATA = "/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"
CAM = "observation.images.image_0"

N_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 8
K = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 4, 8, 16, 23, 30]
SEEDS = [0, 1, 2, 3]
ORACLE_SEEDS = [0, 1]
N_STEPS = 35
N_OBS_LAT = 2  # latent frames encoding the 5 observed frames
N_FUT_PIX = 56  # real future frames, 5 Hz, as in training


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
        cache = {}

        def frame(i):
            i = min(i, hi - 1)  # past the episode end: repeat the last frame (as training's clamping)
            if i not in cache:
                cache[i] = VAMInference._process_image(None, _to_uint8_hwc(ds[i][CAM]))
            return cache[i]

        for t in sorted({lo + 4, (lo + hi) // 2}):
            if t >= hi:
                continue
            obs = np.concatenate([frame(i) for i in range(t - 4, t + 1)], axis=1)
            fut = np.concatenate([frame(t + 1 + j) for j in range(N_FUT_PIX)], axis=1)
            to = lambda a: torch.from_numpy(a[None]).cuda().bfloat16()  # noqa: E731
            clips.append({"episode": int(ids[pos]), "t": t - lo, "len": hi - lo, "task": ds[t]["task"],
                          "obs": to(obs), "fut": to(fut),
                          "fut_real_frames": int(min(N_FUT_PIX, hi - 1 - t))})
    return clips


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    pipe = load_video2world2action_pipeline(EXP, VIDEO, ACTION, STATS, is_hil=False)
    v2w, w2a = pipe.video2world_pipeline, pipe.world2action_pipeline
    layer = w2a.config.xattn_layer_idx
    state = torch.tensor([[[0.3, 0.0, 0.1, 1, 0, 0, 0, 1, 0, 1.0]]]).cuda().bfloat16()
    sched = v2w.scheduler
    sched.set_timesteps(N_STEPS, device="cuda")
    sigmas = [float(s) for s in sched.sigmas]

    def contexts_along_trajectory(obs, task, seed, ks):
        """Same loop as Video2WorldPipeline.generate_video (guidance 0, as the
        policy), but returns (context, sigma, x0_pred) at every step in ks."""
        data_batch = v2w._get_data_batch_input(obs, False, task, None, "", num_latent_conditional_frames=N_OBS_LAT)
        v2w._normalize_video_databatch_inplace(data_batch)
        v2w._augment_image_dim_inplace(data_batch)
        x0_fn = v2w.get_x0_fn_from_batch(data_batch, 0.0, is_negative_prompt=True, use_cuda_graphs=False)
        _T, H, W = data_batch[v2w.input_video_key].shape[-3:]
        f = v2w.tokenizer.spatial_compression_factor
        shape = (1, v2w.config.state_ch, v2w.config.state_t, H // f, W // f)
        sample = (misc.arch_invariant_rand(shape, torch.float32, v2w.tensor_kwargs["device"], seed)
                  * v2w.scheduler.config.sigma_max).to(torch.float32)
        sched.set_timesteps(N_STEPS, device=sample.device)
        out, x0_prev = {}, None
        for i in range(max(ks) + 1):
            sigma_in = sched.sigmas[i].to(sample.device, dtype=torch.float32).repeat(1)
            if i in ks:
                # a finite layer index truncates the forward (no x0); inf keeps the
                # full pass and all hidden states (as generate_video's return_all_context)
                x0_pred, hs = x0_fn(sample, sigma_in, return_only_hidden_states_up_to=float("inf"))
                emb = hs[layer]
                out[i] = (emb.reshape(1, -1, emb.shape[-1]), sigma_in, x0_pred.float())
            else:
                x0_pred = x0_fn(sample, sigma_in, return_only_hidden_states_up_to=None)
            sample, x0_prev = sched.step(x0_pred=x0_pred, i=i, sample=sample, x0_prev=x0_prev)
        return out

    def oracle_context(obs, fut, task, k, seed):
        """Training-style context: the real 61-frame video noised to sigma_k."""
        data_batch = {
            "obs/workspace_rgb": obs, "action/workspace_rgb": fut,
            "obs/language_embedding": v2w.encode_prompt(task).to(v2w.torch_dtype),
            "num_conditional_frames": v2w.tokenizer.get_latent_num_frames(obs.shape[2]),
            "is_preprocessed": True,
        }
        video, condition = v2w.get_mimic_data_and_condition(data_batch)
        g = torch.Generator(device="cuda").manual_seed(1000 + seed)
        eps = torch.randn(video.size(), generator=g, device="cuda", dtype=torch.float32).to(torch.bfloat16)
        sched.set_timesteps(N_STEPS, device="cuda")
        sigma = sched.sigmas[k].repeat(1).unsqueeze(1)
        pred = v2w.denoise(video + eps * rearrange(sigma, "b t -> b 1 t 1 1"), sigma, condition,
                           use_cuda_graphs=False, return_only_hidden_states_up_to=layer, return_decoded_video=False)
        emb = pred.hidden_states[layer]
        return emb.reshape(1, -1, emb.shape[-1]), video.float()

    def decode(ctx, sigma):
        return w2a(state_B_HO_O=state, crossattn_emb=ctx, context_timesteps_B_1=sigma.unsqueeze(1) if sigma.dim() == 1
                   else sigma, seed=0, use_cuda_graphs=False)[0].float().cpu().numpy()

    def rel(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))

    def trel(a, b):
        return float((a - b).norm() / (b.norm() + 1e-8))

    def act_stats(a):
        rot = (np.abs(np.linalg.norm(a[:, 3:6], axis=-1) - 1).mean()
               + np.abs(np.linalg.norm(a[:, 6:9], axis=-1) - 1).mean()) / 2
        return {"translation": float(np.linalg.norm(a[:, :3], axis=-1).mean()), "rot6_norm_err": float(rot),
                "gripper": float(a[:, 9].mean()), "nan": int(np.isnan(a).sum())}

    clips = load_clips()
    print(f"clips: {len(clips)}  K: {K}  sigmas at K: {[round(sigmas[k], 2) for k in K]}", flush=True)
    n_tok_obs = None
    selfcheck = None
    results = []
    for ci, c in enumerate(clips):
        traj = {s: contexts_along_trajectory(c["obs"], c["task"], s, set(K)) for s in SEEDS}
        if n_tok_obs is None:
            n_tok_obs = traj[0][K[0]][0].shape[1] // v2w.config.state_t * N_OBS_LAT

        # self-check: the replicated loop must reproduce the production call
        if ci == 0 and 1 in K:
            prod = pipe(input_vid=c["obs"], state_B_HO_O=state, prompt=c["task"], stop_after_step=1, seed=0,
                        use_cuda_graphs=False)[0].float().cpu().numpy()
            mine = decode(traj[0][1][0], traj[0][1][1])
            selfcheck = float(np.abs(prod - mine).max())
            print(f"self-check vs production call at k=1: max |diff| = {selfcheck:.3e}", flush=True)

        # real and static futures in latent space
        static = c["obs"][:, :, -1:].repeat(1, 1, N_FUT_PIX, 1, 1)
        lat_real = v2w.encode(torch.cat([c["obs"], c["fut"]], 2)).float()
        lat_static = v2w.encode(torch.cat([c["obs"], static], 2)).float()
        fr, fs = lat_real[:, :, N_OBS_LAT:], lat_static[:, :, N_OBS_LAT:]
        err_static = trel(fs, fr)

        rec = {"episode": c["episode"], "t": c["t"], "len": c["len"], "task": c["task"],
               "fut_real_frames": c["fut_real_frames"], "err_static": err_static, "k": {}}
        for k in K:
            ctx = {s: traj[s][k][0] for s in SEEDS}
            sig = traj[0][k][1]
            x0 = {s: traj[s][k][2][:, :, N_OBS_LAT:] for s in SEEDS}
            acts = {s: decode(ctx[s], sig) for s in SEEDS}
            r = {"sigma": float(sig[0]),
                 "forecast_err": float(np.mean([trel(x0[s], fr) for s in SEEDS])),
                 "forecast_err_near": float(np.mean([trel(x0[s][:, :, :3], fr[:, :, :3]) for s in SEEDS])),
                 "forecast_vs_static": float(np.mean([trel(x0[s], fs) for s in SEEDS])),
                 "future_spread": float(np.mean([trel(x0[s], x0[0]) for s in SEEDS[1:]])),
                 "action_spread": float(np.mean([rel(acts[s], acts[0]) for s in SEEDS[1:]])),
                 "acts": {s: act_stats(acts[s]) for s in SEEDS}}
            r["forecast_ratio"] = r["forecast_err"] / err_static
            follow, follow_ctrl = [], []
            for s in SEEDS[1:]:
                d0 = np.linalg.norm(acts[s] - acts[0])
                if d0 < 1e-6:
                    continue
                fut_patch = torch.cat([ctx[0][:, :n_tok_obs], ctx[s][:, n_tok_obs:]], 1)
                obs_patch = torch.cat([ctx[s][:, :n_tok_obs], ctx[0][:, n_tok_obs:]], 1)
                for patch, lst in ((fut_patch, follow), (obs_patch, follow_ctrl)):
                    a = decode(patch, sig)
                    da, db = np.linalg.norm(a - acts[0]), np.linalg.norm(a - acts[s])
                    lst.append(float(da / (da + db + 1e-12)))
            r["follow_future"] = float(np.mean(follow)) if follow else None
            r["follow_observed"] = float(np.mean(follow_ctrl)) if follow_ctrl else None
            orc = []
            for s in ORACLE_SEEDS:
                octx, lat_v = oracle_context(c["obs"], c["fut"], c["task"], k, s)
                orc.append(decode(octx, sched.sigmas[k].repeat(1)))
            r["oracle_vs_forecast"] = float(np.mean([rel(orc[i], acts[s]) for i, s in enumerate(ORACLE_SEEDS)]))
            r["oracle_spread"] = rel(orc[1], orc[0]) if len(orc) > 1 else None
            r["oracle_acts"] = [act_stats(a) for a in orc]
            rec["k"][k] = r
        results.append(rec)
        print(f"clip {ci} ep {c['episode']} t={c['t']} '{c['task'][:35]}' static_err {err_static:.3f} | "
              + " | ".join(f"k{k}: fc {rec['k'][k]['forecast_ratio']:.2f} spr {rec['k'][k]['action_spread']:.3f} "
                           f"fol {rec['k'][k]['follow_future'] if rec['k'][k]['follow_future'] is None else round(rec['k'][k]['follow_future'], 2)} "
                           f"orc {rec['k'][k]['oracle_vs_forecast']:.3f}" for k in K), flush=True)
        del traj
        torch.cuda.empty_cache()

    # latency of the production call per k
    c = clips[0]
    latency = {}
    for k in K:
        ts = []
        for rep in range(4):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipe(input_vid=c["obs"], state_B_HO_O=state, prompt=c["task"], stop_after_step=k, seed=0,
                 use_cuda_graphs=False)
            torch.cuda.synchronize()
            if rep:
                ts.append(time.perf_counter() - t0)
        latency[k] = float(np.median(ts))

    print(f"\nGPU: {torch.cuda.get_device_name()}; self-check max |diff| vs production at k=1: {selfcheck}")
    print("| k | sigma | forecast err / static err | near-future ratio | future spread (seeds) | action spread (seeds) "
          "| follow future | follow observed (ctrl) | oracle vs forecast | oracle spread | translation | rot6 err | latency s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    m = lambda xs: (np.nanmean([np.nan if x is None else x for x in xs]),  # noqa: E731
                    np.nanstd([np.nan if x is None else x for x in xs]))
    summary = {}
    for k in K:
        rows = [r["k"][k] for r in results]
        near = [r["forecast_err_near"] / rr["err_static"] for r, rr in zip(rows, results)]
        cols = {
            "forecast_ratio": m([r["forecast_ratio"] for r in rows]),
            "near_ratio": m(near),
            "future_spread": m([r["future_spread"] for r in rows]),
            "action_spread": m([r["action_spread"] for r in rows]),
            "follow_future": m([r["follow_future"] for r in rows]),
            "follow_observed": m([r["follow_observed"] for r in rows]),
            "oracle_vs_forecast": m([r["oracle_vs_forecast"] for r in rows]),
            "oracle_spread": m([r["oracle_spread"] for r in rows]),
            "translation": m([r["acts"][0]["translation"] for r in rows]),
            "rot6": m([r["acts"][0]["rot6_norm_err"] for r in rows]),
        }
        summary[k] = {kk: list(v) for kk, v in cols.items()} | {"latency_s": latency[k], "sigma": rows[0]["sigma"]}
        f = lambda n: f"{cols[n][0]:.3f} ± {cols[n][1]:.3f}"  # noqa: E731
        print(f"| {k} | {rows[0]['sigma']:.2f} | {f('forecast_ratio')} | {f('near_ratio')} | {f('future_spread')} "
              f"| {f('action_spread')} | {f('follow_future')} | {f('follow_observed')} | {f('oracle_vs_forecast')} "
              f"| {f('oracle_spread')} | {cols['translation'][0]:.4f} | {cols['rot6'][0]:.4f} | {latency[k]:.2f} |")
    nan_total = sum(a["nan"] for r in results for kk in r["k"].values() for a in kk["acts"].values())
    print(f"NaN in actions: {nan_total}")
    out = HERE / f"depth_open_loop_{N_EPISODES}ep.json"
    out.write_text(json.dumps({"K": K, "selfcheck": selfcheck, "gpu": torch.cuda.get_device_name(),
                               "summary": summary, "clips": results}, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
