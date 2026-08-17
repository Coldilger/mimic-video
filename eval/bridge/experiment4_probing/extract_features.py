#!/usr/bin/env python3
"""
Experiment 4 (mimic-video), step 1: extract frozen world-model features.

Runs the real, unmodified VAMInference over real logged BridgeDataV2 moments
and captures `crossattn_emb` -- the exact tensor Experiment 1 ablates
(`../experiment1_ablation/zero_world_model_pipeline.py`) and the only channel
by which the video-diffusion backbone reaches the action decoder. Saves those
features alongside the ground-truth future end-effector pose, for
`train_probe.py` to fit a small probe head on.

Nothing under `model/` is modified: `crossattn_emb` is produced inside
`Video2World2ActionPipeline.__call__` by `video2world_pipeline.generate_video`,
so that one bound method is wrapped on the live instance to record its return
value while the call proceeds normally.

Probe target: the *delta* end-effector pose K steps ahead (pose[t+K] - pose[t]),
not the absolute pose. Experiment 1's offline probe showed absolute
single-step position sits essentially at the no-motion floor (every condition
0.0098-0.0100 vs a 0.0096 floor), i.e. that target measures "the arm barely
moves" rather than representation quality. The delta target removes that
degeneracy; a no-motion reference (predicting all-zero delta) is still
recorded by train_probe.py for comparison.

Usage:
  python extract_features.py --variant finetuned --out features_finetuned.npz
  python extract_features.py --variant pretrained --out features_pretrained.npz
"""

import argparse
import contextlib
import os
import pathlib
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

MIMIC_ROOT = pathlib.Path("/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video")
sys.path.insert(0, str(MIMIC_ROOT / "eval/bridge/SimplerEnv"))
sys.path.insert(0, str(MIMIC_ROOT / "model"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402


# The two Bridge checkpoints differ exactly in the video backbone -- the
# component that produces the probed `crossattn_emb`. Both ship a
# Bridge-trained action decoder, which this experiment never probes, so the
# comparison isolates what Bridge finetuning built into the representation.
VARIANTS = {
    "finetuned": dict(
        experiment_name=(
            "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043"
            "_fused_lr1.000e-04_layer20_bsz256"
        ),
        video_model="video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt",
        action_model=(
            "action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043"
            "_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
        ),
    ),
    # Filled in from the actual downloaded filenames at runtime -- the
    # pretrained release's iter/lr suffixes aren't known ahead of the download.
    "pretrained": dict(
        experiment_name=None,
        video_model="video_backbone/v2w_pretrained_cosmos.pt",
        action_model=None,
    ),
}


@contextlib.contextmanager
def _fast_path_is_file():
    """LeRobot stats a file per frame; cache directory listings instead.

    Same helper as ../experiment1_ablation/ablation_offline_probe.py -- on
    this cluster's shared filesystem the per-frame stat calls otherwise
    dominate runtime.
    """
    original_is_file = pathlib.Path.is_file
    dir_listing_cache = {}

    def fast_is_file(self):
        parent = self.parent
        if parent not in dir_listing_cache:
            try:
                dir_listing_cache[parent] = set(os.listdir(parent))
            except (FileNotFoundError, NotADirectoryError):
                dir_listing_cache[parent] = set()
        return self.name in dir_listing_cache[parent]

    pathlib.Path.is_file = fast_is_file
    try:
        yield
    finally:
        pathlib.Path.is_file = original_is_file


class _PoseProxy:
    def __init__(self, p, q):
        self.p = p
        self.q = q


def pose_from_bridge_state(state: np.ndarray) -> _PoseProxy:
    pos = state[0:3].astype(np.float64)
    rpy = state[3:6].astype(np.float64)
    quat_xyzw = Rotation.from_euler("XYZ", rpy).as_quat()
    return _PoseProxy(p=pos, q=quat_xyzw[[3, 0, 1, 2]])


def to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.shape[0] in (1, 3) and arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def pose_vector(state: np.ndarray) -> np.ndarray:
    """9-D pose: xyz position + first two rows of the rotation matrix (6D rot).

    6D rotation rather than euler/quaternion because it is continuous -- a
    regression target with no wraparound discontinuity for the probe to fit
    around.
    """
    pos = state[0:3].astype(np.float64)
    rot6 = Rotation.from_euler("XYZ", state[3:6].astype(np.float64)).as_matrix()[:2].reshape(6)
    return np.concatenate([pos, rot6])


def resolve_variant(variant: str, checkpoint_dir: pathlib.Path) -> dict:
    cfg = dict(VARIANTS[variant])
    if variant == "pretrained":
        # Resolve the action decoder + experiment name from whatever the
        # download actually produced, rather than hardcoding a guessed suffix.
        matches = sorted((checkpoint_dir / "action_decoder").glob("w2a_bridge_v2w_pretrained_cosmos*.pt"))
        if not matches:
            msg = (
                f"no pretrained action decoder found in {checkpoint_dir / 'action_decoder'}; "
                "run download_pretrained.slurm first"
            )
            raise FileNotFoundError(msg)
        cfg["action_model"] = f"action_decoder/{matches[0].name}"
        # VAMInference derives config from experiment_name, which is the
        # decoder filename minus its trailing `_iter_XXXXXXXXX` chunk.
        stem = matches[0].stem
        cfg["experiment_name"] = stem.rsplit("_iter_", 1)[0]
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, default=40)
    ap.add_argument("--samples-per-episode", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=5, help="K: how many steps ahead the probe target looks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dataset-root", default="/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"
    )
    ap.add_argument("--camera-key", default="observation.images.image_0")
    args = ap.parse_args()

    checkpoint_dir = MIMIC_ROOT / "model" / "checkpoints"
    cfg = resolve_variant(args.variant, checkpoint_dir)
    print(f"variant={args.variant}", flush=True)
    print(f"  experiment_name={cfg['experiment_name']}", flush=True)
    print(f"  video_model={cfg['video_model']}", flush=True)
    print(f"  action_model={cfg['action_model']}", flush=True)

    model = VAMInference(
        is_hil=False,
        experiment_name=cfg["experiment_name"],
        video_model_path=str(checkpoint_dir / cfg["video_model"]),
        action_model_path=str(checkpoint_dir / cfg["action_model"]),
        dataset_statistics_path=checkpoint_dir / "dataset_statistics" / "bridge.json",
        img_horizon=5,
        lowdim_horizon=1,
        # Matches every Experiment 1 condition, so the captured representation
        # is the one the action decoder actually reads at eval time.
        stop_video_denoising_step=23,
        num_execute_actions=5,
    )

    # Capture crossattn_emb where the real pipeline produces it. generate_video
    # returns (crossattn_emb, video_sigma); the pipeline then reshapes the
    # first to [B, tokens, dim] before handing it to the action decoder, so the
    # same reshape is applied here to record exactly what the decoder sees.
    captured = {}
    v2w = model.model.video2world_pipeline
    original_generate_video = v2w.generate_video

    def capturing_generate_video(*a, **kw):
        emb, sigma = original_generate_video(*a, **kw)
        captured["emb"] = emb.reshape(emb.shape[0], -1, emb.shape[-1])
        return emb, sigma

    v2w.generate_video = capturing_generate_video

    rng = np.random.default_rng(args.seed)
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=args.dataset_root)
        episode_ids = rng.choice(meta.total_episodes, size=args.n_episodes, replace=False).tolist()
        ds = LeRobotDataset(
            repo_id="bridge_orig_lerobot",
            root=args.dataset_root,
            episodes=episode_ids,
            video_backend="pyav",
        )
    print(f"dataset loaded: {len(episode_ids)} of {meta.total_episodes} episodes", flush=True)

    feats, targets, nomove_ref, ep_of_sample = [], [], [], []

    for pos, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        # Need t+K to exist inside the same episode.
        if ep_to - ep_from < args.horizon + 2:
            continue

        task_description = ds[ep_from]["task"]
        candidate_ts = np.arange(ep_from, ep_to - args.horizon)
        ts = rng.choice(
            candidate_ts, size=min(args.samples_per_episode, len(candidate_ts)), replace=False
        )

        for t in ts:
            t0 = time.time()
            t = int(t)
            frame_t = ds[t]
            state_t = frame_t["observation.state"].numpy()
            state_future = ds[t + args.horizon]["observation.state"].numpy()

            image_t = to_uint8_hwc(frame_t[args.camera_key])
            proprio_t = pose_from_bridge_state(state_t)
            gripper_t = float(state_t[7])

            captured.clear()
            model.reset(task_description)
            model.step(image_t, task_description, proprio_t, gripper_t)
            if "emb" not in captured:
                msg = "crossattn_emb was never captured -- the wrapped generate_video did not run"
                raise RuntimeError(msg)

            emb = captured["emb"]
            if not feats:
                print(f"crossattn_emb shape (per sample): {tuple(emb.shape)}", flush=True)
            # Pool over the token axis. The raw tensor is [1, 19200, 2048] --
            # ~157MB per sample in fp32, so keeping every token is not an
            # option (a 400-sample run would be ~63GB).
            #
            # Mean *and* std rather than mean alone: with 19200 tokens a plain
            # mean is a very lossy summary, and if the probe then finds no
            # signal, "pooling destroyed it" becomes an alternative explanation
            # that the data cannot rule out. The std channel costs nothing and
            # carries how much each dimension varies across tokens, which a
            # mean cannot express. Spatial structure is still discarded -- a
            # real limitation, recorded in README.md.
            emb_f = emb.float()
            feat = (
                torch.cat([emb_f.mean(dim=1), emb_f.std(dim=1)], dim=-1).squeeze(0).cpu().numpy()
            )

            delta = pose_vector(state_future) - pose_vector(state_t)

            feats.append(feat)
            targets.append(delta)
            nomove_ref.append(np.zeros_like(delta))
            ep_of_sample.append(ep_id)
            print(
                f"sample {len(feats)} (ep {ep_id}, t {t}): {time.time() - t0:.1f}s, feat dim {feat.shape[0]}",
                flush=True,
            )

    feats = np.stack(feats).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    ep_of_sample = np.array(ep_of_sample)

    np.savez_compressed(
        args.out,
        features=feats,
        targets=targets,
        episode_ids=ep_of_sample,
        horizon=args.horizon,
        variant=args.variant,
    )
    print(f"\nsaved {args.out}: features {feats.shape}, targets {targets.shape}", flush=True)
    print(f"target delta magnitude (mean |.|): {np.abs(targets).mean():.5f}", flush=True)


if __name__ == "__main__":
    main()
