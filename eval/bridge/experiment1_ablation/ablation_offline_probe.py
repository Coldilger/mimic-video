#!/usr/bin/env python3
"""
Offline probe for Experiment 1, variant 1 (mimic-video): "remove the module".

Same protocol/infrastructure as ../experiment2_oracle/oracle_offline_probe.py
(real BridgeDataV2 episodes, no SimplerEnv/ManiSkill2). For a sampled
(episode, t), predicts the action chunk twice on the SAME real logged
moment -- baseline (VAMInference, is_hil=False, self-imagined future via the
real video backbone) and ablated (VAMAblatedInference, crossattn_emb forced
to zeros, see zero_world_model_pipeline.py) -- and compares both against the
real logged action. Metric matches oracle_offline_probe.py's (position L1
against the next observation.state, gripper L1), for direct comparability.
"""

import contextlib
import os
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

MIMIC_ROOT = pathlib.Path("/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video")
sys.path.insert(0, str(MIMIC_ROOT / "eval/bridge/SimplerEnv"))
sys.path.insert(0, str(MIMIC_ROOT / "model"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402

from vam_ablated_inference import VAMAblatedInference  # noqa: E402


@contextlib.contextmanager
def _fast_path_is_file():
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


@dataclass
class PoseProxy:
    p: np.ndarray
    q: np.ndarray


def pose_from_bridge_state(state: np.ndarray) -> PoseProxy:
    pos = state[0:3].astype(np.float64)
    rpy = state[3:6].astype(np.float64)
    quat_xyzw = Rotation.from_euler("XYZ", rpy).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return PoseProxy(p=pos, q=quat_wxyz)


def to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.shape[0] in (1, 3) and arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _predict(model, image_t, task_description, proprio_t, gripper_t):
    model.reset(task_description)
    model.step(image_t, task_description, proprio_t, gripper_t)
    cur_rot = Rotation.from_quat(proprio_t.q, scalar_first=True)
    ee_rot_t = cur_rot.as_matrix()[:2].reshape(6)
    lowdim_t = np.concatenate([proprio_t.p, ee_rot_t, [gripper_t]]).astype(np.float64)
    obs_T = model._pose_from_lowdim(lowdim_t)
    pred_delta_T = model._pose_from_lowdim(model.action_buffer[0].astype(np.float64))
    pred_abs_T = obs_T @ pred_delta_T
    pred_pos = pred_abs_T[:3, 3]
    pred_grip = model.action_buffer[0, 9]
    return pred_pos, pred_grip


def run_ablation_probe(
    checkpoint_dir: pathlib.Path,
    dataset_root: pathlib.Path,
    camera_key: str = "observation.images.image_0",
    n_episodes: int = 24,
    samples_per_episode: int = 5,
    seed: int = 0,
):
    experiment_name = (
        "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043"
        "_fused_lr1.000e-04_layer20_bsz256"
    )
    common_kwargs = dict(
        experiment_name=experiment_name,
        video_model_path=str(
            checkpoint_dir / "video_backbone" / "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
        ),
        action_model_path=str(
            checkpoint_dir
            / "action_decoder"
            / "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043"
              "_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
        ),
        dataset_statistics_path=checkpoint_dir / "dataset_statistics" / "bridge.json",
        img_horizon=5,
        lowdim_horizon=1,
        stop_video_denoising_step=23,
        num_execute_actions=5,
    )

    print("Loading baseline model...", flush=True)
    model_baseline = VAMInference(is_hil=False, **common_kwargs)
    print("Loading ablated model...", flush=True)
    model_ablated = VAMAblatedInference(**common_kwargs)

    rng = np.random.default_rng(seed)
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
        episode_ids = rng.choice(meta.total_episodes, size=n_episodes, replace=False).tolist()
        ds = LeRobotDataset(
            repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav"
        )
    print(f"dataset loaded, {len(episode_ids)} episodes selected of {meta.total_episodes} total", flush=True)

    pos_errs_base, grip_errs_base = [], []
    pos_errs_abl, grip_errs_abl = [], []
    pos_errs_baseline_nomove = []

    for pos, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        if ep_to - ep_from < 3:
            continue

        task_description = ds[ep_from]["task"]
        candidate_ts = np.arange(ep_from, ep_to - 1)
        ts = rng.choice(candidate_ts, size=min(samples_per_episode, len(candidate_ts)), replace=False)

        for t in ts:
            t0 = time.time()
            t = int(t)
            frame_t = ds[t]
            frame_next = ds[t + 1]
            image_t = to_uint8_hwc(frame_t[camera_key])
            state_t = frame_t["observation.state"].numpy()
            state_next = frame_next["observation.state"].numpy()
            proprio_t = pose_from_bridge_state(state_t)
            gripper_t = float(state_t[7])

            pred_pos_base, pred_grip_base = _predict(model_baseline, image_t, task_description, proprio_t, gripper_t)
            pred_pos_abl, pred_grip_abl = _predict(model_ablated, image_t, task_description, proprio_t, gripper_t)

            pos_errs_base.append(np.abs(pred_pos_base - state_next[:3]).mean())
            grip_errs_base.append(abs(pred_grip_base - state_next[7]))
            pos_errs_abl.append(np.abs(pred_pos_abl - state_next[:3]).mean())
            grip_errs_abl.append(abs(pred_grip_abl - state_next[7]))
            pos_errs_baseline_nomove.append(np.abs(state_t[:3] - state_next[:3]).mean())
            print(f"sample {len(pos_errs_base)} (ep {ep_id}, t {t}): {time.time() - t0:.1f}s", flush=True)

    pos_errs_base, grip_errs_base, pos_errs_abl, grip_errs_abl, pos_errs_baseline_nomove = map(
        np.array, (pos_errs_base, grip_errs_base, pos_errs_abl, grip_errs_abl, pos_errs_baseline_nomove)
    )
    print(f"n samples: {len(pos_errs_base)}")
    print(f"position L1, baseline (self-imagined):  {pos_errs_base.mean():.4f}  (sd {pos_errs_base.std():.4f})")
    print(f"position L1, ablated (zero crossattn):  {pos_errs_abl.mean():.4f}  (sd {pos_errs_abl.std():.4f})")
    print(f"position L1, no-motion baseline:        {pos_errs_baseline_nomove.mean():.4f}")
    print(f"gripper L1, baseline:                   {grip_errs_base.mean():.4f}")
    print(f"gripper L1, ablated:                    {grip_errs_abl.mean():.4f}")


if __name__ == "__main__":
    run_ablation_probe(
        checkpoint_dir=MIMIC_ROOT / "model" / "checkpoints",
        dataset_root=pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"),
    )
