#!/usr/bin/env python3
"""
Offline oracle probe for mimic-video on real BridgeDataV2 episodes.

Bypasses SimplerEnv/ManiSkill2 entirely. Instead of maniskill2_hil_evaluator.py's
live keyboard teleop (_teleop_segment -> hil_images -> model.ingest_video), this
feeds ingest_video() real recorded future frames from the SAME bridge_orig_lerobot
dataset F1-VLA and LDA-1B were finetuned on. ingest_video() only ever consumes a
plain image sequence (video_action_model.py:134-140) -- it does not care where the
frames came from, so this substitution is mechanically valid.

Metric mirrors LDA-1B's own open-loop probe (LDA-1B/lda/eval/eval_bridge_openloop.py):
position/gripper L1 against the *actual next observation.state*, not the raw
`action` column. This sidesteps reconstructing Bridge's delta-action / rotation
convention (see WARNING in pose_from_bridge_state below) and keeps the metric
identical to what LDA-1B already reports, so the two are comparable out of the box.

STATUS: first-draft harness. Two things are marked VERIFY below and must be
checked before trusting the numbers -- see inline comments. (A third,
image tensor dtype/layout, is now confirmed against lerobot source, not
just assumed -- see to_uint8_hwc.)
"""

import contextlib
import os
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

MIMIC_ROOT = pathlib.Path("/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video")
sys.path.insert(0, str(MIMIC_ROOT / "eval/bridge/SimplerEnv"))
sys.path.insert(0, str(MIMIC_ROOT / "model"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402


@contextlib.contextmanager
def _fast_path_is_file():
    """Same patch used in F1-VLA/eval/bridge/oracle_offline_probe.py.
    LeRobotDataset.__init__ calls Path.is_file() once per episode/video file;
    on beegfs each call is an uncached network round-trip, which for 53,192
    Bridge episodes takes HOURS unpatched (confirmed on the F1 run of this
    same protocol: hung silently for the full 1h SLURM wall-time with zero
    output). This lists each parent dir once instead."""
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
    """Stand-in for ManiSkill2's sapien.Pose. VAMInference.step() only touches
    .p and .q (video_action_model.py:156-169), so a plain object works fine --
    it never needs a live simulator behind it."""

    p: np.ndarray  # (3,) xyz position
    q: np.ndarray  # (4,) quaternion, scalar-first [w,x,y,z]


def pose_from_bridge_state(state: np.ndarray) -> PoseProxy:
    # bridge_orig_lerobot observation.state layout (meta/info.json):
    #   [x, y, z, roll, pitch, yaw, pad, gripper]   (index 6 is a zero pad column)
    pos = state[0:3].astype(np.float64)
    rpy = state[3:6].astype(np.float64)

    # WARNING -- unverified orientation frame. mimic-video/data_preprocessing/
    # action/process_bridge.py documents that Bridge's raw `full_state`
    # orientation is rotated by [[0,0,-1],[0,1,0],[1,0,0]] (right-multiplied)
    # relative to `eef_transform`, and THAT is the convention their action
    # decoder was trained against. bridge_orig_lerobot is a different
    # (HuggingFace) repackaging of the same underlying Berkeley release, and
    # it is not verified here whether it already carries that correction or
    # not. F1-VLA hit an analogous but distinct frame bug (RESULTS.md fix #4,
    # SimplerEnv's absolute ee_pose_at_base vs Bridge's reset-relative euler).
    # Treat orientation-derived numbers from this script as unverified.
    # Position and gripper below are NOT affected by this warning.
    quat_xyzw = Rotation.from_euler("XYZ", rpy).as_quat()  # scipy order: [x,y,z,w]
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return PoseProxy(p=pos, q=quat_wxyz)


def to_uint8_hwc(img) -> np.ndarray:
    """Confirmed against lerobot source (decode_video_frames_torchvision:
    "convert to the pytorch format which is float32 in [0,1] range (and
    channel first)") -- bridge_orig_lerobot frames are float32 CHW [0,1], and
    _process_image (video_action_model.py:113-120) expects uint8 HWC RGB, so
    this conversion is necessary and its branch condition is not a guess."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.shape[0] in (1, 3) and arr.ndim == 3:  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def run_oracle_probe(
    checkpoint_dir: pathlib.Path,
    dataset_root: pathlib.Path,
    camera_key: str = "observation.images.image_0",  # VERIFY: matches mimic's training camera
    n_episodes: int = 24,
    samples_per_episode: int = 5,
    future_horizon: int = 12,  # frames of true future handed to ingest_video
    seed: int = 0,
):
    experiment_name = (
        "w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043"
        "_fused_lr1.000e-04_layer20_bsz256"
    )
    model = VAMInference(
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
        img_horizon=5,  # matches eval/bridge/eval_hil.sh ftcosmos[img_h]
        lowdim_horizon=1,  # matches eval/bridge/eval_hil.sh ftcosmos[lowdim_h]
        stop_video_denoising_step=23,  # matches eval_hil.sh stop_steps
        num_execute_actions=5,
        is_hil=True,  # activates the gt_future_vid / oracle branch (step():180-190)
    )

    # LeRobotDataset(repo_id=..., root=...) with no episodes= arg loads/
    # generates the HF `datasets` split for ALL 53,192 episodes and never
    # finished inside a 1h SLURM job (confirmed via a standalone diagnostic on
    # the F1 side of this same protocol: 50 explicit episodes built in 7.3s,
    # the full unfiltered dataset didn't finish in 1h). We only ever sample
    # n_episodes, so pick ids from the lightweight metadata object first and
    # hand LeRobotDataset only those.
    rng = np.random.default_rng(seed)
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
        episode_ids = rng.choice(meta.total_episodes, size=n_episodes, replace=False).tolist()
        ds = LeRobotDataset(
            repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav"
        )  # default backend (torchcodec) needs FFmpeg .so's not installed here; pyav worked in the diagnostic

    pos_errs, grip_errs, pos_errs_baseline = [], [], []

    # episode_data_index is positional (0..len(episode_ids)-1 in episode_ids'
    # own order) once episodes= filters the dataset, not indexed by the
    # original global episode id -- see lerobot's get_episode_data_index.
    for pos, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        if ep_to - ep_from < future_horizon + 2:
            continue

        task_description = ds[ep_from]["task"]
        candidate_ts = np.arange(ep_from, ep_to - future_horizon - 1)
        ts = rng.choice(candidate_ts, size=min(samples_per_episode, len(candidate_ts)), replace=False)

        for t in ts:
            t0 = time.time()
            t = int(t)
            frame_t = ds[t]
            frame_next = ds[t + 1]
            future_frames = [to_uint8_hwc(ds[t + k][camera_key]) for k in range(1, future_horizon + 1)]
            image_t = to_uint8_hwc(frame_t[camera_key])

            state_t = frame_t["observation.state"].numpy()
            state_next = frame_next["observation.state"].numpy()
            proprio_t = pose_from_bridge_state(state_t)
            gripper_t = float(state_t[7])

            model.reset(task_description)
            model.ingest_video(np.stack(future_frames))  # <-- the actual swap vs teleop
            model.step(image_t, task_description, proprio_t, gripper_t)

            # FIX (first run compared these directly and got a position L1
            # *worse* than the no-motion baseline -- a red flag that turned out
            # to be real): model.action_buffer[:, :3] is NOT a world-frame
            # position or delta. _pose_from_lowdim (video_action_model.py:
            # 259-270) treats it as the translation of a body-frame SE(3)
            # transform, and step() itself only ever uses it composed as
            # obs_T @ Td (line 207) -- never compared to a world-frame target
            # directly anywhere in this class. Reproduce that composition here
            # instead of comparing the raw buffer to observation.state.
            cur_rot = Rotation.from_quat(proprio_t.q, scalar_first=True)
            ee_rot_t = cur_rot.as_matrix()[:2].reshape(6)
            lowdim_t = np.concatenate([proprio_t.p, ee_rot_t, [gripper_t]]).astype(np.float64)
            obs_T = model._pose_from_lowdim(lowdim_t)
            pred_delta_T = model._pose_from_lowdim(model.action_buffer[0].astype(np.float64))
            pred_abs_T = obs_T @ pred_delta_T
            pred_pos = pred_abs_T[:3, 3]
            pred_grip = model.action_buffer[0, 9]

            pos_errs.append(np.abs(pred_pos - state_next[:3]).mean())
            grip_errs.append(abs(pred_grip - state_next[7]))
            pos_errs_baseline.append(np.abs(state_t[:3] - state_next[:3]).mean())  # "arm never moves"
            print(f"sample {len(pos_errs)} (ep {ep_id}, t {t}): {time.time() - t0:.1f}s", flush=True)

    pos_errs, grip_errs, pos_errs_baseline = map(np.array, (pos_errs, grip_errs, pos_errs_baseline))
    print(f"n samples: {len(pos_errs)}")
    print(f"position L1 (oracle):             {pos_errs.mean():.4f}  (sd {pos_errs.std():.4f})")
    print(f"position L1 (no-motion baseline): {pos_errs_baseline.mean():.4f}")
    print(f"gripper L1 (oracle):              {grip_errs.mean():.4f}")


if __name__ == "__main__":
    run_oracle_probe(
        checkpoint_dir=MIMIC_ROOT / "model" / "checkpoints",
        dataset_root=pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"),
    )
