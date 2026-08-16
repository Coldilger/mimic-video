#!/usr/bin/env python3
"""
Offline probe for Experiment 1, variant 2 (mimic-video): "shuffle the
world-model input across episodes" -- the clean version of variant 1,
without its "model has literally zero content in that slot" confound.

mimic's world-model connection is a plain tensor, not F1's KV-cache
(../README.md's mechanism section), so unlike F1's variant 2 this needs no
new hook at all: the video-diffusion backbone's *only* input is the image
history it's given (video_action_model.py's VAMInference.step -- `image` ->
`_add_image_to_history` -> `input_vid` fed to the model; state comes from
`ee_pose_proprio`/`gripper_proprio` separately and never touches `image`).
So "shuffle" is just calling the UNMODIFIED baseline VAMInference with a
different episode's real frame standing in for the current one, while state
and task description stay matched to the real, current sample -- no custom
pipeline class needed, unlike variant 1's ZeroWorldModelPipeline.

Same sampling protocol as variant 1 (24 episodes x 5 moments, seed=0) and
the same shift-by-samples_per_episode pairing scheme as F1's
kv_shuffle_offline_probe.py, so results are directly comparable across both
models and both variants.
"""

import contextlib
import os
import pathlib
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

MIMIC_ROOT = pathlib.Path("/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video")
sys.path.insert(0, str(MIMIC_ROOT / "eval/bridge/SimplerEnv"))
sys.path.insert(0, str(MIMIC_ROOT / "model"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402

from ablation_offline_probe import pose_from_bridge_state, to_uint8_hwc  # noqa: E402


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


def run_shuffle_probe(
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

    print("Loading model...", flush=True)
    model = VAMInference(is_hil=False, **common_kwargs)

    rng = np.random.default_rng(seed)
    with _fast_path_is_file():
        meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
        episode_ids = rng.choice(meta.total_episodes, size=n_episodes, replace=False).tolist()
        ds = LeRobotDataset(
            repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav"
        )
    print(f"dataset loaded, {len(episode_ids)} episodes selected of {meta.total_episodes} total", flush=True)

    # Same (episode, t) sampling as ablation_offline_probe.py, same seed/order.
    samples = []  # (pos, ep_id, t)
    for pos, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos])
        ep_to = int(ds.episode_data_index["to"][pos])
        if ep_to - ep_from < 3:
            continue
        candidate_ts = np.arange(ep_from, ep_to - 1)
        ts = rng.choice(candidate_ts, size=min(samples_per_episode, len(candidate_ts)), replace=False)
        for t in ts:
            samples.append((pos, ep_id, int(t)))

    # Pair each sample with a "world-model input source" from a DIFFERENT
    # episode -- shift by samples_per_episode so consecutive samples (same
    # episode's block) never pair with themselves; wraps around the list.
    n = len(samples)
    shuffle_offset = samples_per_episode if samples_per_episode < n else 1
    pairs = [(i, (i + shuffle_offset) % n) for i in range(n)]
    for i, j in pairs:
        assert samples[i][1] != samples[j][1], f"sample {i} paired with its own episode {samples[i][1]}"

    pos_errs_shuf, grip_errs_shuf = [], []

    for i, j in pairs:
        t0 = time.time()
        pos_idx, ep_id, t = samples[i]
        _, src_ep_id, src_t = samples[j]

        frame_t = ds[t]
        frame_next = ds[t + 1]
        task_description = frame_t["task"]
        state_t = frame_t["observation.state"].numpy()
        state_next = frame_next["observation.state"].numpy()
        proprio_t = pose_from_bridge_state(state_t)
        gripper_t = float(state_t[7])

        # Wrong-episode world-model input: a real frame, but from a
        # different episode/moment than the one we're predicting for. State
        # and task description stay matched to the real, current sample.
        wrong_image = to_uint8_hwc(ds[src_t][camera_key])

        pred_pos, pred_grip = _predict(model, wrong_image, task_description, proprio_t, gripper_t)

        pos_errs_shuf.append(np.abs(pred_pos - state_next[:3]).mean())
        grip_errs_shuf.append(abs(pred_grip - state_next[7]))
        print(f"sample {len(pos_errs_shuf)} (ep {ep_id}, t {t}, world-model input from ep {src_ep_id}): "
              f"{time.time() - t0:.1f}s", flush=True)

    pos_errs_shuf, grip_errs_shuf = map(np.array, (pos_errs_shuf, grip_errs_shuf))
    print(f"n samples: {len(pos_errs_shuf)}")
    print(f"position L1, shuffled (wrong-episode world-model input): {pos_errs_shuf.mean():.4f}  "
          f"(sd {pos_errs_shuf.std():.4f})")
    print(f"gripper L1, shuffled: {grip_errs_shuf.mean():.4f}")


if __name__ == "__main__":
    run_shuffle_probe(
        checkpoint_dir=MIMIC_ROOT / "model" / "checkpoints",
        dataset_root=pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot"),
    )
