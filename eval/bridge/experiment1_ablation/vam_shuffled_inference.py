"""VAMInference, but the world model's image history is drawn from a fixed
pool of real Bridge frames from OTHER episodes instead of the current
rollout's own frames -- the closed-loop analogue of shuffle_offline_probe.py,
following the same design as F1-VLA's F1VLAShuffledInference
(F1-VLA/eval/bridge/experiment1_ablation/f1_vla_policy_shuffled.py).

VAMInference.step() is long (replanning, action-buffer bookkeeping, absolute
pose targeting) and none of it should change here -- only the image the
world-model backbone actually sees. Its ONLY route into step() is
`_add_image_to_history(image)`, called once per step with the real,
just-observed frame; state comes from ee_pose_proprio/gripper_proprio
directly and never touches this. Overriding just that one method -- to push
a pooled real-but-wrong-episode frame instead of the real `image` argument
it's given -- swaps the world model's visual input while leaving proprio,
replanning, and pose-targeting completely real. No changes to
video_action_model.py.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys

import numpy as np

MIMIC_ROOT = pathlib.Path("/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video")
sys.path.insert(0, str(MIMIC_ROOT / "eval/bridge/SimplerEnv"))
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv")

from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402


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


def _to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.shape[0] in (1, 3) and arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


class VAMShuffledInference(VAMInference):
    def __init__(
        self,
        *args,
        frame_pool_size: int = 64,
        frame_pool_seed: int = 123,
        dataset_root: str = "/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot",
        camera_key: str = "observation.images.image_0",
        **kwargs,
    ):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        self._frame_pool = self._load_frame_pool(frame_pool_size, frame_pool_seed, dataset_root, camera_key)
        # Separate, seed-tied RNG for the per-step draw -- deterministic
        # given frame_pool_seed, independent of anything else consuming
        # randomness (e.g. the diffusion sampler).
        self._shuffle_rng = np.random.default_rng(frame_pool_seed)

    def _load_frame_pool(self, size, seed, dataset_root, camera_key):
        # Local import: only needed here, and this whole file already
        # assumes the SimplerEnv-side sys.path is set up by the caller.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        rng = np.random.default_rng(seed)
        with _fast_path_is_file():
            meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=dataset_root)
            episode_ids = rng.choice(meta.total_episodes, size=size, replace=False).tolist()
            ds = LeRobotDataset(
                repo_id="bridge_orig_lerobot", root=dataset_root, episodes=episode_ids, video_backend="pyav"
            )
        pool = []
        for pos in range(len(episode_ids)):
            ep_from = int(ds.episode_data_index["from"][pos])
            frame = _to_uint8_hwc(ds[ep_from][camera_key])
            pool.append(self._process_image(frame))
        return pool

    def _add_image_to_history(self, image: np.ndarray) -> None:
        # `image` is the real, just-observed frame (already processed by
        # VAMInference.step()) -- ignored on purpose. Push a fixed-pool,
        # real-but-wrong-episode frame instead.
        shuffled = self._frame_pool[self._shuffle_rng.integers(len(self._frame_pool))]
        super()._add_image_to_history(shuffled)
