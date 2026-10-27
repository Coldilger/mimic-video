"""VAMInference driving a normal baseline rollout, with SanityPipeline
(sanity_pipeline.py) swapped in as self.model to evaluate every ablation
condition on identical inputs at each replan.

Also provides the two shuffled clips:
- perstep:  every one of the img_horizon frames is the FIRST frame of an
            independently drawn pool episode -- the same distribution
            Experiment 1's VAMShuffledInference produces (it pushes a fresh
            pool frame on every control step into a 5-frame deque);
- coherent: img_horizon CONSECUTIVE frames of ONE other pool episode (fixed
            per sim episode), aligned to the current control step -- wrong
            content, but a temporally coherent clip.
Pool: same 64 bridge_orig_lerobot episodes and seed as Experiment 1.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402
from sanity_pipeline import SanityPipeline  # noqa: E402


@contextlib.contextmanager
def _fast_path_is_file():
    # Same NFS speed-up as experiment1_ablation/vam_shuffled_inference.py.
    original_is_file = pathlib.Path.is_file
    cache = {}

    def fast_is_file(self):
        if self.parent not in cache:
            try:
                cache[self.parent] = set(os.listdir(self.parent))
            except (FileNotFoundError, NotADirectoryError):
                cache[self.parent] = set()
        return self.name in cache[self.parent]

    pathlib.Path.is_file = fast_is_file
    try:
        yield
    finally:
        pathlib.Path.is_file = original_is_file


def _to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


class _FrameSource:
    def __init__(self, process_image, horizon, size=64, seed=123,
                 dataset_root="/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot",
                 camera_key="observation.images.image_0"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        self._process = process_image
        self._horizon = horizon
        rng = np.random.default_rng(seed)
        with _fast_path_is_file():
            meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=dataset_root)
            episode_ids = rng.choice(meta.total_episodes, size=size, replace=False).tolist()
            ds = LeRobotDataset(repo_id="bridge_orig_lerobot", root=dataset_root,
                                episodes=episode_ids, video_backend="pyav")
        # uint8 at dataset resolution; processed to model input only when used
        self.episodes = []
        for pos in range(len(episode_ids)):
            lo = int(ds.episode_data_index["from"][pos])
            hi = int(ds.episode_data_index["to"][pos])
            self.episodes.append([_to_uint8_hwc(ds[i][camera_key]) for i in range(lo, hi)])
        self._rng = np.random.default_rng(seed)
        self.global_step = 0
        self._coherent_ep = None

    def start_episode(self, episode_counter: int) -> None:
        self.global_step = 0
        self._coherent_ep = int(np.random.default_rng(1000 + episode_counter).integers(len(self.episodes)))

    def _clip(self, frames: list[np.ndarray], like: torch.Tensor) -> torch.Tensor:
        vid = np.concatenate([self._process(f) for f in frames], axis=1)  # (C, T, H, W)
        return torch.from_numpy(vid[None]).to(device=like.device, dtype=like.dtype)

    def perstep_clip(self, like: torch.Tensor) -> torch.Tensor:
        idx = self._rng.integers(len(self.episodes), size=self._horizon)
        return self._clip([self.episodes[i][0] for i in idx], like)

    def coherent_clip(self, like: torch.Tensor) -> torch.Tensor:
        ep = self.episodes[self._coherent_ep]
        t = min(self.global_step, len(ep) - 1)
        idx = np.clip(np.arange(t - self._horizon + 1, t + 1), 0, len(ep) - 1)
        return self._clip([ep[i] for i in idx], like)


class VAMSanityInference(VAMInference):
    def __init__(self, *args, out_dir: str, **kwargs):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        self.frames = _FrameSource(self._process_image, self._image_horizon)
        self.model = SanityPipeline(self.model, self.frames)
        self._out_dir = pathlib.Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._episode_counter = -1

    def _flush(self) -> None:
        if self._episode_counter < 0 or not self.model.records:
            return
        path = self._out_dir / f"episode_{self._episode_counter:03d}.npz"
        np.savez_compressed(path, records=np.array(self.model.records, dtype=object),
                            task=self.task_description)
        self.model.records = []

    def reset(self, task_description):
        self._flush()
        super().reset(task_description)
        self._episode_counter += 1
        self.model.start_episode()
        self.frames.start_episode(self._episode_counter)

    def step(self, *args, **kwargs):
        out = super().step(*args, **kwargs)
        self.frames.global_step += 1
        return out

    def close(self) -> None:
        self._flush()
        # pooled real context, reused by closed-loop mean-ablation
        total, n = self.model.real_context_pool()
        if total is not None:
            torch.save({"sum": total.cpu(), "count": n}, self._out_dir / "real_context_pool.pt")
