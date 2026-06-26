from functools import partial
import argparse
import pathlib
from multiprocessing import Pool

import numpy as np
import safetensors.numpy as st
import tqdm
from cosmos_predict2.data.action.types import S_TO_NS


def get_video_embeddings_file(
    ep: pathlib.Path, action_data_dir: pathlib.Path, video_embeddings_dir: pathlib.Path
) -> pathlib.Path:
    name = "__".join([p for p in ep.relative_to(action_data_dir).with_suffix("").parts if p not in {"raw"}])
    return video_embeddings_dir / f"{name}0.safetensors"


def add_video_embedding(
    ep: pathlib.Path,
    dataset_path: pathlib.Path,
    video_embeddings_dir: pathlib.Path,
    overwrite: bool,
) -> None:

    if not overwrite:
        with st.safe_open(ep, "np") as f:
            if "workspace_rgb_embedding" in f.keys():
                return

    action_data = st.load_file(ep)

    video_embeddings_file = get_video_embeddings_file(ep, dataset_path, video_embeddings_dir)
    video_embeddings_data = st.load_file(video_embeddings_file)

    video_len = video_embeddings_data["video_len"]
    if not len(action_data["workspace_rgb"]) == video_len:
        msg = f"Length of video in mp4 and safetensors does not match for {ep}."
        raise ValueError(msg)

    padding_idx = video_embeddings_data["video_embeddings_idxs"] - video_len + 1
    extrapolated_timestamp = (
        action_data["workspace_rgb_timestamps"][-1] + S_TO_NS * padding_idx.clip(min=0) / video_embeddings_data["fps"]
    ).astype(np.uint64)
    embedding_timestamps = np.where(
        padding_idx > 0,
        extrapolated_timestamp,
        action_data["workspace_rgb_timestamps"][video_embeddings_data["video_embeddings_idxs"].clip(max=video_len - 1)],
    )

    action_data["workspace_rgb_embedding"] = video_embeddings_data["video_embeddings"]
    action_data["workspace_rgb_embedding_timestamps"] = embedding_timestamps

    action_data["num_conditional_frames"] = np.array([video_embeddings_data["num_conditional_frames"]])
    action_data["num_conditional_frames_timestamps"] = np.array([0], dtype=np.uint64)

    st.save_file(action_data, ep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=pathlib.Path, required=True)
    p.add_argument("--video-embeddings-path", type=pathlib.Path, required=True)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing video embeddings.")
    p.add_argument("--num-workers", type=int, default=1)
    args = p.parse_args()

    paths = args.dataset_path.glob("**/*.safetensors")

    with Pool(processes=args.num_workers) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(
                partial(
                    add_video_embedding,
                    dataset_path=args.dataset_path,
                    video_embeddings_dir=args.video_embeddings_path,
                    overwrite=args.overwrite,
                ),
                paths,
            ),
            desc="Transferring video embeddings.",
        ):
            pass


if __name__ == "__main__":
    main()
