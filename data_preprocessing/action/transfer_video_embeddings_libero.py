import argparse
import pathlib

import numpy as np
import safetensors.numpy as st
import tqdm
from cosmos_predict2.data.action.types import S_TO_NS


def get_video_embeddings_file(ep: pathlib.Path, video_embeddings_dir: pathlib.Path) -> pathlib.Path:
    ep_name, idx = ep.stem.rsplit("_", 1)
    return video_embeddings_dir / f"{ep.parent.name}_{ep_name}_demo_demo_{idx}_agentview_rgb.safetensors"


def add_video_embedding(ep: pathlib.Path, video_embeddings_file: pathlib.Path) -> None:
    action_data = st.load_file(ep)
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
    args = p.parse_args()

    paths = args.dataset_path.glob("**/*.safetensors")

    for path in tqdm.tqdm(paths, desc="Transferring video embeddings."):
        add_video_embedding(path, get_video_embeddings_file(path, args.video_embeddings_path))


if __name__ == "__main__":
    main()
