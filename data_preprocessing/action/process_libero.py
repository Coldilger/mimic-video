import argparse
import pathlib
from functools import partial
from multiprocessing import Pool

import h5py
import numpy as np
import safetensors.numpy as st
import tqdm
from scipy.spatial.transform import Rotation

NS_PER_SEC = 1_000_000_000


def extract_name(h5_stem: str) -> str:
    # rename black bowl to just bowl bc it isn't black
    return h5_stem.replace("_", " ").replace("black bowl", "bowl") + "."


def _convert(
    h5_path: pathlib.Path, out_dir: pathlib.Path, fps: int, overwrite: bool
) -> str:
    with h5py.File(h5_path, "r") as f:
        demos: list[str] = list(f["data"].keys())
        for demo in demos:
            stem = h5_path.stem.removesuffix("_demo")
            out_path = (
                out_dir
                / h5_path.parent.name
                / f"{stem}_{demo.removeprefix('demo_')}.safetensors"
            )
            if out_path.exists() and not overwrite:
                return f"skip (exists): {out_path}"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            g = f["data"][demo]
            obs = g["obs"]

            agent = obs["agentview_rgb"][()]  # (T,H,W,3)
            ee_pos = obs["ee_pos"][()].astype(np.float32)  # (T,3) xyz
            ee_ori = obs["ee_ori"][()].astype(np.float32)  # (T,4) quaternion
            actions = g["actions"][()].astype(np.float32)  # (T,7) see below
            gripper = obs[
                "gripper_states"
            ][
                ()
            ][
                :, 0
            ].astype(
                np.float32
            )  # (T,1), second dimension is mostly negative of first dimension. goes between 0.001 and 0.04

            # actions are
            # - xyz delta (goal - current proprio). note that goal is usually not achieved.
            # - rotation axisangle left delta (goal * cur_proprio.T). note that goal is usually not achieved.
            # - gripper binary: -1 is open and +1 is closed
            # ie not internal deltas, don't make sense on their own!

            T = agent.shape[0]
            # Synthesize timestamps from FPS (nanoseconds), must match T
            assert fps > 0, "fps must be > 0"
            dt_ns = NS_PER_SEC / fps
            timestamps = (np.arange(T) * dt_ns).astype(np.uint64)

            res = {
                "language_instruction": np.frombuffer(
                    extract_name(stem).encode("utf-8"), dtype=np.uint8
                )[None],
                "language_instruction_timestamps": np.array([0], dtype=np.uint64),
                "workspace_rgb": agent,
                "workspace_rgb_timestamps": timestamps,
                "eef_pos_ref_delta_lowdim": actions[:, :3],
                "eef_pos_ref_delta_lowdim_timestamps": timestamps.copy(),
                "eef_rot_ref_delta_lowdim": Rotation.from_rotvec(
                    actions[:, 3:6]
                ).as_matrix(),
                "eef_rot_ref_delta_lowdim_timestamps": timestamps.copy(),
                "gripper_action_lowdim": actions[:, 6:],
                "gripper_action_lowdim_timestamps": timestamps.copy(),
                "eef_pos_lowdim": ee_pos,
                "eef_pos_lowdim_timestamps": timestamps.copy(),
                "eef_rot_lowdim": ee_ori,
                "eef_rot_lowdim_timestamps": timestamps.copy(),
                "gripper_lowdim": gripper,
                "gripper_lowdim_timestamps": timestamps.copy(),
            }
            st.save_file(res, out_path)

    return f"ok: {h5_path}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Folder with regenerated LIBERO .hdf5 files",
    )
    ap.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Folder to write per-demo .safetensors files",
    )
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Used to artificially create timestamps (ns)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = sorted(
        list(args.input_dir.glob("**/*.hdf5")) + list(args.input_dir.glob("**/*.h5"))
    )
    assert len(h5_paths) > 0, f"No HDF5 files found in {args.input_dir}"

    with Pool(processes=args.num_workers) as pool:
        for msg in tqdm.tqdm(
            pool.imap_unordered(
                partial(
                    _convert,
                    out_dir=args.output_dir,
                    fps=args.fps,
                    overwrite=args.overwrite,
                ),
                iterable=h5_paths,
            ),
            total=len(h5_paths),
            desc="libero h5 -> safetensors",
        ):
            print(msg)


if __name__ == "__main__":
    main()
