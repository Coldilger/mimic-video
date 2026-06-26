import os
import re
import pathlib

import tyro
import tqdm


def extract_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]

    # remove suffix
    base = re.sub(r"_demo_demo_\d+_[a-z_]+_rgb$", "", base)
    # remove suite prefix
    base = (
        base.removeprefix("libero_10_")
        .removeprefix("libero_90_")
        .removeprefix("libero_goal_")
        .removeprefix("libero_object_")
        .removeprefix("libero_spatial_")
    )
    # remove env prefix
    m = re.search(r"[a-z]", base)
    base = base[m.start() :]  # ty:ignore[unresolved-attribute]
    # remove underscores
    base = base.replace("_", " ")
    # rename black bowl to just bowl bc it isn't black
    base = base.replace("black bowl", "bowl")
    return base + "."


def main(dataset_dir: pathlib.Path):
    video_dir = dataset_dir / "video"
    out_meta = dataset_dir / "metas"
    out_meta.mkdir(parents=True, exist_ok=True)
    for mp4 in tqdm.tqdm(video_dir.iterdir()):
        if mp4.suffix != ".mp4":
            print(f"Skipping {mp4}.")
        meta = out_meta / mp4.relative_to(video_dir).with_suffix(".txt")
        meta.write_text(extract_name(meta.stem))

    print(f"Done. Wrote to {out_meta}")


if __name__ == "__main__":
    tyro.cli(main)
