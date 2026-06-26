from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
import torch
from functools import partial
import argparse
import pathlib
from multiprocessing import Pool

import numpy as np
import safetensors.numpy as st_np
import safetensors.torch as st_torch
import tqdm


def get_language_embeddings_file(
    ep: pathlib.Path,
    action_data_dir: pathlib.Path,
    language_embeddings_dir: pathlib.Path,
) -> pathlib.Path:
    name = "__".join([
        p
        for p in ep.relative_to(action_data_dir).with_suffix("").parts
        if p not in {"raw"}
    ])
    return language_embeddings_dir / f"{name}.safetensors"


def add_language_embedding(
    ep: pathlib.Path,
    dataset_path: pathlib.Path,
    language_embeddings_dir: pathlib.Path,
    overwrite: bool,
) -> None:

    if not overwrite:
        with st_np.safe_open(ep, "np") as f:
            if "language_embedding" in f.keys():
                return

    action_data = st_np.load_file(ep)

    language_embeddings_file = get_language_embeddings_file(
        ep, dataset_path, language_embeddings_dir
    )
    language_embedding = st_torch.load_file(language_embeddings_file)["encoded_text"]

    n_tokens = language_embedding.shape[0]
    if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
        language_embedding = torch.cat(
            [
                language_embedding,
                torch.zeros(
                    (
                        CosmosTextEncoderConfig.NUM_TOKENS - n_tokens,
                        CosmosTextEncoderConfig.EMBED_DIM,
                    ),
                    dtype=language_embedding.dtype,
                    device=language_embedding.device,
                ),
            ],
            dim=0,
        )

    action_data["language_embedding"] = language_embedding.to(torch.float32).numpy()[
        None
    ]
    action_data["language_embedding_timestamps"] = np.array([0], dtype=np.uint64)

    st_np.save_file(action_data, ep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=pathlib.Path, required=True)
    p.add_argument("--language-embeddings-path", type=pathlib.Path, required=True)
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing video embeddings."
    )
    p.add_argument("--num-workers", type=int, default=1)
    args = p.parse_args()

    paths = args.dataset_path.glob("**/*.safetensors")

    with Pool(processes=args.num_workers) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(
                partial(
                    add_language_embedding,
                    dataset_path=args.dataset_path,
                    language_embeddings_dir=args.language_embeddings_path,
                    overwrite=args.overwrite,
                ),
                paths,
            ),
            desc="Transferring language embeddings.",
        ):
            pass


if __name__ == "__main__":
    main()
