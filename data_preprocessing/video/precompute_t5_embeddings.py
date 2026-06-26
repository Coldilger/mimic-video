# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pathlib

import tqdm
import safetensors.torch as st

from imaginaire.auxiliary.text_encoder import (
    CosmosT5TextEncoder,
    CosmosT5TextEncoderConfig,
)
from imaginaire.constants import T5_MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute T5 embeddings for text prompts"
    )
    parser.add_argument(
        "--dataset-path", type=pathlib.Path, help="Root path to the dataset"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=T5_MODEL_DIR,
        help="Directory to cache the T5 model",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    metas_dir = args.dataset_path / "metas"
    metas_list = sorted(metas_dir.glob("*.txt"), key=str)[args.rank :: args.world_size]

    out_dir = args.dataset_path / "language_embeddings"
    out_dir.mkdir(exist_ok=True)

    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    for meta_filename in tqdm.tqdm(
        metas_list, total=len(metas_list), desc="Precomputing t5 embeddings."
    ):
        out_file = out_dir / f"{meta_filename.stem}.safetensors"
        if out_file.exists() and not args.overwrite:
            continue

        with open(meta_filename) as fp:
            prompt = fp.read().strip()

        encoded_text = encoder.encode_prompts(prompt, max_length=512, return_mask=False)

        st.save_file({"encoded_text": encoded_text[0]}, out_file)


if __name__ == "__main__":
    main()
