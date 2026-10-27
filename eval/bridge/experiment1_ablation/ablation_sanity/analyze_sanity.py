"""Summarize ablation-sanity records (records/<TASK>/episode_*.npz) into one
table per question. CPU only:

    python analyze_sanity.py records/

Per condition, over all replans of all tasks:
- ctx after ctx_norm: per-token std and mean pairwise cosine between tokens
  (zero-ablation should show cos == 1: every token identical);
- cross-attention entropy, averaged over decoder blocks (1.0 == uniform);
- decoded action: relative L2 distance to the real-context action on the
  same inputs, translation magnitude, rotation validity (mean |column norm
  - 1| of the 6D rotation; 0 for a valid rotation), gripper mean, NaN count.
"""

import pathlib
import sys

import numpy as np

CONDITIONS = ("real", "zero", "mean", "noise", "stale", "shuffle_perstep", "shuffle_coherent")


def main(root: pathlib.Path) -> None:
    rows = {c: [] for c in CONDITIONS}
    n_replans = 0
    for path in sorted(root.glob("*/episode_*.npz")):
        for rec in np.load(path, allow_pickle=True)["records"]:
            n_replans += 1
            real = rec["conditions"]["real"]["action"]
            real_norm = np.linalg.norm(real) + 1e-8
            for name, c in rec["conditions"].items():
                a = c["action"]
                rows[name].append({
                    "normed_token_std": c["ctx"]["normed_token_std"],
                    "pairwise_cos": c["ctx"]["normed_mean_pairwise_cos"],
                    "raw_token_norm": c["ctx"]["raw_token_norm"],
                    "attn_entropy": float(np.mean(c["attn_entropy_per_block"])),
                    "rel_l2_to_real": c["l2_to_real"] / real_norm,
                    "translation": float(np.linalg.norm(a[:, :3], axis=-1).mean()),
                    "gripper": float(a[:, 9].mean()),
                    # rot6 = two rotation-matrix columns; both should have unit norm
                    "rot6_norm_err": float(np.abs(np.linalg.norm(a[:, 3:6], axis=-1) - 1).mean()
                                           + np.abs(np.linalg.norm(a[:, 6:9], axis=-1) - 1).mean()) / 2,
                    "nan": c["action_nan"],
                })
    print(f"replans: {n_replans}  episodes: {len(list(root.glob('*/episode_*.npz')))}\n")
    cols = ["raw_token_norm", "normed_token_std", "pairwise_cos", "attn_entropy",
            "rel_l2_to_real", "translation", "rot6_norm_err", "gripper", "nan"]
    print("| condition | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for name in CONDITIONS:
        r = rows[name]
        if not r:
            continue
        vals = []
        for k in cols:
            x = np.array([d[k] for d in r], dtype=float)
            vals.append(f"{int(x.sum())}" if k == "nan" else f"{x.mean():.3f} ± {x.std():.3f}")
        print(f"| {name} | {len(r)} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "records"))
