"""Experiment 8 (mimic-video): summarise the probe records
(records/<task>/episode_NNN.npz, vam_conflict_inference.py, mode probe) and
the closed-loop runs (SimplerEnv result dirs tagged
exp8_cl_obs{A,B}_fut{A,B}_<alt>_stop1_seed0; the obs A / fut A cell is the
production policy, i.e. Experiment 1's exp1_baseline_stop1_seed0).

Probe numbers: mean over episodes of the per-episode mean, 95% bootstrap
over episodes. d = mean over the chunk's 15 steps of the L2 distance between
two 10-dim action rows (position m, 6D rotation, gripper); follow(AB) =
d(AB,AA) / (d(AB,AA) + d(AB,BB)): 0 = the action stays with the prompt of
the observed-frame tokens, 1 = it goes with the prompt of the future tokens.

    python analyze_conflict.py [--records records] [--alt mirror]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_stats import _NAME, paired_report  # noqa: E402

RESULTS = HERE.parent / "SimplerEnv" / "results"
SHORT = {"PutCarrotOnPlateInScene-v0": "Carrot", "PutSpoonOnTableClothInScene-v0": "Spoon",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0": "Stack", "PutEggplantInBasketScene-v0": "Eggplant"}
FLAGS = ["moved_correct_obj", "moved_wrong_obj", "is_src_obj_grasped", "src_on_target"]


def boot(x, n=4000):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ci(t, nd=3):
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def probe(records_dir: pathlib.Path) -> dict:
    out = {}
    for task_dir in sorted(p for p in records_dir.iterdir() if p.is_dir()):
        seed, alts, meta, n_dec = collections.defaultdict(list), collections.defaultdict(lambda: collections.defaultdict(list)), {}, 0
        files = sorted(task_dir.glob("episode_*.npz"))
        for f in files:
            recs = np.load(f, allow_pickle=True)["records"]
            n_dec += len(recs)
            for k in ("d_AA2_AA", "seed_ctx_obs", "seed_ctx_fut"):
                seed[k].append(np.mean([r[k] for r in recs]))
            for name in recs[0]["alts"]:
                meta[name] = recs[0]["alts"][name]["instruction"]
                for k in ("d_BB_AA", "d_AB_AA", "d_BA_BB", "follow_AB", "follow_BA", "ctx_obs_AB", "ctx_fut_AB"):
                    alts[name][k].append(np.mean([r["alts"][name][k] for r in recs]))
        out[SHORT.get(task_dir.name, task_dir.name)] = {
            "episodes": len(files), "decisions": n_dec, "n_tokens": int(recs[0]["n_tokens"]), "n_obs": int(recs[0]["n_obs"]),
            "seed": {k: boot(v) for k, v in seed.items()},
            "alts": {n: {"instruction": meta[n], **{k: boot(v) for k, v in a.items()}} for n, a in alts.items()}}
    return out


def load_exact(stem: str):
    """(task, seed, episode) -> (success, flags) from result dirs named exactly <task>_<stem>."""
    out = {}
    for mp4 in RESULTS.rglob("*.mp4"):
        m = _NAME.match(mp4.name)
        if not m:
            continue
        d = next((p for p in mp4.parents if p.name.endswith("_" + stem) and p.name.split("_")[0] in SHORT), None)
        if d is None:
            continue
        flags = {f: f"_{f}_True" in mp4.name for f in FLAGS}
        out[(d.name.split("_")[0], 0, int(m.group(2)))] = (m.group(1) == "success", flags)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="records")
    ap.add_argument("--alt", default="mirror")
    a = ap.parse_args()
    summary = {}
    rd = HERE / a.records
    if rd.exists():
        summary["probe"] = p = probe(rd)
        print("### Single decisions: noise floor (same prompt, video seed 1 vs 0)\n")
        print("| task | episodes | decisions | tokens (observed) | context change observed / future | d(AA2, AA) |")
        print("|---|---|---|---|---|---|")
        for t, d in p.items():
            s = d["seed"]
            print(f"| {t} | {d['episodes']} | {d['decisions']} | {d['n_tokens']} ({d['n_obs']}) | {s['seed_ctx_obs'][0]:.3f} / {s['seed_ctx_fut'][0]:.3f} | {ci(s['d_AA2_AA'], 4)} |")
        print("\n### Single decisions: prompt A vs. prompt B\n")
        print("| task | B | context change observed / future | d(BB, AA) | d(AB, AA) | d(BA, BB) | follow(AB) | follow(BA) |")
        print("|---|---|---|---|---|---|---|---|")
        for t, d in p.items():
            for n, x in d["alts"].items():
                print(f"| {t} | {n}: \"{x['instruction']}\" | {x['ctx_obs_AB'][0]:.3f} / {x['ctx_fut_AB'][0]:.3f} | {ci(x['d_BB_AA'])} "
                      f"| {ci(x['d_AB_AA'])} | {ci(x['d_BA_BB'])} | {ci(x['follow_AB'])} | {ci(x['follow_BA'])} |")

    cells = {("A", "A"): "exp1_baseline_stop1_seed0"}
    for o, f in (("A", "B"), ("B", "A"), ("B", "B")):
        cells[(o, f)] = f"exp8_cl_obs{o}_fut{f}_{a.alt}_stop1_seed0"
    res = {c: load_exact(stem) for c, stem in cells.items()}
    if any(res[c] for c in res if c != ("A", "A")):
        print(f"\n### Closed loop (alternative: {a.alt})\n")
        print("| observed tokens from | future tokens from | episodes | success (task A) | moved A's object | moved the other object | grasped A's object |")
        print("|---|---|---|---|---|---|---|")
        summary["closed_loop"] = {}
        for (o, f), r in res.items():
            if not r:
                continue
            n = len(r)
            cnt = {k: sum(v[1][k] for v in r.values()) for k in FLAGS}
            succ = sum(v[0] for v in r.values())
            summary["closed_loop"][f"obs{o}_fut{f}"] = {"n": n, "success": succ, **cnt}
            print(f"| {o} | {f} | {n} | {succ}/{n} | {cnt['moved_correct_obj']}/{n} | {cnt['moved_wrong_obj']}/{n} | {cnt['is_src_obj_grasped']}/{n} |")
        print("\n| comparison | base | cond | diff [95% CI] | McNemar p |")
        print("|---|---|---|---|---|")
        for b, c in ((("A", "A"), ("A", "B")), (("B", "B"), ("B", "A")), (("A", "A"), ("B", "B"))):
            if res[b] and res[c]:
                r = paired_report({k: v[0] for k, v in res[b].items()}, {k: v[0] for k, v in res[c].items()})
                lo, hi = r["diff_ci95"]
                print(f"| obs {b[0]}/fut {b[1]} → obs {c[0]}/fut {c[1]} | {r['base_successes']}/{r['n']} | {r['cond_successes']}/{r['n']} "
                      f"| {100 * r['diff']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] | {r['mcnemar_p']:.4f} |")
    (HERE / "conflict_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
