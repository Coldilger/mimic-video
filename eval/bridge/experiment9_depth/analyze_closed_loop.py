"""Experiment 9, closed loop: success vs. denoising depth k, paired with the
Experiment 1 stop=1 baseline episode by episode (../paired_stats.py: Wilson
CIs, stratified paired bootstrap, exact McNemar).

Latency per decision is taken from the open-loop table in README.md (same
production call, H100 NVL, no CUDA graphs); the closed-loop logs do not
record per-call timing.

    python analyze_closed_loop.py   # writes closed_loop_summary.json + .png
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_stats import load_flags, load_outcomes, paired_report  # noqa: E402

RESULTS = HERE.parent / "SimplerEnv" / "results"
BASE_TAG = "exp1_baseline_stop1_seed0/"
DEPTHS = [0, 1, 2, 4, 8, 16]
# open-loop latency (s) per decision, README.md results table
LATENCY = {0: 0.82, 1: 1.36, 2: 1.89, 4: 2.95, 8: 5.09, 16: 9.30}


def main() -> None:
    base = load_outcomes(RESULTS, BASE_TAG)
    rows = []
    for k in DEPTHS:
        tag = BASE_TAG if k == 1 else f"exp9_stop{k}_seed0/"
        cond = load_outcomes(RESULTS, tag)
        r = paired_report(base, cond)
        flags = load_flags(RESULTS, tag)
        rows.append({
            "k": k,
            "latency_s": LATENCY[k],
            "n": r["n"],
            "successes": r["cond_successes"],
            "wilson95": r["cond_wilson"],
            "diff_vs_k1": r["diff"],
            "diff_ci95": r["diff_ci95"],
            "k1_only": r["discordant_base_only"],
            "k_only": r["discordant_cond_only"],
            "mcnemar_p": r["mcnemar_p"],
            "per_task": {t: d["cond"] for t, d in r["per_task"].items()},
            "flags": {f: sum(v) for f, v in flags.items()},
        })
    (HERE / "closed_loop_summary.json").write_text(json.dumps(rows, indent=1))

    print("| k | latency (s) | success | Wilson 95% | diff vs k=1 [95% CI] | McNemar p | moved correct obj | grasped |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = r["wilson95"]
        dlo, dhi = r["diff_ci95"]
        diff = "—" if r["k"] == 1 else f"{100 * r['diff_vs_k1']:+.1f} pp [{100 * dlo:+.1f}, {100 * dhi:+.1f}]"
        p = "—" if r["k"] == 1 else f"{r['mcnemar_p']:.4f}"
        print(f"| {r['k']} | {r['latency_s']:.2f} | {r['successes']}/{r['n']} = {100 * r['successes'] / r['n']:.1f}% "
              f"| [{100 * lo:.1f}, {100 * hi:.1f}] | {diff} | {p} "
              f"| {r['flags'].get('moved_correct_obj', 0)}/{r['n']} | {r['flags'].get('is_src_obj_grasped', 0)}/{r['n']} |")
    tasks = sorted(rows[0]["per_task"])
    print("\n| k | " + " | ".join(t.replace("InScene-v0", "").replace("Scene-v0", "") for t in tasks) + " |")
    print("|---|" + "---|" * len(tasks))
    for r in rows:
        print(f"| {r['k']} | " + " | ".join(f"{r['per_task'][t]}/24" for t in tasks) + " |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = [r["latency_s"] for r in rows]
    y = [100 * r["successes"] / r["n"] for r in rows]
    err = [[y[i] - 100 * rows[i]["wilson95"][0] for i in range(len(rows))],
           [100 * rows[i]["wilson95"][1] - y[i] for i in range(len(rows))]]
    ax.errorbar(x, y, yerr=err, fmt="o-", color="#2a6fdb", capsize=3)
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(f"k={r['k']}", (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("latency per decision (s, open-loop H100)")
    ax.set_ylabel("success (%), Wilson 95% CI")
    ax.set_title("mimic-video, SimplerEnv Bridge, 96 paired episodes")
    ax.set_ylim(0, 70)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "closed_loop_success_vs_latency.png", dpi=150)


if __name__ == "__main__":
    main()
