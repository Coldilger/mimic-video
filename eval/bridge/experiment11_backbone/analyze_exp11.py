"""Experiment 11 (mimic-video): deployed pair vs the authors' generic-Cosmos
pair, 96 episodes paired by (task, episode) with ../paired_stats.py.
Result dirs are matched by exact name (<task>_<tag>), so tags that merely
contain the baseline tag are not picked up.

    python analyze_exp11.py     # prints markdown, writes exp11_summary.json
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_stats import _FLAG, _NAME, paired_report  # noqa: E402

RESULTS = HERE.parent / "SimplerEnv" / "results" / "mimic_exp1_closed_loop"
TAGS = {"deployed (Bridge video finetune)": "exp1_baseline_stop1_seed0",
        "generic Cosmos backbone": "exp1_baseline_pretrainedbackbone_stop1_seed0"}


def load(tag: str):
    out, flags = {}, defaultdict(lambda: defaultdict(int))
    for d in RESULTS.rglob(f"*_{tag}"):
        if not d.is_dir() or d.name.split("_", 1)[1] != tag:
            continue
        task = d.name.split("_", 1)[0]
        for mp4 in d.rglob("*.mp4"):
            m = _NAME.match(mp4.name)
            if not m:
                continue
            out[(task, 0, int(m.group(2)))] = m.group(1) == "success"
            for name, val in _FLAG.findall(m.group(3) or ""):
                flags[task][name] += val == "True"
    return out, flags


def main() -> None:
    (base, bflags), (cond, cflags) = (load(t) for t in TAGS.values())
    r = paired_report(base, cond)
    lo, hi = r["diff_ci95"]
    tasks = sorted({t for t, _, _ in base})
    summary = {"n": r["n"], "deployed": r["base_successes"], "generic": r["cond_successes"],
               "diff": r["diff"], "diff_ci95": r["diff_ci95"], "mcnemar_p": r["mcnemar_p"],
               "lost_gained": [r["discordant_base_only"], r["discordant_cond_only"]],
               "per_task": {t: {"deployed": sum(v for (tt, _, _), v in base.items() if tt == t),
                                "generic": sum(v for (tt, _, _), v in cond.items() if tt == t),
                                "n": sum(1 for (tt, _, _) in cond if tt == t),
                                "flags_deployed": dict(bflags[t]), "flags_generic": dict(cflags[t])} for t in tasks}}
    (HERE / "exp11_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"| pair | success | diff vs deployed [95% CI] | McNemar p | lost / gained |\n|---|---|---|---|---|")
    print(f"| deployed | {r['base_successes']}/{r['n']} | — | — | — |")
    print(f"| generic Cosmos | {r['cond_successes']}/{r['n']} | {100 * r['diff']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] "
          f"| {r['mcnemar_p']:.4g} | {r['discordant_base_only']} / {r['discordant_cond_only']} |")
    print("\n| task | deployed | generic | flags deployed | flags generic |\n|---|---|---|---|---|")
    for t, v in summary["per_task"].items():
        print(f"| {t} | {v['deployed']}/{v['n']} | {v['generic']}/{v['n']} | {v['flags_deployed']} | {v['flags_generic']} |")


if __name__ == "__main__":
    main()
