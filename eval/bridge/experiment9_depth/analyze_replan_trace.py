"""Summarise traces/*.jsonl written by vam_replan_trace.py: how far ahead of
the arm the commanded target is, and how fast the arm actually moves.

    python analyze_replan_trace.py [name-substring ...]
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def summarise(path: pathlib.Path) -> dict:
    eps = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        eps[r["ep"]].append(r)
    lead, speed, path_len, net, closed = [], [], [], [], []
    for rows in eps.values():
        obs = np.array([r["obs"] for r in rows])
        tgt = np.array([r["target"] for r in rows])
        lead.append(np.linalg.norm(tgt - obs, axis=1).mean())
        steps = np.linalg.norm(np.diff(obs, axis=0), axis=1)
        speed.append(steps.mean())
        path_len.append(steps.sum())
        net.append(np.linalg.norm(obs[-1] - obs[0]))
        closed.append(np.mean([r["grip_cmd"] < 0 for r in rows]))
    return {"episodes": len(eps), "steps": sum(len(v) for v in eps.values()),
            "lead_cm": 100 * float(np.mean(lead)), "speed_cm_per_step": 100 * float(np.mean(speed)),
            "path_cm": 100 * float(np.mean(path_len)), "net_cm": 100 * float(np.mean(net)),
            "gripper_closed_frac": float(np.mean(closed))}


if __name__ == "__main__":
    keys = sys.argv[1:]
    print("| trace | episodes | target lead over arm (cm) | arm speed (cm/step) | path per episode (cm) | net displacement (cm) | gripper-close commands |")
    print("|---|---|---|---|---|---|---|")
    for p in sorted((HERE / "traces").glob("*.jsonl")):
        if keys and not any(k in p.name for k in keys):
            continue
        s = summarise(p)
        print(f"| {p.stem} | {s['episodes']} | {s['lead_cm']:.2f} | {s['speed_cm_per_step']:.2f} | "
              f"{s['path_cm']:.1f} | {s['net_cm']:.1f} | {100 * s['gripper_closed_frac']:.0f}% |")
