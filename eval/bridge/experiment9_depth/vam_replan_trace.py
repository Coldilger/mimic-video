"""Experiment 9, replanning-rate diagnostic.

Why does executing only 1 or 2 actions per decision give 0/96? The production
policy (`SimplerEnv/simpler_env/policies/vam/video_action_model.py`) turns the
plan into absolute target poses anchored at the *observed* end-effector pose
of the replanning step, and the simulator's arm controller
(`arm_pd_ee_target_delta_pose`) moves towards a commanded target. With
`num_execute_actions=1` the commanded target is always plan index 0 — one
plan step ahead of where the arm currently is.

`VAMReplanTrace` is the production policy plus:
  * a per-step trace (observed position, commanded target position) written
    to `traces/<TRACE_NAME>.jsonl`, so the target's lead over the arm and
    the arm's speed can be compared between replanning rates;
  * optional `LEAD=n`: the plan returned by the model is shifted by n steps
    before it is used, so plan index 0 holds the pose originally at index n
    (the gripper channel is shifted too only with `LEAD_GRIPPER=1`). With `num_execute_actions=1` and
    `LEAD=2` the policy still replans at every step but commands a target
    three plan steps ahead — the average lead of the deployed setting
    (5 executed actions: indices 0..4).
"""

from __future__ import annotations

import json
import os
import pathlib

import torch
from simpler_env.policies.vam.video_action_model import VAMInference

HERE = pathlib.Path(__file__).resolve().parent


class LeadShiftPipeline(torch.nn.Module):
    def __init__(self, pipeline, lead: int, shift_gripper: bool = False):
        super().__init__()
        self.pipeline = pipeline
        self.lead = lead
        self.dims = 10 if shift_gripper else 9

    def forward(self, **kwargs):
        actions = self.pipeline(**kwargs)  # (1, H, 10): pos 3, rot6d 6, gripper 1
        if self.lead == 0:
            return actions
        shifted = actions.clone()
        n = actions.shape[1] - self.lead
        shifted[:, :n, : self.dims] = actions[:, self.lead :, : self.dims]
        shifted[:, n:, : self.dims] = actions[:, -1:, : self.dims]
        return shifted


class VAMReplanTrace(VAMInference):
    def __init__(self, *args, **kwargs):
        kwargs["is_hil"] = False
        super().__init__(*args, **kwargs)
        lead = int(os.environ.get("LEAD", "0"))
        if lead:
            self.model = LeadShiftPipeline(self.model, lead, os.environ.get("LEAD_GRIPPER", "0") == "1").cuda()
        out = HERE / "traces"
        out.mkdir(exist_ok=True)
        self._trace = open(out / f"{os.environ['TRACE_NAME']}.jsonl", "w")
        self._episode = -1
        print(f"[trace] lead={lead} gripper_shift={os.environ.get('LEAD_GRIPPER', '0')} exec={self.num_execute_actions}", flush=True)

    def reset(self, task_description):
        super().reset(task_description)
        self._episode += 1

    def step(self, image, task_description, ee_pose_proprio, gripper_proprio):
        replanned = self.action_buffer is None
        obs = [float(x) for x in ee_pose_proprio.p]
        out = super().step(image, task_description, ee_pose_proprio, gripper_proprio)
        self._trace.write(json.dumps({
            "ep": self._episode, "t": self._global_step - 1, "replan": bool(replanned),
            "obs": obs, "target": [float(x) for x in self._last_target_p],
            "grip_cmd": float(out["gripper"][0]),
        }) + "\n")
        self._trace.flush()
        return out
