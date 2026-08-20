"""Live oracle probe for mimic-video: same oracle mechanism as
oracle_offline_probe.py (VAMInference's is_hil=True + ingest_video(), used
completely unmodified on a second, oracle-only model instance), sourced
from a live, randomized SimplerEnv-Bridge rollout instead of replaying
bridge_orig_lerobot -- the exact dataset this checkpoint was fine-tuned on,
with no held-out split. Verbatim recall of a specific trajectory is
impossible here: each episode's object placement is randomized by
SimplerEnv itself (the same mechanism Experiment 1's own closed-loop eval
already uses), so this tests the oracle mechanism against scenes that
cannot literally be memorized (distributional overfitting to the task
family is a separate, softer question this doesn't rule out).

Mirrors the offline probe's own metric exactly: the oracle's predicted
ABSOLUTE position (composing its predicted delta with the current pose,
same as VAMInference.step() does internally) against the REAL achieved
position one step later.

Only over SUCCESSFUL episodes (filtered after maniskill2_evaluator returns
success_arr). Without this filter, "what actually happened next" during a
FAILED episode is not a meaningful target -- on the ~50-89% of episodes the
unmodified policy fails (task-dependent), what it achieved wasn't good, so
an oracle prediction that *diverges* from that outcome could be an
improvement, not an error, and L1-against-a-failed-trajectory can't tell
the two apart (raised by the user, 2026-08-19 -- correct, same reasoning as
F1-VLA's copy of this file). Restricting to successful episodes makes the
reference "a real outcome that actually worked," the closed analogue of why
the offline probe's own reference (expert human teleop demonstrations) is
meaningful in the first place. Not a full fix -- not every moment inside a
successful episode is necessarily optimal -- but a real improvement over
comparing against unfiltered outcomes.

Runs a normal closed-loop rollout with the REAL (non-oracle) VAMInference
driving the robot -- behavior is completely unaffected, pure side
computation, same principle as F1-VLA's copy of this script and
server_policy_oracle_probe.py's approach for LDA-1B. Wraps step() (called
every control tick, not just every replan, since mimic's oracle needs a
FUTURE_HORIZON=12-frame video clip per sample -- a bigger rolling buffer
than F1's single-frame case) and reset() (called once per episode by
maniskill2_evaluator, wrapped purely to count episode boundaries for the
success-filter join at the end). Throttled to roughly one oracle sample
every ~12 ticks (matching the offline probe's own samples_per_episode
density) -- mimic's own world-model forward pass costs ~10s/call
(Experiment 3's own measurement), so querying it every tick would be far
too slow.
"""

import os
import sys
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402
from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402

FUTURE_HORIZON = 12
SAMPLE_STRIDE = 12


class PoseProxy:
    def __init__(self, p, q):
        self.p = p
        self.q = q


def add_live_oracle_probe(model: VAMInference, oracle_model: VAMInference):
    original_step = model.step
    original_reset = model.reset

    tick_buf = deque(maxlen=FUTURE_HORIZON + 1)
    records = []  # each: dict(oracle_l1, zero_l1, episode_idx)
    tick_count = {"n": 0}
    episode_idx = {"n": -1}

    def wrapped_reset(task_description):
        episode_idx["n"] += 1
        tick_buf.clear()  # never compare across an episode boundary
        return original_reset(task_description)

    def wrapped_step(image, task_description, ee_pose_proprio, gripper_proprio):
        tick_buf.append(
            (image.copy(), ee_pose_proprio.p.copy(), ee_pose_proprio.q.copy(),
             float(gripper_proprio), task_description)
        )
        tick_count["n"] += 1

        if len(tick_buf) == tick_buf.maxlen and tick_count["n"] % SAMPLE_STRIDE == 0:
            image_t, p_t, q_t, g_t, task_t = tick_buf[0]
            future_frames = np.stack([e[0] for e in list(tick_buf)[1:]])
            p_next, g_next = tick_buf[1][1], tick_buf[1][3]

            oracle_model.reset(task_t)
            oracle_model.ingest_video(future_frames)
            oracle_model.step(image_t, task_t, PoseProxy(p=p_t, q=q_t), g_t)

            cur_rot = Rotation.from_quat(q_t, scalar_first=True)
            ee_rot_t = cur_rot.as_matrix()[:2].reshape(6)
            lowdim_t = np.concatenate([p_t, ee_rot_t, [g_t]]).astype(np.float64)
            obs_T = oracle_model._pose_from_lowdim(lowdim_t)
            pred_delta_T = oracle_model._pose_from_lowdim(oracle_model.action_buffer[0].astype(np.float64))
            pred_abs_T = obs_T @ pred_delta_T
            pred_pos = pred_abs_T[:3, 3]

            oracle_l1 = float(np.abs(pred_pos - p_next).mean())
            zero_l1 = float(np.abs(p_t - p_next).mean())  # "arm doesn't move" baseline
            records.append(dict(oracle_l1=oracle_l1, zero_l1=zero_l1, episode_idx=episode_idx["n"]))
            print(f"LIVE_ORACLE_SAMPLE n={len(records)} ep={episode_idx['n']} "
                  f"oracle_l1={oracle_l1:.5f} zero_l1={zero_l1:.5f}", flush=True)

        return original_step(image, task_description, ee_pose_proprio, gripper_proprio)

    model.reset = wrapped_reset
    model.step = wrapped_step
    return records


def report(records, label):
    if not records:
        print(f"{label}: no samples recorded")
        return
    o = np.array([r["oracle_l1"] for r in records])
    z = np.array([r["zero_l1"] for r in records])
    print(f"{label} n={len(o)} oracle_mean={o.mean():.5f} oracle_sd={o.std():.5f} zero_mean={z.mean():.5f}")


if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ""

    model = VAMInference(
        args.vam_experiment_name,
        args.vam_video_model_path,
        args.vam_action_model_path,
        args.vam_dataset_statistics_path,
        args.vam_img_horizon,
        args.vam_lowdim_horizon,
        args.vam_stop_video_denoising_step,
        args.vam_num_execute_actions,
        is_hil=False,
    )
    oracle_model = VAMInference(
        args.vam_experiment_name,
        args.vam_video_model_path,
        args.vam_action_model_path,
        args.vam_dataset_statistics_path,
        args.vam_img_horizon,
        args.vam_lowdim_horizon,
        args.vam_stop_video_denoising_step,
        args.vam_num_execute_actions,
        is_hil=True,
    )
    records = add_live_oracle_probe(model, oracle_model)

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    print()
    report(records, "LIVE_ORACLE_FINAL_ALL")
    success_arr = np.asarray(success_arr)
    filtered = [r for r in records if r["episode_idx"] < len(success_arr) and success_arr[r["episode_idx"]]]
    report(filtered, "LIVE_ORACLE_FINAL_SUCCESSFUL_EPISODES_ONLY")
