"""Closed-loop entry point for Experiment 9. CONDITION environment variable:
baseline (the production policy; depth and replanning come from the usual
--vam-* flags) | seedavg (see vam_depth_inference.py, needs VIDEO_SEEDS) | trace (see
vam_replan_trace.py, needs TRACE_NAME, optional LEAD).
Mirrors SimplerEnv/simpler_env/main_inference.py.
"""

import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, str(HERE))

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ""
    condition = os.environ["CONDITION"]
    if condition == "baseline":
        from simpler_env.policies.vam.video_action_model import VAMInference

        cls = lambda *a: VAMInference(*a, is_hil=False)  # noqa: E731
    elif condition == "seedavg":
        from vam_depth_inference import VAMSeedAverageInference as cls
    elif condition == "trace":
        from vam_replan_trace import VAMReplanTrace as cls
    else:
        raise SystemExit(f"unknown CONDITION: {condition}")
    model = cls(
        args.vam_experiment_name,
        args.vam_video_model_path,
        args.vam_action_model_path,
        args.vam_dataset_statistics_path,
        args.vam_img_horizon,
        args.vam_lowdim_horizon,
        args.vam_stop_video_denoising_step,
        args.vam_num_execute_actions,
    )
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
