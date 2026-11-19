"""Closed-loop entry point for Experiment 8 (mimic-video, instruction-foresight
conflict; see vam_conflict_inference.py). Mirrors
../experiment9_depth/main_inference_depth.py with VAMConflictInference.
Env: TASK, CONFLICT_MODE (probe|cl), CONFLICT_OUT (probe), CONFLICT_OBS /
CONFLICT_FUT / CONFLICT_ALT (cl).
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
    from vam_conflict_inference import VAMConflictInference

    model = VAMConflictInference(
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
    model.close()
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
