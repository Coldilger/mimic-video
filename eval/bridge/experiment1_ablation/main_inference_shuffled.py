"""Closed-loop entry point for Experiment 1 variant 2 (mimic-video, shuffle
across episodes). Mirrors SimplerEnv/simpler_env/main_inference.py exactly,
swapping in VAMShuffledInference. Nothing under SimplerEnv/ is touched.
"""

import os
import sys
import pathlib

import numpy as np

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

from vam_shuffled_inference import VAMShuffledInference  # noqa: E402


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""

    model = VAMShuffledInference(
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
