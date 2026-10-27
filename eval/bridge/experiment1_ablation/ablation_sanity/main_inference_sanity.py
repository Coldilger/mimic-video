"""Closed-loop entry point for the Experiment 1 ablation sanity check.
Mirrors SimplerEnv/simpler_env/main_inference.py exactly, swapping in
VAMSanityInference (drives a normal baseline rollout; see sanity_pipeline.py).
Records go to $SANITY_OUT_DIR. Nothing under SimplerEnv/ is touched.
"""

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

from vam_sanity_inference import VAMSanityInference  # noqa: E402


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""

    model = VAMSanityInference(
        args.vam_experiment_name,
        args.vam_video_model_path,
        args.vam_action_model_path,
        args.vam_dataset_statistics_path,
        args.vam_img_horizon,
        args.vam_lowdim_horizon,
        args.vam_stop_video_denoising_step,
        args.vam_num_execute_actions,
        out_dir=os.environ["SANITY_OUT_DIR"],
    )
    success_arr = maniskill2_evaluator(model, args)
    model.close()
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
