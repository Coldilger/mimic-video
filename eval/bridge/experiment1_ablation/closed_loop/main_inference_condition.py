"""Closed-loop entry point for Experiment 1 conditions, selected by the
CONDITION environment variable: baseline | zero | shuffle_perstep | mean |
coherent | stale. Mirrors SimplerEnv/simpler_env/main_inference.py exactly, swapping
in the matching VAMInference subclass. Nothing under SimplerEnv/ is touched.
"""

import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, "/home/3295540/mimic-video/eval/bridge/SimplerEnv")
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402


def policy_class(condition: str):
    if condition == "baseline":
        from simpler_env.policies.vam.video_action_model import VAMInference

        return lambda *a: VAMInference(*a, is_hil=False)
    if condition == "zero":
        from vam_ablated_inference import VAMAblatedInference

        return VAMAblatedInference
    if condition == "shuffle_perstep":
        from vam_shuffled_inference import VAMShuffledInference

        return VAMShuffledInference
    from vam_condition_inference import VAMCoherentShuffledInference, VAMMeanInference, VAMStaleInference

    return {"mean": VAMMeanInference, "coherent": VAMCoherentShuffledInference, "stale": VAMStaleInference}[condition]


if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ""
    model = policy_class(os.environ["CONDITION"])(
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
