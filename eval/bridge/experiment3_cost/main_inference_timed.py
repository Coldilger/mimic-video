"""Closed-loop eval entrypoint for Experiment 3 (mimic-video): measures real
per-replan wall-clock latency for each of the three conditions already
built for Experiment 1 (baseline, ablated, shuffled), applying
timing_wrapper.add_timing to whichever class --vam-variant selects. Mirrors
../main_inference.py / ../experiment1_ablation/main_inference_ablated.py /
main_inference_shuffled.py exactly otherwise -- same real SimplerEnv
rollout, nothing about the model's actual behavior changes.
"""

import argparse
import os
import statistics
import sys

import numpy as np

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

REPO_ROOT = "/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video"
sys.path.insert(0, os.path.join(REPO_ROOT, "eval/bridge/experiment1_ablation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simpler_env.policies.vam.video_action_model import VAMInference  # noqa: E402

from vam_ablated_inference import VAMAblatedInference  # noqa: E402
from vam_shuffled_inference import VAMShuffledInference  # noqa: E402

from timing_wrapper import add_timing  # noqa: E402

add_timing(VAMInference)
add_timing(VAMAblatedInference)
add_timing(VAMShuffledInference)

VARIANTS = {
    "baseline": VAMInference,
    "ablated": VAMAblatedInference,
    "shuffled": VAMShuffledInference,
}


def parse_vam_variant_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vam-variant", type=str, required=True, choices=list(VARIANTS.keys()))
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


if __name__ == "__main__":
    variant_args, remaining_argv = parse_vam_variant_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_args()

    os.environ["DISPLAY"] = ""

    cls = VARIANTS[variant_args.vam_variant]
    # VAMInference's own constructor takes is_hil; the two Experiment 1
    # subclasses set it internally and reject it being passed again.
    ctor_kwargs = dict(
        experiment_name=args.vam_experiment_name,
        video_model_path=args.vam_video_model_path,
        action_model_path=args.vam_action_model_path,
        dataset_statistics_path=args.vam_dataset_statistics_path,
        img_horizon=args.vam_img_horizon,
        lowdim_horizon=args.vam_lowdim_horizon,
        stop_video_denoising_step=args.vam_stop_video_denoising_step,
        num_execute_actions=args.vam_num_execute_actions,
    )
    if cls is VAMInference:
        ctor_kwargs["is_hil"] = False

    model = cls(**ctor_kwargs)
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    lat = getattr(model, "_chunk_latencies_s", [])
    if lat:
        lat_ms = sorted(x * 1000 for x in lat)
        n = len(lat_ms)
        median = statistics.median(lat_ms)
        p95 = lat_ms[int(0.95 * (n - 1))]
        mean = statistics.mean(lat_ms)
        print(f"LATENCY variant={variant_args.vam_variant} n={n} mean_ms={mean:.1f} "
              f"median_ms={median:.1f} p95_ms={p95:.1f} min_ms={lat_ms[0]:.1f} max_ms={lat_ms[-1]:.1f}")
    else:
        print("LATENCY: no chunk predictions recorded")
