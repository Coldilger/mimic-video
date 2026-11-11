#!/bin/bash
#SBATCH --job-name=mimic_depth_ol
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:10:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/mimic-video/eval/bridge/experiment9_depth/depth-ol-%j.out
# Experiment 9 open-loop depth test, see depth_open_loop.py.
set -euo pipefail
ulimit -t unlimited
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
# importing simpler_env loads sapien, which needs libvulkan (same as the eval scripts)
export LD_LIBRARY_PATH=/mnt/beegfsnew/scratch/3295540/local_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
cd /home/3295540/mimic-video/eval/bridge/SimplerEnv
/home/3295540/mimic-video/model/.venv/bin/python3 \
  /home/3295540/mimic-video/eval/bridge/experiment9_depth/depth_open_loop.py "$@"
echo "DEPTH_OL_EXIT=$?"
