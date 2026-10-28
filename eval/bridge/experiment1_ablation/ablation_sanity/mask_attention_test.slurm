#!/bin/bash
#SBATCH --job-name=mimic_mask_test
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/mimic-video/eval/bridge/experiment1_ablation/ablation_sanity/mask-test-%j.out
# Empirical mask/attention test for the decoder's cross-attention, see
# mask_attention_test.py. No simulator; loads the model directly.
set -euo pipefail
ulimit -t unlimited
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
# importing simpler_env loads sapien, which needs libvulkan (same as the eval scripts)
export LD_LIBRARY_PATH=/mnt/beegfsnew/scratch/3295540/local_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
cd /home/3295540/mimic-video/eval/bridge/SimplerEnv
/home/3295540/mimic-video/model/.venv/bin/python3 \
  /home/3295540/mimic-video/eval/bridge/experiment1_ablation/ablation_sanity/mask_attention_test.py
echo "MASK_TEST_EXIT=$?"
