#!/bin/bash
#SBATCH --job-name=mimic_recover_pose
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing/recover-%j.out

# Recovers each extracted sample's CURRENT pose by replaying the extraction's
# RNG draws -- no GPU, no model. See recover_current_pose.py's docstring for
# why this control is sharper than the episode-identity one.
set -u
DIR=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing
PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3
cd "$DIR"
"$PY" recover_current_pose.py \
  --features features_finetuned.npz \
  --out current_pose.npz
echo "RECOVER_EXIT=$?"
