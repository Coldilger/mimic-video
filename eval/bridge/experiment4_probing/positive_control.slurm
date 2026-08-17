#!/bin/bash
#SBATCH --job-name=mimic_pos_control
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing/poscontrol-%j.out

# Positive control for Experiment 4 -- see positive_control.py's docstring.
set -u
DIR=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing
PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3
cd "$DIR"
for VARIANT in finetuned pretrained; do
  echo ""
  echo "############ $VARIANT ############"
  "$PY" positive_control.py --features "features_${VARIANT}.npz"
done
echo "POSCONTROL_EXIT=$?"
