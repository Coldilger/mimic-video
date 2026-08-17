#!/bin/bash
#SBATCH --job-name=mimic_cmp_targets
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing/cmptargets-%j.out

# Decisive control for Experiment 4 -- see compare_targets.py's docstring.
set -u
DIR=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing
PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3
cd "$DIR"
for VARIANT in finetuned pretrained; do
  echo ""
  echo "############ $VARIANT ############"
  "$PY" compare_targets.py --features "features_${VARIANT}.npz"
done
echo "CMP_EXIT=$?"
