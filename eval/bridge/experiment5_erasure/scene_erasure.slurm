#!/bin/bash
#SBATCH --job-name=mimic_scene_erasure
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment5_erasure/erasure-%j.out

# No GPU, no model: pure linear algebra on already-extracted Experiment 4
# features. leace.py's own self-test (run separately, see README) validates
# the erasure method on synthetic data with a known answer before this ever
# touches real features.
set -u
DIR=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment5_erasure
E4=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing
PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3
cd "$DIR"
"$PY" scene_erasure_diagnostic.py \
  --features "$E4/features_${VARIANT}.npz" \
  --current-pose "$E4/current_pose.npz" \
  --json-out "result_${VARIANT}.json"
echo "ERASURE_EXIT=$? VARIANT=$VARIANT"
