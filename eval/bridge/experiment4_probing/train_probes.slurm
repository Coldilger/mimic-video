#!/bin/bash
#SBATCH --job-name=mimic_train_probes
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing/probes-%j.out

# Trains the probe head on both checkpoint variants' frozen features and
# prints each against its own no-motion and mean-predictor references.
#
# CPU is enough (a 4096->256->256->9 MLP over 400 samples), but not the login
# node -- 4 cores shared across users, and the run exceeds a 2-minute budget
# there.
#
# The two runs share seed, split fraction and every hyperparameter, and their
# inputs share targets and episode ids exactly (verified: targets identical,
# episode_ids identical, features differ), so any gap between them is
# attributable to the backbone that produced the features.

set -u
DIR=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing
PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3
cd "$DIR"

for VARIANT in finetuned pretrained; do
  echo ""
  echo "############ $VARIANT ############"
  "$PY" train_probe.py \
    --features "features_${VARIANT}.npz" \
    --json-out "probe_${VARIANT}.json"
done

echo ""
echo "############ side by side ############"
"$PY" - <<'PY'
import json

rows = []
for v in ("finetuned", "pretrained"):
    with open(f"probe_{v}.json") as f:
        rows.append(json.load(f))

hdr = f"{'variant':<12}{'ridge L1':>12}{'MLP L1':>12}{'mean-pred':>12}{'no-motion':>12}{'beats mean':>12}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['variant']:<12}{r['ridge_val_l1']:>12.5f}{r['mlp_val_l1']:>12.5f}"
        f"{r['mean_pred_val_l1']:>12.5f}{r['nomotion_val_l1']:>12.5f}"
        f"{str(r['beats_mean_predictor']):>12}"
    )

ft, pre = rows[0]["ridge_val_l1"], rows[1]["ridge_val_l1"]
print()
print(f"finetuned - pretrained: {ft - pre:+.5f}  "
      f"({'finetuned better' if ft < pre else 'pretrained better'})")
PY

echo "PROBES_EXIT=$?"
