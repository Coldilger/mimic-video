#!/bin/bash
#SBATCH --job-name=mimic_timing_smoke
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment3_cost/timing-smoke-%j.out

LIB_DIR="/mnt/beegfsnew/scratch/3295540/local_libs"
mkdir -p "$LIB_DIR"
cd "$LIB_DIR"
if [ ! -f "usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]; then
    wget -qO libvulkan.deb http://ftp.debian.org/debian/pool/main/v/vulkan-loader/libvulkan1_1.3.239.0-1_amd64.deb
    ar x libvulkan.deb
    tar -xf data.tar.xz
fi
export LD_LIBRARY_PATH="$LIB_DIR/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export DISPLAY=""

CKPT_DIR="/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/checkpoints"
EXPERIMENT_NAME="w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
VIDEO_MODEL="${CKPT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_MODEL="${CKPT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CKPT_DIR}/dataset_statistics/bridge.json"

VARIANT="${VARIANT:-baseline}"

cd /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment3_cost
VENV_PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3

"$VENV_PY" main_inference_timed.py \
  --vam-variant "$VARIANT" \
  --ckpt-path "mimic_timing_${VARIANT}" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 3 \
  --additional-env-save-tags "timing_smoke_${VARIANT}" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name "$EXPERIMENT_NAME" \
  --vam-video-model-path "$VIDEO_MODEL" \
  --vam-action-model-path "$ACTION_MODEL" \
  --vam-dataset-statistics-path "$STATS" \
  --vam-num-execute-actions 5 \
  --vam-img-horizon 5 \
  --vam-lowdim-horizon 1 \
  --vam-stop-video-denoising-step 23
