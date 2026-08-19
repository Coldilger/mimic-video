#!/bin/bash
#SBATCH --job-name=mimic_timing_full
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment3_cost/timing-full-%j.out

# Full-scale version of timing_smoke.slurm: 4 episodes instead of 3
# (~48 replans instead of 36, matching F1-VLA's n=45 -- see
# experiment3_cost/README.md's "Not yet done"). Otherwise identical:
# same task/checkpoint/settings as the smoke run, just more episodes.
# Time bumped from 30 to 50 min: smoke runs (3 episodes) took 14-29 min
# observed via sacct, so this leaves headroom for the +33% episode count.

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

# Default soft RLIMIT_CPU on this partition is 600s (confirmed via srun
# probe); baseline/shuffled (~10s/replan, unlike ablated's ~130ms) exceed it
# on 4 episodes even though wall-clock stays well under the SBATCH --time
# limit -- "CPU time limit exceeded (core dumped)" is bash's SIGXCPU message,
# not an OOM or wall-clock timeout. Hard limit is unlimited, so just raise it.
ulimit -t unlimited

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
  --ckpt-path "mimic_timing_full_${VARIANT}" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 4 \
  --additional-env-save-tags "timing_full_${VARIANT}_${SLURM_JOB_ID}" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name "$EXPERIMENT_NAME" \
  --vam-video-model-path "$VIDEO_MODEL" \
  --vam-action-model-path "$ACTION_MODEL" \
  --vam-dataset-statistics-path "$STATS" \
  --vam-num-execute-actions 5 \
  --vam-img-horizon 5 \
  --vam-lowdim-horizon 1 \
  --vam-stop-video-denoising-step 23
