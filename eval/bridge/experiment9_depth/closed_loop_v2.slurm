#!/bin/bash
#SBATCH --job-name=mimic_exp9_cl2
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Experiment 9 closed loop: baseline policy with the context read after denoising
# step STOP (0-34) instead of the deployed 1. Same 24 episodes per task as
# Experiment 1, paired with its stop=1 baseline via ../paired_stats.py.
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/mimic-video/eval/bridge/experiment9_depth/cl2-%j.out

# Required env vars: TASK, STOP, TAG (save tag, must contain "seed0" for pairing).
# Optional: CONDITION (baseline|seedavg|trace, default baseline), VIDEO_SEEDS (seedavg),
# LEAD (trace: shift the plan by n steps, default 0),
# EXEC (actions executed per decision, default 5), EP_START/EP_END (default 0 24).

set -u
: "${TASK:?set TASK}"
: "${TAG:?set TAG}"
: "${STOP:?set STOP}"
CONDITION="${CONDITION:-baseline}"; EXEC="${EXEC:-5}"; EP_START="${EP_START:-0}"; EP_END="${EP_END:-24}"
export CONDITION
export LEAD="${LEAD:-0}" LEAD_GRIPPER="${LEAD_GRIPPER:-0}" TRACE_NAME="${TAG}_${TASK}_ep${EP_START}-${EP_END}"

LIB_DIR="/mnt/beegfsnew/scratch/3295540/local_libs"
mkdir -p "$LIB_DIR"
cd "$LIB_DIR"
if [ ! -f "usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]; then
    wget -qO libvulkan.deb http://ftp.debian.org/debian/pool/main/v/vulkan-loader/libvulkan1_1.3.239.0-1_amd64.deb
    ar x libvulkan.deb
    tar -xf data.tar.xz
fi
export LD_LIBRARY_PATH="$LIB_DIR/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export DISPLAY=""
ulimit -t unlimited

CKPT_DIR="/home/3295540/mimic-video/model/checkpoints"
EXPERIMENT_NAME="w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
VIDEO_MODEL="${CKPT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_MODEL="${CKPT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CKPT_DIR}/dataset_statistics/bridge.json"

case "$TASK" in
  PutCarrotOnPlateInScene-v0|PutSpoonOnTableClothInScene-v0|StackGreenCubeOnYellowCubeBakedTexInScene-v0)
    ROBOT=widowx
    SCENE=bridge_table_1_v1
    OVERLAY=/home/3295540/mimic-video/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
    INIT_X=0.147; INIT_Y=0.028; MAX_STEPS=60
    ;;
  PutEggplantInBasketScene-v0)
    ROBOT=widowx_sink_camera_setup
    SCENE=bridge_table_1_v2
    OVERLAY=/home/3295540/mimic-video/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
    INIT_X=0.127; INIT_Y=0.06; MAX_STEPS=120
    ;;
  *)
    echo "unknown TASK: $TASK" >&2; exit 2
    ;;
esac

cd /home/3295540/mimic-video/eval/bridge/SimplerEnv
VENV_PY=/home/3295540/mimic-video/model/.venv/bin/python3

"$VENV_PY" /home/3295540/mimic-video/eval/bridge/experiment9_depth/main_inference_depth.py \
  --ckpt-path "mimic_exp1_closed_loop" \
  --robot "$ROBOT" --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps "$MAX_STEPS" \
  --env-name "$TASK" --scene-name "$SCENE" \
  --rgb-overlay-path "$OVERLAY" \
  --robot-init-x $INIT_X $INIT_X 1 --robot-init-y $INIT_Y $INIT_Y 1 \
  --obj-variation-mode episode --obj-episode-range "$EP_START" "$EP_END" \
  --additional-env-save-tags "$TAG" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name "$EXPERIMENT_NAME" \
  --vam-video-model-path "$VIDEO_MODEL" \
  --vam-action-model-path "$ACTION_MODEL" \
  --vam-dataset-statistics-path "$STATS" \
  --vam-num-execute-actions "$EXEC" \
  --vam-img-horizon 5 \
  --vam-lowdim-horizon 1 \
  --vam-stop-video-denoising-step "$STOP"

echo "EVAL_EXIT=$? TASK=$TASK STOP=$STOP TAG=$TAG CONDITION=$CONDITION EXEC=$EXEC EP=$EP_START-$EP_END VIDEO_SEEDS=${VIDEO_SEEDS:-} LEAD=$LEAD LEAD_GRIPPER=$LEAD_GRIPPER"
