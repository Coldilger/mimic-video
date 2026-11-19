#!/bin/bash
#SBATCH --job-name=mimic_exp8_conflict
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/mimic-video/eval/bridge/experiment8_conflict/conflict-%j.out

# Experiment 8 (instruction-foresight conflict), mimic-video, deployed policy
# (stop=1, 5 of 15 actions executed). Same 24 episodes per task as
# Experiment 1. Required env vars: TASK, MODE (probe|cl).
# cl: OBS, FUT (A|B: whose prompt the observed / future tokens come from),
# optional ALT (default mirror). Optional: EP_START/EP_END (0 24), TAG_SUFFIX.

set -u
: "${TASK:?set TASK}"
: "${MODE:?set MODE}"
EP_START="${EP_START:-0}"; EP_END="${EP_END:-24}"; STOP=1; EXEC=5
export TASK CONFLICT_MODE=$MODE
if [ "$MODE" = cl ]; then
  : "${OBS:?set OBS}"; : "${FUT:?set FUT}"; ALT="${ALT:-mirror}"
  export CONFLICT_OBS=$OBS CONFLICT_FUT=$FUT CONFLICT_ALT=$ALT
  TAG="exp8_cl_obs${OBS}_fut${FUT}_${ALT}_stop1_seed0${TAG_SUFFIX:-}"
else
  export CONFLICT_OUT=/home/3295540/mimic-video/eval/bridge/experiment8_conflict/records${TAG_SUFFIX:-}
  TAG="exp8_probe_stop1_seed0${TAG_SUFFIX:-}"
fi

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

"$VENV_PY" /home/3295540/mimic-video/eval/bridge/experiment8_conflict/main_inference_conflict.py \
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

echo "EVAL_EXIT=$? TASK=$TASK MODE=$MODE TAG=$TAG EP=$EP_START-$EP_END"
