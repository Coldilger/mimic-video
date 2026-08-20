#!/bin/bash
#SBATCH --job-name=mimic_eval_shuffled
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# shuffled runs the real video-diffusion backbone every replan, same cost
# as baseline (~10s/decision per experiment3_cost's timing measurement,
# vs ablated's ~130ms) -- the original 1h limit (copied from
# eval_ablated.slurm, which IS nearly free) timed out at ~20/24 episodes
# on the 60-step tasks and would fare worse on Eggplant's 120-step budget.
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment1_ablation/eval-shuffled-stop1-%j.out

# One task x one seed x 24 episodes of closed-loop SimplerEnv rollout,
# shuffled (wrong-episode real input) condition -- variant 2 closed-loop. Matches the
# 4-task x 3-seed x 24-episode protocol F1-VLA and LDA-1B were measured on
# (see eval.sh's stop_video_denoising_step=23, num_execute_actions=5,
# img_horizon=5, lowdim_horizon=1 -- the checkpoint's chosen config,
# unchanged from the offline probes run earlier this session).
#
# Required env vars: TASK, SEED

set -u
: "${TASK:?set TASK}"
: "${SEED:?set SEED}"

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

CKPT_DIR="/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/checkpoints"
EXPERIMENT_NAME="w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256"
VIDEO_MODEL="${CKPT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt"
ACTION_MODEL="${CKPT_DIR}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt"
STATS="${CKPT_DIR}/dataset_statistics/bridge.json"

case "$TASK" in
  PutCarrotOnPlateInScene-v0|PutSpoonOnTableClothInScene-v0|StackGreenCubeOnYellowCubeBakedTexInScene-v0)
    ROBOT=widowx
    SCENE=bridge_table_1_v1
    OVERLAY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
    INIT_X=0.147; INIT_Y=0.028; MAX_STEPS=60
    ;;
  PutEggplantInBasketScene-v0)
    ROBOT=widowx_sink_camera_setup
    SCENE=bridge_table_1_v2
    OVERLAY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
    INIT_X=0.127; INIT_Y=0.06; MAX_STEPS=120
    ;;
  *)
    echo "unknown TASK: $TASK" >&2; exit 2
    ;;
esac

cd /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment1_ablation
VENV_PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3

"$VENV_PY" main_inference_shuffled.py \
  --ckpt-path "mimic_shuffled" \
  --robot "$ROBOT" --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps "$MAX_STEPS" \
  --env-name "$TASK" --scene-name "$SCENE" \
  --rgb-overlay-path "$OVERLAY" \
  --robot-init-x $INIT_X $INIT_X 1 --robot-init-y $INIT_Y $INIT_Y 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --additional-env-save-tags "mimic_shuffled_stop1_seed${SEED}" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name "$EXPERIMENT_NAME" \
  --vam-video-model-path "$VIDEO_MODEL" \
  --vam-action-model-path "$ACTION_MODEL" \
  --vam-dataset-statistics-path "$STATS" \
  --vam-num-execute-actions 5 \
  --vam-img-horizon 5 \
  --vam-lowdim-horizon 1 \
  --vam-stop-video-denoising-step 1

echo "EVAL_EXIT=$? TASK=$TASK SEED=$SEED"
