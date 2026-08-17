#!/bin/bash
#SBATCH --job-name=mimic_probe_extract
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# medium_gpua100's MIG slices cannot initialise the Vulkan renderer that the
# mimic stack pulls in (segfault, job 629618) -- H100/H200 only.
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing/extract-%j.out

# Extracts frozen crossattn_emb features + future-pose targets for one
# checkpoint variant. Each sample runs the full video-diffusion backbone
# (~10s per Experiment 3's timing), so 400 samples is roughly an hour.
#
# Required env vars: VARIANT (finetuned|pretrained)
# Optional: N_EPISODES, SAMPLES_PER_EPISODE, HORIZON, OUT

set -u
: "${VARIANT:?set VARIANT to finetuned or pretrained}"
N_EPISODES="${N_EPISODES:-40}"
SAMPLES_PER_EPISODE="${SAMPLES_PER_EPISODE:-10}"
HORIZON="${HORIZON:-5}"
DIR=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing
OUT="${OUT:-$DIR/features_${VARIANT}.npz}"

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
export DISPLAY=""

# Importing VAMInference pulls in simpler_env, which links libvulkan even
# though nothing here renders -- same shim the eval launchers use
# (../experiment1_ablation/eval_baseline.slurm).
LIB_DIR="/mnt/beegfsnew/scratch/3295540/local_libs"
mkdir -p "$LIB_DIR"
cd "$LIB_DIR"
if [ ! -f "usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]; then
    wget -qO libvulkan.deb http://ftp.debian.org/debian/pool/main/v/vulkan-loader/libvulkan1_1.3.239.0-1_amd64.deb
    ar x libvulkan.deb
    tar -xf data.tar.xz
fi
export LD_LIBRARY_PATH="$LIB_DIR/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

cd "$DIR"
/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3 \
  extract_features.py \
  --variant "$VARIANT" \
  --out "$OUT" \
  --n-episodes "$N_EPISODES" \
  --samples-per-episode "$SAMPLES_PER_EPISODE" \
  --horizon "$HORIZON"

echo "EXTRACT_EXIT=$? VARIANT=$VARIANT"
