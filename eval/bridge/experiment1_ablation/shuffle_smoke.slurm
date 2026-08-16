#!/bin/bash
#SBATCH --job-name=mimic_shuffle_smoke
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment1_ablation/shuffle-smoke-%j.out

# Only one model instance (baseline VAMInference, unmodified) is loaded --
# unlike ablation_smoke.slurm's two (baseline + ablated), which needed
# 128G. 64G matches what experiment2_oracle's single-model probes used.

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

cd /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment1_ablation
VENV_PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3

"$VENV_PY" -c "
from shuffle_offline_probe import run_shuffle_probe
import pathlib
run_shuffle_probe(
    checkpoint_dir=pathlib.Path('/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/checkpoints'),
    dataset_root=pathlib.Path('/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot'),
    n_episodes=4,
    samples_per_episode=2,
)
"
