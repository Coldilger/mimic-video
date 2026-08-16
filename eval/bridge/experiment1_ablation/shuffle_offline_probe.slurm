#!/bin/bash
#SBATCH --job-name=mimic_shuffle_probe
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment1_ablation/shuffle-offline-%j.out

# Full run: 24 episodes x 5 samples/episode (script defaults), matching
# ablation_offline_probe.py's protocol for direct comparability. Only one
# model instance loaded (64G is enough -- see shuffle_smoke.slurm).
# medium_* not short_* -- ablation_offline_probe.slurm's equivalent run
# took ~27 min, which exceeds short_*'s 1h10m cap only if queueing eats
# into the budget; use medium for the same safety margin that probe used.

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

"$VENV_PY" shuffle_offline_probe.py
