#!/bin/bash
#SBATCH --job-name=mimic_ablation_smoke
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment1_ablation/ablation-smoke-%j.out

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
deactivate 2>/dev/null
source ../../../model/.venv/bin/activate

python -c "
from ablation_offline_probe import run_ablation_probe
import pathlib
run_ablation_probe(
    checkpoint_dir=pathlib.Path('/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/checkpoints'),
    dataset_root=pathlib.Path('/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot'),
    n_episodes=2,
    samples_per_episode=2,
)
"
