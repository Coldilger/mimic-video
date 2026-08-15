#!/bin/bash
#SBATCH --job-name=mimic_oracle_offline
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment2_oracle/oracle-offline-%j.out

# Offline oracle probe (Experiment B). Never touches SimplerEnv/ManiSkill2 at
# runtime (no rendering, no simulator stepping) -- but VAMInference lives in
# simpler_env.policies.vam.video_action_model, and simpler_env/__init__.py
# unconditionally imports ManiSkill2/sapien at import time, which needs
# libvulkan.so.1 just to *import*, regardless of whether it's ever used.
# First run of this script failed on exactly that ImportError. Same fix
# submit_eval.slurm already uses for its real (rendering) eval runs.
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

cd /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment2_oracle
deactivate 2>/dev/null
source ../../../model/.venv/bin/activate

python oracle_offline_probe.py
