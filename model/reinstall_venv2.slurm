#!/bin/bash
#SBATCH --job-name=mimic_venv_reinstall2
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/reinstall-venv2-%j.out

# Third attempt. Attempts 1/2 (jobs 627691/627693) failed building apex:
# torch resolves to 2.8.0+cu129 on x86_64 regardless of --extra chosen
# (pyproject.toml's own comment: "this file has become a bit fucked up"),
# and apex's build script hard-requires an EXACT nvcc major.minor match --
# 12.8 (closest available via `module avail cuda`) was NOT tolerated, only
# an exact 12.9 will do. This step: create a dedicated conda env with the
# exact CUDA 12.9 toolkit (confirmed available via `conda search -c nvidia
# cuda-toolkit=12.9`) -- done via sbatch, not the login node (heavy
# download).

conda create -n cuda129 -c nvidia cuda-toolkit=12.9 -y
echo "CONDA_CREATE_EXIT=$?"

/home/3295540/.conda/envs/cuda129/bin/nvcc --version
