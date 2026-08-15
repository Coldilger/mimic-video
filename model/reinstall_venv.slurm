#!/bin/bash
#SBATCH --job-name=mimic_venv_reinstall
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/reinstall-venv-%j.out

# The venv's site-packages got silently gutted by the /scratch 30-day purge
# (directory skeletons survived, actual package files did not -- torch,
# numpy, transformers all showed 0 files; discovered 2026-08-15 while
# building Experiment 1's offline probe, whose only actual dependency is
# scipy, but the damage turned out to be venv-wide). Reconstructs from
# scratch following README.md's own documented setup exactly (hopper =
# H100/H200, our hardware -> cu126 extra). No uv.lock survived either (or
# never existed committed) -- pyproject.toml pins exact versions for most
# deps, so this should still be a faithful rebuild even resolving fresh.

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export PATH="/home/3295540/.local/bin:$PATH"

# Third attempt. Attempt 1 (627691) failed: torch resolves to 2.8.0+cu129 on
# x86_64 regardless of --extra chosen (pyproject.toml's own comment: "this
# file has become a bit fucked up"), but default nvcc was 13.3 -- a major
# mismatch apex's build refuses to tolerate. Attempt 2 (627693) tried module
# cuda/12.8, closer but apex's check apparently wants an EXACT major.minor
# match, not just same-major -- still failed. This uses a dedicated conda
# env with the exact CUDA 12.9.86 toolkit (built via reinstall_venv2.slurm,
# confirmed present).
export CUDA_HOME=/home/3295540/.conda/envs/cuda129
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
# apex built fine with the above, but transformer-engine-torch's build
# needs headers apex didn't (cuda_runtime_api.h, nvtx3/nvToolsExt.h,
# cudnn.h) -- the conda cuda-toolkit package puts its headers under
# targets/x86_64-linux/include, not plain include/, and cudnn is a
# separate package entirely (installed below, once, idempotent).
# nvtx3/nvToolsExt.h is bundled under nsight-compute's own tree, not the
# standard include dir -- need the *parent* of the nvtx3/ dir on CPATH so
# `#include <nvtx3/nvToolsExt.h>` resolves.
NVTX_HEADER=$(find "$CUDA_HOME" -iname 'nvToolsExt.h' 2>/dev/null | head -1)
NVTX_INCLUDE=$(dirname "$(dirname "$NVTX_HEADER")")
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:${NVTX_INCLUDE:-}:${CPATH:-}"
export C_INCLUDE_PATH="$CPATH"
export CPLUS_INCLUDE_PATH="$CPATH"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"

if [ ! -f "$CUDA_HOME/include/cudnn.h" ]; then
  conda install -n cuda129 -c nvidia cudnn -y
fi

cd /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model

echo "=== uv sync --extra cu129 (matches what x86_64 actually resolves to) ==="
uv sync --extra cu129
echo "SYNC_EXIT=$?"

echo "=== sanity check ==="
.venv/bin/python3 -c "
import torch, numpy, scipy, transformers, accelerate
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
"
echo "SANITY_EXIT=$?"
