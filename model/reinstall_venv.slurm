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

# Attempt 4 (after this same script's earlier "successful" run silently
# regressed): plain `uv sync --extra cu129` trusts already-installed
# dist-info/RECORD metadata and skips reinstalling anything it thinks is
# already satisfied -- but the underlying uv package-cache blobs these
# venvs hardlink from can get purged independently of the venv itself
# (their own inode mtime, not the venv's), so a venv can go from a
# confirmed-working sanity check straight to `torch` missing its
# __init__.py without any command of ours touching it. `uv cache clean`
# + `--reinstall` forces every package to be re-fetched/re-extracted from
# scratch instead of trusting stale links.
echo "=== uv cache clean (purge may have silently gutted cached blobs) ==="
uv cache clean
echo "CACHE_CLEAN_EXIT=$?"

echo "=== uv sync --extra cu129 --reinstall (matches what x86_64 actually resolves to) ==="
uv sync --extra cu129 --reinstall
echo "SYNC_EXIT=$?"

echo "=== sanity check (before lerobot) ==="
.venv/bin/python3 -c "
import torch, numpy, scipy, transformers, accelerate
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
"
echo "SANITY_EXIT=$?"

# Fourth attempt at getting lerobot in (needed only for
# LeRobotDataset, to read bridge_orig_lerobot in Experiment 1's offline
# probe -- not part of pyproject.toml, was installed ad-hoc before the
# purge). `uv pip install --python .venv/bin/python3 ...` kept resolving
# to the externally-managed BASE interpreter instead of this venv
# (confirmed via "Using Python ... environment at: /home/.../cpython-3.10.19...",
# not .venv) -- bin/activate was also a purge casualty, missing entirely,
# so uv had no VIRTUAL_ENV to detect. Rebuilt bin/activate from a throwaway
# `uv venv` template (repo's own README.md says to `source .venv/bin/activate`
# before plain `uv pip install`, exactly what's done here) with VIRTUAL_ENV
# pointed at this venv. That's also what silently corrupted torch down to
# 2.6.0+cu126 with a missing __init__.py in the process -- one of the failed
# --python attempts must have partially written into the shared uv cache
# this venv hardlinks from, hence redoing the sync above first.
echo "=== source activate + uv pip install lerobot ==="
source .venv/bin/activate
uv pip install --reinstall lerobot==0.3.3 av pyav
echo "LEROBOT_INSTALL_EXIT=$?"

echo "=== sanity check (after lerobot) ==="
.venv/bin/python3 -c "
import torch, numpy, scipy, transformers, accelerate, lerobot, av
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('lerobot', lerobot.__version__)
print('av', av.__version__)
"
echo "SANITY_EXIT=$?"
