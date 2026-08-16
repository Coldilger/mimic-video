#!/bin/bash
#SBATCH --job-name=mimic_venv_reinstall
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/reinstall-venv-%j.out

# Full from-scratch rebuild recipe for model/.venv, proven working
# 2026-08-16 after model/.venv got gutted twice by the /scratch purge
# hitting uv's shared package cache (directory skeletons survive, actual
# package files/dist-info RECORDs vanish independently of the venv's own
# age -- see the "mimic-video venv reinstall" memory note for the full
# root-cause writeup). Earlier attempts patched individual corrupted
# dist-info dirs one package at a time (torch, then packaging, ...) --
# that's whack-a-mole, since ANY package whose old dist-info lost its
# RECORD file breaks the same way on its next reinstall. Wiping .venv
# entirely and rebuilding fresh is decisive: apex/transformer-engine stay
# cached in ~/.cache/uv from prior builds, so this relinks from cache
# rather than recompiling CUDA extensions from source (fast, ~10-15 min
# total instead of the original multi-hour build).

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export PATH="/home/3295540/.local/bin:$PATH"

# apex / transformer-engine-torch need an EXACT CUDA 12.9 nvcc match
# (neither system default 13.3 nor `module load cuda/12.8` sufficed) --
# use the dedicated conda env built via reinstall_venv2.slurm.
export CUDA_HOME=/home/3295540/.conda/envs/cuda129
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
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

echo "=== wiping .venv entirely (not git-tracked, safe to rebuild) ==="
rm -rf .venv

echo "=== uv sync --extra cu129 (x86_64 always resolves to cu129 regardless of extra) ==="
uv sync --extra cu129
echo "SYNC_EXIT=$?"

echo "=== sanity check (project deps only) ==="
.venv/bin/python3 -c "
import torch, numpy, scipy, transformers, accelerate
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
"
echo "SANITY_EXIT=$?"

# lerobot (needed only for LeRobotDataset, to read bridge_orig_lerobot in
# Experiment 1's offline probe -- not part of pyproject.toml). lerobot
# 0.3.3 declares "torch<2.8.0", which conflicts with our pinned
# 2.8.0+cu129 -- a real `uv pip install lerobot` (with normal dependency
# resolution) downgrades torch to satisfy that ceiling and silently
# breaks the from-source-built apex/transformer-engine's ABI match. Since
# we only touch lerobot's narrow LeRobotDataset reader, not its training
# code, `--no-deps` + manually installing its actually-imported
# submodules' deps is the correct tradeoff. `source .venv/bin/activate`
# first is required -- `uv pip install --python .venv/bin/python3`
# unreliably resolves to the externally-managed base interpreter instead
# of this venv.
echo "=== install lerobot with --no-deps (avoid re-pulling a conflicting torch) ==="
source .venv/bin/activate
uv pip install --no-deps lerobot==0.3.3 av
echo "LEROBOT_INSTALL_EXIT=$?"

# The rest of lerobot's declared deps that LeRobotDataset actually imports
# at runtime, confirmed via iterative import checks -- none of these pull
# in torch/torchvision as a transitive dependency.
echo "=== install lerobot's runtime deps (torch-free, confirmed via import checks) ==="
uv pip install \
  "datasets<=3.6.0,>=2.19.0" \
  "jsonlines>=4.0.0" \
  "draccus==0.10.0" \
  "deepdiff<9.0.0,>=7.0.1" \
  "rerun-sdk<0.23.0,>=0.21.0" \
  "gymnasium<1.0.0,>=0.29.1" \
  "pynput>=1.7.7" \
  "pyserial>=3.5" \
  "cmake>=3.29.0.1"
echo "DEPS_INSTALL_EXIT=$?"

echo "=== final sanity check ==="
.venv/bin/python3 -c "
import torch, numpy, scipy, transformers, accelerate, lerobot, av, datasets
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('lerobot', lerobot.__version__)
print('av', av.__version__)
print('datasets', datasets.__version__)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('LeRobotDataset import OK')
"
echo "FINAL_SANITY_EXIT=$?"
