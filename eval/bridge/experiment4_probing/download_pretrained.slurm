#!/bin/bash
#SBATCH --job-name=mimic_dl_pretrained
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing/download-%j.out

# Downloads the pretrained_cosmos_bridge checkpoint for Experiment 4's
# memorization-robustness comparison. The two Bridge checkpoints differ
# exactly in the component this experiment probes:
#
#   pretrained_cosmos_bridge -> video_backbone/v2w_pretrained_cosmos.pt
#                               (generic Cosmos, never finetuned on Bridge)
#   finetuned_cosmos_bridge  -> video_backbone/v2w_bridge_lora*.pt
#                               (Bridge-LoRA-finetuned; already on disk,
#                                used by Experiments 1-3)
#
# Both ship a Bridge-trained action decoder, so probing `crossattn_emb` (which
# comes from the video backbone) under each isolates how much of the probed
# representation Bridge finetuning actually built -- the check the offline
# metrics can't otherwise separate from memorization, since both models were
# finetuned on all of Bridge with no held-out split.

set -euo pipefail
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1

cd /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model
.venv/bin/python scripts/download_checkpoints.py --models pretrained_cosmos_bridge

echo "DOWNLOAD_EXIT=$?"
ls -la checkpoints/video_backbone/ checkpoints/action_decoder/
