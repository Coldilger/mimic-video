#!/bin/bash

#SBATCH --job-name="mimic_eval"
#SBATCH --partition=medium_gpuh200
#SBATCH --gpus=H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=mimic_eval_log_%j.out
#SBATCH --error=mimic_eval_err_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ilia.koldyshev@studbocconi.it

echo -e "Job ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID:-none}) allocated on compute node $(hostname)"

# Highest stop_steps value to sweep up to (inclusive). Override at submit
# time, e.g.: STOP_MAX=10 sbatch --export=ALL,STOP_MAX=10 --array=0-2%3 submit_eval.slurm
STOP_MAX="${STOP_MAX:-35}"
all_stops=($(seq 0 "$STOP_MAX"))
n_total=${#all_stops[@]}

# If submitted as a job array (e.g. `sbatch --array=0-7%3 submit_eval.slurm`),
# split the stop_steps sweep (0..STOP_MAX) into one contiguous chunk per array
# task and hand it to eval.sh via STOP_STEPS, so chunks run in parallel across
# separate SLURM allocations instead of one job crawling through them all
# serially. Submitted without --array, this runs the full 0..STOP_MAX sweep
# in one job, same as before. eval.sh skips anything already rendered in
# results/, so reruns and overlapping chunks are safe.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  n_chunks="${SLURM_ARRAY_TASK_COUNT:-$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))}"
  chunk_idx=$(( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN ))
  chunk_size=$(( (n_total + n_chunks - 1) / n_chunks ))
  start=$(( chunk_idx * chunk_size ))
  if (( start < n_total )); then
    end=$(( start + chunk_size ))
    (( end > n_total )) && end=$n_total
    export STOP_STEPS="${all_stops[*]:start:end-start}"
  else
    export STOP_STEPS=" "
  fi
  echo "Array task ${SLURM_ARRAY_TASK_ID} (of ${n_chunks} chunks, stop_max=${STOP_MAX}) covers stop_steps: ${STOP_STEPS}"
else
  export STOP_STEPS="${all_stops[*]}"
  echo "Single job covers stop_steps 0..${STOP_MAX}: ${STOP_STEPS}"
fi

LIB_DIR="/mnt/beegfsnew/scratch/3295540/local_libs"
mkdir -p "$LIB_DIR"
cd "$LIB_DIR"
if [ ! -f "usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]; then
    echo "Vulkan not found. Downloading..."
    wget -qO libvulkan.deb http://ftp.debian.org/debian/pool/main/v/vulkan-loader/libvulkan1_1.3.239.0-1_amd64.deb
    ar x libvulkan.deb
    tar -xf data.tar.xz
    echo "Vulkan restored."
else
    echo "Vulkan already present."
fi
if [ ! -f "bin/ffmpeg" ]; then
    echo "FFmpeg not found. Downloading..."
    wget -qO ffmpeg.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    tar -xf ffmpeg.tar.xz
    mkdir -p bin
    cp ffmpeg-*-amd64-static/ffmpeg bin/
    cp ffmpeg-*-amd64-static/ffprobe bin/
    echo "FFmpeg restored."
else
    echo "FFmpeg already present."
fi
# 2. Go to the evaluation directory
cd /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge
# 3. Activate the environment (suppress errors if any)
deactivate 2>/dev/null
source ../../model/.venv/bin/activate
# 4. Run the render
echo "Starting evaluation..."
bash eval.sh
echo "Evaluation fully completed!"
