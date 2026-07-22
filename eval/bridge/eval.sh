GPUS=(0)

#added by Ilia
export LD_LIBRARY_PATH="/mnt/beegfsnew/scratch/3295540/local_libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export PATH="/mnt/beegfsnew/scratch/3295540/local_libs/bin:$PATH"

fifo="/tmp/gpuq.$$"
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"

for i in $(seq 0 $(( 2 * ${#GPUS[@]} - 1 ))); do
  g=$((i % ${#GPUS[@]}))
  echo "$g" >&3
done

launch() {
  local gpu
  read -u 3 gpu || exit 1
  (
    if [[ "$gpu" -eq 0 ]]; then
      export CUDA_VISIBLE_DEVICES=0
      export SAPIEN_RENDER_CUDA_ORDINAL=0
    else
      export CUDA_VISIBLE_DEVICES="${gpu},0"
      export SAPIEN_RENDER_CUDA_ORDINAL=1
    fi
    export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select

    "$@"
    rc=$?
    echo "$gpu" >&3
    exit $rc
  ) &
}

checkpoint_dir="/scratch/3295540/mimic-video-project/mimic-video/model/checkpoints"

declare -A ftcosmos=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256
  [action_model]=${checkpoint_dir}/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/bridge.json
)

# ptcosmos  deleted by Ilia
models=(ftcosmos)

execute_actions=(5)

# Which stop-steps this run should cover. Override via STOP_STEPS (space-separated)
# to split the sweep across parallel SLURM jobs, e.g.:
#   STOP_STEPS="12 13 14 15" bash eval.sh
if [[ -n "${STOP_STEPS:-}" ]]; then
  read -ra stop_steps <<< "$STOP_STEPS"
else
  stop_steps=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
fi

results_root="results"

# True if every episode in [lo, hi) already has a rendered mp4 for this
# variation+env, so the launch can be skipped on resume.
is_done() {
  local variation="$1" env_name="$2" lo="$3" hi="$4"
  local dir="${results_root}/${variation}"
  [[ -d "$dir" ]] || return 1
  local n=0 f ep
  while IFS= read -r f; do
    ep="${f##*_episode_}"
    ep="${ep%%_*}"
    if (( ep >= lo && ep < hi )); then
      n=$((n + 1))
    fi
  done < <(find "$dir" -path "*/${env_name}/*" -name "*.mp4" 2>/dev/null)
  (( n == hi - lo ))
}

run_task() {
  local variation="$1" model_name="$2" execute_steps="$3" stop="$4"
  local env_name="$5" scene_name="$6" robot="$7" rgb_overlay_path="$8"
  local robot_init_x="$9" robot_init_y="${10}" max_steps="${11}" lo="${12}" hi="${13}"
  local -n M="$model_name"

  if is_done "$variation" "$env_name" "$lo" "$hi"; then
    echo "[skip] ${variation} ${env_name} episodes ${lo}-${hi} already complete"
    return
  fi

  launch python SimplerEnv/simpler_env/main_inference.py --ckpt-path "$variation" \
    --robot "$robot" --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps "$max_steps" \
    --env-name "$env_name" --scene-name "$scene_name" \
    --rgb-overlay-path "$rgb_overlay_path" \
    --robot-init-x "$robot_init_x" "$robot_init_x" 1 --robot-init-y "$robot_init_y" "$robot_init_y" 1 --obj-variation-mode episode --obj-episode-range "$lo" "$hi" \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
    --vam-experiment-name "${M[experiment_name]}" \
    --vam-video-model-path "${M[video_model]}" \
    --vam-action-model-path "${M[action_model]}" \
    --vam-dataset-statistics-path "${M[stats]}" \
    --vam-num-execute-actions "$execute_steps" \
    --vam-img-horizon "${M[img_h]}" \
    --vam-lowdim-horizon "${M[lowdim_h]}" \
    --vam-stop-video-denoising-step "$stop"
}

episode_ranges=("0 8" "8 16" "16 24")

for model_name in "${models[@]}"; do
  declare -n M="$model_name"

  action_stem="$(basename -- "${M[action_model]}")"
  action_stem="${action_stem%.pt}"

  for execute_steps in "${execute_actions[@]}"; do
    for stop in "${stop_steps[@]}"; do
      variation_name="${action_stem}_stopafter${stop}_execute${execute_steps}"

      # bridge_table_1_v1 tasks
      scene_name=bridge_table_1_v1
      robot=widowx
      rgb_overlay_path=SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
      robot_init_x=0.147
      robot_init_y=0.028

      for env_name in PutCarrotOnPlateInScene-v0 PutSpoonOnTableClothInScene-v0 StackGreenCubeOnYellowCubeBakedTexInScene-v0; do
        for range in "${episode_ranges[@]}"; do
          read -r lo hi <<< "$range"
          run_task "$variation_name" "$model_name" "$execute_steps" "$stop" \
            "$env_name" "$scene_name" "$robot" "$rgb_overlay_path" \
            "$robot_init_x" "$robot_init_y" 60 "$lo" "$hi"
        done
      done

      # bridge_table_1_v2 (sink) task
      scene_name=bridge_table_1_v2
      robot=widowx_sink_camera_setup
      rgb_overlay_path=SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
      robot_init_x=0.127
      robot_init_y=0.06

      for range in "${episode_ranges[@]}"; do
        read -r lo hi <<< "$range"
        run_task "$variation_name" "$model_name" "$execute_steps" "$stop" \
          "PutEggplantInBasketScene-v0" "$scene_name" "$robot" "$rgb_overlay_path" \
          "$robot_init_x" "$robot_init_y" 120 "$lo" "$hi"
      done
    done
  done
done

wait
exec 3>&- 3<&-
