#!/usr/bin/env bash
set -euo pipefail
umask 027
: "${SLURM_JOB_ID:?Run through Slurm}" "${PRISM_V3_SOURCE_DIR:?}" "${PRISM_COMMIT:?}"
source "$PRISM_V3_SOURCE_DIR/scripts/yancheng/common.sh"
[[ "${PRISM_ALLOW_SUBMIT:-0}" == 1 && "${PRISM_TEST_MODE:-0}" != 1 ]] || prism_die "Compute execution disabled"
prism_check_paths
prism_path_under "$PRISM_STORAGE" "$PRISM_V3_SOURCE_DIR"
[[ "$(cat "$PRISM_V3_SOURCE_DIR/.source.commit")" == "$PRISM_COMMIT" ]] || prism_die "V3 source commit differs"
printf '%s  %s\n' "${PRISM_V3_SOURCE_MANIFEST_SHA256:?}" "$PRISM_V3_SOURCE_DIR/.source.sha256" | sha256sum --check --status
(cd "$PRISM_V3_SOURCE_DIR" && sha256sum --check --status .source.sha256)
printf '%s  %s\n' "${PRISM_IMAGE_SHA256:?}" "$PRISM_IMAGE" | sha256sum --check --status
stage="${1:?Supply a V3 stage}"; shift
distributed_gpus=1
if [[ "$stage" == lora-ppo-smoke || "$stage" == lora-ppo-policy \
   || "$stage" == linear-ppo-smoke || "$stage" == linear-ppo-policy ]]; then
  distributed_gpus="${PRISM_DISTRIBUTED_GPUS:-4}"
  if [[ "$stage" == linear-ppo-smoke || "$stage" == linear-ppo-policy ]]; then
    [[ "$distributed_gpus" == 4 ]] || prism_die "Matched linear PPO uses exactly 4 homogeneous GPUs per method"
  else
    [[ "$distributed_gpus" == 4 || "$distributed_gpus" == 8 ]] || prism_die "Distributed LoRA PPO uses 4 or 8 homogeneous GPUs"
  fi
fi
if [[ "$stage" != cpu-smoke ]]; then
  printf '%s  %s\n' "${PRISM_CONFIG_SHA256:?}" "${PRISM_CONFIG_PATH:?}" | sha256sum --check --status
fi
jobroot="$PRISM_WORK/jobs/$SLURM_JOB_ID"
[[ ! -e "$jobroot" ]] || prism_die "Preserve existing job scratch before restarting"
mkdir -p "$jobroot/tmp" "$jobroot/home"
prism_load_apptainer
if [[ "$PRISM_CONTAINER_RUNTIME" == singularity ]]; then
  export SINGULARITY_CACHEDIR="$PRISM_APPTAINER_CACHE" SINGULARITY_TMPDIR="$jobroot/tmp"
else
  export APPTAINER_CACHEDIR="$PRISM_APPTAINER_CACHE" APPTAINER_TMPDIR="$jobroot/tmp"
fi
unset OPENAI_API_KEY APPTAINERENV_OPENAI_API_KEY SINGULARITYENV_OPENAI_API_KEY
threads="${SLURM_CPUS_PER_TASK:-1}"
if [[ "$distributed_gpus" -gt 1 ]]; then
  [[ "$threads" -ge "$distributed_gpus" && $((threads%distributed_gpus)) -eq 0 ]] || prism_die "CPU allocation must divide evenly across PPO ranks"
  threads=$((threads/distributed_gpus))
fi
opts=(--cleanenv --no-home --bind "$PRISM_V3_SOURCE_DIR:/opt/prism:ro"
  --bind "$PRISM_STORAGE:$PRISM_STORAGE" --bind "$jobroot/home:/prism-home" --bind "$PRISM_HF_CACHE:/hf"
  --pwd /opt/prism --env HOME=/prism-home --env XDG_CACHE_HOME=/prism-home/.cache
  --env PYTHONPATH=/opt/prism/src --env HF_HOME=/hf --env TMPDIR="$jobroot/tmp"
  --env "OMP_NUM_THREADS=$threads" --env "OPENBLAS_NUM_THREADS=$threads"
  --env "MKL_NUM_THREADS=$threads" --env "SLURM_CPUS_PER_TASK=$threads"
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env TOKENIZERS_PARALLELISM=false)
if [[ "${PRISM_STAGE_RESOURCE:?}" == gpu ]]; then
  prism_check_gpu_partition
  opts+=(--nv --env "PRISM_GPU_MODEL=${PRISM_GPU_MODEL:-A100}"
         --env "PRISM_EXPECTED_GPU_COUNT=$distributed_gpus")
  if [[ "$distributed_gpus" -gt 1 ]]; then
    opts+=(--env "PRISM_DISTRIBUTED_GPUS=$distributed_gpus")
  fi
  for name in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
    if [[ -v "$name" ]]; then opts+=(--env "$name=${!name}"); fi
  done
else
  [[ "$SLURM_JOB_PARTITION" == "$PRISM_CPU_PARTITION" ]] || prism_die "Unexpected CPU partition"
fi
if [[ "$stage" == prepare-data || "$stage" == download-data ]]; then
  prism_forward_proxy_envs
  opts+=(--env HF_HUB_OFFLINE=0 --env TRANSFORMERS_OFFLINE=0)
else
  opts+=(--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1)
fi
test_assets="${PRISM_ASSET_ROOT:-$PRISM_STORAGE/PRISM}"
if [[ "$stage" == cpu-smoke && -f "$test_assets/assets/ifeval/source-manifest.json" ]]; then
  prism_path_under "$PRISM_STORAGE" "$test_assets"
  opts+=(--env "PRISM_TEST_IFEVAL_SOURCE=$test_assets/assets/ifeval"
         --env "PRISM_TEST_NLTK_DATA=$test_assets/assets/nltk_data")
fi
runner=("$PRISM_CONTAINER_RUNTIME" exec "${opts[@]}" "$PRISM_IMAGE")
echo "stage=$stage commit=$PRISM_COMMIT image=$PRISM_IMAGE_SHA256 job=$SLURM_JOB_ID resource=$PRISM_STAGE_RESOURCE"
if [[ "$stage" == cpu-smoke ]]; then
  "${runner[@]}" python -m pytest -q /opt/prism/tests -p no:cacheprovider --basetemp "$jobroot/pytest"
  exit 0
fi
actual="$("${runner[@]}" python -m prism.v3 resource --config "$PRISM_CONFIG_PATH" --for-stage "$stage")"
[[ "$actual" == "$PRISM_STAGE_RESOURCE" ]] || prism_die "V3 Python/Bash resource classification differs"
if [[ "$PRISM_STAGE_RESOURCE" == gpu ]]; then
  "${runner[@]}" python /opt/prism/scripts/smoke_gpu.py
fi
cli=("$@")
if [[ "$stage" == lora-ppo-policy || "$stage" == linear-ppo-policy ]]; then
  stop_after=""
  for ((i=0; i<${#cli[@]}; i++)); do
    if [[ "${cli[$i]}" == --stop-after-episodes && $((i+1)) -lt ${#cli[@]} ]]; then stop_after="${cli[$((i+1))]}"; fi
  done
  [[ "$stop_after" =~ ^[1-9][0-9]*$ ]] || prism_die "Distributed PPO policy requires an explicit positive --stop-after-episodes checkpoint"
  [[ "$stage" != linear-ppo-policy || "$stop_after" == 16384 ]] \
    || prism_die "Formal linear PPO horizon is exactly 16384 episodes"
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  [[ "$stage" == extract-features || "$stage" == evaluate-generate || "$stage" == prepare-shared || "$stage" == lora-reward-score || "$stage" == lora-ppo-ifeval-generate ]] || prism_die "Unexpected V3 array stage"
  cli+=(--shard-index "$SLURM_ARRAY_TASK_ID")
fi
if [[ "$distributed_gpus" -gt 1 ]]; then
  "${runner[@]}" torchrun --standalone --nproc_per_node="$distributed_gpus" -m prism.v3 "$stage" "${cli[@]}"
else
  "${runner[@]}" python -m prism.v3 "$stage" "${cli[@]}"
fi
