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
opts=(--cleanenv --no-home --bind "$PRISM_V3_SOURCE_DIR:/opt/prism:ro"
  --bind "$PRISM_STORAGE:$PRISM_STORAGE" --bind "$jobroot/home:/prism-home" --bind "$PRISM_HF_CACHE:/hf"
  --pwd /opt/prism --env HOME=/prism-home --env XDG_CACHE_HOME=/prism-home/.cache
  --env PYTHONPATH=/opt/prism/src --env HF_HOME=/hf --env TMPDIR="$jobroot/tmp"
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}" --env "OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}"
  --env "MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}" --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}"
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env TOKENIZERS_PARALLELISM=false)
if [[ "${PRISM_STAGE_RESOURCE:?}" == gpu ]]; then
  prism_check_gpu_partition
  opts+=(--nv --env "PRISM_GPU_MODEL=${PRISM_GPU_MODEL:-A100}")
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
if [[ "$stage" == lora-ppo-policy ]]; then
  stop_after=""
  for ((i=0; i<${#cli[@]}; i++)); do
    if [[ "${cli[$i]}" == --stop-after-episodes && $((i+1)) -lt ${#cli[@]} ]]; then stop_after="${cli[$((i+1))]}"; fi
  done
  [[ "$stop_after" =~ ^[1-9][0-9]*$ ]] || prism_die "LoRA PPO policy requires an explicit positive --stop-after-episodes checkpoint"
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  [[ "$stage" == extract-features || "$stage" == evaluate-generate || "$stage" == prepare-shared || "$stage" == lora-reward-score ]] || prism_die "Unexpected V3 array stage"
  cli+=(--shard-index "$SLURM_ARRAY_TASK_ID")
fi
"${runner[@]}" python -m prism.v3 "$stage" "${cli[@]}"
