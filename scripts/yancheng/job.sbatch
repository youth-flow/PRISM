#!/usr/bin/env bash
set -euo pipefail
umask 027
: "${SLURM_JOB_ID:?Run under Slurm}" "${PRISM_REPO:?}" "${PRISM_COMMIT:?}"
source "$PRISM_REPO/scripts/yancheng/common.sh"
[[ "${PRISM_ALLOW_SUBMIT:-0}" == 1 && "${PRISM_TEST_MODE:-0}" != 1 ]] || prism_die "Computations disabled"
prism_check_paths
stage="${1:?}"; shift
if prism_stage_gpu "$stage"; then gpu=1; else gpu=0; fi
[[ "$(prism_git rev-parse HEAD)" == "$PRISM_COMMIT" ]]
[[ -z "$(prism_git status --porcelain)" ]]
printf '%s  %s\n' "$PRISM_IMAGE_SHA256" "$PRISM_IMAGE" | sha256sum --check --status
if [[ "$stage" != smoke && "$stage" != cpu-smoke ]]; then
  : "${PRISM_CONFIG_PATH:?}" "${PRISM_CONFIG_SHA256:?}"
  printf '%s  %s\n' "$PRISM_CONFIG_SHA256" "$PRISM_CONFIG_PATH" | sha256sum --check --status
fi
jobroot="$PRISM_WORK/jobs/$SLURM_JOB_ID"
[[ ! -e "$jobroot" ]] || prism_die "Job scratch exists; preserve it and inspect before restarting"
mkdir -p "$jobroot/source" "$jobroot/tmp" "$jobroot/home" "$PRISM_APPTAINER_CACHE"
prism_git archive "$PRISM_COMMIT" | tar -x -C "$jobroot/source"
prism_load_apptainer
if [[ "$PRISM_CONTAINER_RUNTIME" == singularity ]]; then
  export SINGULARITY_CACHEDIR="$PRISM_APPTAINER_CACHE" SINGULARITY_TMPDIR="$jobroot/tmp"
else
  export APPTAINER_CACHEDIR="$PRISM_APPTAINER_CACHE" APPTAINER_TMPDIR="$jobroot/tmp"
fi
runtime_prefix="$(prism_runtime_env_prefix)"
export "${runtime_prefix}PYTHONPATH=/opt/prism/src"
opts=(--cleanenv --no-home
  --bind "$jobroot/source:/opt/prism:ro" --bind "$PRISM_STORAGE:$PRISM_STORAGE"
  --bind "$jobroot/home:/prism-home" --bind "$PRISM_HF_CACHE:/hf"
  --pwd /opt/prism --env HOME=/prism-home --env XDG_CACHE_HOME=/prism-home/.cache
  --env PYTHONPATH=/opt/prism/src --env HF_HOME=/hf --env TMPDIR="$jobroot/tmp"
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" --env TOKENIZERS_PARALLELISM=false
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "PRISM_GPU_MODEL=${PRISM_GPU_MODEL:-A100}")
opts+=(--env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}")
opts+=(--env "OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}" --env "MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}")
if [[ "$gpu" == 1 ]]; then
  prism_check_gpu_partition
  opts+=(--nv)
  for var in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
    if [[ -v "$var" ]]; then opts+=(--env "$var=${!var}"); fi
  done
else
  [[ "$SLURM_JOB_PARTITION" == "$PRISM_CPU_PARTITION" ]] || prism_die "Unexpected CPU partition"
fi
if [[ "$stage" == cluster-label ]]; then prism_forward_proxy_envs; fi
if [[ "$stage" == cpu-smoke || "$stage" == stage-assets || "$stage" == prepare-data ]]; then
  prism_forward_proxy_envs
  opts+=(--env HF_HUB_OFFLINE=0 --env TRANSFORMERS_OFFLINE=0)
else
  opts+=(--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1)
fi
runner=("$PRISM_CONTAINER_RUNTIME" exec "${opts[@]}" "$PRISM_IMAGE")

if [[ "$stage" == cpu-smoke && -f "$PRISM_REPO/assets/ifeval/source-manifest.json" ]]; then
  export "${runtime_prefix}PRISM_TEST_IFEVAL_SOURCE=$PRISM_REPO/assets/ifeval"
  export "${runtime_prefix}PRISM_TEST_NLTK_DATA=$PRISM_REPO/assets/nltk_data"
fi

if [[ "$stage" == label || "$stage" == evaluate-judge ]]; then
  role=training
  [[ "$stage" != evaluate-judge ]] || role=evaluation
  # Query names in the clean container without keys or model loading. Forward
  # only the role's configured key through Apptainer's environment mechanism.
  api_env_names="$("${runner[@]}" python -m prism.cli api-env-names --config "$PRISM_CONFIG_PATH" --judge-role "$role")"
  if [[ -n "$api_env_names" ]]; then
    read -r -a api_envs <<< "$api_env_names"
    prism_forward_api_envs "${api_envs[@]}"
  fi
fi
echo "stage=$stage commit=$PRISM_COMMIT image=$PRISM_IMAGE_SHA256 runtime=$PRISM_CONTAINER_RUNTIME job=$SLURM_JOB_ID"
if [[ "$gpu" == 1 && "$stage" != smoke && "$stage" != model-smoke ]]; then
  echo "gpu_partition=$SLURM_JOB_PARTITION expected_model=${PRISM_GPU_MODEL:-A100}"
  "${runner[@]}" python /opt/prism/scripts/smoke_gpu.py
fi
if [[ "$stage" == smoke ]]; then
  "${runner[@]}" python /opt/prism/scripts/smoke_gpu.py
  "${runner[@]}" python -m pytest -q /opt/prism/tests -p no:cacheprovider --basetemp "$jobroot/pytest"
elif [[ "$stage" == cpu-smoke ]]; then
  "${runner[@]}" python /opt/prism/scripts/smoke_cpu.py "$@"
elif [[ "$stage" == model-smoke ]]; then
  "${runner[@]}" python /opt/prism/scripts/smoke_gpu.py
  "${runner[@]}" python -m prism.cli "$stage" "$@"
elif [[ "$stage" == rollout-shard ]]; then
  "${runner[@]}" python -m prism.cli "$stage" "$@" --shard-index "${SLURM_ARRAY_TASK_ID:?Use a rollout shard array}"
elif [[ "$stage" == cluster-label ]]; then
  key_file="${PRISM_TRAINING_KEY_FILE:-$PRISM_STORAGE/prism/secrets/deepseek-api.key}"
  prism_path_under "$PRISM_STORAGE" "$key_file"
  "${runner[@]}" python -m prism.cluster_judge "$@" --key-file "$key_file" --workers "${PRISM_JUDGE_WORKERS:-4}" --execute
elif [[ "$stage" == judge-export || "$stage" == judge-import ]]; then
  "${runner[@]}" python -m prism.local_judge "${stage#judge-}" "$@"
else
  "${runner[@]}" python -m prism.cli "$stage" "$@"
fi
