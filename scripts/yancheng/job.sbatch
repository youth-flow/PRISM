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
if [[ "$stage" != smoke ]]; then
  : "${PRISM_CONFIG_PATH:?}" "${PRISM_CONFIG_SHA256:?}"
  printf '%s  %s\n' "$PRISM_CONFIG_SHA256" "$PRISM_CONFIG_PATH" | sha256sum --check --status
fi
jobroot="$PRISM_WORK/jobs/$SLURM_JOB_ID"
[[ ! -e "$jobroot" ]] || prism_die "Job scratch exists; preserve it and inspect before restarting"
mkdir -p "$jobroot/source" "$jobroot/tmp" "$jobroot/home" "$PRISM_APPTAINER_CACHE"
prism_git archive "$PRISM_COMMIT" | tar -x -C "$jobroot/source"
prism_load_apptainer
export APPTAINER_CACHEDIR="$PRISM_APPTAINER_CACHE" APPTAINER_TMPDIR="$jobroot/tmp"
export APPTAINERENV_PYTHONPATH=/opt/prism/src
opts=(--cleanenv --no-home
  --bind "$jobroot/source:/opt/prism:ro" --bind "$PRISM_STORAGE:$PRISM_STORAGE"
  --bind "$jobroot/home:/prism-home" --bind "$PRISM_HF_CACHE:/hf"
  --pwd /opt/prism --env HOME=/prism-home --env XDG_CACHE_HOME=/prism-home/.cache
  --env PYTHONPATH=/opt/prism/src --env HF_HOME=/hf --env TMPDIR="$jobroot/tmp"
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" --env TOKENIZERS_PARALLELISM=false
  --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "PRISM_GPU_MODEL=${PRISM_GPU_MODEL:-A100}")
if [[ "$gpu" == 1 ]]; then
  [[ "$SLURM_JOB_PARTITION" == "$PRISM_GPU_PARTITION" ]] || prism_die "Unexpected GPU partition"
  opts+=(--nv)
  for var in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
    if [[ -v "$var" ]]; then opts+=(--env "$var=${!var}"); fi
  done
else
  [[ "$SLURM_JOB_PARTITION" == "$PRISM_CPU_PARTITION" ]] || prism_die "Unexpected CPU partition"
fi
if [[ "$stage" == stage-assets || "$stage" == prepare-data ]]; then
  opts+=(--env HF_HUB_OFFLINE=0 --env TRANSFORMERS_OFFLINE=0)
else
  opts+=(--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1)
fi
runner=(apptainer exec "${opts[@]}" "$PRISM_IMAGE")

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
echo "stage=$stage commit=$PRISM_COMMIT image=$PRISM_IMAGE_SHA256 job=$SLURM_JOB_ID"
if [[ "$stage" == smoke ]]; then
  "${runner[@]}" python /opt/prism/scripts/smoke_gpu.py
  "${runner[@]}" python -m pytest -q /opt/prism/tests -p no:cacheprovider --basetemp "$jobroot/pytest"
else
  "${runner[@]}" python -m prism.cli "$stage" "$@"
fi
