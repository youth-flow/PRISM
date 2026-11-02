#!/usr/bin/env bash
set -euo pipefail
umask 027
: "${SLURM_JOB_ID:?Must run under Slurm}"
: "${PRISM_REPO:?}"
: "${PRISM_IMAGE:?}"
: "${PRISM_IMAGE_SHA256:?}"
: "${PRISM_COMMIT:?}"
: "${PRISM_HF_CACHE:?}"
: "${PRISM_WORK:?}"
stage="${1:?}"; shift
[[ "$(git -C "$PRISM_REPO" rev-parse HEAD)" == "$PRISM_COMMIT" ]]
[[ -z "$(git -C "$PRISM_REPO" status --porcelain)" ]]
printf '%s  %s\n' "$PRISM_IMAGE_SHA256" "$PRISM_IMAGE" | sha256sum --check --status
if [[ "$stage" != smoke ]]; then
  : "${PRISM_CONFIG_PATH:?}" "${PRISM_CONFIG_SHA256:?}"
  printf '%s  %s\n' "$PRISM_CONFIG_SHA256" "$PRISM_CONFIG_PATH" | sha256sum --check --status
fi
# Copy the committed source into this job's scratch space, so queued jobs cannot
# read edits made to the shared checkout halfway through a stage.
jobroot="$PRISM_WORK/jobs/$SLURM_JOB_ID"
mkdir -p "$jobroot/source" "$jobroot/tmp"
git -C "$PRISM_REPO" archive "$PRISM_COMMIT" | tar -x -C "$jobroot/source"
export APPTAINER_TMPDIR="$jobroot/tmp"
opts=(--cleanenv --no-mount /opt/knem-1.1.4.90mlnx3
  --bind "$jobroot/source:/opt/prism:ro" --bind "$PRISM_HF_CACHE:/hf"
  --bind "/project/sigroup:/project/sigroup" --bind "$PRISM_WORK:$PRISM_WORK"
  --pwd "$PRISM_REPO" --env PYTHONPATH=/opt/prism/src --env HF_HOME=/hf
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" --env TOKENIZERS_PARALLELISM=false
  --env "SLURM_JOB_ID=$SLURM_JOB_ID")
case "$stage" in
  smoke|model-smoke|rollout|label|policy|evaluate-generate|evaluate-judge)
    [[ "${SLURM_JOB_PARTITION}" == "${PRISM_GPU_PARTITION:-gpu-l20}" ]]
    opts+=(--nv)
    for var in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
      if [[ -v "$var" ]]; then opts+=(--env "$var=${!var}"); fi
    done ;;
esac
if [[ "$stage" == stage-assets ]]; then
  opts+=(--env HF_HUB_OFFLINE=0 --env TRANSFORMERS_OFFLINE=0)
else
  opts+=(--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1)
fi
runner=(apptainer exec "${opts[@]}" "$PRISM_IMAGE")
echo "stage=$stage commit=$PRISM_COMMIT image=$PRISM_IMAGE_SHA256 job=$SLURM_JOB_ID"
if [[ "$stage" == smoke ]]; then
  "${runner[@]}" python /opt/prism/scripts/smoke_gpu.py
  "${runner[@]}" python -m pytest -q /opt/prism/tests --basetemp "$jobroot/pytest"
else
  "${runner[@]}" python -m prism.cli "$stage" "$@"
fi
