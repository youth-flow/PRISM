#!/usr/bin/env bash
# Usage: bash scripts/hpc4/submit.sh [--dry-run] STAGE [CLI args...]
set -euo pipefail
dry=0
if [[ "${1:-}" == --dry-run ]]; then dry=1; shift; fi
stage="${1:?Supply a stage}"; shift
case "$stage" in
  smoke|model-smoke|rollout|label|policy|evaluate-generate|evaluate-judge) gpu=1 ;;
  stage-assets|design|sample|reward|aggregate|validate) gpu=0 ;;
  *) echo "Unsupported stage: $stage" >&2; exit 2 ;;
esac
: "${PRISM_REPO:?Source env.local.sh first}"
: "${PRISM_IMAGE:?}"
: "${PRISM_IMAGE_SHA256:?}"
: "${PRISM_HF_CACHE:?}"
: "${PRISM_WORK:?}"
: "${PRISM_LOGS:?}"
[[ "$PRISM_IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "Need verified image SHA256" >&2; exit 2; }
export PRISM_CONFIG_PATH="" PRISM_CONFIG_SHA256=""
if [[ "$stage" != smoke ]]; then
  cli=("$@")
  for ((i=0; i<${#cli[@]}; i++)); do
    if [[ "${cli[$i]}" == --config && $((i+1)) -lt ${#cli[@]} ]]; then
      PRISM_CONFIG_PATH="${cli[$((i+1))]}"
    fi
  done
  [[ "$PRISM_CONFIG_PATH" == /* && -f "$PRISM_CONFIG_PATH" ]] || { echo "HPC jobs require --config /absolute/path/to/frozen.yaml" >&2; exit 2; }
  PRISM_CONFIG_SHA256="$(sha256sum "$PRISM_CONFIG_PATH" | cut -d ' ' -f 1)"
fi
export PRISM_COMMIT
PRISM_COMMIT="$(git -C "$PRISM_REPO" rev-parse HEAD)"
[[ -z "$(git -C "$PRISM_REPO" status --porcelain)" ]] || { echo "Commit source before submitting; checkout must be clean" >&2; exit 2; }
args=(--parsable --account="${PRISM_ACCOUNT:-sigroup}" --nodes=1 --ntasks=1 --cpus-per-task="${PRISM_CPUS:-8}" --mem="${PRISM_MEMORY:-64G}" --time="${PRISM_TIME:-08:00:00}" --job-name="prism-$stage" --output="$PRISM_LOGS/%x-%j.log")
if [[ "$gpu" == 1 ]]; then
  args+=(--partition="${PRISM_GPU_PARTITION:-gpu-l20}" --gpus-per-node=1)
else
  args+=(--partition="${PRISM_CPU_PARTITION:-amd}")
fi
if [[ -n "${PRISM_EXCLUDE:-}" ]]; then args+=(--exclude="$PRISM_EXCLUDE"); fi
if [[ -n "${PRISM_DEPENDENCY:-}" ]]; then
  [[ "$PRISM_DEPENDENCY" =~ ^[0-9]+$ ]] || { echo "Dependency must be one job ID" >&2; exit 2; }
  args+=(--dependency="afterok:$PRISM_DEPENDENCY")
fi
command=(sbatch "${args[@]}" "$PRISM_REPO/scripts/hpc4/job.sbatch" "$stage" "$@")
if [[ "$dry" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
mkdir -p "$PRISM_LOGS" "$PRISM_WORK" "$PRISM_HF_CACHE"
"${command[@]}"
