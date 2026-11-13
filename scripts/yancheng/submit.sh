#!/usr/bin/env bash
# --dry-run prints only; default profile prevents all real submission.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dry=0
resource="${PRISM_STAGE_RESOURCE:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) dry=1; shift ;;
    --resource) resource="${2:?Supply cpu or gpu}"; shift 2 ;;
    *) break ;;
  esac
done
[[ -z "$resource" || "$resource" == cpu || "$resource" == gpu ]] || prism_die "Resource must be cpu or gpu"
export PRISM_STAGE_RESOURCE="$resource"
stage="${1:?Supply a stage}"; shift
if prism_stage_gpu "$stage"; then gpu=1; else gpu=0; fi
prism_check_paths
: "${PRISM_ACCOUNT:?}" "${PRISM_GPU_PARTITION:?}" "${PRISM_CPU_PARTITION:?}"
export PRISM_COMMIT PRISM_CONFIG_PATH="" PRISM_CONFIG_SHA256=""
PRISM_COMMIT="$(prism_git rev-parse HEAD)"
[[ -z "$(prism_git status --porcelain)" ]] || prism_die "Commit source first; checkout must be clean"
if [[ "$stage" != smoke ]]; then
  cli=("$@")
  for ((i=0; i<${#cli[@]}; i++)); do
    if [[ "${cli[$i]}" == --config && $((i+1)) -lt ${#cli[@]} ]]; then PRISM_CONFIG_PATH="${cli[$((i+1))]}"; fi
  done
  [[ "$PRISM_CONFIG_PATH" == /* && -f "$PRISM_CONFIG_PATH" ]] || prism_die "Supply --config /absolute/path/to/frozen.yaml"
  prism_path_under "$PRISM_STORAGE" "$PRISM_CONFIG_PATH"
  PRISM_CONFIG_SHA256="$(sha256sum "$PRISM_CONFIG_PATH" | cut -d ' ' -f 1)"
fi
args=(--parsable --export=ALL --account="$PRISM_ACCOUNT" --nodes=1 --ntasks=1
  --cpus-per-task="${PRISM_CPUS:-8}" --mem="${PRISM_MEMORY:-64G}" --time="${PRISM_TIME:-08:00:00}"
  --chdir="$PRISM_REPO" --job-name="prism-$stage" --output="$PRISM_LOGS/%x-%j.out" --error="$PRISM_LOGS/%x-%j.err")
if [[ "$gpu" == 1 ]]; then
  [[ "${PRISM_GPU_GRES:-gpu:1}" =~ ^gpu(:[A-Za-z0-9_-]+)?:1$ ]] || prism_die "v1 requires exactly one GPU"
  args+=(--partition="$PRISM_GPU_PARTITION" --gres="${PRISM_GPU_GRES:-gpu:1}")
else
  args+=(--partition="$PRISM_CPU_PARTITION")
fi
if [[ -n "${PRISM_QOS:-}" ]]; then args+=(--qos="$PRISM_QOS"); fi
if [[ -n "${PRISM_EXCLUDE:-}" ]]; then args+=(--exclude="$PRISM_EXCLUDE"); fi
if [[ -n "${PRISM_DEPENDENCY:-}" ]]; then
  [[ "$PRISM_DEPENDENCY" =~ ^[0-9]+$ ]] || prism_die "Dependency must be one numeric job ID"
  args+=(--dependency="afterok:$PRISM_DEPENDENCY")
fi
command=(sbatch "${args[@]}" "$PRISM_REPO/scripts/yancheng/job.sbatch" "$stage" "$@")
if [[ "$dry" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
[[ "${PRISM_TEST_MODE:-0}" != 1 ]] || prism_die "Test mode can only construct dry runs"
[[ "${PRISM_ALLOW_SUBMIT:-0}" == 1 ]] || prism_die "Submission disabled during platform setup; use --dry-run"
[[ "${PRISM_IMAGE_SHA256:-}" =~ ^[0-9a-f]{64}$ && -f "$PRISM_IMAGE" ]] || prism_die "Install and verify the runtime image first"
printf '%s  %s\n' "$PRISM_IMAGE_SHA256" "$PRISM_IMAGE" | sha256sum --check --status
mkdir -p "$PRISM_LOGS" "$PRISM_WORK" "$PRISM_HF_CACHE" "$PRISM_APPTAINER_CACHE" "$PRISM_APPTAINER_TMP"
prism_exec_from_repo "${command[@]}"
