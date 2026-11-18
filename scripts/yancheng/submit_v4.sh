#!/usr/bin/env bash
# Compute-only V4 submission. Dry runs never create snapshots or call Slurm.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dry=0
if [[ "${1:-}" == --dry-run ]]; then dry=1; shift; fi
stage="${1:?Supply a V4 compute stage}"; shift
gpu=0; cpus=2; memory=3G; duration=01:00:00
case "$stage" in
  extract-features|linear-test-features|prepare-shared|evaluate-generate) gpu=1; cpus=4; memory=64G; duration=12:00:00 ;;
  policy)
    [[ -f "${PRISM_V4_POLICY_ADMISSION:-}" ]] || prism_die "V4 formal PPO admission is closed; run policy-smoke for engineering"
    gpu=1; cpus=8; memory=64G; duration=48:00:00 ;;
  model-smoke|shared-smoke|policy-smoke) gpu=1; cpus=8; memory=64G; duration=01:00:00 ;;
  download-model|download-data) cpus=4; memory=6G; duration=04:00:00 ;;
  geometry|geometry-audit|design|freeze-study|audit) cpus=32; memory=48G; duration=12:00:00 ;;
  prepare-data) cpus=16; memory=24G; duration=04:00:00 ;;
  merge-features) cpus=16; memory=24G; duration=04:00:00 ;;
  reward|audit-training|full-pool-reward|full-pool-reward-report) cpus=32; memory=48G; duration=04:00:00 ;;
  linear-test-data) cpus=8; memory=12G; duration=02:00:00 ;;
  linear-reward-train|linear-reward-report) cpus=16; memory=24G; duration=02:00:00 ;;
  cpu-smoke|runtime-bootstrap) cpus=4; memory=6G; duration=01:00:00 ;;
  validate|plan|status|runtime-check|cohort-audit|design-report|sample|merge-shared|normalize-reward|merge-evaluation|judge-export|judge-import|aggregate) ;;
  *) prism_die "Unsupported V4 compute stage; paid Judges run locally: $stage" ;;
esac
prism_check_paths
: "${PRISM_ACCOUNT:?}" "${PRISM_GPU_PARTITION:?}" "${PRISM_CPU_PARTITION:?}"
[[ -z "$(prism_git status --porcelain --untracked-files=no)" ]] || prism_die "Commit tracked V4 changes before submission"
export PRISM_COMMIT PRISM_CONFIG_PATH="" PRISM_CONFIG_SHA256="" PRISM_V4_SOURCE_DIR PRISM_STAGE_RESOURCE
PRISM_COMMIT="$(prism_git rev-parse HEAD)"
PRISM_V4_SOURCE_DIR="$PRISM_WORK/releases/$PRISM_COMMIT"
PRISM_STAGE_RESOURCE=cpu
[[ "$gpu" == 0 ]] || PRISM_STAGE_RESOURCE=gpu
if [[ "$stage" != cpu-smoke ]]; then
  cli=("$@")
  for ((i=0; i<${#cli[@]}; i++)); do
    if [[ "${cli[$i]}" == --config && $((i+1)) -lt ${#cli[@]} ]]; then PRISM_CONFIG_PATH="${cli[$((i+1))]}"; fi
  done
  [[ "$PRISM_CONFIG_PATH" == /* && -f "$PRISM_CONFIG_PATH" ]] || prism_die "Supply --config /absolute/path/to/v4.yaml"
  prism_path_under "$PRISM_STORAGE" "$PRISM_CONFIG_PATH"
  PRISM_CONFIG_SHA256="$(sha256sum "$PRISM_CONFIG_PATH" | cut -d ' ' -f 1)"
fi
args=(--parsable --export=ALL --account="$PRISM_ACCOUNT" --nodes=1 --ntasks=1
  --cpus-per-task="${PRISM_CPUS:-$cpus}" --mem="${PRISM_MEMORY:-$memory}" --time="${PRISM_TIME:-$duration}"
  --chdir="$PRISM_V4_SOURCE_DIR" --job-name="prism-v4-$stage"
  --output="$PRISM_LOGS/%x-%j.out" --error="$PRISM_LOGS/%x-%j.err")
if [[ "$gpu" == 1 ]]; then
  [[ "$PRISM_GPU_PARTITION" != *,* ]] || prism_die "V4 uses one verified GPU partition per submission"
  [[ "${PRISM_GPU_GRES:-gpu:1}" =~ ^gpu(:[A-Za-z0-9_-]+)?:1$ ]] || prism_die "Exactly one GPU per V4 task"
  args+=(--partition="$PRISM_GPU_PARTITION" --gres="${PRISM_GPU_GRES:-gpu:1}")
else
  [[ "$PRISM_CPU_PARTITION" != FILL_* && "$PRISM_CPU_PARTITION" != "$PRISM_GPU_PARTITION" ]] || prism_die "Use a verified CPU partition"
  args+=(--partition="$PRISM_CPU_PARTITION")
fi
if [[ "$gpu" == 0 && "$PRISM_CPU_PARTITION" == q_amd_share ]]; then
  requested_cpus="${PRISM_CPUS:-$cpus}"; requested_memory="${PRISM_MEMORY:-$memory}"
  [[ "$requested_cpus" =~ ^[1-9][0-9]*$ && "$requested_memory" =~ ^([1-9][0-9]*)([MG])$ ]] || prism_die "Use integer CPU cores and M/G memory for the verified CPU profile"
  memory_mib="${BASH_REMATCH[1]}"; unit="${BASH_REMATCH[2]}"
  [[ "$unit" != G ]] || memory_mib=$((memory_mib*1024))
  [[ "$memory_mib" -le $((requested_cpus*1536)) ]] || prism_die "q_amd_share memory/core exceeds the verified 1.5GiB profile; choose a matching CPU/memory allocation"
fi
[[ -z "${PRISM_QOS:-}" ]] || args+=(--qos="$PRISM_QOS")
[[ -z "${PRISM_EXCLUDE:-}" ]] || args+=(--exclude="$PRISM_EXCLUDE")
if [[ -n "${PRISM_DEPENDENCY:-}" ]]; then
  [[ "$PRISM_DEPENDENCY" =~ ^[0-9]+(:[0-9]+)*$ ]] || prism_die "Dependency must be colon-separated numeric job IDs"
  args+=(--dependency="afterok:$PRISM_DEPENDENCY")
fi
new_tasks=1
if [[ -n "${PRISM_ARRAY:-}" ]]; then
  [[ "$stage" == extract-features || "$stage" == evaluate-generate || "$stage" == prepare-shared ]] || prism_die "Only independent feature/generation/scoring shards use arrays"
  [[ "$PRISM_ARRAY" =~ ^([0-9]+)-([0-9]+)%([0-9]+)$ ]] || prism_die "Array needs FIRST-LAST%CONCURRENCY"
  first=$((10#${BASH_REMATCH[1]})); last=$((10#${BASH_REMATCH[2]})); concurrent=$((10#${BASH_REMATCH[3]}))
  [[ "$first" -le "$last" && "$concurrent" -ge 1 && "$concurrent" -le 50 ]] || prism_die "Invalid array range/concurrency"
  shards=""
  for ((i=0; i<${#cli[@]}; i++)); do
    if [[ "${cli[$i]}" == --shards && $((i+1)) -lt ${#cli[@]} ]]; then shards="${cli[$((i+1))]}"; fi
    [[ "${cli[$i]}" != --shard-index ]] || prism_die "Array shard index comes from Slurm"
  done
  [[ "$shards" =~ ^[0-9]+$ ]] || prism_die "Array requires --shards"
  [[ "$last" -lt "$shards" ]] || prism_die "Array outside global shard coverage"
  new_tasks=$((last-first+1))
  [[ "$new_tasks" -le 50 ]] || prism_die "More than 50 queued V4 array tasks"
  args+=(--array="$PRISM_ARRAY")
fi
command=(sbatch "${args[@]}" "$PRISM_V4_SOURCE_DIR/scripts/yancheng/run_v4.sh" "$stage" "$@")
if [[ "$dry" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
[[ "${PRISM_TEST_MODE:-0}" != 1 ]] || prism_die "Test mode allows dry runs only"
[[ "${PRISM_ALLOW_SUBMIT:-0}" == 1 ]] || prism_die "Submission disabled; inspect --dry-run first"
prism_path_under "$PRISM_STORAGE" "${PRISM_V4_RUNTIME:?Set the isolated V4 runtime path}"
if [[ "$stage" == runtime-bootstrap ]]; then
  [[ ! -e "$PRISM_V4_RUNTIME" ]] || prism_die "Never overwrite a V4 runtime layer"
  prism_path_under "$PRISM_STORAGE" "${PRISM_V4_WHEELS:?}"
  printf '%s  %s\n' "${PRISM_V4_WHEELS_SHA256:?}" "$PRISM_V4_WHEELS/wheels.sha256" | sha256sum --check --status
else
  [[ "${PRISM_V4_RUNTIME_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || prism_die "Verify the V4 runtime manifest first"
  printf '%s  %s\n' "$PRISM_V4_RUNTIME_SHA256" "$PRISM_V4_RUNTIME/runtime.sha256" | sha256sum --check --status
fi
command -v squeue >/dev/null || prism_die "Slurm squeue unavailable"
active="$(squeue -h -r -u "${USER:?}" -o '%i' | awk 'NF {n++} END {print n+0}')"
[[ "$active" =~ ^[0-9]+$ && $((active+new_tasks)) -le 50 ]] || prism_die "Personal running/pending tasks would exceed 50"
[[ "${PRISM_IMAGE_SHA256:-}" =~ ^[0-9a-f]{64}$ && -f "$PRISM_IMAGE" ]] || prism_die "Verified runtime image required"
printf '%s  %s\n' "$PRISM_IMAGE_SHA256" "$PRISM_IMAGE" | sha256sum --check --status
mkdir -p "$PRISM_LOGS" "$PRISM_WORK/releases" "$PRISM_HF_CACHE" "$PRISM_APPTAINER_CACHE" "$PRISM_APPTAINER_TMP"
prism_path_under "$PRISM_STORAGE" "$PRISM_V4_SOURCE_DIR"
if [[ ! -d "$PRISM_V4_SOURCE_DIR" ]]; then
  staging="$(mktemp -d "$PRISM_WORK/releases/.${PRISM_COMMIT}.XXXXXX")"
  prism_git archive "$PRISM_COMMIT" | tar -x -C "$staging"
  (cd "$staging" && find . -type f ! -name .source.sha256 ! -name .source.commit -print0 | sort -z | xargs -0 sha256sum > .source.sha256)
  printf '%s\n' "$PRISM_COMMIT" > "$staging/.source.commit"
  mv -- "$staging" "$PRISM_V4_SOURCE_DIR"
fi
[[ "$(cat "$PRISM_V4_SOURCE_DIR/.source.commit")" == "$PRISM_COMMIT" ]] || prism_die "Immutable V4 source commit differs"
(cd "$PRISM_V4_SOURCE_DIR" && sha256sum --check --status .source.sha256)
export PRISM_V4_SOURCE_MANIFEST_SHA256
PRISM_V4_SOURCE_MANIFEST_SHA256="$(sha256sum "$PRISM_V4_SOURCE_DIR/.source.sha256" | cut -d ' ' -f 1)"
# Never forward an OpenAI credential into compute jobs or the container.
unset OPENAI_API_KEY APPTAINERENV_OPENAI_API_KEY SINGULARITYENV_OPENAI_API_KEY
prism_exec_from_repo "${command[@]}"
