#!/usr/bin/env bash
prism_die() { echo "PRISM: $*" >&2; exit 2; }

# Login nodes currently provide Git 1.8.3.1, predating git -C.
prism_git() { (cd "$PRISM_REPO" && git "$@"); }

# The site submission hook rejects jobs launched from the home-side online1
# symlink even when sbatch --chdir points at shared storage.  Run the submit
# process itself from the canonical repository path under /online1.
prism_exec_from_repo() { (cd "$PRISM_REPO" && "$@"); }

prism_forward_api_envs() {
  # Names come from the validated config. Values remain in the process
  # environment, never Apptainer command-line flags or output.
  local name target
  for name in "$@"; do
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || prism_die "Invalid API environment variable name"
    target="APPTAINERENV_${name}"
    if [[ -v "$name" ]]; then
      export "$target=${!name}"
    elif [[ ! -v "$target" ]]; then
      prism_die "Missing API environment variable: $name"
    fi
  done
}

prism_forward_proxy_envs() {
  # The site proxy is not a credential, but cleanenv still removes it unless
  # it is explicitly forwarded. Keep the value out of constructed argv/logs.
  local name target
  for name in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; do
    target="APPTAINERENV_${name}"
    if [[ -v "$name" ]]; then export "$target=${!name}"; fi
  done
}

prism_stage_gpu() {
  local gpu=0
  case "$1" in
    smoke|model-smoke|rollout|label|policy|evaluate-reference|evaluate-generate|evaluate-judge) gpu=1 ;;
    cpu-smoke|stage-assets|prepare-data|design|sample|reward|aggregate|validate|judge-export|judge-import) gpu=0 ;;
    *) prism_die "Unknown stage: $1" ;;
  esac
  # The Python config/backend layer computes resource needs. Bash receives that
  # result explicitly instead of duplicating provider-specific YAML parsing.
  case "${PRISM_STAGE_RESOURCE:-}" in
    gpu) gpu=1 ;;
    cpu) gpu=0 ;;
    "") ;;
    *) prism_die "Stage resource must be cpu or gpu" ;;
  esac
  [[ "$gpu" == 1 ]]
}

prism_stage_resources() {
  # Set conservative one-process defaults. CPU memory follows q_amd_share's
  # published 1500 MB/core limit; GPU jobs retain 8 CPUs and 64 GB host RAM.
  local stage="$1" gpu="$2"
  if [[ "$gpu" == 1 ]]; then
    PRISM_DEFAULT_CPUS=8 PRISM_DEFAULT_MEMORY=64G
    case "$stage" in
      smoke|model-smoke) PRISM_DEFAULT_TIME=01:00:00 ;;
      rollout) PRISM_DEFAULT_TIME=24:00:00 ;;
      policy|evaluate-reference|evaluate-generate|label|evaluate-judge) PRISM_DEFAULT_TIME=12:00:00 ;;
      *) prism_die "No GPU resource profile for stage: $stage" ;;
    esac
  else
    case "$stage" in
      cpu-smoke) PRISM_DEFAULT_CPUS=1 PRISM_DEFAULT_MEMORY=1500M PRISM_DEFAULT_TIME=00:05:00 ;;
      stage-assets) PRISM_DEFAULT_CPUS=4 PRISM_DEFAULT_MEMORY=6G PRISM_DEFAULT_TIME=04:00:00 ;;
      prepare-data) PRISM_DEFAULT_CPUS=16 PRISM_DEFAULT_MEMORY=24G PRISM_DEFAULT_TIME=06:00:00 ;;
      design) PRISM_DEFAULT_CPUS=16 PRISM_DEFAULT_MEMORY=24G PRISM_DEFAULT_TIME=12:00:00 ;;
      sample) PRISM_DEFAULT_CPUS=1 PRISM_DEFAULT_MEMORY=1500M PRISM_DEFAULT_TIME=00:30:00 ;;
      reward) PRISM_DEFAULT_CPUS=8 PRISM_DEFAULT_MEMORY=12G PRISM_DEFAULT_TIME=04:00:00 ;;
      label) PRISM_DEFAULT_CPUS=1 PRISM_DEFAULT_MEMORY=1500M PRISM_DEFAULT_TIME=12:00:00 ;;
      evaluate-judge) PRISM_DEFAULT_CPUS=1 PRISM_DEFAULT_MEMORY=1500M PRISM_DEFAULT_TIME=18:00:00 ;;
      aggregate|validate|judge-export|judge-import) PRISM_DEFAULT_CPUS=1 PRISM_DEFAULT_MEMORY=1500M PRISM_DEFAULT_TIME=01:00:00 ;;
      *) prism_die "No CPU resource profile for stage: $stage" ;;
    esac
  fi
}

prism_path_under() {
  # realpath -m follows existing symlinks and also handles paths not yet created.
  local base target
  base="$(realpath -m -- "$1")"; target="$(realpath -m -- "$2")"
  [[ "$target" == "$base/"* ]] || prism_die "Path outside shared storage: $2"
}

prism_check_paths() {
  : "${PRISM_STORAGE:?Source the Yancheng env file}"
  [[ -d "$PRISM_STORAGE" ]] || prism_die "Shared storage missing: $PRISM_STORAGE"
  [[ "$(realpath -m "$PRISM_STORAGE")" == /online1/* || "${PRISM_TEST_MODE:-0}" == 1 ]] || prism_die "Storage must resolve under /online1"
  local key
  for key in PRISM_REPO PRISM_IMAGE PRISM_HF_CACHE PRISM_WORK PRISM_LOGS PRISM_APPTAINER_CACHE PRISM_APPTAINER_TMP; do
    [[ -n "${!key:-}" && "${!key}" == /* ]] || prism_die "Need absolute $key"
    prism_path_under "$PRISM_STORAGE" "${!key}"
  done
}

prism_load_apptainer() {
  if ! type module >/dev/null 2>&1; then
    [[ -r /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
  fi
  type module >/dev/null 2>&1 || prism_die "Environment Modules unavailable"
  module load "${PRISM_APPTAINER_MODULE:-amd/apptainer/1.2.5}"
  command -v apptainer >/dev/null || prism_die "Apptainer module did not provide executable"
}
