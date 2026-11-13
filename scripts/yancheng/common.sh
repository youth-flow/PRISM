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

prism_stage_gpu() {
  local gpu=0
  case "$1" in
    smoke|model-smoke|rollout|label|policy|evaluate-reference|evaluate-generate|evaluate-judge) gpu=1 ;;
    stage-assets|prepare-data|design|sample|reward|aggregate|validate) gpu=0 ;;
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
