#!/usr/bin/env bash
prism_die() { echo "PRISM: $*" >&2; exit 2; }

# Login nodes currently provide Git 1.8.3.1, predating git -C.
prism_git() { (cd "$PRISM_REPO" && git "$@"); }

prism_stage_gpu() {
  case "$1" in
    smoke|model-smoke|rollout|label|policy|evaluate-generate|evaluate-judge) return 0 ;;
    stage-assets|design|sample|reward|aggregate|validate) return 1 ;;
    *) prism_die "Unknown stage: $1" ;;
  esac
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
