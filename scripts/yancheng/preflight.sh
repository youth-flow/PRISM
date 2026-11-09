#!/usr/bin/env bash
# Metadata only: never invokes sbatch, srun, Apptainer exec, Torch, or a model.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
prism_check_paths
echo "host=$(hostname)"
echo "storage=$(readlink -f "$PRISM_STORAGE")"
echo "source_commit=$(prism_git rev-parse HEAD)"
prism_git status --short --branch
df -h "$PRISM_STORAGE"
sinfo -h -o '%P|%a|%l|%G|%f'
sacctmgr -nP show assoc where user="$USER" format=Account,Partition,QOS
if type module >/dev/null 2>&1; then module show "${PRISM_APPTAINER_MODULE:-amd/apptainer/1.2.5}"; fi
if [[ -f "$PRISM_IMAGE" && "${PRISM_IMAGE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s  %s\n' "$PRISM_IMAGE_SHA256" "$PRISM_IMAGE" | sha256sum --check
else
  echo "runtime_image=pending (no container executed)"
fi
echo "submission_enabled=${PRISM_ALLOW_SUBMIT:-0}"
