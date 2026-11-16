#!/usr/bin/env bash
# V2 cluster compute only. OpenAI always uses the local budgeted executor.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dry=0
if [[ "${1:-}" == --dry-run ]]; then dry=1; shift; fi
stage="${1:?Supply a V2 stage}"; shift
case "$stage" in
  label|evaluate-judge|cluster-label) prism_die "V2 OpenAI runs locally with prism.budgeted_judge; never reserve a GPU for API waiting" ;;
  cpu-smoke|model-smoke|validate|fixed-candidates|rollout-shard|merge-rollout|design|reward|policy|evaluate-reference|evaluate-generate|aggregate|judge-export|judge-import|v2-sample-export|v2-materialize|v2-audit-reward|v2-audit-policy) ;;
  *) prism_die "Unsupported V2 stage: $stage" ;;
esac
if prism_stage_gpu "$stage"; then
  [[ "${PRISM_GPU_PARTITION:-}" != *,* ]] || prism_die "V2 submits one verified GPU partition per job"
fi
export PRISM_V2_EXECUTION=1
if [[ "$dry" == 1 ]]; then
  exec bash "$(dirname "${BASH_SOURCE[0]}")/submit.sh" --dry-run "$stage" "$@"
else
  exec bash "$(dirname "${BASH_SOURCE[0]}")/submit.sh" "$stage" "$@"
fi
