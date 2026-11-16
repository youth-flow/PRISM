#!/usr/bin/env bash
# Deploy an immutable tracked-source bundle. Does not submit a job.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
bundle="" commit="" target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) bundle="${2:?}"; shift 2 ;;
    --commit) commit="${2:?}"; shift 2 ;;
    --target) target="${2:?}"; shift 2 ;;
    *) prism_die "Unknown release argument: $1" ;;
  esac
done
: "${PRISM_STORAGE:?Source private env.sh first}"
[[ "$commit" =~ ^[0-9a-f]{40}$ && -f "$bundle" && "$target" == /* ]] || prism_die "Need --bundle, full --commit and absolute --target"
prism_path_under "$PRISM_STORAGE" "$bundle"
prism_path_under "$PRISM_STORAGE" "$target"
[[ "$(realpath -m "$target")" == "$(realpath -m "$PRISM_STORAGE")/prism/source-releases/"* ]] || prism_die "Release target must be under prism/source-releases"
[[ ! -e "$target" ]] || prism_die "Release already exists; never overwrite queued/running source"
mkdir -p "$(dirname "$target")"
git clone --no-checkout "$bundle" "$target"
(cd "$target" && git checkout --detach "$commit" && test "$(git rev-parse HEAD)" = "$commit" && test -z "$(git status --porcelain)")
printf 'release=%s commit=%s\n' "$target" "$commit"
