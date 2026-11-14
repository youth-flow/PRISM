#!/usr/bin/env bash
# Source a copy kept OUTSIDE the Git checkout. No passwords belong in this file.
export PRISM_STORAGE="$(readlink -f "${HOME}/online1")"
export PRISM_ROOT="${PRISM_STORAGE}/prism"
export PRISM_REPO="${PRISM_STORAGE}/PRISM"
export PRISM_IMAGE="${PRISM_ROOT}/images/prism.sif"
export PRISM_IMAGE_SHA256="FILL_FROM_VERIFIED_RUNTIME_BUILD"
export PRISM_HF_CACHE="${PRISM_ROOT}/hf-cache"
export PRISM_WORK="${PRISM_ROOT}/work"
export PRISM_LOGS="${PRISM_ROOT}/logs"
export PRISM_APPTAINER_CACHE="${PRISM_ROOT}/apptainer-cache"
export PRISM_APPTAINER_TMP="${PRISM_ROOT}/apptainer-tmp"
export PRISM_APPTAINER_MODULE="amd/apptainer/1.2.5"
# Site-provided setuid Singularity can be selected on nodes where the rootless
# Apptainer user-namespace workflow is unavailable. The SIF stays unchanged.
export PRISM_CONTAINER_RUNTIME="singularity"  # Verified on gpu056 with the unchanged SIF.
export PRISM_CONTAINER_MODULE="amd/singularity/3.10.4"
# Verified account/partitions on 2026-11-14, unlike the older manual examples.
export PRISM_ACCOUNT="sc001"
export PRISM_GPU_PARTITION="q_amd_gpu_nvidia_A100_share"
export PRISM_CPU_PARTITION="q_amd_share"
export PRISM_QOS=""                 # Account offers normal,standby; keep site default.
export PRISM_GPU_GRES="gpu:tesla:1" # One process, one A100; no exclusive node.
export PRISM_GPU_MODEL="A100"
# submit.sh supplies audited per-stage defaults. Set any of these only as an
# explicit one-invocation override after inspecting the relevant pilot log.
# export PRISM_CPUS=8
# export PRISM_MEMORY=64G
# export PRISM_TIME=08:00:00
export http_proxy="http://174.0.250.13:3128"
export https_proxy="http://174.0.250.13:3128"
export PRISM_EXCLUDE=""
# Platform preparation only. Enable explicitly when ready for scheduled work.
export PRISM_ALLOW_SUBMIT=0
