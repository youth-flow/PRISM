#!/usr/bin/env bash
# Copy OUTSIDE Git; fill verified paths and hashes. No credentials here.
export PRISM_STORAGE="$(readlink -f "${HOME}/online1")"
export PRISM_ROOT="$PRISM_STORAGE/prism"
export PRISM_REPO="$PRISM_ROOT/v4-source-repository" # immutable checkout of pushed v4 commit
export PRISM_IMAGE="$PRISM_ROOT/images/prism.sif" # existing, unchanged image
export PRISM_IMAGE_SHA256="c1b4e80d731cbb3124825ce81c3dcd5226865478cf051db165ec27a0856d6896"
export PRISM_V4_RUNTIME="$PRISM_ROOT/v4-runtime/tf4571-01"
export PRISM_V4_RUNTIME_SHA256="FILL_FROM_RUNTIME_BOOTSTRAP"
export PRISM_V4_WHEELS="$PRISM_ROOT/v4-inputs/wheels"
export PRISM_V4_WHEELS_SHA256="FILL_WITH_SHA256_OF_WHEELS_MANIFEST"
export PRISM_HF_CACHE="$PRISM_ROOT/hf-cache"
export PRISM_WORK="$PRISM_ROOT/v4-work"
export PRISM_LOGS="$PRISM_ROOT/v4-logs"
export PRISM_APPTAINER_CACHE="$PRISM_ROOT/apptainer-cache"
export PRISM_APPTAINER_TMP="$PRISM_ROOT/v4-tmp"
export PRISM_CONTAINER_RUNTIME=singularity
export PRISM_CONTAINER_MODULE=amd/singularity/3.10.4
export PRISM_ACCOUNT=sc001
export PRISM_CPU_PARTITION=q_amd_share
export PRISM_GPU_PARTITION=q_amd_gpu_nvidia_A100_share
export PRISM_GPU_GRES=gpu:tesla:1
export PRISM_GPU_MODEL=A100
export PRISM_EXCLUDE=bn002,bn009
export PRISM_ALLOW_SUBMIT=0
# Configure the official site HTTP proxy privately if required for downloads.
# Alternate cards require their own real 7B acceptance; do not reuse 1B receipts.
# Keep PRISM_V4_POLICY_ADMISSION unset until the V4 Reward review authorizes PPO.
