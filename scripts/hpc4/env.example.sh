#!/usr/bin/env bash
# Source a copy outside the tracked checkout. These are the prior HPC4 account paths.
export PRISM_REPO="/project/sigroup/${USER}/PRISM"
export PRISM_IMAGE="/project/sigroup/${USER}/prism/images/prism.sif"
export PRISM_IMAGE_SHA256="FILL_FROM_VERIFIED_BUILD_ARTIFACT"
export PRISM_HF_CACHE="/project/sigroup/${USER}/prism/hf-cache"
export PRISM_WORK="/scratch/${USER}/prism"
export PRISM_LOGS="/project/sigroup/${USER}/prism/logs"
export PRISM_ACCOUNT="sigroup"
export PRISM_GPU_PARTITION="gpu-l20"
export PRISM_CPU_PARTITION="amd"
# Site availability can change. Do not carry forward a historical bad-node list blindly.
export PRISM_EXCLUDE=""
export PRISM_CPUS=8
export PRISM_MEMORY=64G
export PRISM_TIME=08:00:00
