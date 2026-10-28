#!/usr/bin/env bash
set -euo pipefail
# Default: use the installed GPU renderer. For CPU rendering, supply a working
# Mesa Vulkan ICD through VK_ICD_FILENAMES and set ROBOMME_CPU_RENDER=1.
if [[ "${ROBOMME_CPU_RENDER:-0}" == "1" ]]; then
    : "${VK_ICD_FILENAMES:?Set VK_ICD_FILENAMES to your installed Mesa lavapipe ICD}"
    export SAPIEN_RENDER_DEVICE=cpu
    export ROBOMME_RENDER_BACKEND="${ROBOMME_RENDER_BACKEND:-pci:0000:00:00.0}"
    export MUJOCO_GL=osmesa
fi
exec "$@"
