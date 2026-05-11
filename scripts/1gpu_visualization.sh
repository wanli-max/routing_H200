#!/bin/bash
# Single-GPU launcher for response + reasoning-weight visualization.
#
# Usage:
#   cd /projects_vol/gp_boan/EasyR1
#   bash scripts/1gpu_visualization.sh /path/to/global_step_80 [OUTPUT_DIR]

set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] || [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
      awk -F', ' -v uuids="$CUDA_VISIBLE_DEVICES" \
      'BEGIN{split(uuids,u,",")} {for(i in u) if($2==u[i]) printf "%s%s",(n++?",":""),$1}')
    echo "[INFO] Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

CHECKPOINT_PATH=${1:-""}
OUTPUT_DIR=${2:-""}
MAX_SAMPLES=${MAX_SAMPLES:-3}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}

if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "Usage:"
    echo "  bash scripts/1gpu_visualization.sh CHECKPOINT_PATH [OUTPUT_DIR]"
    exit 1
fi

if [[ "$(basename "${CHECKPOINT_PATH}")" == "actor" ]]; then
    ACTOR_DIR="${CHECKPOINT_PATH}"
else
    ACTOR_DIR="${CHECKPOINT_PATH}/actor"
fi

if [[ ! -d "${ACTOR_DIR}" ]]; then
    echo "[ERROR] Actor checkpoint directory not found: ${ACTOR_DIR}"
    exit 1
fi

MERGED_DIR="${ACTOR_DIR}/huggingface"
HAS_MERGED_WEIGHTS=0
if compgen -G "${MERGED_DIR}/*.safetensors" > /dev/null || compgen -G "${MERGED_DIR}/pytorch_model*.bin" > /dev/null; then
    HAS_MERGED_WEIGHTS=1
fi

if [[ "${HAS_MERGED_WEIGHTS}" != "1" ]]; then
    echo "[INFO] No merged Hugging Face weights found under ${MERGED_DIR}"
    echo "[INFO] Running model merger first..."
    python3 scripts/model_merger.py --local_dir "${ACTOR_DIR}"
fi

echo "=================================================="
echo "  EasyR1 Reasoning Weight Visualization - 1 GPU"
echo "=================================================="
echo "  Checkpoint  : ${CHECKPOINT_PATH}"
echo "  Output dir  : ${OUTPUT_DIR:-<default beside checkpoint>}"
echo "  Max samples : ${MAX_SAMPLES}"
echo "=================================================="

CMD=(
  torchrun
  --nproc_per_node=1
  scripts/visualize_reasoning_weights.py
  --checkpoint "${CHECKPOINT_PATH}"
  --max-samples "${MAX_SAMPLES}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  CMD+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi

"${CMD[@]}"
