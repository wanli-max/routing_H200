#!/bin/bash
#PBS -q gpu_as
#PBS -P gs_ccds_boan
#PBS -l select=1:ncpus=16:ngpus=1:mem=128gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /projects_vol/gp_boan/routing_H200/visualization_1gpu.log

# Single-GPU launcher for response + reasoning-weight visualization.
#
# Usage:
#   cd /projects_vol/gp_boan/routing_H200
#   bash scripts/1gpu_visualization.sh
#   bash scripts/1gpu_visualization.sh /path/to/global_step_80 [OUTPUT_DIR]
#
# Or submit as a PBS job, following the same pattern as job_train_*.sh:
#   qsub scripts/1gpu_visualization.sh

set -euo pipefail

module load anaconda/2025
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate easyr1_virl39k_jh

REPO_ROOT="/projects_vol/gp_boan/routing_H200"
cd "${REPO_ROOT}"
git pull

if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] || [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
      awk -F', ' -v uuids="$CUDA_VISIBLE_DEVICES" \
      'BEGIN{split(uuids,u,",")} {for(i in u) if($2==u[i]) printf "%s%s",(n++?",":""),$1}')
    echo "[INFO] Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

DEFAULT_CHECKPOINT_PATH="/projects_vol/gp_boan/routing_H200/checkpoints/easy_r1/virl39k_4gpu_3b_reasoning_20260428_004314/global_step_80"
DEFAULT_OUTPUT_DIR="/projects_vol/gp_boan/routing_H200/visualization_1gpu"
CHECKPOINT_PATH=${1:-"${DEFAULT_CHECKPOINT_PATH}"}
OUTPUT_DIR=${2:-"${DEFAULT_OUTPUT_DIR}"}
MAX_SAMPLES=${MAX_SAMPLES:-3}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-}

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
echo "  Default ckpt: ${DEFAULT_CHECKPOINT_PATH}"
echo "  Output dir  : ${OUTPUT_DIR}"
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

CMD+=(--output-dir "${OUTPUT_DIR}")

if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  CMD+=(--max-new-tokens "${MAX_NEW_TOKENS}")
fi

"${CMD[@]}"
