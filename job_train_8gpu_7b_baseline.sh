#!/bin/bash
#PBS -q gpu_as
#PBS -P gs_ccds_boan
#PBS -l select=1:ncpus=64:ngpus=8:mem=512gb
#PBS -l walltime=36:00:00
#PBS -j oe
#PBS -o /projects_vol/gp_boan/routing_H200/virl39k_8gpu_7b_baseline.log

set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] || [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
      awk -F', ' -v uuids="$CUDA_VISIBLE_DEVICES" \
      'BEGIN{split(uuids,u,",")} {for(i in u) if($2==u[i]) printf "%s%s",(n++?",":""),$1}')
    echo "[INFO] Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

module load anaconda/2025
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate easyr1_virl39k_jh

cd /projects_vol/gp_boan/routing_H200
git pull

# ── config ────────────────────────────────────────────────────────────────────
# GPU layout: 8 GPUs, TP=2 for rollout → 4 vLLM replicas (DP=4)
#             FSDP training across all 8 GPUs
# Batch math: rollout_batch_size=512 / DP=4 = 128 per replica ✓
#             global_batch_size=512 / 8 GPUs = 64 per device ✓
MODEL_PATH=${1:-"Qwen/Qwen2.5-VL-7B-Instruct"}
DATASET_ROOT=${2:-"/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1"}
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="virl39k_8gpu_7b_baseline_${RUN_TIMESTAMP}"

echo "============================================"
echo "  ViRL39K Training - 8x GPU (7B) Baseline"
echo "  (no routing weight, no perception loss)"
echo "============================================"
echo "  Experiment : ${EXPERIMENT_NAME}"
echo "  Model      : ${MODEL_PATH}"
echo "  Data       : ${DATASET_ROOT}"
echo "============================================"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${DATASET_ROOT}/train" \
    data.val_files="${DATASET_ROOT}/val" \
    data.prompt_key=prompt \
    data.format_prompt=null \
    data.filter_overlong_prompts=false \
    data.max_prompt_length=8192 \
    data.rollout_batch_size=512 \
    data.val_batch_size=512 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=512 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.use_answer_chain_routing=false \
    worker.actor.perception_loss_coef=0.0 \
    trainer.total_epochs=2 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.logger='["file","tensorboard"]' \
    2>&1 | tee "${EXPERIMENT_NAME}.log"

echo "[FINISH] Baseline training completed: ${EXPERIMENT_NAME}"
