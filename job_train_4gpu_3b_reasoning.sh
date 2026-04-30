#!/bin/bash
#PBS -q gpu_as
#PBS -P gs_ccds_boan
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o /projects_vol/gp_boan/routing_H200/virl39k_4gpu_3b_reasoning.log

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
MODEL_PATH=${1:-"Qwen/Qwen2.5-VL-3B-Instruct"}
DATASET_ROOT=${2:-"/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1"}
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="virl39k_4gpu_3b_reasoning_${RUN_TIMESTAMP}"

echo "============================================"
echo "  ViRL39K Training - 4x GPU (3B) Reasoning"
echo "  (reasoning routing weight only, no perception loss)"
echo "============================================"
echo "  Experiment : ${EXPERIMENT_NAME}"
echo "  Model      : ${MODEL_PATH}"
echo "  Data       : ${DATASET_ROOT}"
echo "============================================"

RESUME_CKPT=${RESUME_CKPT:-""}

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
    worker.rollout.max_num_batched_tokens=16384 \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=512 \
    worker.actor.clip_ratio_low=0 \
    worker.actor.clip_ratio_high=10.0 \
    worker.actor.clip_ratio_dual=3.0 \
    worker.actor.reasoning_loss_weight_clip_min=0.2 \
    worker.actor.reasoning_loss_weight_clip_max=10.0 \
    worker.actor.answer_chain_local_window_size=128 \
    trainer.total_epochs=3 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=4 \
    trainer.logger='["file","tensorboard"]' \
    ${RESUME_CKPT:+trainer.load_checkpoint_path="${RESUME_CKPT}"} \
    2>&1 | tee "${EXPERIMENT_NAME}.log"

echo "[FINISH] Training completed: ${EXPERIMENT_NAME}"
