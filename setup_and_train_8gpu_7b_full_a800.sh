#!/bin/bash
# ============================================================
#  Train: 8x A800, 7B full method (reasoning routing + perception loss)
#
#  Prerequisites (run once manually):
#    git clone https://github.com/wanli-max/routing_H200.git
#    cd routing_H200
#
#  Then launch:
#    bash setup_and_train_8gpu_7b_full_a800.sh
#
#  To submit via SLURM:
#    sbatch --nodes=1 --gres=gpu:8 --cpus-per-task=64 --mem=512G \
#           setup_and_train_8gpu_7b_full_a800.sh
# ============================================================

set -euo pipefail

# ── [EDIT THESE] ─────────────────────────────────────────────────────────────
DATASET_ROOT="/path/to/virl39k_mmk12_easyr1"         # TODO: set preprocessed dataset root
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"            # HF model ID or local path

# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="virl39k_8gpu_7b_full_${RUN_TIMESTAMP}"

echo "============================================"
echo "  ViRL39K Training - 8x A800 (7B) Full Method"
echo "  (reasoning routing weight + perception loss)"
echo "============================================"
echo "  Experiment : ${EXPERIMENT_NAME}"
echo "  Repo root  : ${REPO_ROOT}"
echo "  Model      : ${MODEL_PATH}"
echo "  Data       : ${DATASET_ROOT}"
echo "============================================"

cd "${REPO_ROOT}"
git pull

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
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.limit_images=10 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=512 \
    worker.actor.reasoning_loss_weight_clip_min=0.2 \
    worker.actor.reasoning_loss_weight_clip_max=5.0 \
    worker.actor.answer_chain_local_window_size=128 \
    worker.actor.perception_loss_coef=0.001 \
    worker.actor.perception_success_threshold=0.8 \
    trainer.total_epochs=3 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.logger='["file","tensorboard"]' \
    2>&1 | tee "${REPO_ROOT}/${EXPERIMENT_NAME}.log"

echo "[FINISH] Training completed: ${EXPERIMENT_NAME}"
