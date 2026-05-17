#!/bin/bash
# ============================================================
# Train selected A800 runs sequentially on one 8-GPU node.
#
# Order:
#   1. 7B full
#   2. 3B full
#   3. 3B reasoning only
#   4. 3B perception only
#
# Launch after acquiring an 8x A800 allocation, or submit with SLURM:
#   sbatch --nodes=1 --gres=gpu:8 --cpus-per-task=64 --mem=512G \
#          setup_and_train_8gpu_a800_all.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/virl39k_mmk12_easyr1}"
MODEL_7B_PATH="${MODEL_7B_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_3B_PATH="${MODEL_3B_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"

echo "============================================"
echo "  A800 sequential training"
echo "  Repo root : ${REPO_ROOT}"
echo "  Data      : ${DATASET_ROOT}"
echo "  7B model  : ${MODEL_7B_PATH}"
echo "  3B model  : ${MODEL_3B_PATH}"
echo "  Timestamp : ${RUN_TIMESTAMP}"
echo "============================================"

cd "${REPO_ROOT}"
git pull

if [[ ! -d "${DATASET_ROOT}/train" || ! -d "${DATASET_ROOT}/val" ]]; then
    echo "[ERROR] DATASET_ROOT must contain train/ and val/: ${DATASET_ROOT}" >&2
    exit 1
fi

run_train() {
    local model_label="$1"
    local model_path="$2"
    local method="$3"
    shift 3

    local experiment_name="${model_label}_${method}_${RUN_TIMESTAMP}"
    local log_path="${REPO_ROOT}/${experiment_name}.log"

    echo "============================================"
    echo "  Launching: ${experiment_name}"
    echo "  Model    : ${model_path}"
    echo "  Method   : ${method}"
    echo "  Log      : ${log_path}"
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
        worker.rollout.max_num_batched_tokens=16384 \
        worker.rollout.gpu_memory_utilization=0.5 \
        worker.rollout.tensor_parallel_size=2 \
        worker.rollout.limit_images=10 \
        worker.actor.model.model_path="${model_path}" \
        worker.actor.global_batch_size=512 \
        trainer.total_epochs=3 \
        trainer.experiment_name="${experiment_name}" \
        trainer.n_gpus_per_node=8 \
        trainer.logger='["file","tensorboard"]' \
        "$@" \
        2>&1 | tee "${log_path}"
}

run_baseline() {
    local model_label="$1"
    local model_path="$2"
    run_train "${model_label}" "${model_path}" "baseline" \
        worker.actor.use_answer_chain_routing=false \
        worker.actor.perception_loss_coef=0.0
}

run_reasoning_only() {
    local model_label="$1"
    local model_path="$2"
    run_train "${model_label}" "${model_path}" "reasoning" \
        worker.actor.reasoning_loss_weight_clip_min=0.2 \
        worker.actor.reasoning_loss_weight_clip_max=5.0 \
        worker.actor.answer_chain_local_window_size=128 \
        worker.actor.perception_loss_coef=0.0
}

run_perception_only() {
    local model_label="$1"
    local model_path="$2"
    run_train "${model_label}" "${model_path}" "perception_only" \
        worker.actor.use_answer_chain_routing=false \
        worker.actor.answer_chain_local_window_size=128 \
        worker.actor.perception_loss_coef=0.001 \
        worker.actor.perception_success_threshold=0.8
}

run_full() {
    local model_label="$1"
    local model_path="$2"
    run_train "${model_label}" "${model_path}" "full" \
        worker.actor.reasoning_loss_weight_clip_min=0.2 \
        worker.actor.reasoning_loss_weight_clip_max=5.0 \
        worker.actor.answer_chain_local_window_size=128 \
        worker.actor.perception_loss_coef=0.001 \
        worker.actor.perception_success_threshold=0.8
}

run_full "7b" "${MODEL_7B_PATH}"

run_full "3b" "${MODEL_3B_PATH}"
run_reasoning_only "3b" "${MODEL_3B_PATH}"
run_perception_only "3b" "${MODEL_3B_PATH}"

echo "[FINISH] All A800 training jobs completed."
