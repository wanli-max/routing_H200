#!/bin/bash
# ============================================================
# Eval an EasyR1 VLM checkpoint/model on one adapted benchmark.
#
# Launch after entering a GPU environment:
#   bash scripts/eval_benchmark.sh TEST_MODEL_PATH TEST_DATASET_PATH
#
# Example:
#   bash scripts/eval_benchmark.sh \
#     checkpoints/easy_r1/exp/global_step_80 \
#     datasets/easyr1_eval_benchmarks/MathVista/test.parquet
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TEST_MODEL_PATH="${1:-${TEST_MODEL_PATH:-}}"
TEST_DATASET_PATH="${2:-${TEST_DATASET_PATH:-}}"

if [[ -z "${TEST_MODEL_PATH}" || -z "${TEST_DATASET_PATH}" ]]; then
    echo "Usage:" >&2
    echo "  bash scripts/eval_benchmark.sh TEST_MODEL_PATH TEST_DATASET_PATH" >&2
    exit 1
fi

if [[ ! -e "${TEST_DATASET_PATH}" ]]; then
    echo "[ERROR] TEST_DATASET_PATH does not exist: ${TEST_DATASET_PATH}" >&2
    exit 1
fi

if [[ "${SKIP_GIT_PULL:-0}" != "1" ]]; then
    git pull
fi

lower_model_path="$(echo "${TEST_MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
    if [[ "${lower_model_path}" == *"7b"* ]]; then
        BASE_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
    else
        BASE_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
    fi
fi

LOAD_CHECKPOINT_ARG=()
MODEL_PATH="${TEST_MODEL_PATH}"
if [[ "$(basename "${TEST_MODEL_PATH}")" == global_step_* ]]; then
    if [[ ! -d "${TEST_MODEL_PATH}" ]]; then
        echo "[ERROR] Checkpoint directory does not exist: ${TEST_MODEL_PATH}" >&2
        exit 1
    fi
    MODEL_PATH="${BASE_MODEL_PATH}"
    LOAD_CHECKPOINT_ARG=(trainer.load_checkpoint_path="${TEST_MODEL_PATH}")
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
MODEL_LABEL="$(basename "${TEST_MODEL_PATH}" | tr -cs '[:alnum:]_.-' '_')"
DATASET_LABEL="$(basename "$(dirname "${TEST_DATASET_PATH}")" | tr -cs '[:alnum:]_.-' '_')"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_${MODEL_LABEL}_${DATASET_LABEL}_${RUN_TIMESTAMP}}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/${EXPERIMENT_NAME}.log}"

N_GPUS="${N_GPUS:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${N_GPUS}}"
REWARD_FUNCTION="${REWARD_FUNCTION:-./examples/reward_function/math.py:compute_score}"

echo "============================================"
echo "  EasyR1 benchmark eval"
echo "============================================"
echo "  Experiment     : ${EXPERIMENT_NAME}"
echo "  Model input    : ${TEST_MODEL_PATH}"
echo "  Base model     : ${MODEL_PATH}"
echo "  Dataset        : ${TEST_DATASET_PATH}"
echo "  Reward function: ${REWARD_FUNCTION}"
echo "  GPUs           : ${N_GPUS}"
echo "  TP size        : ${TENSOR_PARALLEL_SIZE}"
echo "  Log            : ${LOG_PATH}"
echo "============================================"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TEST_DATASET_PATH}" \
    data.val_files="${TEST_DATASET_PATH}" \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=images \
    data.format_prompt=null \
    data.filter_overlong_prompts=false \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE:-64}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    worker.rollout.limit_images="${LIMIT_IMAGES:-10}" \
    worker.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-16384}" \
    worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.5}" \
    worker.reward.reward_function="${REWARD_FUNCTION}" \
    trainer.val_before_train=true \
    trainer.val_only=true \
    trainer.val_generations_to_log="${VAL_GENERATIONS_TO_LOG:-20}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.logger='["file","tensorboard"]' \
    trainer.save_freq=-1 \
    "${LOAD_CHECKPOINT_ARG[@]}" \
    2>&1 | tee "${LOG_PATH}"

echo "[FINISH] Eval completed: ${EXPERIMENT_NAME}"
