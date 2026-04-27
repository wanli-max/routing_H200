#!/bin/bash
# ============================================================
#  Setup + train: 8x A800, 7B baseline
#
#  Prerequisites (run once manually):
#    git clone https://github.com/wanli-max/routing_H200.git
#    cd routing_H200
#
#  Then launch:
#    bash setup_and_train_8gpu_7b_baseline_a800.sh
#
#  To submit via SLURM:
#    sbatch --nodes=1 --gres=gpu:8 --cpus-per-task=64 --mem=512G \
#           setup_and_train_8gpu_7b_baseline_a800.sh
# ============================================================

set -euo pipefail

# ── [EDIT THESE] ─────────────────────────────────────────────────────────────
VIRL39K_DIR="/data/raw/ViRL39K"        # directory containing 39Krelease.parquet + images/
MMK12_DIR="/data/raw/MMK12/data"       # directory containing test-*.parquet

MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # HF model ID or local path

# ─────────────────────────────────────────────────────────────────────────────

# ── derived paths (relative to repo root) ────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${REPO_ROOT}/datasets/virl39k_mmk12_easyr1"

RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="virl39k_8gpu_7b_baseline_${RUN_TIMESTAMP}"

echo "============================================"
echo "  ViRL39K Training - 8x A800 (7B) Baseline"
echo "  (no routing weight, no perception loss)"
echo "============================================"
echo "  Experiment : ${EXPERIMENT_NAME}"
echo "  Repo root  : ${REPO_ROOT}"
echo "  Model      : ${MODEL_PATH}"
echo "  Data       : ${DATASET_ROOT}"
echo "============================================"

cd "${REPO_ROOT}"
git pull

# ── preprocess dataset ────────────────────────────────────────────────────────
if [[ -f "${DATASET_ROOT}/train/part-00000.parquet" && \
      -f "${DATASET_ROOT}/val/part-00000.parquet" ]]; then
    echo "[1/2] Dataset already exists at ${DATASET_ROOT}, skipping preprocessing."
    echo "      (delete the directory and re-run to force reprocessing)"
else
    echo "[1/2] Preprocessing datasets ..."
    python3 scripts/adapt_virl39k_mmk12.py \
        --train-dir    "${VIRL39K_DIR}" \
        --val-dir      "${MMK12_DIR}" \
        --output-root  "${DATASET_ROOT}" \
        --overwrite
fi

# ── train ─────────────────────────────────────────────────────────────────────
echo "[2/2] Launching training: ${EXPERIMENT_NAME}"

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
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=512 \
    trainer.total_epochs=2 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.logger='["file","tensorboard"]' \
    2>&1 | tee "${REPO_ROOT}/${EXPERIMENT_NAME}.log"

echo "[FINISH] Training completed: ${EXPERIMENT_NAME}"
