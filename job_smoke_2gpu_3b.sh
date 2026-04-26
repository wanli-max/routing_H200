#!/bin/bash
#PBS -q gpu_as
#PBS -P gs_ccds_boan
#PBS -l select=1:ncpus=16:ngpus=2:mem=128gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /projects_vol/gp_boan/routing_H200/virl39k_smoke_2gpu_3b.log

set -euo pipefail

module load anaconda/2025
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate easyr1_virl39k_jh

if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] || [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | \
      awk -F', ' -v uuids="$CUDA_VISIBLE_DEVICES" \
      'BEGIN{split(uuids,u,",")} {for(i in u) if($2==u[i]) printf "%s%s",(n++?",":""),$1}')
    echo "[INFO] Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

cd /projects_vol/gp_boan/routing_H200
git pull

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH=${1:-"Qwen/Qwen2.5-VL-3B-Instruct"}
DATASET_ROOT="/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1"
SMOKE_ROOT="/projects_vol/gp_boan/easyr1_assets/datasets/virl39k_mmk12_easyr1_smoke"
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="virl39k_smoke_2gpu_3b_${RUN_TIMESTAMP}"

echo "============================================"
echo "  ViRL39K Smoke Test - 2x GPU (3B)"
echo "============================================"
echo "  Experiment : ${EXPERIMENT_NAME}"
echo "  Model      : ${MODEL_PATH}"
echo "  Data       : ${DATASET_ROOT}"
echo "============================================"

# ── build smoke subset (8 train / 4 val) ─────────────────────────────────────
export DATASET_ROOT SMOKE_ROOT
python3 - <<'PY'
import glob, os
from pathlib import Path
from datasets import load_dataset

src  = Path(os.environ["DATASET_ROOT"])
dst  = Path(os.environ["SMOKE_ROOT"])

train_files = sorted(glob.glob(str(src / "train" / "*.parquet")))
val_files   = sorted(glob.glob(str(src / "val"   / "*.parquet")))
if not train_files: raise FileNotFoundError(f"No train parquet under {src}/train")
if not val_files:   raise FileNotFoundError(f"No val parquet under {src}/val")

train_ds = load_dataset("parquet", data_files=train_files, split="train").select(range(8))
val_ds   = load_dataset("parquet", data_files=val_files,   split="train").select(range(4))

(dst / "train").mkdir(parents=True, exist_ok=True)
(dst / "val").mkdir(parents=True, exist_ok=True)
train_ds.to_parquet(str(dst / "train" / "part-00000.parquet"))
val_ds.to_parquet(str(dst / "val"   / "part-00000.parquet"))
print(f"Smoke subset ready: {len(train_ds)} train, {len(val_ds)} val")
PY

# ── train ─────────────────────────────────────────────────────────────────────
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${SMOKE_ROOT}/train" \
    data.val_files="${SMOKE_ROOT}/val" \
    data.prompt_key=prompt \
    data.format_prompt=null \
    data.filter_overlong_prompts=false \
    data.rollout_batch_size=4 \
    data.val_batch_size=4 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.global_batch_size=4 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.35 \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.logger='["file","tensorboard"]' \
    trainer.max_steps=3 \
    trainer.val_freq=3 \
    trainer.save_freq=3 \
    2>&1 | tee "${EXPERIMENT_NAME}.log"

echo "[FINISH] Smoke run completed: ${EXPERIMENT_NAME}"
