#!/bin/bash
# Temporary preprocessing smoke test — deletes itself and output on exit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_OUTPUT="${REPO_ROOT}/_test_preprocess_tmp"
SCRIPT_PATH="${BASH_SOURCE[0]}"

cleanup() {
    echo ""
    echo "[cleanup] Removing test output: ${TEST_OUTPUT}"
    rm -rf "${TEST_OUTPUT}"
    echo "[cleanup] Removing this script: ${SCRIPT_PATH}"
    rm -f "${SCRIPT_PATH}"
}
trap cleanup EXIT

cd "${REPO_ROOT}"

module load anaconda/2025 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate easyr1_virl39k_jh

echo "============================================"
echo "  Preprocessing smoke test (H200)"
echo "  ViRL39K : /projects_vol/gp_boan/easyr1_assets/datasets/train/ViRL39K"
echo "  MMK12   : /projects_vol/gp_boan/easyr1_assets/datasets/val/MMK12/data"
echo "  Output  : ${TEST_OUTPUT}"
echo "============================================"

python3 scripts/adapt_virl39k_mmk12.py \
    --train-dir "/projects_vol/gp_boan/easyr1_assets/datasets/train/ViRL39K" \
    --val-dir   "/projects_vol/gp_boan/easyr1_assets/datasets/val/MMK12/data" \
    --output-root "${TEST_OUTPUT}" \
    --overwrite

echo ""
echo "[verify] Train rows: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('${TEST_OUTPUT}/train/part-00000.parquet')))")"
echo "[verify] Val rows  : $(python3 -c "import pandas as pd; print(len(pd.read_parquet('${TEST_OUTPUT}/val/part-00000.parquet')))")"
echo ""
echo "[verify] Train image sample:"
python3 -c "
import pandas as pd
df = pd.read_parquet('${TEST_OUTPUT}/train/part-00000.parquet')
img = df['images'].iloc[0]
print('  type:', type(img[0]) if img else 'empty')
print('  value:', img[0] if img else 'empty')
"
echo "[verify] Val image sample:"
python3 -c "
import pandas as pd
df = pd.read_parquet('${TEST_OUTPUT}/val/part-00000.parquet')
img = df['images'].iloc[0]
item = img[0] if img else None
if isinstance(item, dict):
    print('  bytes:', len(item['bytes']) if item.get('bytes') else 'NULL')
    print('  path:', item.get('path'))
else:
    print('  value:', item)
"
echo ""
echo "[DONE] All checks passed."
