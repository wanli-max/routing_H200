#!/bin/bash
# Extract eval reward curves and upload the small JSON results to GitHub.
#
# Run from anywhere inside this repository:
#   bash scripts/upload_extracted_rewards.sh
#
# Optional override:
#   COMMIT_MESSAGE="Upload latest reward curves" bash scripts/upload_extracted_rewards.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-extract_results}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload extracted eval reward curves}"

cd "${REPO_ROOT}"

if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
    python scripts/extract_eval_reward_json.py --output-dir "${OUTPUT_DIR}"
fi

if ! compgen -G "${OUTPUT_DIR}/*_reward.json" > /dev/null; then
    echo "[ERROR] No reward JSON files found under ${OUTPUT_DIR}/" >&2
    exit 1
fi

git add -- "${OUTPUT_DIR}"/*_reward.json

if git diff --cached --quiet -- "${OUTPUT_DIR}"; then
    echo "[INFO] No reward JSON changes to upload."
    exit 0
fi

git commit -m "${COMMIT_MESSAGE}"

if ! git remote get-url routing-h200 > /dev/null 2>&1; then
    echo "[ERROR] Required git remote not found: routing-h200" >&2
    exit 1
fi

echo "[INFO] Pushing reward JSON results to routing-h200/main"
git push routing-h200 HEAD:main

echo "[DONE] Uploaded extracted reward curves."
