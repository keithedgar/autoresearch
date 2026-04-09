#!/usr/bin/env bash
# autoresearch/nightly.sh — nightly batch launcher for autoresearch
#
# Designed to be called by cron, systemd timer, or manually.
# Runs a bounded batch of experiments (default 12 ≈ 1 hour), then exits.
# State is persisted in state.json between runs.
#
# Usage:
#   From host (recommended):
#     docker exec -w /workspace/autoresearch crsai-pytorch bash nightly.sh
#
#   With overrides:
#     docker exec -w /workspace/autoresearch crsai-pytorch \
#       env AUTORESEARCH_BATCH_SIZE=6 bash nightly.sh
#
# Crontab example (run at 1 AM every night):
#   0 1 * * * docker exec -w /workspace/autoresearch crsai-pytorch bash nightly.sh \
#             >> /mnt/d/cotton-prod/logs/autoresearch-nightly.log 2>&1
#
set -euo pipefail

TAG="${AUTORESEARCH_TAG:-apr}"
BATCH_SIZE="${AUTORESEARCH_BATCH_SIZE:-12}"
PROFILE="${AUTORESEARCH_PROFILE:-rtx5060}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
LOG_DIR="${AUTORESEARCH_LOG_DIR:-/workspace/autoresearch/logs}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nightly-$(date -u +%Y%m%d-%H%M%S).log"

echo "=== Autoresearch Nightly Run ===" | tee "$LOG_FILE"
echo "Date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "Tag:        $TAG" | tee -a "$LOG_FILE"
echo "Batch size: $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "Profile:    $PROFILE" | tee -a "$LOG_FILE"
echo "GPU:        $GPU" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify vLLM is reachable before starting a potentially long batch
LLM_URL="${LLM_URL:-http://crsai-vllm:8000/v1}"
if ! curl -sf "${LLM_URL}/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM not reachable at ${LLM_URL}. Aborting." | tee -a "$LOG_FILE"
    exit 1
fi
echo "vLLM healthy at ${LLM_URL}" | tee -a "$LOG_FILE"

cd /workspace/autoresearch

export AUTORESEARCH_PROFILE="$PROFILE"
export CUDA_VISIBLE_DEVICES="$GPU"

python3 -u orchestrator.py \
    --tag "$TAG" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"

# Prune logs older than 30 days
find "$LOG_DIR" -name "nightly-*.log" -mtime +30 -delete 2>/dev/null || true

exit "$EXIT_CODE"
