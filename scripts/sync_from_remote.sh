#!/usr/bin/env bash
# Pull evaluation reports, logs, and lightweight figures back to local repo
# Usage: ./scripts/sync_from_remote.sh [user@host] [port] [remote_path]

REMOTE_HOST="${1:-remote-gpu}"
REMOTE_PORT="${2:-22}"
REMOTE_PATH="${3:-/root/autodl-tmp/egsq/}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Pulling results from ${REMOTE_HOST}:${REMOTE_PORT}:${REMOTE_PATH}/reports/ to ${LOCAL_DIR}/reports/"

rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    --include '*/' \
    --include '*.csv' \
    --include '*.json' \
    --include '*.npz' \
    --include '*.png' \
    --include '*.pdf' \
    --include '*.log' \
    --exclude '*' \
    "${REMOTE_HOST}:${REMOTE_PATH}/reports/" "${LOCAL_DIR}/reports/"

echo "Sync from remote completed successfully."
