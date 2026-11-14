#!/usr/bin/env bash
# Push local code, configs and docs to remote GPU server
# Usage: ./scripts/sync_to_remote.sh [user@host] [port] [remote_path]

REMOTE_HOST="${1:-remote-gpu}"
REMOTE_PORT="${2:-22}"
REMOTE_PATH="${3:-/root/autodl-tmp/egsq/}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Pushing code from ${LOCAL_DIR} to ${REMOTE_HOST}:${REMOTE_PORT} -> ${REMOTE_PATH}"

rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '/data/' \
    --exclude 'checkpoints/' \
    "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_PATH}/"

echo "Sync to remote completed successfully."
