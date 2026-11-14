#!/usr/bin/env bash
# Prepare lightweight EGSQ delivery bundle (EXCLUDING raw datasets)
set -e

REMOTE_DIR="/root/autodl-tmp/egsq"
BUNDLE_DIR="/root/autodl-tmp/egsq_migration_bundle"

echo "=== Preparing Clean EGSQ Migration Bundle (Code + Checkpoints + Reports) ==="
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

# Copy essential components, explicitly omitting data/
cp -r "$REMOTE_DIR/src" "$BUNDLE_DIR/"
cp -r "$REMOTE_DIR/scripts" "$BUNDLE_DIR/"
cp -r "$REMOTE_DIR/checkpoints" "$BUNDLE_DIR/"
cp -r "$REMOTE_DIR/reports" "$BUNDLE_DIR/"

# Remove Python cache
find "$BUNDLE_DIR" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$BUNDLE_DIR" -name "*.pyc" -delete 2>/dev/null || true

# Package as single archive
cd /root/autodl-tmp
tar -czvf egsq_clean_bundle.tar.gz -C egsq_migration_bundle .

echo "=== Migration Bundle Ready ==="
ls -lh /root/autodl-tmp/egsq_clean_bundle.tar.gz
