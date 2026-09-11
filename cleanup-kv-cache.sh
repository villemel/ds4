#!/bin/bash
# Clean up stale KV cache entries that haven't been touched in N hours.
#
# Purpose: keep disk usage BELOW the server's --kv-disk-space-mb budget so that
# disk-cache-full eviction stays dormant. When the budget fills, the server's
# eviction cycles *intermediate continued waypoints* (32k/64k/128k-token
# checkpoints of active conversations) — the very checkpoints that make a
# resume-fallback cheap. Their loss collapses a failed checkpoint match down to
# an ancient cold checkpoint and forces huge re-prefills (see
# .pi/notes/kv-cache-interleave-loss.md). Proactively removing genuinely stale
# files (<relevance window) frees budget headroom so active waypoints survive.
#
# Safe to run while the server is running: it deletes only .kv files; the
# server rescans the directory on every cache operation (kv_cache_refresh) and
# will simply re-prefill if a deleted entry was ever needed again.
#
# Usage:
#   cleanup-kv-cache.sh [kv-dir] [hours] [--dry-run]
#   Defaults: kv-dir=$HOME/.ds4, hours=72
#   --dry-run  report what would be deleted without deleting anything
set -euo pipefail

KV_DIR="${1:-$HOME/.ds4}"
HOURS="${2:-72}"
DRY_RUN=0
for a in "$@"; do
    if [ "$a" = "--dry-run" ]; then DRY_RUN=1; fi
done

if [ ! -d "$KV_DIR" ]; then
    echo "KV cache directory not found: $KV_DIR"
    exit 1
fi

MIN_AGE_MIN=$((HOURS * 60))

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] Scanning $KV_DIR for KV cache files older than $HOURS hours..."
else
    echo "Scanning $KV_DIR for KV cache files older than $HOURS hours..."
fi

count=0
bytes=0

# Use a null-delimited loop so filenames with spaces are safe.
while IFS= read -r -d '' f; do
    sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$DRY_RUN" = "1" ]; then
        echo "  would delete: $f ($(( sz / 1024 / 1024 )) MiB)"
    else
        rm -f "$f"
    fi
    count=$((count + 1))
    bytes=$((bytes + sz))
done < <(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -mmin "+${MIN_AGE_MIN}" -print0 2>/dev/null)

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] Would delete $count files ($(( bytes / 1024 / 1024 / 1024 )) GiB)"
else
    echo "Deleted $count files ($(( bytes / 1024 / 1024 / 1024 )) GiB)"
fi
