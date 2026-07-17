#!/bin/bash
# Clean up stale KV cache entries that haven't been touched in 48+ hours.
# Matches the 48h stale threshold — short-term acceleration layer, not long-term storage.
#
# Safe to run while the server is running — the server re-scans on restart
# and will simply re-prefill if a deleted entry was needed.
set -e

KV_DIR="${1:-$HOME/.ds4}"
HOURS="${2:-48}"

if [ ! -d "$KV_DIR" ]; then
    echo "KV cache directory not found: $KV_DIR"
    exit 1
fi

echo "Scanning $KV_DIR for KV cache files older than $HOURS hours..."
count=0
bytes=0

while IFS= read -r -d '' f; do
    # Skip files less than $HOURS hours old
    if [ -z "$(find "$f" -mmin "+$((HOURS * 60))" -print 2>/dev/null)" ]; then
        continue
    fi
    sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)
    rm -f "$f"
    count=$((count + 1))
    bytes=$((bytes + sz))
done < <(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -print0 2>/dev/null)

echo "Deleted $count files ($(( bytes / 1024 / 1024 / 1024 )) GiB)"
