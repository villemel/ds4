#!/bin/bash
# ds4-server log rotation — zero-downtime copytruncate approach
# Copies current logs to dated archives, truncates live files in-place,
# purges archives older than retention period. Server never restarts.

set -euo pipefail

LOG_DIR="/Users/max/Library/Logs/ds4-server"
DATE=$(date +%Y%m%d)
RETENTION_DAYS=7

# --- Rotate current logs (copy + truncate — server keeps its fd) ---
for f in err.log out.log; do
    filepath="$LOG_DIR/$f"
    if [ -f "$filepath" ] && [ -s "$filepath" ]; then
        archive="$LOG_DIR/${f%.log}-$DATE.log"
        # Only rotate if we haven't already rotated today
        if [ ! -f "$archive" ]; then
            cp "$filepath" "$archive"
            # Truncate in-place — same inode, server fd stays valid
            : > "$filepath"
            size=$(stat -f%z "$archive" | awk '{printf "%.1f", $1/1024}')
            echo "Rotated: $f → ${f%.log}-$DATE.log (${size}K)"
        else
            echo "Skip: $archive already exists (already rotated today)"
        fi
    fi
done

# --- Purge logs older than retention period ---
# Only purge archive files (err-YYYYMMDD.log, out-YYYYMMDD.log), never the live files
count=0
while IFS= read -r -d '' oldfile; do
    rm "$oldfile"
    count=$((count + 1))
done < <(find "$LOG_DIR" -name "*.log" ! -name "err.log" ! -name "out.log" -mtime +$RETENTION_DAYS -print0)
echo "Purged $count stale archive(s) (retention: $RETENTION_DAYS days)."

# --- Summarise current archives ---
echo "--- Archives on disk ---"
ls -lh "$LOG_DIR"/err-*.log "$LOG_DIR"/out-*.log 2>/dev/null || echo "(no archives yet)"
