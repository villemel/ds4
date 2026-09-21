#!/bin/bash
# ds4-server log rotation — zero-downtime copytruncate approach, fleet-wide.
#
# One launchd job (~/Library/LaunchAgents/com.ds4.log-rotation.plist, daily at
# 03:00) rotates the log directory of EVERY ds4 worktree model. Each model
# writes to a directory of its own so one noisy trial can never fill the disk
# of the workhorse, and no model is left unrotated:
#
#   ds4-server             legacy DeepSeek V4 Flash store (port 8000 plists in
#                          ~/ds4/misc/swap, and the pre-launchd nohup era)
#   ds4-glm-5.3-server     GLM 5.3 Flash workhorse, launchd
#                          com.ds4-glm53-flash.server, port 8000
#   ds4-server-glmfull     GLM 5.3 full IQ2_XXS (79 layers), ~/ds4-glmfull:
#                          run-server.sh and com.ds4-glm53-full.server.plist,
#                          port 8008
#   ds4-server-v41         DeepSeek V4.1 Flash Q2 trial, ~/ds4-v41, port 8005
#   ds4-server-qfn         Qwen3.8 Flash Next Q4 trial, ~/ds4-qfn, port 8007
#   ds4-server-glm53-trial retired llama.cpp-era trial; listed so its 137 KiB
#                          of live logs get archived and age out
#
# A new worktree = one line in LOG_DIRS. Missing directories are skipped, so
# deleting a worktree never breaks the job.
#
# Copytruncate, not rename: the servers keep their write offsets and launchd
# keeps the inode it opened at bootstrap, so rotation needs no restart and
# costs nothing on a machine where a restart means re-reading ~200 GiB of
# weights. Live files are always truncated in place, never unlinked.
#
# DRY_RUN=1 /Users/max/ds4/rotate-logs.sh   # print every decision, touch nothing

set -euo pipefail

RETENTION_DAYS=7
DRY_RUN="${DRY_RUN:-0}"
DATE=$(date +%Y%m%d)

LOG_DIRS=(
    "/Users/max/Library/Logs/ds4-server"
    "/Users/max/Library/Logs/ds4-glm-5.3-server"
    "/Users/max/Library/Logs/ds4-server-glmfull"
    "/Users/max/Library/Logs/ds4-server-v41"
    "/Users/max/Library/Logs/ds4-server-qfn"
    "/Users/max/Library/Logs/ds4-server-glm53-trial"
)

# This script's own stdout/stderr target (set by the plist). It is written by
# launchd in append mode, so truncating in place is safe for its fd.
SELF_LOG="/Users/max/Library/Logs/ds4-rotation.log"
SELF_LOG_MAX=$((2 * 1024 * 1024))

log() { printf '%s\n' "$*"; }

rotate_dir() {
    local dir="$1"
    log "=== $dir"

    if [ ! -d "$dir" ]; then
        log "    skipped: no such directory"
        return 0
    fi

    # --- Rotate live logs (copy + truncate, server fd stays valid) ---
    local f filepath archive size
    for f in err.log out.log; do
        filepath="$dir/$f"
        if [ ! -f "$filepath" ] || [ ! -s "$filepath" ]; then
            continue
        fi
        archive="$dir/${f%.log}-$DATE.log"
        if [ -f "$archive" ]; then
            log "    skip: $(basename "$archive") already exists (already rotated today)"
            continue
        fi
        if [ "$DRY_RUN" = "1" ]; then
            log "    DRY: cp $filepath $archive && truncate $filepath"
            continue
        fi
        cp "$filepath" "$archive"
        : > "$filepath"
        size=$(stat -f%z "$archive" | awk '{printf "%.1f", $1/1024}')
        log "    rotated: $f -> ${f%.log}-$DATE.log (${size}K)"
    done

    # --- Purge stale archives (dated archives only, never a live log) ---
    local count=0 oldfile
    while IFS= read -r -d '' oldfile; do
        if [ "$DRY_RUN" = "1" ]; then
            log "    DRY: rm $oldfile"
        else
            rm "$oldfile"
        fi
        count=$((count + 1))
    done < <(find "$dir" -maxdepth 1 \
                 \( -name 'err-[0-9]*.log' -o -name 'out-[0-9]*.log' \) \
                 -mtime +"$RETENTION_DAYS" -print0)
    log "    purged $count stale archive(s) (retention: $RETENTION_DAYS days)"

    # --- Summarise what is on disk here ---
    if [ "$DRY_RUN" = "1" ]; then
        log "    DRY: summary skipped"
        return 0
    fi
    log "    $(du -sh "$dir" 2>/dev/null | cut -f1) on disk; archives:"
    local archives archive_path
    archives=$(find "$dir" -maxdepth 1 \
                   \( -name 'err-[0-9]*.log' -o -name 'out-[0-9]*.log' \) | sort)
    if [ -z "$archives" ]; then
        log "        (no archives yet)"
    else
        while IFS= read -r archive_path; do
            log "        $(ls -lh "$archive_path" | awk '{print $5, $9}')"
        done <<< "$archives"
    fi
    return 0
}

for LOG_DIR in "${LOG_DIRS[@]}"; do
    rotate_dir "$LOG_DIR"
done

log "=== fleet total"
if [ "$DRY_RUN" != "1" ]; then
    du -sh "${LOG_DIRS[@]}" 2>/dev/null || true
else
    log "    DRY: total skipped"
fi

# --- Keep the rotation job's own log bounded ---
if [ -f "$SELF_LOG" ]; then
    self_size=$(stat -f%z "$SELF_LOG")
    if [ "$self_size" -gt "$SELF_LOG_MAX" ]; then
        if [ "$DRY_RUN" = "1" ]; then
            log "DRY: truncate $SELF_LOG ($self_size bytes)"
        else
            : > "$SELF_LOG"
            log "Truncated own log (grew past $((SELF_LOG_MAX / 1024 / 1024)) MiB)."
        fi
    fi
fi
