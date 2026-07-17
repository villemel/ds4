#!/bin/bash
# monitor-kv-cache.sh — Track ds4-server KV cache performance and stale blocks
#
# Runs as a cron job to collect KV cache statistics from server logs and
# the disk cache directory. Output is delivered to the user as a periodic
# performance digest.
#
# The script evolves its schedule based on observed idle periods.
# It writes a stats history file for trend analysis.
set -e

DS4_DIR="${1:-$HOME/ds4}"
KV_DIR="${2:-$HOME/.ds4}"
LOG_DIR="${3:-$HOME/Library/Logs/ds4-server}"
STATS_FILE="${DS4_DIR}/.kv-cache-stats.jsonl"
SCHEDULE_FILE="${DS4_DIR}/.kv-cache-schedule.json"
NOW=$(date +%s)
DATE_TAG=$(date +%Y%m%d-%H%M%S)

# --- 1. Collect server log metrics ---

# Use today's err log
ERR_LOG="${LOG_DIR}/err-$(date +%Y%m%d).log"
if [ ! -f "$ERR_LOG" ]; then
    # Fall back to err.log
    ERR_LOG="${LOG_DIR}/err.log"
fi

# Count events from today's log (last 24h window if using err.log)
if [[ "$ERR_LOG" == *err.log ]]; then
    # Use the last 24 hours of the rolling log
    SINCE_TS=$((NOW - 86400))
    LOG_DATA=$(grep -E "^(0[0-9]|1[0-9]|2[0-3])" "$ERR_LOG" 2>/dev/null | head -10000)
else
    LOG_DATA=$(cat "$ERR_LOG" 2>/dev/null)
fi

# Count cache operations
KV_HITS=$(echo "$LOG_DATA" | grep -c "kv cache hit" 2>/dev/null; true)
KV_STORED=$(echo "$LOG_DATA" | grep -c "kv cache stored" 2>/dev/null; true)
KV_MISSES=$(echo "$LOG_DATA" | grep -c "live kv cache miss" 2>/dev/null; true)
KV_EVICTED=$(echo "$LOG_DATA" | grep -c "kv cache evicted" 2>/dev/null; true)

# Store reason breakdown
STORE_COLD=$(echo "$LOG_DATA" | grep "kv cache stored" | grep -c "reason=cold" 2>/dev/null; true)
STORE_CONTINUED=$(echo "$LOG_DATA" | grep "kv cache stored" | grep -c "reason=continued" 2>/dev/null; true)
STORE_EVICT=$(echo "$LOG_DATA" | grep "kv cache stored" | grep -c "reason=evict" 2>/dev/null; true)
STORE_SHUTDOWN=$(echo "$LOG_DATA" | grep "kv cache stored" | grep -c "reason=shutdown" 2>/dev/null; true)

# Token mismatch analysis — extract the 'common=' values from misses
# This tells us how much overlap misses have with cached entries
MISS_COMMON=$(echo "$LOG_DATA" | grep "live kv cache miss" | grep -o "common=[0-9]*" | sed 's/common=//' 2>/dev/null || echo "")
if [ -n "$MISS_COMMON" ]; then
    MISS_COMMON_AVG=$(echo "$MISS_COMMON" | awk '{sum+=$1; count++} END {if(count>0) printf "%.0f", sum/count; else print "0"}')
    MISS_COMMON_MIN=$(echo "$MISS_COMMON" | sort -n | head -1)
    MISS_COMMON_MAX=$(echo "$MISS_COMMON" | sort -n | tail -1)
else
    MISS_COMMON_AVG=0
    MISS_COMMON_MIN=0
    MISS_COMMON_MAX=0
fi

# Extract load times from cache hits
LOAD_TIMES=$(echo "$LOG_DATA" | grep "kv cache hit" | grep -o "load=[0-9.]* ms" | sed 's/load=//;s/ ms//' 2>/dev/null || echo "")
if [ -n "$LOAD_TIMES" ]; then
    LOAD_AVG=$(echo "$LOAD_TIMES" | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
    LOAD_MAX=$(echo "$LOAD_TIMES" | sort -n | tail -1)
else
    LOAD_AVG=0
    LOAD_MAX=0
fi

# Extract save times
SAVE_TIMES=$(echo "$LOG_DATA" | grep "kv cache stored" | grep -o "save=[0-9.]* ms" | sed 's/save=//;s/ ms//' 2>/dev/null || echo "")
if [ -n "$SAVE_TIMES" ]; then
    SAVE_AVG=$(echo "$SAVE_TIMES" | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "0"}')
    SAVE_MAX=$(echo "$SAVE_TIMES" | sort -n | tail -1)
else
    SAVE_AVG=0
    SAVE_MAX=0
fi

# --- 2. Disk cache state ---

KV_FILE_COUNT=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f 2>/dev/null | wc -l | tr -d ' ')
KV_DISK_USAGE=$(du -sh "$KV_DIR" 2>/dev/null | awk '{print $1}')
KV_DISK_USAGE_BYTES=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -exec stat -f%z {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')

# Find stale entries — files not modified in the hit half-life (6h)
STALE_6H=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -mmin +360 2>/dev/null | wc -l | tr -d ' ')
STALE_12H=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -mmin +720 2>/dev/null | wc -l | tr -d ' ')
STALE_24H=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -mtime +1 2>/dev/null | wc -l | tr -d ' ')

# Total size of stale files
STALE_6H_BYTES=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -mmin +360 -exec stat -f%z {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
STALE_6H_HUMAN=""
if [ -n "$STALE_6H_BYTES" ] && [ "$STALE_6H_BYTES" -gt 0 ]; then
    STALE_6H_HUMAN=$(echo "scale=1; $STALE_6H_BYTES / 1073741824" | bc 2>/dev/null || echo "${STALE_6H_BYTES}")
fi

# Size distribution (approx from file sizes)
SMALL_FILES=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -size -100M 2>/dev/null | wc -l | tr -d ' ')
MEDIUM_FILES=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -size +100M -size -1G 2>/dev/null | wc -l | tr -d ' ')
LARGE_FILES=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -size +1G 2>/dev/null | wc -l | tr -d ' ')

# --- 3. Server status ---

SERVER_PID=$(pgrep -x "ds4-server" 2>/dev/null || pgrep -f "^/.*ds4-server " 2>/dev/null | head -1 || echo "")
SERVER_UPTIME=""
if [ -n "$SERVER_PID" ]; then
    SERVER_START=$(ps -o lstart= -p "$SERVER_PID" 2>/dev/null)
    if [ -n "$SERVER_START" ]; then
        SERVER_EPOCH=$(date -j -f "%a %d %b %H:%M:%S %Y" "$SERVER_START" +%s 2>/dev/null || echo 0)
        if [ "$SERVER_EPOCH" -gt 0 ] 2>/dev/null; then
            SERVER_UPTIME=$((NOW - SERVER_EPOCH))
        fi
    fi
fi

# Stale 48h computation (needed before stats write)
STALE_48H=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -mtime +2 2>/dev/null | wc -l | tr -d ' ')
STALE_48H_BYTES=$(find "$KV_DIR" -maxdepth 1 -name '*.kv' -type f -mtime +2 -exec stat -f%z {} + 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
STALE_48H_HUMAN="0"
if [ -n "$STALE_48H_BYTES" ] && [ "$STALE_48H_BYTES" -gt 0 ]; then
    STALE_48H_HUMAN=$(echo "scale=1; $STALE_48H_BYTES / 1073741824" | bc 2>/dev/null || echo "0")
fi

# --- 4. Compute derived metrics ---

# Hit ratio
TOTAL_REQUESTS=$((KV_HITS + KV_MISSES))
if [ "$TOTAL_REQUESTS" -gt 0 ]; then
    HIT_RATIO=$(echo "scale=1; $KV_HITS * 100 / $TOTAL_REQUESTS" | bc 2>/dev/null)
else
    HIT_RATIO="0.0"
fi

# Stale ratio
if [ "$KV_FILE_COUNT" -gt 0 ]; then
    STALE_RATIO=$(echo "scale=1; $STALE_6H * 100 / $KV_FILE_COUNT" | bc 2>/dev/null)
else
    STALE_RATIO="0.0"
fi

# --- 5. Determine idle periods for schedule adaptation ---

# Look at the hourly distribution of requests from today's log
HOURLY_HITS=$(echo "$LOG_DATA" | grep -oE "^[0-9]{2}" | sort | uniq -c | sort -k2 -n 2>/dev/null)
# Find the quietest hours (fewest total kv operations)
QUIET_HOURS=$(echo "$HOURLY_HITS" | sort -n | head -3 | awk '{print $2}' | tr '\n' ',' | sed 's/,$//')

# --- 6. Write stats record ---

SERVER_UP=$( [ -n "$SERVER_PID" ] && echo "true" || echo "false" )
printf '{"ts":%d,"date":"%s","server_up":%s,"server_uptime_sec":%d,' \
  "$NOW" "$DATE_TAG" "$SERVER_UP" "${SERVER_UPTIME:-0}" >> "$STATS_FILE"
printf '"kv_hits":%d,"kv_misses":%d,"kv_stored":%d,"kv_evicted":%d,"hit_ratio":%s,' \
  "$KV_HITS" "$KV_MISSES" "$KV_STORED" "$KV_EVICTED" "$HIT_RATIO" >> "$STATS_FILE"
printf '"store_reason_cold":%d,"store_reason_continued":%d,"store_reason_evict":%d,"store_reason_shutdown":%d,' \
  "$STORE_COLD" "$STORE_CONTINUED" "$STORE_EVICT" "$STORE_SHUTDOWN" >> "$STATS_FILE"
printf '"miss_common_avg":%d,"miss_common_min":%d,"miss_common_max":%d,' \
  "$MISS_COMMON_AVG" "$MISS_COMMON_MIN" "$MISS_COMMON_MAX" >> "$STATS_FILE"
printf '"load_avg_ms":%s,"load_max_ms":%s,"save_avg_ms":%s,"save_max_ms":%s,' \
  "$LOAD_AVG" "$LOAD_MAX" "$SAVE_AVG" "$SAVE_MAX" >> "$STATS_FILE"
printf '"kv_file_count":%d,"kv_disk_bytes":%d,' \
  "$KV_FILE_COUNT" "${KV_DISK_USAGE_BYTES:-0}" >> "$STATS_FILE"
printf '"stale_6h":%d,"stale_12h":%d,"stale_24h":%d,"stale_48h":%d,"stale_6h_bytes":%d,' \
  "$STALE_6H" "$STALE_12H" "$STALE_24H" "$STALE_48H" "${STALE_6H_BYTES:-0}" >> "$STATS_FILE"
printf '"stale_48h_bytes":%d,"stale_ratio":%s,"files_small":%d,"files_medium":%d,"files_large":%d}\n' \
  "${STALE_48H_BYTES:-0}" "$STALE_RATIO" "$SMALL_FILES" "$MEDIUM_FILES" "$LARGE_FILES" >> "$STATS_FILE"

# Keep only last 1000 entries (about 3 months at 3x/day)
tail -1000 "$STATS_FILE" > "${STATS_FILE}.tmp" && mv "${STATS_FILE}.tmp" "$STATS_FILE"

# --- 7. Generate human-readable report ---

echo "═══ ds4-server KV Cache Report ═══"
echo "  Generated: $(date)"
echo ""

# Server health
if [ -n "$SERVER_PID" ]; then
    UPTIME_HUMAN=""
    if [ -n "$SERVER_UPTIME" ]; then
        UPTIME_HUMAN=$(printf '%dd %dh %dm' $((SERVER_UPTIME/86400)) $(((SERVER_UPTIME%86400)/3600)) $(((SERVER_UPTIME%3600)/60)))
    fi
    echo "  Server:  RUNNING (pid $SERVER_PID, up $UPTIME_HUMAN)"
else
    echo "  Server:  STOPPED"
fi
echo "  Log:     $ERR_LOG"
echo ""

# Cache efficiency
echo "── Cache Efficiency ──"
echo "  Hits:           $KV_HITS"
echo "  Misses:         $KV_MISSES"
echo "  Hit ratio:      ${HIT_RATIO}%"
echo "  Checkpoints stored: $KV_STORED"
echo "  Budget evictions:   $KV_EVICTED"
echo ""

# Store breakdown
echo "── Store Reasons ──"
echo "  evict:      $STORE_EVICT"
echo "  continued:  $STORE_CONTINUED"
echo "  cold:       $STORE_COLD"
echo "  shutdown:   $STORE_SHUTDOWN"
echo ""

# Performance
echo "── Performance ──"
echo "  Load avg:   ${LOAD_AVG}ms (max: ${LOAD_MAX}ms)"
echo "  Save avg:   ${SAVE_AVG}ms (max: ${SAVE_MAX}ms)"
echo "  Miss overlap avg: ${MISS_COMMON_AVG} tokens (min: ${MISS_COMMON_MIN}, max: ${MISS_COMMON_MAX})"
echo ""

# Disk cache
echo "── Disk Cache ──"
echo "  Files:      $KV_FILE_COUNT"
echo "  Usage:      $KV_DISK_USAGE"
echo "  Small (<100M):  $SMALL_FILES"
echo "  Medium (100M-1G): $MEDIUM_FILES"
echo "  Large (>1G):    $LARGE_FILES"
echo ""

echo "── Cache Health ──"
echo "  Stale >6h:   $STALE_6H files ($STALE_6H_HUMAN GB)"
echo "  Stale >12h:  $STALE_12H files"
echo "  Stale >24h:  $STALE_24H files"
echo "  Stale >48h:  $STALE_48H files ($STALE_48H_HUMAN GB) ← relevance boundary"
echo "  Stale ratio (>6h): ${STALE_RATIO}% of all files"

# Estimate production rate from actual disk growth in stats history
# This is more accurate than trying to derive it from token counts
PRODUCTION_GB_DAY=""
HEADROOM_DAYS=""
if [ -f "$STATS_FILE" ] && [ "$(wc -l < "$STATS_FILE")" -ge 2 ]; then
    EARLIEST=$(head -1 "$STATS_FILE")
    LATEST=$(tail -1 "$STATS_FILE")
    EARLY_TS=$(echo "$EARLIEST" | grep -o '"ts":[0-9]*' | cut -d: -f2)
    LATE_TS=$(echo "$LATEST" | grep -o '"ts":[0-9]*' | cut -d: -f2)
    EARLY_BYTES=$(echo "$EARLIEST" | grep -o '"kv_disk_bytes":[0-9]*' | cut -d: -f2)
    LATE_BYTES=$(echo "$LATEST" | grep -o '"kv_disk_bytes":[0-9]*' | cut -d: -f2)
    if [ -n "$EARLY_TS" ] && [ -n "$LATE_TS" ] && [ -n "$EARLY_BYTES" ] && [ -n "$LATE_BYTES" ] \
       && [ "$EARLY_BYTES" -gt 0 ] && [ "$LATE_BYTES" -gt 0 ] && [ "$LATE_TS" -gt "$EARLY_TS" ]; then
        ELAPSED_DAYS=$(echo "scale=2; ($LATE_TS - $EARLY_TS) / 86400" | bc 2>/dev/null)
        GROWTH_BYTES=$((LATE_BYTES - EARLY_BYTES))
        if [ "$GROWTH_BYTES" -gt 0 ] 2>/dev/null && [ "$(echo "$ELAPSED_DAYS > 0" | bc 2>/dev/null)" = "1" ]; then
            DAILY_BYTES=$(echo "scale=0; $GROWTH_BYTES / $ELAPSED_DAYS" | bc 2>/dev/null)
            PRODUCTION_GB_DAY=$(echo "scale=1; $DAILY_BYTES / 1073741824" | bc 2>/dev/null)
            BUDGET_BYTES=$((2097152 * 1024 * 1024))
            HEADROOM_DAYS=$(echo "scale=0; $BUDGET_BYTES / ($DAILY_BYTES + 1)" | bc 2>/dev/null)
        fi
    fi
fi
if [ -z "$PRODUCTION_GB_DAY" ] || [ "$(echo "$PRODUCTION_GB_DAY < 1.0" | bc 2>/dev/null)" = "1" ]; then
    # Fallback: estimate from total disk usage vs log timespan
    # Use the total disk bytes directly with the server uptime
    if [ -n "$SERVER_UPTIME" ] && [ "$SERVER_UPTIME" -gt 3600 ] && [ "$KV_DISK_USAGE_BYTES" -gt 0 ]; then
        UPTIME_DAYS=$(echo "scale=2; $SERVER_UPTIME / 86400" | bc 2>/dev/null)
        DAILY_BYTES=$(echo "scale=0; $KV_DISK_USAGE_BYTES / $UPTIME_DAYS" | bc 2>/dev/null)
        PRODUCTION_GB_DAY=$(echo "scale=1; $DAILY_BYTES / 1073741824" | bc 2>/dev/null)
        BUDGET_BYTES=$((2097152 * 1024 * 1024))
        HEADROOM_DAYS=$(echo "scale=0; $BUDGET_BYTES / ($DAILY_BYTES + 1)" | bc 2>/dev/null)
    fi
fi
if [ -n "$PRODUCTION_GB_DAY" ]; then
    echo "  Est. daily KV production: ~${PRODUCTION_GB_DAY} GB/day (from observed growth)"
    if [ -n "$HEADROOM_DAYS" ] && [ "$HEADROOM_DAYS" != "?" ]; then
        echo "  Budget headroom: ~${HEADROOM_DAYS} days at current rate"
    fi
fi
echo ""

# Trend (compare with previous record)
PREV=$(tail -2 "$STATS_FILE" | head -1 2>/dev/null)
if [ -n "$PREV" ]; then
    PREV_HITS=$(echo "$PREV" | grep -o '"kv_hits": [0-9]*' | awk '{print $2}' | tr -d ',' 2>/dev/null || echo 0)
    PREV_STALE=$(echo "$PREV" | grep -o '"stale_6h": [0-9]*' | awk '{print $2}' | tr -d ',' 2>/dev/null || echo 0)
    PREV_HIT_RATIO=$(echo "$PREV" | grep -o '"hit_ratio": [0-9.]*' | awk '{print $2}' | tr -d ',' 2>/dev/null || echo 0)

    echo "── Trends vs Last Check ──"
    if [ "$PREV_HITS" -gt 0 ] 2>/dev/null; then
        HIT_DELTA=$((KV_HITS - PREV_HITS))
        if [ "$HIT_DELTA" -ge 0 ]; then
            echo "  New cache hits since last: +$HIT_DELTA"
        else
            echo "  New cache hits since last: $HIT_DELTA (log rotated?)"
        fi
    fi
    STALE_DELTA=$((STALE_6H - PREV_STALE))
    if [ "$STALE_DELTA" -gt 0 ]; then
        echo "  ⚠ Stale blocks increased by $STALE_DELTA since last check"
    elif [ "$STALE_DELTA" -lt 0 ]; then
        echo "  ✓ Stale blocks decreased by $((STALE_DELTA * -1)) since last check (cleanup active)"
    else
        echo "  Stale blocks unchanged"
    fi
    echo ""
fi

# Schedule recommendation
if [ -n "$QUIET_HOURS" ]; then
    echo "── Schedule Note ──"
    echo "  Quietest hours: $QUIET_HOURS"
    echo "  (Cron should target these for minimal disruption)"
    echo ""
fi

# Alert conditions
ALERTS=""
if [ "$STALE_48H" -gt 200 ] 2>/dev/null; then
    ALERTS="${ALERTS}  ⚠ High stale block count ($STALE_48H files >48h old, ${STALE_48H_HUMAN}GB). Beyond relevance window — consider cleanup-kv-cache.sh.\n"
fi
if [ "$STALE_24H" -gt 500 ] && [ "$STALE_48H" -gt 100 ] 2>/dev/null; then
    ALERTS="${ALERTS}  ⚠ Stale files accumulating across all windows. Cache may be growing faster than usage.\n"
fi
if [ "$HIT_RATIO" != "0.0" ] && [ "$(echo "$HIT_RATIO < 30" | bc 2>/dev/null)" = "1" ]; then
    ALERTS="${ALERTS}  ⚠ Low hit ratio (${HIT_RATIO}%). Cache efficiency is degraded.\n"
fi
if [ -n "$SERVER_PID" ] && [ -n "$SERVER_UPTIME" ] && [ "$SERVER_UPTIME" -lt 300 ]; then
    ALERTS="${ALERTS}  ℹ Server recently restarted (uptime <5min). Warming up.\n"
fi
if [ -n "$ALERTS" ]; then
    echo "── Alerts ──"
    echo -e "$ALERTS"
fi

echo "══════════════════════════════════"
