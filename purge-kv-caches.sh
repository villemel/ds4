#!/bin/sh
# One stale-checkpoint purge pass over every ds4 KV store on this machine.
#
# Each worktree runs its own server with its own --kv-disk-dir and its own
# --kv-disk-space-mb. Those budgets are per-instance: the engine never adds them
# up, so configured in isolation they are simply cumulative, and the sum can
# exceed the free space on the volume long before any single server sees budget
# pressure. Retention is therefore the fleet-level control, and this script is
# what enforces it: purge every store by age, then print the budget sum against
# the envelope so configuration drift shows up in the log instead of in a full
# disk. (The eventual fix is hot-cache-only, which removes the disk store.)
#
# Driven by ~/Library/LaunchAgents/com.ds4.kv-cache-cleanup.plist, every 12h.
# ~/.ds4/kvcache is deliberately NOT purged: it holds /save session payloads,
# not checkpoints.
#
# Usage: purge-kv-caches.sh [--dry-run]
set -eu

CLEANUP="$(dirname "$0")/cleanup-kv-cache.sh"
DRY_RUN="${1:-}"

# Ceiling for the sum of the per-instance budgets. The volume has ~2.1 TiB free
# with the KV fleet empty; weights, datasets and backups own the rest. Raise it
# only after re-checking `df -k`.
ENVELOPE_MB=1572864   # 1.5 TiB

# hours  kv-dir                                    where that instance declares its budget
TABLE="
72 $HOME/.ds4/server-kv      $HOME/ds4/run-server.sh
48 $HOME/.ds4-v41/server-kv  $HOME/ds4-v41/run-server.sh
48 $HOME/.ds4-qfn/server-kv  $HOME/ds4-qfn/run-server.sh
"

# Read the budget from the launcher rather than restating it here: KV_SPACE_MB=
# in run-server.sh, or the value following --kv-disk-space-mb in a launchd
# plist. A copy of the number in this file would drift from what the servers
# actually enforce.
budget_mb() {
    { sed -n 's/^KV_SPACE_MB=\([0-9][0-9]*\).*/\1/p' "$1" 2>/dev/null
      awk '/--kv-disk-space-mb/ { f = 1; next }
           f && /<string>/ { gsub(/[^0-9]/, ""); if ($0 != "") { print; exit } }' "$1" 2>/dev/null
    } | head -1
}

total_budget_mb=0
total_used_mb=0
problems=0

while read -r hours dir cfg _; do
    [ -n "${dir:-}" ] || continue
    mb=$(du -sk "$dir" 2>/dev/null | cut -f1 | awk '{print int($1/1024)}')
    mb=${mb:-0}   # a store that has not run yet has no directory; report 0, keep going
    files=$(find "$dir" -maxdepth 1 -name '*.kv' -type f 2>/dev/null | wc -l | tr -d ' ')
    budget=$(budget_mb "$cfg")
    total_used_mb=$((total_used_mb + mb))
    if [ -z "$budget" ]; then
        echo "WARN  $dir: no --kv-disk-space-mb found in $cfg (unbounded?)"
        budget=0
        problems=$((problems + 1))
    fi
    total_budget_mb=$((total_budget_mb + budget))
    printf '%-30s %6s GiB used of %6s GiB budget, %3s files, purge >%sh\n' \
        "$dir" "$((mb / 1024))" "$((budget / 1024))" "$files" "$hours"

    status=0
    "$CLEANUP" "$dir" "$hours" $DRY_RUN || status=$?
    if [ "$status" != "0" ]; then
        echo "WARN  purge failed for $dir (exit $status), continuing"
        problems=$((problems + 1))
    fi
done <<EOF
$TABLE
EOF

free_gib=$(df -k "$HOME" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')
echo "Fleet: $((total_used_mb / 1024)) GiB used, budgets sum to $((total_budget_mb / 1024)) GiB (envelope $((ENVELOPE_MB / 1024)) GiB), volume free ${free_gib} GiB"
if [ "$total_budget_mb" -gt "$ENVELOPE_MB" ]; then
    echo "WARN  per-instance budgets exceed the envelope: they are cumulative on one disk"
    problems=$((problems + 1))
fi
if [ "$problems" != "0" ]; then
    echo "$problems problem(s) detected"
    exit 1
fi
