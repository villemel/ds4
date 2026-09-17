#!/bin/sh
# Launch/stop/status for the Qwen3.8 Flash Next Q4 trial server.
#
# Trial setup: lives in the ds4-qfn worktree (branch try-qwen38) so it never
# touches the GLM deployment in /Users/max/ds4, the V4.1 trial in
# /Users/max/ds4-v41, nor the V4 trial in /Users/max/ds4-v4.
#
# Q4 main/MTP weights (69.74 GiB) run fully resident on this 256 GB machine --
# no --ssd-streaming. The Qwen GGUF keeps its original BF16 n-grams on disk
# (the runtime reads the selected rows directly from the file). Built-in MTP
# is enabled with --mtp. The native Qwen context is 262144 tokens.
#
# The "SSD KV cache" is the disk KV checkpoint store (--kv-disk-dir), kept in
# its own directory so Qwen checkpoints (which include recurrent state) can
# never be selected by another server's cache.
#
# Usage: ./run-server.sh [start|stop|status]   (default: start)

set -eu
cd "$(dirname "$0")"

PORT=8007
PIDFILE="/tmp/ds4-qfn-server.pid"
LOGDIR="$HOME/Library/Logs/ds4-server-qfn"
KVDIR="$HOME/.ds4-qfn/server-kv"
MODEL="gguf/Qwen3.8-Flash-Next-Q4.gguf"

# Separate KV store from the other servers: different model, different
# routed-expert quant -- sharing a dir would invite cross-model checkpoint
# loads. Budgets are per-instance and therefore cumulative across the fleet, so
# this trial takes 192 GiB (it is the hottest producer at ~20 GiB/h, but it is a
# trial: 48h retention purged by ds4/purge-kv-caches.sh, not deep resume
# history). The four stores must sum under 1.5 TiB -- see that script. If this
# instance takes over the workhorse role, give it the GLM store's 1 TiB / 72h
# instead of raising both.
KV_SPACE_MB=196608

start() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "ds4-qfn server already running (pid $(cat "$PIDFILE"))"
        exit 1
    fi
    # Instance lock: one huge model process at a time. The GLM server
    # (com.ds4-glm5.3-flash.server, port 8000) holds it while running -- stop it first.
    if pgrep -x ds4-server >/dev/null 2>&1; then
        echo "ERROR: another ds4-server is running (pgrep -x ds4-server):"
        pgrep -lx ds4-server
        echo "Stop it first: launchctl bootout gui/$(id -u)/com.ds4-glm5.3-flash.server"
        echo "Two resident huge-model processes are not supported."
        exit 1
    fi
    mkdir -p "$KVDIR" "$LOGDIR"
    nohup ./ds4-server \
        --metal \
        -m "$MODEL" \
        --mtp \
        --ctx 262144 \
        --tokens 384000 \
        --threads 24 \
        --power 100 \
        --host 0.0.0.0 \
        --port "$PORT" \
        --batched-session 4 \
        --prefill-chunk 8192 \
        --kv-disk-dir "$KVDIR" \
        --kv-disk-space-mb "$KV_SPACE_MB" \
        >> "$LOGDIR/out.log" 2>> "$LOGDIR/err.log" &
    echo $! > "$PIDFILE"
    echo "ds4-qfn server starting on port $PORT (pid $(cat "$PIDFILE"))"
    echo "logs: $LOGDIR/{out,err}.log   kv cache: $KVDIR"
}

stop() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        kill "$(cat "$PIDFILE")"
        rm -f "$PIDFILE"
        echo "ds4-qfn server stopped"
    else
        rm -f "$PIDFILE"
        echo "ds4-qfn server not running"
    fi
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "running (pid $(cat "$PIDFILE"), port $PORT)"
    else
        echo "not running"
    fi
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) echo "usage: $0 [start|stop|status]"; exit 2 ;;
esac
