#!/bin/sh
# fleet-server.sh — single switcher for the ds4 model fleet (launchd-managed).
#
# Five models, one resident at a time (the engine's instance lock):
#   ds4f        DeepSeek V4 Flash 0731          ~/ds4-baseline  port 8006
#   ds41f       DeepSeek V4.1 Flash Q2          ~/ds4-v41       port 8005
#   glm53flash  GLM 5.3 Flash Q4_K  (workhorse) ~/ds4           port 8000
#   glm53full   GLM 5.3 full slim-speed (BF16src IQ2 pool, Q4K dense)  ~/ds4-glmfull   port 8008
#   qwen38fn    Qwen3.8 Flash Next Q4           ~/ds4-qfn       port 8007
#
# What it does: identifies the running ds4-server (its launchd label via the
# pid; legacy nohup instances are killed), boots the job out, waits for the
# process AND its wired memory to go away, generates the target plist
# (~/ds4/fleet/<label>.plist), and loads it. The script runs and terminates;
# launchd owns the server afterwards.
#
# At login only ~/Library/LaunchAgents/com.ds4-glm53-flash.server.plist
# auto-starts: glm53flash stays the boot workhorse. Every other plist lives
# in ~/ds4/fleet/ and is runtime-only, so a reboot always comes back to the
# workhorse and never tries to resident two huge models at once.
#
# SSD cold KV cache is OFF by default for every model. Pass --kv to add
# --kv-disk-dir/--kv-disk-space-mb for that model's own store. The budgets
# below (KV_SPACE_MB_*) sum to the 1.5 TiB fleet envelope and are read back
# by ~/ds4/purge-kv-caches.sh (launchd com.ds4.kv-cache-cleanup, every 12h;
# 72h workhorse retention, 48h trials). Do not raise a budget without
# re-checking the envelope math there.
#
# Usage:
#   fleet-server.sh <ds4f|ds41f|glm53flash|glm53full|qwen38fn> [--kv]
#   fleet-server.sh status
#   fleet-server.sh stop      # halt whatever is resident: boots out every
#                             # loaded ds4 job (never plain-kill — KeepAlive
#                             # would relaunch) and kills legacy nohup procs.
#                             # Stays down until the next login, when the
#                             # glm53flash LaunchAgent re-loads via RunAtLoad.
#   DS4_FLEET_DRYRUN=1 fleet-server.sh <model> [--kv]  # write plists, print actions, no launchctl
#   DS4_FLEET_SLEEP=<sec>                              # minimum settle sleep after bootout (default 8)
#
# Pitfalls encoded here (see the ds4-server-fleet-ops skill):
#   - pgrep -x, never pgrep -f ("less ds4-server.log" must not match).
#   - The new server takes 45-90s to map its weights; "loaded" means launchd
#     has the job, not that the HTTP port answers yet.

set -eu

GUI="gui/$(id -u)"
FLEET_DIR="$HOME/ds4/fleet"
MIN_SLEEP="${DS4_FLEET_SLEEP:-8}"
DRYRUN="${DS4_FLEET_DRYRUN:-0}"
WAIT_TIMEOUT=180

# Symmetric fleet geometry (identical for every model).
THREADS=24
POWER=100
HOST=192.168.1.232
TOKENS=384000
THROTTLE=4

# One log file for the whole fleet (rotated daily 03:00 by ~/ds4/rotate-logs.sh,
# copytruncate — the models' write offsets survive, no restart needed). Which
# model a line belongs to is in the startup banner and the port lines.
LOGFILE="$HOME/Library/Logs/ds4-fleet/server.log"

# Per-model SSD KV budgets (MB). Read by purge-kv-caches.sh; the five sum to
# 1572864 MB = the 1.5 TiB envelope exactly. Keep it that way.
KV_SPACE_MB_DS4F=196608
KV_SPACE_MB_DS41F=65536
KV_SPACE_MB_GLM53FLASH=1048576
KV_SPACE_MB_GLM53FULL=65536
KV_SPACE_MB_QWEN38FN=196608

ALL_LABELS="com.ds4-glm5.3-flash.server com.ds4-glm53-full.server com.ds4-ds4f.server com.ds4-v41.server com.ds4-qfn.server"

do_run() {
    if [ "$DRYRUN" = "1" ]; then echo "DRYRUN + $*"; else "$@"; fi
}

# ---- status ----------------------------------------------------------------
status_all() {
    echo "launchd jobs:"
    for L in $ALL_LABELS; do
        info=$(launchctl print "$GUI/$L" 2>/dev/null | grep -E '^\s(state|pid) =' | tr -s ' \t' ' ' | tr '\n' ' ' || true)
        if [ -n "$info" ]; then echo "  $L  $info"; else echo "  $L  not loaded"; fi
    done
    echo "processes:"
    pgrep -lx ds4-server || echo "  no ds4-server process"
    echo "listening ports:"
    lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk '/ds4-serve/ {print "  pid "$2" -> "$9}' || true
}

# ---- stop ------------------------------------------------------------------
# Halt the fleet regardless of which weights are resident. Bootout is the only
# correct front end: KeepAlive(SuccessfulExit=false) relaunches a job whose
# process merely dies, so killing without bootout restart-loops the server.
stop_all() {
    booted=""
    for L in $ALL_LABELS; do
        if launchctl print "$GUI/$L" >/dev/null 2>&1; then
            do_run launchctl bootout "$GUI/$L"
            booted="$booted $L"
        fi
    done
    for pid in $(pgrep -x ds4-server || true); do
        if [ "$DRYRUN" != "1" ]; then
            echo "killing legacy (non-launchd) ds4-server pid $pid"
            kill "$pid" 2>/dev/null || true
        else
            echo "DRYRUN + kill legacy ds4-server pid $pid"
        fi
    done
    rm -f /tmp/ds4-v41-server.pid /tmp/ds4-qfn-server.pid

    if [ "$DRYRUN" = "1" ]; then
        echo "DRYRUN: would boot out:$booted"
        return 0
    fi
    if [ -z "$booted" ] && ! pgrep -x ds4-server >/dev/null 2>&1; then
        echo "nothing to stop: no ds4 launchd job loaded and no ds4-server process"
        return 0
    fi
    # Same settle/wait as the switch path: launchd SIGTERMs the booted job and
    # the huge wired mapping takes seconds to leave.
    sleep "$MIN_SLEEP"
    waited=$MIN_SLEEP
    while pgrep -x ds4-server >/dev/null 2>&1; do
        [ "$waited" -ge "$WAIT_TIMEOUT" ] && {
            echo "ERROR: ds4-server still alive after ${waited}s — the job is booted out (it will not be relaunched) but the process is not gone" >&2
            exit 1
        }
        sleep 2
        waited=$((waited + 2))
    done
    echo "stopped:$booted"
    echo "no ds4-server process; the glm53flash workhorse returns at next login (RunAtLoad)"
}

MODEL="${1:-}"
if [ "$MODEL" = "status" ]; then status_all; exit 0; fi
if [ "$MODEL" = "stop" ]; then stop_all; exit 0; fi

WANT_KV=0
[ "${2:-}" = "--kv" ] && WANT_KV=1
[ "${2:-}" = "--kv" ] || [ $# -le 1 ] || { echo "usage: $0 <model> [--kv]  |  $0 status  |  $0 stop" >&2; exit 2; }

# ---- per-model configuration ----------------------------------------------
# Kept exactly as each model is run today; only the management layer (launchd,
# host binding, log/KV policy) is symmetric across the fleet.
case "$MODEL" in
ds4f)
    LABEL=com.ds4-ds4f.server
    PORT=8006
    BIN="$HOME/ds4-baseline/ds4-server"
    WD="$HOME/ds4-baseline"
    MODEL_GGUF="$HOME/ds4/gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf"
    VISION=""
    CTX=1048576
    BATCHED=4
    PREFILL_CHUNK=8192
    MTP=""
    KVDIR="$HOME/.ds4-v4/server-kv"
    KVSPACE=$KV_SPACE_MB_DS4F
    ;;
ds41f)
    LABEL=com.ds4-v41.server
    PORT=8005
    BIN="$HOME/ds4-v41/ds4-server"
    WD="$HOME/ds4-v41"
    MODEL_GGUF="$HOME/ds4-v41/gguf/DeepSeek-V4.1-Flash-Q2.gguf"
    VISION=""
    CTX=1048576
    BATCHED=3
    PREFILL_CHUNK=8192
    MTP=""
    KVDIR="$HOME/.ds4-v41/server-kv"
    KVSPACE=$KV_SPACE_MB_DS41F
    ;;
glm53flash)
    LABEL=com.ds4-glm5.3-flash.server
    PORT=8000
    BIN="$HOME/ds4/ds4-server"
    WD="$HOME/ds4"
    MODEL_GGUF="$HOME/ds4/gguf/GLM-5.3-Flash-Q4_K.gguf"
    VISION="$HOME/ds4/gguf/GLM-5.3-Flash-Vision-Encoder.gguf"
    CTX=1048576
    BATCHED=3
    PREFILL_CHUNK=""
    MTP=""
    KVDIR="$HOME/.ds4/server-kv"
    KVSPACE=$KV_SPACE_MB_GLM53FLASH
    ;;
glm53full)
    LABEL=com.ds4-glm53-full.server
    PORT=8008
    BIN="$HOME/ds4-glmfull/ds4-server"
    WD="$HOME/ds4-glmfull"
    MODEL_GGUF="$HOME/ds4/gguf/GLM-5.3-BF16src-RoutedIQ2XXS-blk78IQ2XXS-Q4Kattn-shexp-Q4Kslim.gguf"
    VISION=""
    CTX=450000
    BATCHED=""
    PREFILL_CHUNK=""   # GLM full rejects --prefill-chunk (graph-selected chunks)
    MTP=""             # MTP measured at 0.47x resident decode — stays OFF
    KVDIR="$HOME/.ds4-glmfull/server-kv"
    KVSPACE=$KV_SPACE_MB_GLM53FULL
    ;;
qwen38fn)
    LABEL=com.ds4-qfn.server
    PORT=8007
    BIN="$HOME/ds4-qfn/ds4-server"
    WD="$HOME/ds4-qfn"
    MODEL_GGUF="$HOME/ds4-qfn/gguf/Qwen3.8-Flash-Next-Q4.gguf"
    VISION=""
    CTX=262144
    BATCHED=4
    PREFILL_CHUNK=8192
    MTP=1
    KVDIR="$HOME/.ds4-qfn/server-kv"
    KVSPACE=$KV_SPACE_MB_QWEN38FN
    ;;
status)
    ;;
"")
    echo "usage: $0 <ds4f|ds41f|glm53flash|glm53full|qwen38fn> [--kv]" >&2
    echo "       $0 status  |  $0 stop" >&2
    exit 2
    ;;
*)
    echo "unknown model: $MODEL (known: ds4f ds41f glm53flash glm53full qwen38fn)" >&2
    exit 2
    ;;
esac

PLIST_PATH="$FLEET_DIR/$LABEL.plist"

# ---- plist generation ------------------------------------------------------
write_plist() {
    mkdir -p "$FLEET_DIR"
    ARGS="$BIN
--metal
-m
$MODEL_GGUF"
    if [ -n "$VISION" ]; then
        ARGS="$ARGS
--vision
$VISION"
    fi
    ARGS="$ARGS
--ctx
$CTX
--tokens
$TOKENS
--threads
$THREADS
--power
$POWER
--host
$HOST
--port
$PORT"
    if [ -n "$BATCHED" ]; then
        ARGS="$ARGS
--batched-session
$BATCHED"
    fi
    if [ -n "$PREFILL_CHUNK" ]; then
        ARGS="$ARGS
--prefill-chunk
$PREFILL_CHUNK"
    fi
    if [ -n "$MTP" ]; then
        ARGS="$ARGS
--mtp"
    fi
    if [ "$WANT_KV" = "1" ]; then
        mkdir -p "$KVDIR"
        ARGS="$ARGS
--kv-disk-dir
$KVDIR
--kv-disk-space-mb
$KVSPACE"
    fi

    ARG_XML=$(printf '%s\n' "$ARGS" | awk '{ printf "\t\t<string>%s</string>\n", $0 }')

    cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
$ARG_XML	</array>
	<key>WorkingDirectory</key>
	<string>$WD</string>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key>
		<false/>
	</dict>
	<key>ThrottleInterval</key>
	<integer>$THROTTLE</integer>
	<key>StandardOutPath</key>
	<string>$LOGFILE</string>
	<key>StandardErrorPath</key>
	<string>$LOGFILE</string>
	<key>EnvironmentVariables</key>
	<dict>
		<key>HOME</key>
		<string>$HOME</string>
	</dict>
</dict>
</plist>
EOF
    mkdir -p "$(dirname "$LOGFILE")"
    plutil -lint "$PLIST_PATH" >/dev/null   # a broken plist must abort before bootout
}

# ---- switch ----------------------------------------------------------------
[ -x "$BIN" ] || { echo "ERROR: $BIN not executable — build the worktree the plist points at (make ds4-server)" >&2; exit 1; }
[ -f "$MODEL_GGUF" ] || { echo "ERROR: model file missing: $MODEL_GGUF" >&2; exit 1; }

# Generate the target plist FIRST so a generation failure aborts before the
# running server is touched.
write_plist
KVSTATE="no ssd kv cache"
[ "$WANT_KV" = "1" ] && KVSTATE="ssd kv cache: $KVDIR (budget ${KVSPACE} MB)"

# Boot out EVERY loaded ds4 server job (at most one should be loaded) and kill
# any legacy nohup instance.
booted=""
for L in $ALL_LABELS; do
    if launchctl print "$GUI/$L" >/dev/null 2>&1; then
        do_run launchctl bootout "$GUI/$L"
        booted="$booted $L"
    fi
done
for pid in $(pgrep -x ds4-server || true); do
    if [ "$DRYRUN" != "1" ]; then
        echo "killing legacy (non-launchd) ds4-server pid $pid"
        kill "$pid" 2>/dev/null || true
    else
        echo "DRYRUN + kill legacy ds4-server pid $pid"
    fi
done
rm -f /tmp/ds4-v41-server.pid /tmp/ds4-qfn-server.pid

if [ "$DRYRUN" != "1" ]; then
    if [ -n "$booted" ]; then
        # Minimum settle (the proven 5-12s window), then wait until the old
        # process is truly gone — two resident huge models are not supported.
        sleep "$MIN_SLEEP"
        waited=$MIN_SLEEP
        while pgrep -x ds4-server >/dev/null 2>&1; do
            [ "$waited" -ge "$WAIT_TIMEOUT" ] && {
                echo "ERROR: old ds4-server still alive after ${waited}s — aborting, not loading over it" >&2
                exit 1
            }
            sleep 2
            waited=$((waited + 2))
        done
    fi
    do_run launchctl load "$PLIST_PATH"
fi

if [ "$DRYRUN" != "1" ]; then
    launchctl print "$GUI/$LABEL" >/dev/null 2>&1 && \
        echo "loaded $LABEL (port $PORT, $KVSTATE)" || \
        echo "WARNING: $LABEL not visible in launchctl print after load" >&2
    echo "weights take 45-90s to map; watch $LOGFILE for the startup banner"
else
    echo "DRYRUN: plist written at $PLIST_PATH ($KVSTATE)"
fi
