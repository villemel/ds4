#!/bin/sh
# fleet-server.sh — single switcher for the ds4 model fleet (launchd-managed).
#
# Five models, one resident at a time (the engine's instance lock):
#   ds4f        DeepSeek V4 Flash 0731          ~/ds4-baseline  port 8006
#   ds41f       DeepSeek V4.1 Flash Q2          ~/ds4-v41       port 8005
#   glm53flash  GLM 5.3 Flash Q4_K  (workhorse) ~/ds4           port 8000
#   glm53full   GLM 5.3 full attrib-E (unsloth IQ2 pool, Q4K dense cuts)  ~/ds4-glmfull   port 8008
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
# WB STAGE-PROFILE RE-RUN (diagnostics-only one-off; decode-round close-out):
#   GLM53FULL_PROFILE=wb fleet-server.sh glm53full
# Serves the standing production set + DS4_METAL_DECODE_STAGE_PROFILE=1 for
# exactly ONE load; the next regular glm53full load restores production.
# NOT an arm queue (stood down at e309d04 and stays down) — a named one-off
# profile, no cursor, no cycling.
#
# WHY the first attempt was aborted and treated as an error (2026-10-03, the
# arm-1 window): the pre-fix instrument printed 1212 stage lines per decode
# token (~5.4 MB/min, ~50 MB spoiled) and degraded decode ~13x (18.6 -> 1.4
# t/s) because every stage boundary ran commit+wait on the forced-unfused
# path. BOTH side effects are now accounted for by the fix (f97733d):
#   1. LOG VOLUME is bounded: decode stages aggregate in-memory and print
#      ONE summary line per 8 sampled tokens ("ds4: metal decode stage
#      profile summary pos=A..B tokens=N ... part.stage=ms/tok ..."), i.e.
#      ~1 line per 128 decoded tokens at the default 1-of-16 sampling —
#      KB/min scale, not MB/min. DO NOT abort on log cadence.
#   2. DECODE SPEED lands at ~10 t/s-class (vs ~18 production): one of
#      every 16 tokens runs the instrumented unfused pass; the other 15 run
#      the fused production path. The slowdown is the instrument's price,
#      by design — NOT a regression. DO NOT abort on it.
#   The only abort criteria are the standing ones: warning/error/swap/
#   memory-pressure lines in the log.
#
# WHAT THE CENSUS MEASURES (so arm-1's read-error is not repeated): each
# stage's ms is measured on the INSTRUMENTED (boundary-synced, unfused)
# path and includes a per-boundary CPU/submit/drain overhead — absolute
# stage SHARES are therefore biased toward boundary-heavy stages and are
# upper bounds only. The trustworthy signals are (a) each stage's
# ctx-SCALING (the overhead is ~ctx-independent, so per-stage growth with
# context is real work — the pricing base for C5/C6/C7-class candidates),
# and (b) the cross-check against the fused wall (the non-sampled tokens'
# decode t/s in the server's own chunk lines) which prices the instrument's
# sync tax. Prefill is NOT instrumented by this profile (the prefill-stage
# env DS4_METAL_LAYER_STAGE_PROFILE adds per-stage sync to prefill and is
# deliberately NOT set — prefill runs at production speed).
#
# OPTIONAL TUNING: DS4_METAL_DECODE_STAGE_PROFILE_SAMPLE=N (default 16;
# 32 = faster/thinner, 8 = slower/denser). DS4_DECODE_PROFILE_DETAIL is
# INERT on the Metal arm — never set it.
#
# RUN LENGTH the census needs: >=2k decoded tokens per ctx band visited,
# spanning 10k->150k+ ctx. A normal 60-90 min session at ~10 t/s-class
# generates 30-50k tokens and covers it (the ride-along vehicle). The
# controlled alternative is the fixed ctx-ladder census in one stopped
# window (~45-75 min): ~/ds4-glmfull/misc/run-glm53-wb-census.sh — see the
# runbook header for the trade-offs and the exact ladder.
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
# ENVS: per-model standing envs, space-separated KEY=VALUE, rendered into the
# plist EnvironmentVariables (launchd never reads shell profiles). All
# DS4_GLM_* knobs are presence-gated: =0 still enables; disable by removing.
ENVS=""
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
    MODEL_GGUF="$HOME/ds4/gguf/GLM-5.3-attrib-densecuts-mtpzap.gguf"
    VISION=""
    CTX=450000
    BATCHED=""
    PREFILL_CHUNK=""   # GLM full rejects --prefill-chunk (graph-selected chunks)
    MTP=""             # MTP measured at 0.47x resident decode — stays OFF
    KVDIR="$HOME/.ds4-glmfull/server-kv"
    KVSPACE=$KV_SPACE_MB_GLM53FULL
    # Weights staged 2026-10-03 (operator-directed "add E", config-only): the
    # G3' attribution leg E — the 15 dense-cut extras (Q4_K output head +
    # FFN blk.0-2) + MTP-zap; experts stay byte-exact unsloth IQ2_XXS (the
    # BF16 re-roll buys zero stream bytes and carries the suspect half of
    # the slim +5.058% avg NLL). Size exact 201,803,630,976 B = slim-exact.
    # Rollback = the untouched slim/Q4Kattn/IQ2 files.
    # Standing envs = the immutable production set (A/B arm queue stood down
    # 2026-10-04 — every arm is served; ledger: glm53-ab-queue.md). Radix +
    # wide scorer are HARDCODED release paths on the 846e639+ binary: their
    # env knobs are inert and are NOT set. The measured wide-kernel knobs
    # (EXPERT_GEMV_WIDE / ATTN_GATHER_WIDE) stay unset — both priced at or
    # below standing at every matched window (c3 -0.5..-1.4%, c4 -5..-7%,
    # c34 additive-no-rescue). SCRATCH 8192 = one-pass prefill scoring at
    # ctx 450k (planned ~239.8 GiB; trim CTX ~0.089 GiB/1k, never the
    # scratch). CACHE_ARENA / PREFILL_FLUSH_INTERVAL / DECODE_FLUSH_INTERVAL
    # stay unset (all banked neutral). The banner prints binary_sha256 for
    # era-matching (parser: ~/ds4-glmfull/misc/glm53-ab-parser.py).
    ENVS="DS4_GLM_INDEXED_PREFILL_SCORE_SCRATCH_MB=8192"
    # One-off WB census profile (header WB block above has the full
    # instructions; NOT an arm queue — that machinery stays stood down):
    #   GLM53FULL_PROFILE=wb fleet-server.sh glm53full
    # serves standing + DS4_METAL_DECODE_STAGE_PROFILE=1 for exactly ONE
    # load; any other value (or unset) serves the immutable production set.
    case "${GLM53FULL_PROFILE:-prod}" in
    wb) ENVS="$ENVS DS4_METAL_DECODE_STAGE_PROFILE=1" ;;
    *)  ;;
    esac
    if [ "${GLM53FULL_PROFILE:-prod}" = wb ]; then
        echo "glm53full profile: WB census — decode stage profile ON (~10 t/s-class decode, bounded log cadence; do NOT abort on either — see the header WB block)"
    fi
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
    ENV_XML=$(for _kv in $ENVS; do
        [ -n "$_kv" ] || continue
        _k="${_kv%%=*}" ; _v="${_kv#*=}"
        printf '\t\t<key>%s</key>\n\t\t<string>%s</string>\n' "$_k" "$_v"
    done)

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
$ENV_XML	</dict>
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
